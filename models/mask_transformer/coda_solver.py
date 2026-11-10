"""CODA: the damped Gauss-Newton / Levenberg-Marquardt anchor solver.

Extracted VERBATIM (2026-07-19) from the GN branch of
`control_transformer_t_concat_v8_kv_v4_noqres.py` (lines 1031-1293) so a second
backbone can run the identical solver. The T-Concat class still carries its own
inline copy and is intentionally NOT rewired here: it produces every headline
number in the paper, so it is left byte-for-byte untouched. If the two ever
need to diverge, fix this file and the inline copy together.

The solver operates purely on the POST-transformer per-codebook embeddings
(`emb_list`: Q tensors of [B, C, T_tok]) and the VQ decode path, so it is
independent of how the transformer lays out its attention slots -- which is the
only thing that differs between the T-Concat and C-Concat backbones.
"""

import torch
import torch.nn.functional as F  # noqa: F401  (used by the extracted body)

from utils.motion_process import recover_from_ric


def coda_solve(emb_optim_list, emb_list, vq_model, global_joint,
               global_joint_mask, _mean, _std, control_opt, Q, joints_num):
    """Run the CODA solve in place on `emb_optim_list` (leaves are mutated).

    All arguments mirror the names used at the original call site so the
    extracted body below is unmodified apart from `self.Q` -> `Q` and
    `self.opt.joints_num` -> `joints_num`.
    """
    # Sequence-aware CODA dispatch (feature flag; design doc
    # .codex-research/plan/20260723_seq_aware_coda_design.md): delegate when
    # control_opt["gn"]["seq"] carries a non-zero weight; baseline unchanged.
    _seq_probe = (((control_opt or {}).get("gn", {}) or {}).get("seq")
                  or {}) if control_opt else {}
    if any(float(_seq_probe.get(k, 0.0)) > 0.0
           for k in ("lam_base", "lam_vel", "lam_acc", "lam_contact")):
        from models.mask_transformer.coda_solver_seq import coda_solve_seq
        return coda_solve_seq(emb_optim_list, emb_list, vq_model, global_joint,
                              global_joint_mask, _mean, _std, control_opt, Q,
                              joints_num)

    # ---- Gauss-Newton / Levenberg-Marquardt REPLACEMENT for Stage-2 ----
    # The Stage-2 objective is a small least-squares problem: the residual lives in
    # R^{3A} (A = controlled anchor-frames; 15 dims for 5-anchor control) while the
    # variables are the ~37.6k embedding entries. Solve in the DUAL space:
    #     delta = -J^T (J J^T + lambda I)^{-1} r
    # (minimum-norm LM step; the linear system is 3A x 3A — trivially small).
    # Each LM step costs 3A backward passes for the Jacobian rows + 1 decode.
    # K<=8 steps replace the 600-iteration Adam loop. The Jacobian is per-sample,
    # so batch mode loops samples AFTER the (batched) Stage-1 — this is the
    # "batched-S1 + per-sample-GN" configuration: Stage-1 keeps its batch
    # amortization and stays on-distribution (no batch=1 logit overfitting),
    # GN only performs the small finishing moves it is FID-safe for.
    _gn_cfg = (control_opt or {}).get("gn", {}) if control_opt else {}
    _gn_steps = int(_gn_cfg.get("max_steps", 8))
    _gn_tau = float(_gn_cfg.get("tau", 1e-6))       # target anchor MSE (m^2)
    _gn_lam0 = float(_gn_cfg.get("lam0", 1e-2))
    _gn_amax = int(_gn_cfg.get("anchor_cap", 48))   # Jacobian row budget
    # prox_beta > 0: proximal acceptance — a trial must reduce anchor MSE
    # AND the combined C_beta = mse/mse0 + beta * D_Sigma(e, e0), where
    # D_Sigma is the mean per-dim Mahalanobis distance from the feed-forward
    # start e0. Controls cumulative off-manifold drift (multi-joint FID fix).
    _gn_beta = float(_gn_cfg.get("prox_beta", 0.0))
    # jac_chunk: rows per replicated-batch backward (0 = per-row loop).
    # DEFAULT 0: the fast path is metric-equivalent but NOT bit-equivalent
    # (conv backward float reassociation diverges the LM branch path within
    # solver noise) — published numbers stay reproducible on the row loop;
    # opt in explicitly for latency-critical runs.
    _gn_jchunk = int(_gn_cfg.get("jac_chunk", 0))

    # metric="codebook_cov": minimum-MAHALANOBIS-norm steps instead of minimum-L2.
    # Diagnosis 2026-07-13: plain min-L2 dual steps jump to the constraint surface
    # along directions the decoder never saw in training (FID 0.067->0.134), while
    # Adam's many small steps implicitly follow the loss geometry (0.067->0.083).
    # The codebook covariance Sigma_q (per quantizer, over its 128 code vectors) is
    # a training-free prior over on-manifold directions: solve
    #     (J Sigma J^T + lam*scale*I) z = r,   delta = -Sigma J^T z.
    _gn_metric = str(_gn_cfg.get("metric", "") or "")
    _sigma_q = None
    if _gn_metric == "codebook_cov":
        _gn_ridge = float(_gn_cfg.get("ridge", 0.1))
        _sigma_q = []
        for q in range(Q):
            cb = vq_model.vqvae.quantizers[q].codebook.detach().float()  # (K, C)
            c = cb - cb.mean(0, keepdim=True)
            S = (c.T @ c) / max(1, cb.shape[0] - 1)               # (C, C)
            mu = float(S.diagonal().mean().item())
            if not (mu > 0) or not torch.isfinite(S).all():       # codex P1 guard
                raise RuntimeError(f"quantizer {q}: degenerate codebook covariance "
                                   f"(mean diag {mu}); cannot build the metric")
            # ridge keeps the metric positive-definite (min eig >= ridge/(1+ridge))
            S = S + _gn_ridge * mu * torch.eye(
                S.shape[0], device=S.device, dtype=S.dtype)
            _sigma_q.append(S / S.diagonal().mean())              # scale-normalized
    elif _gn_metric:
        raise ValueError(f"unknown gn metric {_gn_metric!r}")

    _chol_q = None
    if _gn_beta > 0.0 and _sigma_q is not None:
        _chol_q = [torch.linalg.cholesky(S) for S in _sigma_q]

    def _gn_solve_one(_emb, _gj, _gjm, _tag):
        # Solve ONE sample: _emb = list of six (1, C, Tk) leaf tensors,
        # _gj/_gjm = that sample's (1, T, 22[, 3]) control tensors.
        _gn_lam = _gn_lam0
        b_idx, f_idx_full, j_idx_full = torch.where(_gjm)
        if not bool((b_idx == 0).all()):  # runtime check — python -O strips asserts
            raise RuntimeError("GN per-sample slice violated: mask beyond sample 0")
        # Dense control: the JACOBIAN is capped for cost, but acceptance / stopping /
        # reporting use the FULL mask — otherwise the solver could fit the 48
        # collocation rows while degrading the frames between them (codex P1).
        f_idx, j_idx = f_idx_full, j_idx_full
        if len(f_idx) > _gn_amax:
            sel = torch.linspace(0, len(f_idx) - 1, _gn_amax).round().long()
            f_idx, j_idx = f_idx[sel], j_idx[sel]
        _A = len(f_idx)

        def _gn_apply_sigma(vec):
            """Apply the blockwise codebook metric to a flat (P,) vector."""
            if _sigma_q is None:
                return vec
            out, off = [], 0
            for qi, e in enumerate(_emb):
                n = e.numel()
                blk = vec[off:off + n].view(e.shape)              # (1, C, Tk)
                out.append(torch.einsum("cd,bdt->bct", _sigma_q[qi], blk).reshape(-1))
                off += n
            return torch.cat(out)

        _e0 = None
        if _gn_beta > 0.0:
            _e0 = [e.detach().clone() for e in _emb]

        def _gn_prox_dist():
            # mean per-dim Mahalanobis (or L2) distance of e from the start e0
            tot, P = 0.0, 0
            for qi, (e, e0) in enumerate(zip(_emb, _e0)):
                d = (e.detach() - e0).view(e.shape[1], -1)        # (C, Tk)
                if _chol_q is not None:
                    sid = torch.cholesky_solve(d, _chol_q[qi])    # Sigma^-1 d
                else:
                    sid = d
                tot += float((d * sid).sum().item())
                P += d.numel()
            return tot / max(P, 1)

        def _gn_decode():
            pm = vq_model.vqvae.forward_decoder_from_quantized_codes(_emb)
            pm = pm.squeeze(2).permute(0, 2, 1) * _std + _mean
            return recover_from_ric(pm.float(), joints_num)

        def _gn_residual(pj):
            # (3A,) SOLVE residual (capped rows), metres
            return (pj[0, f_idx, j_idx] - _gj[0, f_idx, j_idx]).reshape(-1)

        def _gn_mse_full(pj):
            # scalar anchor MSE over the FULL mask — the accept/stop signal
            d = pj[0, f_idx_full, j_idx_full] - _gj[0, f_idx_full, j_idx_full]
            return float((d ** 2).mean().item())

        _mse0 = None   # anchor MSE at the feed-forward start (C_beta denominator)
        _C_prev = None  # last ACCEPTED combined objective
        with torch.enable_grad():
            for _gn_it in range(_gn_steps):
                pj = _gn_decode()
                r = _gn_residual(pj)
                mse = _gn_mse_full(pj)
                if _mse0 is None:
                    _mse0 = max(mse, 1e-12)
                    _C_prev = mse / _mse0        # = 1.0, D_Sigma(e0,e0)=0
                if mse < _gn_tau:
                    break
                # Jacobian rows. Fast path: REPLICATED-BATCH backward — decode a
                # batch of m identical copies of the sample, each replica selects
                # its own residual component, one backward returns m rows at once
                # through the decoder's native batch parallelism. (A vmap /
                # is_grads_batched variant fails here: the decoder backward hits
                # as_strided, which has no vmap rule.) The sequential version
                # launched 3A tiny backwards and left the GPU at ~40% util.
                _nr = 3 * _A
                J = None
                if _gn_jchunk > 0:
                    try:
                        _a_all = torch.arange(_nr, device=r.device)
                        _chunks = []
                        for _i0 in range(0, _nr, _gn_jchunk):
                            _rows = _a_all[_i0:_i0 + _gn_jchunk]
                            _m = _rows.numel()
                            _f_sel = f_idx[_rows // 3]           # (m,) frames
                            _j_sel = j_idx[_rows // 3]           # (m,) joints
                            _c_sel = _rows % 3                   # (m,) coords
                            _erep = [e.detach().repeat(_m, 1, 1)
                                      .requires_grad_(True) for e in _emb]
                            _pmr = vq_model.vqvae.forward_decoder_from_quantized_codes(_erep)
                            _pmr = _pmr.squeeze(2).permute(0, 2, 1) * _std + _mean
                            _pjr = recover_from_ric(_pmr.float(), joints_num)
                            _sc = _pjr[torch.arange(_m, device=r.device),
                                       _f_sel, _j_sel, _c_sel].sum()
                            _g = torch.autograd.grad(_sc, _erep, allow_unused=True)
                            _chunks.append(torch.cat(
                                [(gi if gi is not None else torch.zeros(
                                    (_m,) + tuple(e.shape[1:]),
                                    device=r.device, dtype=r.dtype))
                                 .reshape(_m, -1)
                                 for gi, e in zip(_g, _emb)], dim=1))
                        J = torch.cat(_chunks)                   # (3A, P)
                    except RuntimeError as _je:
                        print(f"  [Stage2 GN]{_tag} batched-J unavailable "
                              f"({str(_je)[:80]}) — per-row fallback")
                        J = None
                if J is None:
                    J_rows = []
                    for k in range(_nr):
                        grads = torch.autograd.grad(
                            r[k], _emb, retain_graph=(k < _nr - 1),
                            allow_unused=True)
                        J_rows.append(torch.cat(
                            [(g if g is not None else torch.zeros_like(e)).reshape(-1)
                             for g, e in zip(grads, _emb)]))
                    J = torch.stack(J_rows)                      # (3A, P)
                with torch.no_grad():
                    if _sigma_q is not None:
                        # one einsum per quantizer over ALL rows (codex P2: the
                        # per-row loop launched up to 864 kernels per LM step)
                        JS_parts, off = [], 0
                        for qi, e in enumerate(_emb):
                            n = e.numel()
                            blk = J[:, off:off + n].view(J.shape[0], *e.shape[1:])
                            JS_parts.append(torch.einsum(
                                "cd,mdt->mct", _sigma_q[qi], blk).reshape(J.shape[0], -1))
                            off += n
                        JS = torch.cat(JS_parts, dim=1)          # (3A, P) = J Sigma
                        JJt = J @ JS.T                            # J Sigma J^T
                    else:
                        JJt = J @ J.T
                    _scale = float(JJt.diagonal().mean().item())
                    if not (_scale > 1e-20):
                        break     # degenerate Jacobian — LM stalled (codex P1 guard)
                    eye = torch.eye(JJt.shape[0], device=JJt.device, dtype=JJt.dtype)
                    # exact-restore base — subtractive undo accumulates float drift
                    # and later lambda trials would start from a perturbed point
                    _base = [e.detach().clone() for e in _emb]
                    accepted = False
                    for _try in range(4):                        # trust region on lambda
                        for e, bb in zip(_emb, _base):
                            e.copy_(bb)
                        try:
                            z = torch.linalg.solve(JJt + _gn_lam * _scale * eye,
                                                   r.detach())
                        except Exception:
                            _gn_lam *= 10.0
                            continue
                        if not torch.isfinite(z).all():
                            _gn_lam *= 10.0
                            continue
                        delta = -_gn_apply_sigma(J.T @ z)        # (P,) [Sigma J^T z]
                        off = 0
                        for e in _emb:
                            n = e.numel()
                            e.add_(delta[off:off + n].view_as(e))
                            off += n
                        pj2 = _gn_decode()
                        mse2 = _gn_mse_full(pj2)
                        if _gn_beta > 0.0:
                            _C2 = mse2 / _mse0 + _gn_beta * _gn_prox_dist()
                            _ok = (mse2 < mse) and (_C2 < _C_prev)
                        else:
                            _ok = mse2 < mse
                        if _ok:
                            if _gn_beta > 0.0:
                                _C_prev = _C2
                            _gn_lam = max(_gn_lam * 0.5, 1e-6)
                            accepted = True
                            break
                        _gn_lam *= 10.0
                    if not accepted:
                        for e, bb in zip(_emb, _base):
                            e.copy_(bb)                          # exact restore
                        break                                    # LM stalled — stop
                print(f"  [Stage2 GN]{_tag} step {_gn_it+1}/{_gn_steps} "
                      f"full-mse {mse:.2e}->{mse2 if accepted else mse:.2e} "
                      f"lam={_gn_lam:.1e}")
        return [e.detach() for e in _emb]

    if _gn_amax <= 0:
        raise ValueError(f"gn anchor_cap must be positive, got {_gn_amax}")
    _emb_out = [e.detach().clone() for e in emb_list]
    # fix (2026-07-23): _bsz was lost in the 2026-07-19 extraction from the
    # inline copy (which defines it in its enclosing scope) -> NameError on any
    # batched call. Restore the inline copy's definition.
    _bsz = emb_list[0].shape[0]
    for _bi in range(_bsz):
        _gjm_b = global_joint_mask[_bi:_bi + 1]
        if not bool(_gjm_b.any()):
            continue                     # no anchors — keep feed-forward output
        _emb_b = [e[_bi:_bi + 1].clone().detach().contiguous().requires_grad_(True)
                  for e in emb_list]
        _tag = f" s{_bi}" if _bsz > 1 else ""
        _sol = _gn_solve_one(_emb_b, global_joint[_bi:_bi + 1], _gjm_b, _tag)
        for _qi in range(Q):
            _emb_out[_qi][_bi:_bi + 1] = _sol[_qi]
    # codex P0: the common footer below re-derives emb_list from emb_optim_list,
    # so the solutions MUST be copied back into those leaves (in place) —
    # assigning emb_list here would be silently discarded.
    with torch.no_grad():
        for _qi in range(Q):
            emb_optim_list[_qi].data.copy_(_emb_out[_qi])
