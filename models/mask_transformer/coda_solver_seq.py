"""Sequence-aware CODA-Maha: constrained correction with sequence-prior costs.

Design doc: .codex-research/plan/20260723_seq_aware_coda_design.md
Baseline reference: coda_solver.py (verbatim extraction of the inline GN branch
in control_transformer_t_concat_v8_kv_v4_noqres.py). The baseline is NOT
modified; callers dispatch here only when control_opt["gn"]["seq"] is present
with at least one non-zero weight.

Problem solved at each damped Gauss-Newton iterate u_t (u = the Q per-quantizer
continuous post-sampling embeddings, flattened to (P,)):

    min_d  1/2 d^T H d + g_soft^T d      s.t.  J d = -r(u_t)

    H      = Sigma^{-1}                                    (codebook Mahalanobis)
           + lam_base * Sigma^{-1}                         (stay near u0, same metric)
           + lam_vel  * J_phi^T D1^T D1 J_phi / N_vel      (preserve u0's velocities)
           + lam_acc  * J_phi^T D2^T D2 J_phi / N_acc      (preserve u0's accelerations)
           + lam_con  * J_phi^T D1^T M_c D1 J_phi / N_con  (zero feet velocity on
                                                            u0-detected contact frames)
    g_soft = lam_base * Sigma^{-1}(u_t - u0)
           + J_phi^T [ lam_vel D1^T s_vel/N_vel + lam_acc D2^T s_acc/N_acc
                       + lam_con D1^T M_c s_con/N_con ]

where phi(u) = decoded joints (B=1, T, 22, 3) through the SAME differentiable
chain the residual uses (vq decoder -> denorm -> recover_from_ric); D1/D2 are
first/second temporal differences; M_c gates feet joints [10, 11] on frames
where u0's decoded feet are in geometric contact (the exact definition of the
reported skate metric: height < 0.05 m in both adjacent frames); s_* are the
soft residuals at u_t. Every soft quadratic is mean-normalized by its residual
dimension count N_* so the lambda values are comparable across terms (the
2026-07-14 prox lesson: un-normalized doses are not comparable).

Dual (Schur) solve, damping and trust region mirror the baseline exactly:

    (J H^{-1} J^T + lam_damp * s * I) z = r - J H^{-1} g_soft
    d = -H^{-1} (g_soft + J^T z),   s = mean diag(J H^{-1} J^T)

With all lambdas = 0: H = Sigma^{-1}, g_soft = 0, so H^{-1} = Sigma and the
step reduces term-by-term to the baseline (J Sigma J^T + lam s I) z = r,
d = -Sigma J^T z. The regression test asserts this numerically.

H^{-1} v is computed matrix-free by preconditioned CG (preconditioner =
Sigma/(1+lam_base), exact when the observation terms vanish). Each H.v costs
one JVP + one VJP through the decode chain; no dense P x P matrix is formed.

Acceptance (priority order per the spec): a trial is accepted only if
(1) the full-mask anchor MSE strictly decreases (control = highest priority,
same rule as the baseline), and (2) a MODEL-ADEQUACY guard holds: the actual
increase of the soft objective S(u) = 1/2*lam_base*(u-u0)^T Sigma^{-1} (u-u0)
+ 1/2*||s_obs(u)||^2 may not exceed twice the magnitude of the change the QP's
own quadratic model predicted for this step (Delta_pred = g_soft^T d
+ 1/2 d^T H_soft d). Satisfying anchors NECESSARILY costs soft value — the QP
already trades that off optimally — so the guard only rejects steps whose soft
cost CLEARLY exceeds what the local model bargained for (spec: reject when the
sequence costs "明显恶化", not on any increase). With all lambdas = 0, S == 0
and the guard is vacuous: acceptance reduces exactly to the baseline rule.
Rejection grows lam_damp x10 and exactly restores the embeddings; user targets
are never modified — an unreachable/conflicting control set surfaces as a
reported stall, not a fake success. There is NO re-quantization and NO final
overwrite in this module.
"""

import torch

from utils.motion_process import recover_from_ric

# geometric contact definition — matches data_loaders/humanml/utils/metrics.py
# calculate_skating_ratio (joints 10, 11; height < 0.05 m in both frames).
_FEET_JOINTS = (10, 11)
_CONTACT_H = 0.05


def _build_sigma(vq_model, Q, ridge):
    """Per-quantizer codebook covariance metric — identical to the baseline
    (coda_solver.py L68-82): cov over the K code vectors, ridged, then
    scale-normalized to unit mean diagonal. Returns (sigma_q, chol_q)."""
    sigma_q, chol_q = [], []
    for q in range(Q):
        cb = vq_model.vqvae.quantizers[q].codebook.detach().float()   # (K, C)
        c = cb - cb.mean(0, keepdim=True)
        S = (c.T @ c) / max(1, cb.shape[0] - 1)
        mu = float(S.diagonal().mean().item())
        if not (mu > 0) or not torch.isfinite(S).all():
            raise RuntimeError(f"quantizer {q}: degenerate codebook covariance "
                               f"(mean diag {mu}); cannot build the metric")
        S = S + ridge * mu * torch.eye(S.shape[0], device=S.device, dtype=S.dtype)
        S = S / S.diagonal().mean()
        sigma_q.append(S)
        chol_q.append(torch.linalg.cholesky(S))
    return sigma_q, chol_q


class _BlockMetric:
    """Blockwise apply of Sigma and Sigma^{-1} on flat (P,) vectors, given the
    per-quantizer embedding shapes (the same (1, C, Tk) layout the solver
    mutates). Sigma acts on the C axis of every (q, t) block."""

    def __init__(self, sigma_q, chol_q, shapes):
        self.sigma_q, self.chol_q, self.shapes = sigma_q, chol_q, shapes
        self.numels = [1 * c * t for (_, c, t) in shapes]

    def _blocks(self, vec):
        out, off = [], 0
        for (b, c, t), n in zip(self.shapes, self.numels):
            yield vec[off:off + n].view(b, c, t)
            off += n

    def sigma(self, vec):
        return torch.cat([torch.einsum("cd,bdt->bct", self.sigma_q[qi], blk).reshape(-1)
                          for qi, blk in enumerate(self._blocks(vec))])

    def sigma_inv(self, vec):
        out = []
        for qi, blk in enumerate(self._blocks(vec)):
            d = blk.reshape(blk.shape[1], -1)                 # (C, Tk)
            out.append(torch.cholesky_solve(d, self.chol_q[qi]).reshape(-1))
        return torch.cat(out)

    # ---- batched (multi-RHS) twins of sigma / sigma_inv. The K column axis is
    # an independent-RHS axis: every column is bit-identical to the serial
    # single-vector path (Sigma / cholesky_solve act only on the C axis and
    # batch RHS columns order-independently).
    def sigma_batched(self, V):          # V:(P,K) -> (P,K); Sigma on the C axis per column
        K = V.shape[1]; out, off = [], 0
        for qi, ((b, c, t), n) in enumerate(zip(self.shapes, self.numels)):
            blk = V[off:off + n].reshape(c, t, K)
            out.append(torch.einsum("cd,dtk->ctk", self.sigma_q[qi], blk).reshape(n, K))
            off += n
        return torch.cat(out, dim=0)

    def sigma_inv_batched(self, V):      # V:(P,K) -> (P,K)
        K = V.shape[1]; out, off = [], 0
        for qi, ((b, c, t), n) in enumerate(zip(self.shapes, self.numels)):
            d = V[off:off + n].reshape(c, t * K)              # (C, Tk*K); column j=t*K+k
            out.append(torch.cholesky_solve(d, self.chol_q[qi]).reshape(n, K))
            off += n
        return torch.cat(out, dim=0)


def _cg(apply_A, b, apply_Minv, iters, tol):
    """Preconditioned conjugate gradient for SPD apply_A. Fails loudly on
    non-finite iterates (codex P2); returns (x, final_relative_residual)."""
    x = torch.zeros_like(b)
    r = b.clone()
    z = apply_Minv(r)
    p = z.clone()
    rz = torch.dot(r, z)
    b_norm = float(b.norm()) + 1e-30
    rel = 1.0
    for _ in range(iters):
        Ap = apply_A(p)
        pAp = torch.dot(p, Ap)
        if not torch.isfinite(pAp):
            raise RuntimeError("seq-CG: non-finite curvature p^T A p")
        alpha = rz / (pAp + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rel = float(r.norm()) / b_norm
        if not (rel == rel):   # NaN check
            raise RuntimeError("seq-CG: non-finite residual")
        if rel < tol:
            break
        z = apply_Minv(r)
        rz_new = torch.dot(r, z)
        p = z + (rz_new / (rz + 1e-30)) * p
        rz = rz_new
    return x, rel


def coda_solve_seq(emb_optim_list, emb_list, vq_model, global_joint,
                   global_joint_mask, _mean, _std, control_opt, Q, joints_num,
                   m_lens=None):
    """Drop-in variant of coda_solver.coda_solve with sequence-prior costs.

    Same contract: mutates the emb_optim_list leaves in place; no return value
    consumed by callers. Config lives in control_opt["gn"] with the baseline
    keys plus a "seq" sub-dict:
        seq = dict(lam_base=0.0, lam_vel=0.0, lam_acc=0.0, lam_contact=0.0,
                   cg_iters=32, cg_tol=1e-6, cg_block=0)

    cg_block=0 (default) => serial per-column CG (bit-reproducible headline
    path); cg_block>0 => batched chunked parallel-CG with column-chunk width
    cg_block (recommend 256; 128-512 range; larger = fewer launches but more
    decoder-activation memory). Batched conv-backward reassociates vs batch-1 so
    results match serial within solver round-off (~1e-5 rel), not bitwise — keep
    cg_block=0 for bit-exact reproducibility.
    """
    gn_cfg = (control_opt or {}).get("gn", {}) if control_opt else {}
    gn_steps = int(gn_cfg.get("max_steps", 8))
    gn_tau = float(gn_cfg.get("tau", 1e-6))
    gn_lam0 = float(gn_cfg.get("lam0", 1e-2))
    gn_amax = int(gn_cfg.get("anchor_cap", 48))
    if gn_amax <= 0:
        raise ValueError(f"gn anchor_cap must be positive, got {gn_amax}")
    metric = str(gn_cfg.get("metric", "") or "")
    if metric != "codebook_cov":
        # The seq variant is defined on top of the Mahalanobis metric (the
        # design's stated scope). Refuse loudly rather than silently assume I.
        raise ValueError(f"coda_solve_seq requires metric='codebook_cov', got {metric!r}")
    if float(gn_cfg.get("prox_beta", 0.0) or 0.0) != 0.0:
        raise ValueError("coda_solve_seq: prox_beta is not supported here (the "
                         "seq guard replaces it); remove prox_beta or the seq weights")
    seq = dict(gn_cfg.get("seq") or {})
    lam_base = float(seq.get("lam_base", 0.0))
    lam_vel = float(seq.get("lam_vel", 0.0))
    lam_acc = float(seq.get("lam_acc", 0.0))
    lam_con = float(seq.get("lam_contact", 0.0))
    cg_iters = int(seq.get("cg_iters", 32))
    cg_tol = float(seq.get("cg_tol", 1e-6))
    cg_block = int(seq.get("cg_block", 0))
    if cg_block < 0:
        raise ValueError(f"seq cg_block must be >= 0, got {cg_block}")
    import math as _math
    for k, v in (("lam_base", lam_base), ("lam_vel", lam_vel),
                 ("lam_acc", lam_acc), ("lam_contact", lam_con)):
        if not _math.isfinite(v) or v < 0:
            raise ValueError(f"seq weight {k} must be finite and >= 0, got {v}")
    if cg_iters <= 0:
        raise ValueError(f"seq cg_iters must be > 0, got {cg_iters}")
    if not (_math.isfinite(cg_tol) and cg_tol > 0):
        raise ValueError(f"seq cg_tol must be finite and > 0, got {cg_tol}")

    sigma_q, chol_q = _build_sigma(vq_model, Q, float(gn_cfg.get("ridge", 0.1)))

    def _decode(embs):
        pm = vq_model.vqvae.forward_decoder_from_quantized_codes(embs)
        pm = pm.squeeze(2).permute(0, 2, 1) * _std + _mean
        return recover_from_ric(pm.float(), joints_num)      # (1, T, J, 3)

    def _solve_one(_emb, _gj, _gjm, _tag, _vt=None):
        lam_damp = gn_lam0
        shapes = [tuple(e.shape) for e in _emb]
        met = _BlockMetric(sigma_q, chol_q, shapes)
        P = sum(e.numel() for e in _emb)

        b_idx, f_idx_full, j_idx_full = torch.where(_gjm)
        if not bool((b_idx == 0).all()):
            raise RuntimeError("seq-GN per-sample slice violated: mask beyond sample 0")
        f_idx, j_idx = f_idx_full, j_idx_full
        if len(f_idx) > gn_amax:                              # Jacobian row budget;
            sel = torch.linspace(0, len(f_idx) - 1, gn_amax).round().long()
            f_idx, j_idx = f_idx[sel], j_idx[sel]             # accept/stop use full mask
        A = len(f_idx)
        nr = 3 * A

        def _flat(embs):
            return torch.cat([e.reshape(-1) for e in embs])

        def _unflat(vec):
            out, off = [], 0
            for (b, c, t) in shapes:
                n = b * c * t
                out.append(vec[off:off + n].view(b, c, t))
                off += n
            return out

        u0 = _flat([e.detach() for e in _emb]).clone()
        with torch.no_grad():
            phi0 = _decode([e.detach() for e in _emb])        # (1, T, J, 3)
        # valid-length gate (codex P1): sequence costs must not act on padded
        # frames. _vt = this sample's true frame count (None -> all frames,
        # the pre-fix behaviour, kept for callers without length info).
        _T_valid = int(_vt) if _vt is not None else phi0.shape[1]
        _T_valid = max(2, min(_T_valid, phi0.shape[1]))
        phi0 = phi0[:, :_T_valid]
        d1_0 = phi0[:, 1:] - phi0[:, :-1]
        d2_0 = d1_0[:, 1:] - d1_0[:, :-1]
        N_vel = max(d1_0.numel(), 1)
        N_acc = max(d2_0.numel(), 1)

        # u0-detected geometric contact (fixed gate; metric-matched definition):
        # feet joints 10/11 below 0.05 m in BOTH frames of each adjacent pair.
        feet0 = phi0[0, :, list(_FEET_JOINTS), 1]             # (T, 2) heights
        contact = (feet0[:-1] < _CONTACT_H) & (feet0[1:] < _CONTACT_H)   # (T-1, 2)
        con_mask = contact.to(phi0.dtype)                     # gate on D1 feet rows
        N_con = max(int(contact.sum().item()) * 3, 1)         # xyz per gated pair

        use_obs = (lam_vel > 0) or (lam_acc > 0) or (lam_con > 0)
        use_soft_guard = use_obs or (lam_base > 0)

        def _obs_fwd(phi):
            """phi (1,T,J,3) -> the stacked, weighted soft residual pieces
            (restricted to the valid frames)."""
            phi = phi[:, :_T_valid]
            d1 = phi[:, 1:] - phi[:, :-1]
            d2 = d1[:, 1:] - d1[:, :-1]
            pieces = []
            if lam_vel > 0:
                pieces.append((lam_vel / N_vel) ** 0.5 * (d1 - d1_0).reshape(-1))
            if lam_acc > 0:
                pieces.append((lam_acc / N_acc) ** 0.5 * (d2 - d2_0).reshape(-1))
            if lam_con > 0:
                dfeet = d1[0, :, list(_FEET_JOINTS), :]        # (T-1, 2, 3)
                pieces.append((lam_con / N_con) ** 0.5
                              * (con_mask.unsqueeze(-1) * dfeet).reshape(-1))
            return torch.cat(pieces) if pieces else phi.new_zeros(0)

        def _obs_fwd_lin(phi_tangent):
            """The same map on a tangent (all pieces are linear in phi except
            the constant u0 offsets, which drop out of the linearization)."""
            phi_tangent = phi_tangent[:, :_T_valid]
            d1 = phi_tangent[:, 1:] - phi_tangent[:, :-1]
            d2 = d1[:, 1:] - d1[:, :-1]
            pieces = []
            if lam_vel > 0:
                pieces.append((lam_vel / N_vel) ** 0.5 * d1.reshape(-1))
            if lam_acc > 0:
                pieces.append((lam_acc / N_acc) ** 0.5 * d2.reshape(-1))
            if lam_con > 0:
                dfeet = d1[0, :, list(_FEET_JOINTS), :]
                pieces.append((lam_con / N_con) ** 0.5
                              * (con_mask.unsqueeze(-1) * dfeet).reshape(-1))
            return torch.cat(pieces) if pieces else phi_tangent.new_zeros(0)

        # ---- batched (multi-column) twins for the parallel-CG path (cg_block>0).
        # The leading K axis is the column axis; row m of every output equals the
        # serial single-vector map applied to column m (block-diagonal in K).
        def _unflat_batched(V):              # (P,K) -> list of (K,C,Tk)
            out, off, K = [], 0, V.shape[1]
            for (b, c, t) in shapes:
                n = b * c * t
                out.append(V[off:off + n].transpose(0, 1).reshape(K, c, t))
                off += n
            return out

        def _flat_batched(blocks):           # list of (K,C,Tk) -> (P,K); inverse of _unflat_batched
            parts = []
            for blk in blocks:
                K, c, t = blk.shape
                parts.append(blk.reshape(K, c * t).transpose(0, 1))
            return torch.cat(parts, dim=0)

        def _obs_fwd_lin_batched(phi_t):     # (K,T,J,3) -> (K,N_obs); batched twin of _obs_fwd_lin
            phi_t = phi_t[:, :_T_valid]
            d1 = phi_t[:, 1:] - phi_t[:, :-1]
            d2 = d1[:, 1:] - d1[:, :-1]
            K = phi_t.shape[0]; pieces = []
            if lam_vel > 0:
                pieces.append((lam_vel / N_vel) ** 0.5 * d1.reshape(K, -1))
            if lam_acc > 0:
                pieces.append((lam_acc / N_acc) ** 0.5 * d2.reshape(K, -1))
            if lam_con > 0:
                dfeet = d1[:, :, list(_FEET_JOINTS), :]        # (K,T-1,2,3)
                pieces.append((lam_con / N_con) ** 0.5
                              * (con_mask.unsqueeze(0).unsqueeze(-1) * dfeet).reshape(K, -1))
            return torch.cat(pieces, dim=1) if pieces else phi_t.new_zeros(K, 0)

        def anchor_mse_full(phi):
            d = phi[0, f_idx_full, j_idx_full] - _gj[0, f_idx_full, j_idx_full]
            return float((d ** 2).mean().item())

        def soft_S(phi, u_flat):
            """QP-consistent soft objective S(u) (same quadratics as H/g_soft):
            1/2*lam_base*(u-u0)^T Sigma^{-1}(u-u0) + 1/2*||s_obs(u)||^2."""
            base = 0.5 * lam_base * float(
                ((u_flat - u0) @ met.sigma_inv(u_flat - u0)).item())
            s = _obs_fwd(phi)
            obs = 0.5 * float((s ** 2).sum().item()) if s.numel() else 0.0
            return base + obs

        mse0 = None
        # S(u0): the base/vel/acc residuals vanish at u0, but the CONTACT
        # residual does not (u0's feet may slide during contact frames --
        # driving that to zero is the point). Initialize with the true value.
        S_prev = soft_S(phi0, u0)
        accepted_steps = 0
        stop_reason = "max_steps"
        n_accept = n_reject = 0

        with torch.enable_grad():
            for it in range(gn_steps):
                # ---- forward at u_t with graph (shared by residual J and phi VJP)
                for e in _emb:
                    if not e.requires_grad:
                        e.requires_grad_(True)
                phi_g = _decode(_emb)                          # graph-carrying
                r_g = (phi_g[0, f_idx, j_idx] - _gj[0, f_idx, j_idx]).reshape(-1)
                mse = anchor_mse_full(phi_g.detach())
                if mse0 is None:
                    mse0 = max(mse, 1e-12)
                if mse < gn_tau:
                    stop_reason = "tau"
                    break

                # ---- Jacobian rows J = dr/du (baseline row loop, retain graph
                #      so phi_g can also serve the soft-term VJPs below)
                J_rows = []
                for k in range(nr):
                    grads = torch.autograd.grad(r_g[k], _emb, retain_graph=True,
                                                allow_unused=True)
                    J_rows.append(torch.cat(
                        [(g if g is not None else torch.zeros_like(e)).reshape(-1)
                         for g, e in zip(grads, _emb)]))
                J = torch.stack(J_rows).detach()               # (3A, P)

                u_t = _flat([e.detach() for e in _emb])

                # ---- g_soft at u_t
                if use_obs:
                    s_now = _obs_fwd(phi_g)                    # graph-carrying
                    # VJP: J_phi^T (obs^T s_now) in one backward through phi_g
                    g_obs = torch.autograd.grad(
                        0.5 * (s_now ** 2).sum(), _emb, retain_graph=True,
                        allow_unused=True)
                    g_obs = torch.cat(
                        [(g if g is not None else torch.zeros_like(e)).reshape(-1)
                         for g, e in zip(g_obs, _emb)]).detach()
                else:
                    g_obs = torch.zeros_like(u_t)
                g_soft = g_obs
                if lam_base > 0:
                    # base term shares the primary Sigma^{-1} scale (unnormalized;
                    # acceptance cost C uses the mean-per-dim form separately)
                    g_soft = g_soft + lam_base * met.sigma_inv(u_t - u0)

                # ---- H.v operator (matrix-free)
                def H_apply(v):
                    out = met.sigma_inv(v)
                    if lam_base > 0:
                        out = out + lam_base * met.sigma_inv(v)
                    if use_obs:
                        # JVP via double-backward through the SAME phi graph:
                        # phi_tangent = J_phi v  ->  obs_lin  ->  VJP back.
                        v_embs = _unflat(v)
                        jvp = torch.autograd.functional.jvp(
                            lambda *es: _decode(list(es)),
                            tuple(e.detach() for e in _emb),
                            tuple(v_embs), create_graph=False)[1]
                        w = _obs_fwd_lin(jvp)                  # (N_obs,) weighted
                        # A^T w: backward of <obs_lin(phi), w> through phi graph
                        Aw = torch.autograd.grad(
                            (_obs_fwd_lin(phi_g) * w.detach()).sum(), _emb,
                            retain_graph=True, allow_unused=True)
                        Aw = torch.cat(
                            [(g if g is not None else torch.zeros_like(e)).reshape(-1)
                             for g, e in zip(Aw, _emb)]).detach()
                        out = out + Aw
                    return out

                Minv = (lambda v: met.sigma(v) / (1.0 + lam_base)) \
                    if lam_base > 0 else met.sigma

                # ---- batched matvec + lockstep parallel-CG (cg_block>0 path).
                # emb_rep/phi_rep carry the chunk's own decode graph; each column
                # is an independent CG run (column axis never couples), so column
                # m reproduces serial _cg(H_apply, ., ...) at every iteration.
                def _H_apply_b(Pc, emb_rep, phi_rep):        # (P,cb) -> (P,cb) = H @ Pc columnwise
                    si = met.sigma_inv_batched(Pc)
                    out = (si + lam_base * si) if lam_base > 0 else si   # == serial's two sigma_inv adds
                    if use_obs:
                        v_embs = _unflat_batched(Pc)                     # list (cb,C,Tk)
                        jvp = torch.autograd.functional.jvp(
                            lambda *es: _decode(list(es)),
                            tuple(e.detach() for e in emb_rep),
                            tuple(v_embs), create_graph=False)[1]        # (cb,T,J,3) = J_phi @ Pc per column
                        w = _obs_fwd_lin_batched(jvp)                    # (cb,N_obs)
                        Aw = torch.autograd.grad(
                            (_obs_fwd_lin_batched(phi_rep) * w.detach()).sum(),
                            emb_rep, retain_graph=True, allow_unused=True)
                        out = out + _flat_batched(
                            [(g if g is not None else torch.zeros_like(e)) for g, e in zip(Aw, emb_rep)])
                    return out

                def _Minv_b(V):
                    return met.sigma_batched(V) / (1.0 + lam_base) if lam_base > 0 \
                        else met.sigma_batched(V)

                def _cg_batched(Bc, emb_rep, phi_rep):       # lockstep PCG over cb columns; per-column scalars
                    Xc = torch.zeros_like(Bc); Rc = Bc.clone()
                    Zc = _Minv_b(Rc); Pd = Zc.clone()
                    rz = (Rc * Zc).sum(0)                             # (cb,)
                    bnorm = Bc.norm(dim=0) + 1e-30                    # (cb,)
                    active = torch.ones(Bc.shape[1], dtype=torch.bool, device=Bc.device)
                    zeros = torch.zeros_like(rz); rel = torch.zeros_like(rz)
                    for _ in range(cg_iters):
                        APc = _H_apply_b(Pd, emb_rep, phi_rep)
                        pAp = (Pd * APc).sum(0)
                        if not torch.isfinite(pAp[active]).all():
                            raise RuntimeError("seq-CG: non-finite curvature p^T A p")
                        alpha = torch.where(active, rz / (pAp + 1e-30), zeros)
                        Xc = Xc + alpha * Pd
                        Rc = Rc - alpha * APc
                        rel = Rc.norm(dim=0) / bnorm
                        if not torch.isfinite(rel[active]).all():
                            raise RuntimeError("seq-CG: non-finite residual")
                        active = active & (rel >= cg_tol)            # freeze AFTER x-update == serial break timing
                        if not bool(active.any()):
                            break
                        Zc = _Minv_b(Rc)
                        rz_new = (Rc * Zc).sum(0)
                        beta = torch.where(active, rz_new / (rz + 1e-30), zeros)
                        Pd = (Zc + beta * Pd) * active.to(Pd.dtype)  # zero frozen dirs => pAp=0 (finite)
                        rz = rz_new
                    return Xc, rel

                # ---- X = H^{-1} J^T (columns), x_g = H^{-1} g_soft
                with torch.no_grad() if not use_obs else torch.enable_grad():
                    if use_obs and cg_block > 0:
                        # ---- batched chunked parallel-CG (fold g_soft as column
                        #      nr only when nonzero; chunk to bound decoder-
                        #      activation memory; OOM-halving fallback).
                        include_g = float(g_soft.norm()) > 0
                        B = torch.cat([J.t(), g_soft.unsqueeze(1)], dim=1) \
                            if include_g else J.t().contiguous()
                        Mcol = B.shape[1]
                        X_all = B.new_zeros(P, Mcol)
                        _cg_worst = 0.0
                        _cb = cg_block; c0 = 0
                        while c0 < Mcol:
                            cb = min(_cb, Mcol - c0)
                            try:
                                Bc = B[:, c0:c0 + cb].contiguous()
                                emb_rep = [e.detach().repeat(cb, 1, 1).requires_grad_(True)
                                           for e in _emb]
                                phi_rep = _decode(emb_rep)             # (cb,T,J,3), graph reused per chunk
                                Xc, relc = _cg_batched(Bc, emb_rep, phi_rep)
                                X_all[:, c0:c0 + cb] = Xc
                                _cg_worst = max(_cg_worst, float(relc.max()))
                                del emb_rep, phi_rep, Xc, relc
                                c0 += cb
                            except RuntimeError as _oom:
                                if "out of memory" not in str(_oom).lower() or _cb == 1:
                                    raise                              # fail-loud / unshrinkable
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                _cb = max(1, _cb // 2)                  # shrink and retry this chunk
                        X = X_all[:, :nr].detach()
                        x_g = (X_all[:, nr] if include_g
                               else torch.zeros_like(g_soft)).detach()
                        if _cg_worst > 1e-2:
                            print(f"  [Stage2 seqGN]{_tag} WARNING: worst CG rel "
                                  f"residual {_cg_worst:.1e} (>1e-2) — raise "
                                  f"cg_iters for a trustworthy step")
                    elif use_obs:
                        # ---- ORIGINAL serial per-column CG (verbatim) ----
                        _cg_worst = 0.0
                        cols = []
                        for k in range(nr):
                            _xk, _relk = _cg(H_apply, J[k], Minv, cg_iters, cg_tol)
                            cols.append(_xk)
                            _cg_worst = max(_cg_worst, _relk)
                        if float(g_soft.norm()) > 0:
                            x_g, _relg = _cg(H_apply, g_soft, Minv, cg_iters, cg_tol)
                            _cg_worst = max(_cg_worst, _relg)
                        else:
                            x_g = torch.zeros_like(g_soft)
                        if _cg_worst > 1e-2:
                            print(f"  [Stage2 seqGN]{_tag} WARNING: worst CG rel "
                                  f"residual {_cg_worst:.1e} (>1e-2) — raise "
                                  f"cg_iters for a trustworthy step")
                        X = torch.stack(cols, dim=1).detach()          # moved here from old L404
                    else:
                        # exact closed form: H^{-1} = Sigma/(1+lam_base)
                        scale = 1.0 / (1.0 + lam_base)
                        cols = [met.sigma(J[k]) * scale for k in range(nr)]
                        x_g = met.sigma(g_soft) * scale \
                            if float(g_soft.norm()) > 0 else torch.zeros_like(g_soft)
                        X = torch.stack(cols, dim=1).detach()          # moved here from old L404
                x_g = x_g.detach()                                     # old L405, unchanged (no-op for batched)
                import os as _os_dbg
                if _os_dbg.environ.get("CODA_SEQ_DEBUG") and use_obs:
                    _r0 = float((H_apply(X[:, 0]) - J[0]).norm() / (J[0].norm() + 1e-30))
                    _rg = float((H_apply(x_g) - g_soft).norm() / (g_soft.norm() + 1e-30))                         if float(g_soft.norm()) > 0 else -1.0
                    print(f"    [seqGN-dbg] CG quality at u_t: ||H X0 - J0||/||J0|| = {_r0:.2e}"
                          f"  ||H xg - g||/||g|| = {_rg:.2e}  |g_soft|={float(g_soft.norm()):.3e}")

                with torch.no_grad():
                    JHiJt = J @ X                              # (3A, 3A)
                    # CG-approximate columns make J@X slightly nonsymmetric;
                    # the exact Schur complement is symmetric — project back.
                    JHiJt = 0.5 * (JHiJt + JHiJt.T)
                    s_scale = float(JHiJt.diagonal().mean().item())
                    if not (s_scale > 1e-20):
                        stop_reason = "degenerate_jacobian"
                        break
                    rhs_base = r_g.detach() - J @ x_g
                    eye = torch.eye(nr, device=J.device, dtype=J.dtype)
                    base = [e.detach().clone() for e in _emb]  # exact restore
                    accepted = False
                    for _try in range(4):
                        for e, bb in zip(_emb, base):
                            e.data.copy_(bb)
                        try:
                            z = torch.linalg.solve(
                                JHiJt + lam_damp * s_scale * eye, rhs_base)
                        except Exception:
                            lam_damp *= 10.0
                            continue
                        if not torch.isfinite(z).all():
                            lam_damp *= 10.0
                            continue
                        d = -(x_g + X @ z)
                        if use_soft_guard:
                            # model-predicted soft change for this step:
                            # Delta_pred = g_soft^T d + 1/2 d^T H_soft d with
                            # H_soft d = H d - Sigma^{-1} d. H d is computed
                            # EXPLICITLY via H_apply (codex P1: the KKT identity
                            # Hd = -(g_soft+J^T z) does not hold for the
                            # CG-approximate X, x_g), and BEFORE the embeddings
                            # are mutated so the retained phi_g graph is valid.
                            with torch.enable_grad():
                                Hd = H_apply(d)
                            dHsd = float((d @ (Hd - met.sigma_inv(d))).item())
                            pred = float((g_soft @ d).item()) + 0.5 * dHsd
                        off = 0
                        for e in _emb:
                            n = e.numel()
                            e.data.add_(d[off:off + n].view_as(e))
                            off += n
                        phi2 = _decode([e.detach() for e in _emb])
                        mse2 = anchor_mse_full(phi2)
                        u2 = _flat([e.detach() for e in _emb])
                        S2 = soft_S(phi2, u2)
                        if use_soft_guard:
                            guard_ok = (S2 - S_prev) <= (
                                2.0 * abs(pred) + 1e-6 * (1.0 + S_prev))
                        else:
                            guard_ok = True
                        if (mse2 < mse) and guard_ok:
                            S_prev = S2
                            lam_damp = max(lam_damp * 0.5, 1e-6)
                            accepted = True
                            n_accept += 1
                            break
                        import os as _os_dbg
                        if _os_dbg.environ.get("CODA_SEQ_DEBUG"):
                            _rhs_kkt = g_soft + J.T @ z
                            _cgres = float('nan')  # H at u2 != H at u_t; checked above instead
                            _du = u2 - u0
                            _b2 = 0.5 * lam_base * float((_du @ met.sigma_inv(_du)).item())
                            _o2 = S2 - _b2
                            print(f"    [seqGN-dbg] REJECT try{_try}: mse {mse:.3e}->{mse2:.3e} "
                                  f"(drop={mse2<mse}) S {S_prev:.3e}->{S2:.3e} "
                                  f"[base {_b2:.3e} obs {_o2:.3e}] "
                                  f"pred={pred if use_soft_guard else float('nan'):.3e} "
                                  f"dSinvd={float((d @ met.sigma_inv(d)).item()):.3e} "
                                  f"cg_resid={_cgres:.2e} guard_ok={guard_ok} |d|={float(d.norm()):.3e}")
                        lam_damp *= 10.0
                        n_reject += 1
                    if not accepted:
                        for e, bb in zip(_emb, base):
                            e.data.copy_(bb)
                        stop_reason = "trust_region_stall"
                        break
                accepted_steps += 1
                print(f"  [Stage2 seqGN]{_tag} step {accepted_steps}/{gn_steps} "
                      f"full-mse {mse:.2e}->{mse2:.2e} S {S_prev:.3e} "
                      f"lam={lam_damp:.1e}")

        final_phi = _decode([e.detach() for e in _emb])
        print(f"  [Stage2 seqGN]{_tag} done: steps={accepted_steps} "
              f"accept={n_accept} reject={n_reject} "
              f"final-mse {anchor_mse_full(final_phi):.2e} stop={stop_reason}")
        return [e.detach() for e in _emb]

    _bsz = emb_list[0].shape[0]
    _emb_out = [e.detach().clone() for e in emb_list]
    for _bi in range(_bsz):
        _gjm_b = global_joint_mask[_bi:_bi + 1]
        if not bool(_gjm_b.any()):
            continue
        _emb_b = [e[_bi:_bi + 1].clone().detach().contiguous().requires_grad_(True)
                  for e in emb_list]
        _tag = f" s{_bi}" if _bsz > 1 else ""
        _vt_b = None
        if m_lens is not None:
            try:
                _vt_b = int(m_lens[_bi])
            except Exception:
                _vt_b = None
        _sol = _solve_one(_emb_b, global_joint[_bi:_bi + 1], _gjm_b, _tag, _vt_b)
        for _qi in range(Q):
            _emb_out[_qi][_bi:_bi + 1] = _sol[_qi]
    # same footer contract as the baseline: copy back into the caller's leaves.
    with torch.no_grad():
        for _qi in range(Q):
            emb_optim_list[_qi].data.copy_(_emb_out[_qi])
