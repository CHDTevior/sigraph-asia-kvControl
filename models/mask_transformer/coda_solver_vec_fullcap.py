"""CODA-vec: CROSS-SAMPLE BATCHED Gauss-Newton / Levenberg-Marquardt anchor solver.

This is a NEW module -- it does not modify `coda_solver.py` (the verbatim
per-sample serial extraction) or the inline GN block in
`control_transformer_t_concat_v8_kv_v4_noqres.py` (lines ~1031-1293, the copy
that produced every published CODA number). Both stay byte-for-byte untouched.

Where the serial solver loops `for _bi in range(B): _gn_solve_one(sample_bi)`
(each sample doing its own generation call in the fully-serial `--gn` path, or
its own per-sample LM loop after a batched Stage-1 in the `--gn_batch` path),
this module runs Stage-1 generation batched (like `--gn_batch`) AND runs the
LM iterations for ALL samples in the group simultaneously:
  - per-sample damping (`lam`, a (B,) vector instead of a python float)
  - per-sample convergence (`done` mask; converged/stalled samples freeze --
    they stop receiving embedding updates but stay in the batch, padded)
  - ragged anchor counts A_b: capped-anchor rows are padded to R_max = 3 *
    anchor_cap across the batch and masked (padded rows carry an all-zero
    Jacobian row and a lam*scale-regularized diagonal, so they solve to
    z=0 and contribute nothing -- see `_scale computed via masked mean`
    below for why padding is numerically inert rather than merely ignored)
  - per-sample trial/rollback with the same 4-try trust-region retry count as
    serial, restoring the exact pre-trial embeddings on rejection
  - acceptance / stopping / reporting always uses the FULL anchor mask; only
    the Jacobian ROWS are capped (same correctness invariant as serial --
    codex P1: the solver must not fit the capped collocation points while
    degrading the frames between them)
  - the Jacobian itself batches the existing "replicated-batch backward"
    trick (see jfast / GN_JAC_CHUNK in the inline copy) ACROSS samples: each
    row-job (which sample, which capped anchor, which coordinate) becomes one
    replica in a super-batch; the decoder processes a whole chunk of replicas
    -- possibly spanning several different original samples -- in one forward
    pass, and ONE backward recovers every row in the chunk at once (valid
    because the VQ decoder is a per-batch-item-independent conv stack: see
    `forward_decoder_from_quantized_codes` in models/vqvae.py, concatenates
    quantizer outputs along the channel dim and decodes, no batch mixing).

Scope limitation (documented, not silently dropped): `prox_beta` (the
proximal Mahalanobis-acceptance term for multi-joint FID control) is NOT
implemented here. It defaults to 0.0 and is not exercised by the equivalence
protocol this module was built against; batching its cumulative per-sample
distance bookkeeping together with per-sample stalling is materially more
complex than the rest of this solver and was out of scope for this pass.
Passing `gn_cfg["prox_beta"] != 0` raises NotImplementedError rather than
silently ignoring it (use the serial `coda_solve` for prox_beta runs).
"""
import torch
import torch.nn.functional as F

from models.mask_transformer.tools import lengths_to_mask
from utils.motion_process import recover_from_ric

# Default chunk size (replicas per super-batch decode) when gn_cfg["jac_chunk"]
# is unset/0. Unlike the serial per-sample jfast path -- where jac_chunk=0
# validly means "no batching, per-row loop" -- gn_vec's entire purpose is
# batching, so 0 cannot mean "don't batch" here. 256 replicas/chunk keeps
# peak memory well under one 80GB A100 for this model (~37.6k params/sample
# decode) while still processing a whole group's worth of anchor rows in a
# small number of chunks; back off on OOM.
_DEFAULT_JAC_CHUNK = 256
_MIN_JAC_CHUNK = 8


def _build_codebook_sigma(vq_model, Q, metric, ridge):
    """Verbatim logic from coda_solver.coda_solve, unchanged: Sigma_q is
    shared across the whole batch (built from the codebook, not per-sample),
    so no batching is needed here."""
    if not metric:
        return None
    if metric != "codebook_cov":
        raise ValueError(f"unknown gn metric {metric!r}")
    sigma_q = []
    for q in range(Q):
        cb = vq_model.vqvae.quantizers[q].codebook.detach().float()  # (K, C)
        c = cb - cb.mean(0, keepdim=True)
        S = (c.T @ c) / max(1, cb.shape[0] - 1)
        mu = float(S.diagonal().mean().item())
        if not (mu > 0) or not torch.isfinite(S).all():
            raise RuntimeError(f"quantizer {q}: degenerate codebook covariance "
                               f"(mean diag {mu}); cannot build the metric")
        S = S + ridge * mu * torch.eye(S.shape[0], device=S.device, dtype=S.dtype)
        sigma_q.append(S / S.diagonal().mean())
    return sigma_q


def _apply_sigma_batched(vec, sigma_q, shapes):
    """Blockwise per-quantizer metric, applied along the LAST axis of `vec`
    (shape (..., P)), batched over every leading dim via einsum. `shapes` is
    the list of (C_q, Tk) per quantizer (no batch/leading dims)."""
    if sigma_q is None:
        return vec
    lead = vec.shape[:-1]
    parts, off = [], 0
    for qi, (C_q, Tk) in enumerate(shapes):
        n = C_q * Tk
        blk = vec[..., off:off + n].reshape(*lead, C_q, Tk)
        out = torch.einsum("cd,...dt->...ct", sigma_q[qi], blk)
        parts.append(out.reshape(*lead, n))
        off += n
    return torch.cat(parts, dim=-1)


def coda_solve_vec(emb_optim_list, emb_list, vq_model, global_joint,
                    global_joint_mask, _mean, _std, control_opt, Q, joints_num):
    """Cross-sample batched CODA solve. Mutates `emb_optim_list` IN PLACE
    (same external contract as coda_solver.coda_solve): `emb_optim_list` is
    a list of Q leaf tensors (B, C_q, Tk), requires_grad=True, holding the
    Stage-2 starting point (a clone of `emb_list`). Samples with no anchors
    at all are left untouched (their Stage-1 value stands)."""
    device = emb_optim_list[0].device
    B = emb_optim_list[0].shape[0]
    shapes = [tuple(e.shape[1:]) for e in emb_optim_list]     # [(C_q, Tk), ...]
    P = sum(c * t for c, t in shapes)

    gn_cfg = (control_opt or {}).get("gn", {}) if control_opt else {}
    gn_steps = int(gn_cfg.get("max_steps", 8))
    gn_tau = float(gn_cfg.get("tau", 1e-6))
    gn_lam0 = float(gn_cfg.get("lam0", 1e-2))
    gn_amax = int(gn_cfg.get("anchor_cap", 48))
    gn_beta = float(gn_cfg.get("prox_beta", 0.0))
    if gn_beta != 0.0:
        raise NotImplementedError(
            "coda_solve_vec: prox_beta is out of scope for this batched solver "
            "(not exercised by the equivalence protocol) -- use coda_solve "
            "(per-sample, control_transformer_t_concat_v8_kv_v4_noqres.py's "
            "inline copy) for prox_beta runs.")
    jac_chunk = int(gn_cfg.get("jac_chunk", 0))
    if jac_chunk <= 0:
        jac_chunk = _DEFAULT_JAC_CHUNK
    gn_metric = str(gn_cfg.get("metric", "") or "")
    sigma_q = _build_codebook_sigma(vq_model, Q, gn_metric, float(gn_cfg.get("ridge", 0.1))) \
        if gn_metric else None

    if gn_amax <= 0:
        raise ValueError(f"gn anchor_cap must be positive, got {gn_amax}")

    with torch.no_grad():
        # ---- per-sample setup: O(B) index bookkeeping, not the bottleneck ----
        has_anchors = global_joint_mask.any(dim=1).any(dim=1)          # (B,)
        f_cap = [None] * B
        j_cap = [None] * B
        R = torch.zeros(B, dtype=torch.long, device=device)
        for b in range(B):
            if not bool(has_anchors[b]):
                continue
            f_full, j_full = torch.nonzero(global_joint_mask[b], as_tuple=True)
            if len(f_full) > gn_amax:
                sel = torch.linspace(0, len(f_full) - 1, gn_amax, device=device).round().long()
                f_full, j_full = f_full[sel], j_full[sel]
            f_cap[b], j_cap[b] = f_full, j_full
            R[b] = 3 * len(f_full)
        if not bool(has_anchors.any()):
            return                              # nothing to do -- Stage-1 output stands
        R_max = int(R.max().item())

        active = has_anchors.clone()
        done = ~has_anchors
        lam = torch.full((B,), gn_lam0, device=device, dtype=emb_optim_list[0].dtype)
        # persists (and only ever shrinks) across the whole solve -- once a chunk
        # size OOMs we don't want to keep re-trying the larger size every LM step
        chunk = jac_chunk

        def decode_full():
            pm = vq_model.vqvae.forward_decoder_from_quantized_codes(emb_optim_list)
            pm = pm.squeeze(2).permute(0, 2, 1) * _std + _mean
            return recover_from_ric(pm.float(), joints_num)          # (B, T, J, 3)

        def mse_full_all(pj):
            # per-sample mean over (3 * A_full_b) elements -- identical
            # reduction to serial's `(d ** 2).mean()` over the FULL mask
            d2 = (pj - global_joint) ** 2
            m = global_joint_mask.unsqueeze(-1).expand_as(d2)
            s = (d2 * m).sum(dim=(1, 2, 3))
            c = m.sum(dim=(1, 2, 3)).clamp(min=1)
            return s / c                                             # (B,)

        for _gn_it in range(gn_steps):
            if not bool(active.any()):
                break
            pj = decode_full()
            mse = mse_full_all(pj)
            newly_conv = active & (mse < gn_tau)
            done = done | newly_conv
            active = active & ~newly_conv
            if not bool(active.any()):
                break

            # ---- capacity-adaptive chunking: sample sub-batch (nb) x R_max block-tile (rc) ----
            # `idx_active` is captured ONCE here; sub-batches are DISJOINT subsets of it and are
            # solved independently (every reduction below -- Jacobian rows, Gram, scale, solve,
            # decode, mse, accept/lam -- is per-sample; the VQ decoder is a per-item conv stack).
            # At low cap the gate keeps nb == n_act AND tile == False, so this block executes the
            # byte-identical single-shot path (single-block Gram/J^T z == original bmm; single-batch
            # solve == original solve). Chunking (nb < n_act and/or tile) engages ONLY when the
            # full single-shot op would OOM or exceed the CUDA launch grid -- i.e. only at caps
            # where the pre-chunking path could not run at all, so there is no feasible reference
            # to differ from and the chunked result is the exact GN update.
            idx_active = torch.nonzero(active, as_tuple=True)[0]
            n_act = int(idx_active.numel())
            elt = emb_optim_list[0].element_size()
            slab_bytes = R_max * P * elt                               # one sample's (R_max,P) slab
            mem_frac = float(gn_cfg.get("vec_mem_frac", 0.45))         # J_sub gets <=45% of the GPU
            gram_frac = float(gn_cfg.get("vec_gram_frac", 0.30))       # 2nd full copy allowed below this
            grid_cap = int(gn_cfg.get("vec_grid_cap", 60000))          # < 65535 CUDA grid.z limit
            row_block = int(gn_cfg.get("vec_row_block", 512))          # R_max tile width when tiling
            if device.type == "cuda":
                total_mem = torch.cuda.get_device_properties(device).total_memory
                nb = max(1, min(n_act, int(total_mem * mem_frac) // max(1, slab_bytes)))
                tile = (nb * R_max > grid_cap) or (slab_bytes * nb > int(total_mem * gram_frac))
            else:                                                     # CPU / unit-test: single-shot path
                nb, tile = n_act, False
            rc = row_block if tile else R_max
            blocks = [slice(a, min(a + rc, R_max)) for a in range(0, R_max, rc)]

            base = [e.detach().clone() for e in emb_optim_list]        # top-of-iteration embeddings

            for s0 in range(0, n_act, nb):
                sub = idx_active[s0:s0 + nb]
                nbs = int(sub.numel())
                loc = torch.full((B,), -1, dtype=torch.long, device=device)   # global b -> local row
                loc[sub] = torch.arange(nbs, device=device)

                # ---- row-jobs for THIS sub-batch only (mirrors the original job build) ----
                job_b, job_row, job_f, job_j, job_c = [], [], [], [], []
                for b in sub.tolist():
                    Ab = len(f_cap[b])
                    job_b.append(torch.full((3 * Ab,), b, dtype=torch.long, device=device))
                    job_row.append(torch.arange(3 * Ab, device=device))
                    job_f.append(f_cap[b].repeat_interleave(3))
                    job_j.append(j_cap[b].repeat_interleave(3))
                    job_c.append(torch.arange(3, device=device).repeat(Ab))
                job_b = torch.cat(job_b)
                job_row = torch.cat(job_row)
                job_f = torch.cat(job_f)
                job_j = torch.cat(job_j)
                job_c = torch.cat(job_c)
                n_jobs = int(job_b.numel())

                # ---- J_sub = (nbs, R_max, P): built directly, never (B,R_max,P), no J[idx] gather --
                J_sub = torch.zeros(nbs, R_max, P, device=device, dtype=emb_optim_list[0].dtype)
                r_sub = torch.zeros(nbs, R_max, device=device, dtype=emb_optim_list[0].dtype)
                r_sub[loc[job_b], job_row] = pj[job_b, job_f, job_j, job_c] - \
                    global_joint[job_b, job_f, job_j, job_c]

                c0 = 0
                while c0 < n_jobs:                          # existing OOM-adaptive backward loop
                    while True:
                        sl = slice(c0, min(c0 + chunk, n_jobs))
                        b_s, row_s = job_b[sl], job_row[sl]
                        f_s, j_s, c_s = job_f[sl], job_j[sl], job_c[sl]
                        m = int(b_s.numel())
                        erep = pmr = pjr = sc = grads = flat = None
                        try:
                            with torch.enable_grad():
                                erep = [emb_optim_list[q].detach()[b_s].clone().requires_grad_(True)
                                        for q in range(Q)]
                                pmr = vq_model.vqvae.forward_decoder_from_quantized_codes(erep)
                                pmr = pmr.squeeze(2).permute(0, 2, 1) * _std + _mean
                                pjr = recover_from_ric(pmr.float(), joints_num)
                                sc = pjr[torch.arange(m, device=device), f_s, j_s, c_s].sum()
                                grads = torch.autograd.grad(sc, erep, allow_unused=True)
                            flat = torch.cat([
                                (g if g is not None else torch.zeros_like(e)).reshape(m, -1)
                                for g, e in zip(grads, erep)], dim=1)
                            J_sub[loc[b_s], row_s] = flat.detach()
                        except torch.cuda.OutOfMemoryError:
                            if chunk <= _MIN_JAC_CHUNK:
                                raise
                            del erep, pmr, pjr, sc, grads, flat
                            torch.cuda.empty_cache()
                            chunk = max(_MIN_JAC_CHUNK, chunk // 2)
                            continue
                        break
                    c0 += m

                # ---- Gram (J Sigma J^T) assembled block-by-block over R_max (one block if !tile) ---
                # Each JJt[i,p,q] = J_sub[i,p]^T Sigma J_sub[i,q] is a FULL-P dot product; only the
                # output rows/cols are tiled, the P contraction is never split. Sigma symmetric ->
                # exact transpose mirror. tile=False => single block => identical to the original bmm.
                JJt = torch.empty(nbs, R_max, R_max, device=device, dtype=J_sub.dtype)
                for bi, sb in enumerate(blocks):
                    SJb = _apply_sigma_batched(J_sub[:, sb, :], sigma_q, shapes) \
                        if sigma_q is not None else J_sub[:, sb, :]
                    for sa in blocks[bi:]:                  # upper triangle
                        blk = torch.bmm(J_sub[:, sa, :], SJb.transpose(1, 2))   # (nbs,|sa|,|sb|)
                        JJt[:, sa, sb] = blk
                        if sa != sb:
                            JJt[:, sb, sa] = blk.transpose(1, 2)
                    del SJb

                # ---- scale via masked mean over VALID rows; degeneracy freeze (== original) ----
                row_valid = (torch.arange(R_max, device=device).unsqueeze(0) < R[sub].unsqueeze(1))
                rv = row_valid.to(JJt.dtype)
                scale = (JJt.diagonal(dim1=1, dim2=2) * rv).sum(dim=1) / rv.sum(dim=1).clamp(min=1)
                degenerate = ~(scale > 1e-20)
                if bool(degenerate.any()):
                    deg_b = sub[degenerate]
                    done[deg_b] = True
                    active[deg_b] = False
                    keep = ~degenerate
                    sub, J_sub, JJt, scale, r_sub = \
                        sub[keep], J_sub[keep], JJt[keep], scale[keep], r_sub[keep]
                if sub.numel() == 0:
                    del J_sub, JJt
                    continue

                # ---- 4-try trust-region loop (== original, Ja->J_sub, idx_active->sub) ----
                lam_a = lam[sub].clone()
                mse_a = mse[sub]
                r_a = r_sub
                accepted = torch.zeros(sub.numel(), dtype=torch.bool, device=device)
                eye = torch.eye(R_max, device=device, dtype=JJt.dtype).unsqueeze(0)

                for _try in range(4):
                    trial = ~accepted
                    if not bool(trial.any()):
                        break
                    trial_pos = torch.nonzero(trial, as_tuple=True)[0]
                    trial_b = sub[trial_pos]
                    for e, bb in zip(emb_optim_list, base):     # unconditional restore, every try
                        e[trial_b] = bb[trial_b]
                    A_try = JJt[trial_pos] + (lam_a[trial_pos] * scale[trial_pos]).view(-1, 1, 1) * eye
                    r_try = r_a[trial_pos]
                    try:
                        z = torch.linalg.solve(A_try, r_try)
                        ok_solve = torch.isfinite(z).all(dim=1)
                    except Exception as _solve_exc:
                        # Surfaced, behaviour unchanged: the trial still counts as failed. Runners
                        # treat a log containing this marker as a failed evaluation.
                        print(f"[CODA_SOLVE_EXC] {type(_solve_exc).__name__}: {str(_solve_exc)[:200]}",
                              flush=True)
                        z = torch.zeros_like(r_try)
                        ok_solve = torch.zeros(trial_b.numel(), dtype=torch.bool, device=device)
                    if bool(ok_solve.any()):
                        sel = trial_pos[ok_solve]           # sub-local rows that solved OK
                        z_ok = z[ok_solve]
                        # J^T z: bmm over ALL sub samples with a zero-padded z, then 2D-gather
                        # the (nbs,P) RESULT. cuBLAS handles the big-P non-contiguous J_sub view;
                        # we NEVER advanced-index a tensor still carrying the P (~150k) axis --
                        # that is what triggered the >65535 grid-dim "invalid configuration".
                        # Exactly equivalent: z_full is 0 on non-OK rows so their J^T z is 0,
                        # and sJt_z_all[sel] picks the OK rows in trial_pos order.
                        nbs_sub = sub.numel()
                        z_full = torch.zeros(nbs_sub, R_max, device=device, dtype=z.dtype)
                        z_full[sel] = z_ok
                        Jt_z_all = torch.zeros(nbs_sub, P, device=device, dtype=z.dtype)
                        for sa in blocks:
                            Jt_z_all += torch.bmm(J_sub[:, sa, :].transpose(1, 2),
                                                  z_full[:, sa].unsqueeze(-1)).squeeze(-1)
                        sJt_z_all = _apply_sigma_batched(Jt_z_all, sigma_q, shapes) if sigma_q is not None else Jt_z_all
                        delta = torch.zeros(trial_b.numel(), P, device=device, dtype=z.dtype)
                        delta[ok_solve] = -sJt_z_all[sel]
                        off = 0
                        for e in emb_optim_list:
                            n = e.shape[1] * e.shape[2]
                            e[trial_b] = e[trial_b] + delta[:, off:off + n].view(-1, *e.shape[1:])
                            off += n
                    pj2 = decode_full()
                    mse2_all = mse_full_all(pj2)
                    mse2_trial = mse2_all[trial_b]
                    accept_now = ok_solve & (mse2_trial < mse_a[trial_pos])
                    accepted[trial_pos[accept_now]] = True
                    lam_a[trial_pos[accept_now]] = torch.clamp(lam_a[trial_pos[accept_now]] * 0.5, min=1e-6)
                    reject_pos = trial_pos[~accept_now]
                    lam_a[reject_pos] = lam_a[reject_pos] * 10.0

                stalled_b = sub[~accepted]
                if stalled_b.numel() > 0:
                    for e, bb in zip(emb_optim_list, base):
                        e[stalled_b] = bb[stalled_b]
                    done[stalled_b] = True
                    active[stalled_b] = False
                lam[sub] = lam_a
                del J_sub, JJt


def generate_with_control_vec(
    ct2m_transformer, clip_text, m_length, time_steps, cond_scale,
    temperature, topkr, force_mask,
    vq_model, global_joint, global_joint_mask,
    _mean, _std,
    control_opt=None,
    avoid_points=None,
    abitary_func=None,
    is_relative=False,
):
    """NEW orchestration path for --gn_vec. Batched Stage-1 + batched Stage-2:
    calls the model's existing, UNMODIFIED `generate()` method for Stage 1
    (same call the inline `_generate_with_control_impl` makes), rebuilds
    `emb_list` from its logits exactly as that method does (small glue code,
    not the solver under test), then hands Stage 2 off to `coda_solve_vec`
    instead of the model's inline per-sample GN block. Mirrors
    `generate_with_control`'s external signature/return so eval_t2m.py can
    call it as a drop-in.
    """
    m_lens = m_length // 4
    ids_bt6, logits = ct2m_transformer.generate(
        clip_text, m_lens, time_steps, cond_scale,
        temperature=temperature,
        topk_filter_thres=topkr,
        force_mask=force_mask,
        vq_model=vq_model,
        global_joint=global_joint,
        global_joint_mask=global_joint_mask,
        _mean=_mean, _std=_std,
        lr=control_opt.get("each_lr", 6e-2) if control_opt else 6e-2,
        each_iter=control_opt.get("each_iter", 50) if control_opt else 50,
        avoid_points=avoid_points,
        abitary_func=abitary_func,
        is_relative=is_relative,
        rgar=(control_opt or {}).get("rgar"),
    )

    seq_len = ids_bt6.shape[1]
    padding_mask = ~lengths_to_mask(m_lens, seq_len)

    Q = ct2m_transformer.Q
    emb_list = []
    for i in range(Q):
        cb = vq_model.vqvae.quantizers[i].codebook
        probs_i = F.softmax(logits[:, i, :, :] / max(temperature, 1e-8), dim=-1)
        emb_i = probs_i @ cb
        emb_i = emb_i.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        emb_list.append(emb_i.permute(0, 2, 1))

    last_iter = control_opt.get("iter", 0) if control_opt else 0
    if last_iter > 0 and global_joint is not None and global_joint_mask is not None \
            and global_joint_mask.any():
        emb_optim_list = [e.clone().detach().contiguous().requires_grad_(True) for e in emb_list]
        coda_solve_vec(emb_optim_list, emb_list, vq_model, global_joint,
                       global_joint_mask, _mean, _std, control_opt,
                       Q, ct2m_transformer.opt.joints_num)
        emb_list = [e.detach() for e in emb_optim_list]

    pred_motions = vq_model.vqvae.forward_decoder_from_quantized_codes(emb_list)
    pred_motions = pred_motions.squeeze(2).permute(0, 2, 1)
    pred_motions_denorm = pred_motions * _std + _mean
    pred_motions_denorm = recover_from_ric(pred_motions_denorm.float(), ct2m_transformer.opt.joints_num)
    return pred_motions_denorm, pred_motions
