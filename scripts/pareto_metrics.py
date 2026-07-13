#!/usr/bin/env python3
"""Aggregate the accuracy-vs-latency sweep (internal analysis).

Reads /scratch/.../output/aaai_pareto_sweep/ produced by pareto_sweep.sh:
  kv_d{5,dense}/<idx>_pred_joints_<PROTO>.npy   + kv_d*.log   (per-protocol dt lines)
  mc_d{5,dense}/<idx>_baseline_maskcontrol_traj_i<E>x<L>.npy + mc_*.log (dt_gen lines)

Error metric: mean 3D distance on CONTROLLED frames (pelvis), the paper's convention.
Controlled frames = KF_DENSITY anchors (k evenly spaced) or every frame (dense) — recomputed
here exactly as the runners' build_pelvis_cond does (np.linspace(0, T-1, k).round()).
"""
import json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path("/scratch/ts1v23/workspace/MaskControl")
OUT = ROOT / "output/aaai_pareto_sweep"
PICKS = json.load(open(ROOT / "analysis/d1_picks/_index.json"))["picks"]

KV_PROTOS = ["B10x50", "B25x100", "B50x200", "B100x300", "M2", "M3", "RGAR-M3"]
MC_BUDGETS = ["10x50", "25x100", "50x200", "100x300", "100x600"]
ITER_TOTAL = {"B10x50": 150, "B25x100": 350, "B50x200": 700, "B100x300": 1300,
              "M2": 1600, "M3": 2525, "RGAR-M3": 2525,
              "10x50": 150, "25x100": 350, "50x200": 700, "100x300": 1300, "100x600": 1600}


def ctrl_frames(T, dens):
    if dens == "dense":
        return np.arange(T)
    k = int(dens)
    return np.unique(np.linspace(0, T - 1, k).round().astype(int))


def err_cm(gen, gt, dens):
    T = min(len(gen), len(gt))
    fr = ctrl_frames(T, dens)
    fr = fr[fr < T]
    return float(np.linalg.norm(gen[fr, 0] - gt[fr], axis=-1).mean() * 100)


def kv_times(log_path):
    """{proto: [dt, ...]} from D1's '    [PROTO]   12.3s' lines."""
    times = {}
    for line in open(log_path):
        m = re.match(r"\s+\[([\w\-]+)\]\s+([\d.]+)s", line)
        if m:
            times.setdefault(m.group(1), []).append(float(m.group(2)))
    return times


def mc_times(log_path):
    return [float(m.group(1)) for line in open(log_path)
            if (m := re.search(r"dt_gen=([\d.]+)s", line))]


def main():
    rows = []
    for dens in ("5", "dense"):
        kvt = kv_times(OUT / f"kv_d{dens}.log")
        for proto in KV_PROTOS:
            errs = []
            for p in PICKS:
                f = OUT / f"kv_d{dens}" / f"{p['idx']}_pred_joints_{proto}.npy"
                if not f.exists():
                    continue
                gt = np.load(p["saved_paths"]["gt_trajectory"])
                errs.append(err_cm(np.load(f), gt, dens))
            if errs:
                ts = kvt.get(proto, [])
                rows.append(dict(method="KV-Control", dens=dens, budget=proto,
                                 iters=ITER_TOTAL[proto], n=len(errs),
                                 err_cm=float(np.mean(errs)),
                                 t_s=float(np.median(ts)) if ts else float("nan")))
        for bud in MC_BUDGETS:
            errs = []
            for p in PICKS:
                f = OUT / f"mc_d{dens}" / f"{p['idx']}_baseline_maskcontrol_traj_i{bud}.npy"
                if not f.exists():
                    continue
                gt = np.load(p["saved_paths"]["gt_trajectory"])
                errs.append(err_cm(np.load(f), gt, dens))
            if errs:
                ts = mc_times(OUT / f"mc_d{dens}_i{bud}.log")
                rows.append(dict(method="MaskControl", dens=dens, budget=bud,
                                 iters=ITER_TOTAL[bud], n=len(errs),
                                 err_cm=float(np.mean(errs)),
                                 t_s=float(np.median(ts)) if ts else float("nan")))

    print(f"{'method':<12}{'dens':<7}{'budget':<10}{'iters':>6}{'n':>4}{'err_cm':>9}{'t_med_s':>9}")
    for r in rows:
        print(f"{r['method']:<12}{r['dens']:<7}{r['budget']:<10}{r['iters']:>6}{r['n']:>4}"
              f"{r['err_cm']:>9.2f}{r['t_s']:>9.2f}")
    json.dump(rows, open(OUT / "pareto_rows.json", "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, dens, title in zip(axes, ("5", "dense"),
                               ("sparse control (5 anchors)", "dense control (every frame)")):
        for meth, color in (("KV-Control", "#3d6bd8"), ("MaskControl", "#c95f2b")):
            pts = sorted([r for r in rows if r["method"] == meth and r["dens"] == dens],
                         key=lambda r: r["t_s"])
            if not pts:
                continue
            ax.plot([r["t_s"] for r in pts], [r["err_cm"] for r in pts],
                    "o-", color=color, label=meth)
            for r in pts:
                ax.annotate(r["budget"], (r["t_s"], r["err_cm"]), fontsize=7,
                            textcoords="offset points", xytext=(4, 4))
        ax.set_xlabel("median wall-clock per sample (s)")
        ax.set_ylabel("mean 3D control error (cm)")
        ax.set_yscale("log"); ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Refinement budget Pareto — 16 HumanML3D picks, per-method native operating point")
    fig.tight_layout()
    fig.savefig(OUT / "pareto.png", dpi=160)
    print(f"[pareto] plot -> {OUT / 'pareto.png'}")


if __name__ == "__main__":
    main()
