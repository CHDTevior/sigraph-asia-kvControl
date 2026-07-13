#!/usr/bin/env python3
"""A1 probe — per-iteration convergence traces of Stage-1/Stage-2 refinement (RGAR Day-1).

TRAIN-split probe (test split untouched). Anchors follow the TRAINING distribution (codex
019f59cc P1): density category round-robin over {1,2,5,49,196}; for 49/196 the realized count
scales with effective length (k = max(1, round(n_eff*d/196))); positions are seeded-RANDOM per
pick and stored as anchor_frames, which D1 consumes directly (evenly spaced linspace anchors
would bias entry-loss calibration optimistic).

Statistics the RGAR design rests on (each must be re-verified here, not taken from mined logs):
  (a) %% of samples whose Stage-2 EXIT loss is worse than ENTRY;
  (b) %% entering Stage-2 already below tau (anchor MSE, m^2);
  (c) Stage-1 wasted-iteration fraction (past the last >1%% hysteretic improvement —
      an inter-observation metric; the final update has no post-update observation).

Usage:
  python scripts/a1_trace_probe.py --build
  python scripts/a1_trace_probe.py --run          # GPU; writes runs/<stamp>/
  python scripts/a1_trace_probe.py --analyze [--run_dir runs/<stamp>]
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

REPO = Path("/scratch/ts1v23/workspace/MaskControl")
PROBE = REPO / "output/a1_trace_probe"
DS = REPO / "dataset/HumanML3D"
N_SAMPLES = 32
SEED = 20260713
DENSITIES = [1, 2, 5, 49, 196]


def _n_eff(n_frames):
    return 4 * (min(int(n_frames), 196) // 4)      # exactly D1's effective length


def build():
    sys.path.insert(0, str(REPO))
    from utils.motion_process import recover_from_ric
    import torch
    ids = [l.strip() for l in open(DS / "train.txt") if l.strip() and not l.startswith("M")]
    rng = np.random.RandomState(SEED)
    rng.shuffle(ids)
    picks, k = [], 0
    for cid in ids:
        f = DS / "new_joint_vecs" / f"{cid}.npy"
        t = DS / "texts" / f"{cid}.txt"
        if not (f.exists() and t.exists()):
            continue
        feat = np.load(f)
        if not (40 <= len(feat) <= 196) or not np.isfinite(feat).all():
            continue
        text = open(t).readline().split("#")[0].strip()
        if not text:
            continue
        dens = DENSITIES[k % len(DENSITIES)]
        neff = _n_eff(len(feat))
        n_anchor = dens if dens <= 5 else max(1, round(neff * dens / 196))
        if n_anchor > neff:
            continue
        anchors = np.sort(rng.choice(neff, size=n_anchor, replace=False))  # TRAINING-style random
        joints = recover_from_ric(torch.from_numpy(feat).float()[None], 22)[0].numpy()
        d = PROBE / "picks" / cid
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "gt_joints.npy", joints.astype(np.float32))
        np.save(d / "gt_trajectory.npy", joints[:, 0].astype(np.float32))
        (d / "text.txt").write_text(text + "\n")
        picks.append({"idx": cid, "category": f"train-probe-d{dens}", "text": text,
                      "n_frames": int(len(feat)), "density": int(dens),
                      "anchor_frames": anchors.tolist(),
                      "saved_paths": {"gt_joints": str(d / "gt_joints.npy"),
                                      "gt_trajectory": str(d / "gt_trajectory.npy"),
                                      "gt_feature263": str(f), "text": str(d / "text.txt")}})
        k += 1
        if k >= N_SAMPLES:
            break
    if len(picks) != N_SAMPLES:
        raise SystemExit(f"only {len(picks)} eligible probe picks; expected {N_SAMPLES}")
    json.dump({"picks": picks,
               "note": "TRAIN-split probe; anchors random per training distribution, counts "
                       "scaled by length for the 49/196 categories"},
              open(PROBE / "_index.json", "w"), indent=1)
    print(f"[a1] built {len(picks)} probe picks -> {PROBE / 'picks'}")


def run():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = PROBE / "runs" / stamp
    (run_dir / "traces").mkdir(parents=True)
    env = dict(os.environ, TTT_TRACE_DIR=str(run_dir / "traces"),
               USE_V8_V4_CONTROL_NOQRES="1", STAGE2_OPTIMIZER="adam")
    env.pop("KF_DENSITY", None)                    # anchors come from anchor_frames per pick
    cmd = [sys.executable, "-u", str(REPO / "scripts/D1_stage2_kv_inference.py"),
           "--picks_index", str(PROBE / "_index.json"), "--protocols", "M3",
           "--out_dir", str(run_dir / "out"), "--overwrite"]
    print(f"[a1] run -> {run_dir}")
    r = subprocess.run(cmd, cwd=str(REPO), env=env)
    if r.returncode != 0:
        raise SystemExit(f"probe run failed rc={r.returncode}")
    idx = json.load(open(PROBE / "_index.json"))["picks"]
    import shutil
    shutil.copy(PROBE / "_index.json", run_dir / "index_snapshot.json")
    json.dump({"run_dir": str(run_dir), "protocol": "M3", "n_expected": len(idx),
               "pick_order": [p["idx"] for p in idx]},
              open(run_dir / "manifest.json", "w"), indent=1)
    print(f"[a1] manifest -> {run_dir / 'manifest.json'}")


def analyze(run_dir):
    run_dir = Path(run_dir) if run_dir else sorted((PROBE / "runs").glob("*"))[-1]
    mf = json.load(open(run_dir / "manifest.json"))
    snap = run_dir / "index_snapshot.json"
    idx = {p["idx"]: p for p in
           json.load(open(snap if snap.exists() else PROBE / "_index.json"))["picks"]}
    traces = sorted((run_dir / "traces").glob("trace_*.json"))
    if len(traces) != mf["n_expected"]:
        raise SystemExit(f"{len(traces)} traces != expected {mf['n_expected']} — partial run; "
                         f"refusing to compute statistics from an incomplete set")
    # pair by the explicit tag D1 stamped into each trace — never by file timestamp (codex
    # round-2: wall-clock filenames are not guaranteed chronological)
    by_tag = {}
    for tf in traces:
        tr = json.load(open(tf))
        cid = (tr.get("tag") or "").split(":")[0]
        if not cid or cid in by_tag:
            raise SystemExit(f"trace {tf.name}: missing/duplicate tag {tr.get('tag')!r}")
        by_tag[cid] = tr
    if set(by_tag) != set(mf["pick_order"]):
        raise SystemExit(f"trace tags != manifest picks; missing "
                         f"{sorted(set(mf['pick_order']) - set(by_tag))}")
    rows = []
    for order, cid in enumerate(mf["pick_order"]):
        tr = by_tag[cid]
        pick = idx[cid]
        if not tr.get("complete"):
            raise SystemExit(f"trace for {cid} incomplete (exception mid-run)")
        if tr.get("text", "")[:60] != pick["text"][:60]:
            raise SystemExit(f"trace/pick text mismatch at position {order}: "
                             f"{tr.get('text','')[:40]!r} vs {pick['text'][:40]!r}")
        s2 = tr["stage2"]
        if not tr["stage1"] or not s2 or tr["stage2_exit"] is None:
            raise SystemExit(f"trace for {cid}: empty stage data — cannot contribute; aborting")
        waste = []
        for st in tr["stage1"]:
            L = st["losses"]
            if len(L) < 2:
                continue
            best, last_gain = L[0], 0
            for i, v in enumerate(L[1:], 1):
                if v < best * 0.99:
                    best, last_gain = v, i
            waste.append(1.0 - last_gain / (len(L) - 1))
        rows.append(dict(idx=cid, density=pick["density"], entry=s2[0],
                         exit=tr["stage2_exit"], worse=tr["stage2_exit"] > s2[0],
                         s1_waste=float(np.mean(waste)) if waste else float("nan")))
    n = len(rows)
    print(f"[a1] {n} complete traces from {run_dir}")
    for tau in (1e-4, 1e-5, 1e-6):
        m = sum(r["entry"] < tau for r in rows)
        print(f"  entry anchor-MSE < {tau:.0e} m^2: {m}/{n} ({100*m/n:.0f}%)")
    worse = sum(r["worse"] for r in rows)
    print(f"  Stage-2 exit WORSE than entry: {worse}/{n} ({100*worse/n:.0f}%)")
    w = [r["s1_waste"] for r in rows if np.isfinite(r["s1_waste"])]
    print(f"  Stage-1 wasted-interval fraction (hysteretic >1% rule): "
          f"mean {100*np.mean(w):.0f}%  median {100*np.median(w):.0f}%")
    for d in DENSITIES:
        rs = [r for r in rows if r["density"] == d]
        if rs:
            print(f"    density {d:>3}: n={len(rs)}  s2-worse {sum(r['worse'] for r in rs)}/{len(rs)}"
                  f"  median entry {np.median([r['entry'] for r in rs]):.2e}")
    json.dump(rows, open(run_dir / "a1_stats.json", "w"), indent=1)
    print(f"[a1] stats -> {run_dir / 'a1_stats.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--run_dir", type=str, default=None)
    a = ap.parse_args()
    PROBE.mkdir(parents=True, exist_ok=True)
    if a.build:
        build()
    if a.run:
        run()
    if a.analyze:
        analyze(a.run_dir)
    if not (a.build or a.run or a.analyze):
        raise SystemExit("pass --build / --run / --analyze")
