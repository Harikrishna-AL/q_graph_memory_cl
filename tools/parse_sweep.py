"""
Aggregate the seed and target-geometry sweeps written by tools/run_sweeps.sh.

  python tools/parse_sweep.py                  # aggregate whatever exists
  python tools/parse_sweep.py --check-targets  # verify the target frames only

--check-targets is worth running before the sweep: it reports, for each target
geometry and dimension, whether the generated frame is actually what it claims
to be (unit norm, and pairwise cosine -1/(k-1) for a simplex ETF).
"""

import argparse
import json
import os
import pathlib
import statistics as st
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
os.chdir(ROOT)

SWEEP = pathlib.Path("results/sweep")
OUT = pathlib.Path("results/parsed")

# run_ablations.py writes these exact labels into the JSON. Matched by prefix
# so the parser tolerates the alternative naming used in parse_grid_logs.py.
STEP_NCM = "Standard NCM"
STEP_GLOBAL = "+Analytic ETF"
STEP_EPISODIC = "+alpha=1"
STEP_FULL = "Full MAYA"


def check_targets():
    """Report what each target geometry actually produces."""
    import sys
    import types
    sys.path.insert(0, str(ROOT))
    from src.config import Config
    import src.model as M

    cls = next(v for v in vars(M).values()
               if isinstance(v, type) and hasattr(v, "_generate_targets"))
    o = types.SimpleNamespace(device="cpu")
    o._generate_simplex_etf = lambda k, d: cls._generate_simplex_etf(o, k, d)

    k = int(getattr(Config, "BIO_ETF_MAX_CLASSES", 500))
    print(f"Target frames for k={k} slots (BIO_ETF_MAX_CLASSES)\n")
    print(f"{'geometry':9} {'p':>5}  {'shape':13} {'‖v‖ range':>15} {'max cos':>9}  verdict")
    print("-" * 78)
    for geom in ("etf", "onehot", "random"):
        for d in (getattr(Config, "BIO_ALIGN_DIM", 256), 512):
            try:
                Config.SEED = 42
                T = cls._generate_targets(o, k, d, geom)
                n = T.norm(dim=1)
                G = T @ T.t()
                G.fill_diagonal_(-9)
                mx = float(G.max())
                unit = abs(float(n.min()) - 1) < 1e-3 and abs(float(n.max()) - 1) < 1e-3
                if geom == "etf":
                    ideal = -1.0 / (k - 1)
                    ok = unit and abs(mx - ideal) < 1e-2
                    verdict = "true simplex ETF" if ok else \
                        f"NOT an ETF (ideal cos {ideal:+.4f}); random-frame fallback"
                elif geom == "onehot":
                    verdict = "orthonormal" if unit and abs(mx) < 1e-6 else "unexpected"
                else:
                    verdict = "random unit frame" if unit else "unexpected"
                print(f"{geom:9} {d:>5}  {str(tuple(T.shape)):13} "
                      f"{float(n.min()):.3f}-{float(n.max()):.3f}    {mx:+.4f}  {verdict}")
            except ValueError as e:
                print(f"{geom:9} {d:>5}  {'--':13} {'--':>15} {'--':>9}  refused: {str(e)[:34]}...")
    print("\nA simplex ETF over k points spans only k-1 dimensions, so p must be "
          "≥ k-1\nfor the construction to produce a genuine ETF.")


def load_sweep():
    """Group sweep files by (dataset, backbone, geometry, align_dim)."""
    groups = defaultdict(dict)   # key -> {seed: record}
    for f in sorted(SWEEP.glob("ablation_*.json")):
        with open(f) as fh:
            rec = json.load(fh)
        c = rec.get("config")
        if not c:
            print(f"  skipping {f.name}: no config block (pre-sweep file)")
            continue
        key = (c["dataset"], c["backbone"], c.get("target_geometry", "etf"), c.get("align_dim"))
        groups[key][c["seed"]] = rec
    return groups


def agg(vals):
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return vals[0], None, 1
    return st.mean(vals), st.stdev(vals), len(vals)


def fmt(m, s, n):
    if m is None:
        return "   --   "
    return f"{m:5.1f}" + (f" ±{s:4.2f}" if s is not None else f" (n={n})")


def component(rec, step):
    """AIA in percent for a component-ablation row, matched by label prefix."""
    rows = rec.get("component_ablation") or rec.get("component") or []
    for x in rows:
        if x["step"].startswith(step):
            a = x["aia"]
            return a * 100 if a <= 1.0 else a
    return None


def baseline_p256():
    """The existing p=256 results, for comparison against the corrected frame."""
    f = OUT / "all_results.json"
    if not f.exists():
        return {}
    with open(f) as fh:
        parsed = json.load(fh)
    out = {}
    for key, rec in parsed.items():
        comp = {x["step"]: x["aia"] for x in rec["ablation"]["component"]}
        out[key] = {
            "global": next((v for k, v in comp.items() if k.startswith("+Analytic ETF")), None),
            "full": next((v for k, v in comp.items() if k.startswith("Full MAYA")), None),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-targets", action="store_true",
                    help="verify the generated target frames and exit")
    a = ap.parse_args()
    if a.check_targets:
        check_targets()
        return

    if not SWEEP.exists() or not any(SWEEP.glob("ablation_*.json")):
        print(f"No sweep results in {SWEEP}/. Run tools/run_sweeps.sh first.")
        return

    groups = load_sweep()
    base = baseline_p256()

    # ── Everything that has landed, with the p=256 delta ────────────────
    n_files = sum(len(v) for v in groups.values())
    print("=" * 78)
    print(f"LANDED: {n_files} run(s)")
    print("=" * 78)
    print(f"{'config':22} {'geom':>7} {'p':>5} {'seed':>5} "
          f"{'global':>7} {'episod':>7} {'full':>7}  {'vs p=256 full':>13}  {'K=1':>6}")
    print("-" * 78)
    for key in sorted(groups):
        ds, bb, geom, dim = key
        for seed, rec in sorted(groups[key].items()):
            g = component(rec, STEP_GLOBAL)
            e = component(rec, STEP_EPISODIC)
            f = component(rec, STEP_FULL)
            k1 = rec.get("k_sweep", [{}])[0].get("aia")
            k1 = k1 * 100 if k1 is not None and k1 <= 1.0 else k1
            b = base.get(f"{ds}/{bb}", {}).get("full")
            d = f"{f - b:+6.1f}" if (b is not None and f is not None and geom == "etf") else "     --"
            print(f"{ds+'/'+bb:22} {geom:>7} {str(dim):>5} {seed:>5} "
                  f"{g:7.2f} {e:7.2f} {f:7.2f}  {d:>13}  {k1:6.1f}")
    print()

    # ── Item 2: target geometry, at matched dimension ─────────────────────
    geoms = sorted({k[2] for k in groups})
    if len(geoms) > 1:
        print("\n" + "=" * 78)
        print("ITEM 2 -- target geometry (MAYA full, matched align_dim)")
        print("=" * 78)
        print(f"{'config':24} " + " ".join(f"{g:>13}" for g in geoms) + "   ETF-onehot")
        print("-" * 78)
        for ds, bb in sorted({(k[0], k[1]) for k in groups}):
            cells, vals = [], {}
            for g in geoms:
                recs = [r for k, r in groups.items() if k[:3] == (ds, bb, g) for r in [r]]
                seeds = {}
                for k, v in groups.items():
                    if k[:3] == (ds, bb, g):
                        seeds.update(v)
                v = [component(r, STEP_FULL) for r in seeds.values()]
                v = [x for x in v if x is not None]
                m, s, n = agg(v)
                vals[g] = m
                cells.append(fmt(m, s, n))
            d = (vals.get("etf") - vals.get("onehot")
                 if vals.get("etf") is not None and vals.get("onehot") is not None else None)
            print(f"{ds+'/'+bb:24} " + " ".join(f"{c:>13}" for c in cells) +
                  (f"   {d:+6.1f}" if d is not None else "       --"))

    # ── Item 1: seeds ─────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("ITEM 1 -- seed sweep (target geometry = etf, default align_dim)")
    print("=" * 78)
    print(f"{'config':24} {'global only':>14} {'MAYA full':>14}  {'seeds':>6}")
    print("-" * 78)
    summary = {}
    for key in sorted(groups):
        ds, bb, geom, dim = key
        if geom != "etf" or dim not in (None, 256):
            continue
        seeds = groups[key]
        g = [component(r, STEP_GLOBAL) for r in seeds.values()]
        f = [component(r, STEP_FULL) for r in seeds.values()]
        g, f = [x for x in g if x is not None], [x for x in f if x is not None]
        gm, gs, gn = agg(g)
        fm, fs, fn = agg(f)
        print(f"{ds+'/'+bb:24} {fmt(gm, gs, gn):>14} {fmt(fm, fs, fn):>14}  {sorted(seeds):>6}"
              if False else
              f"{ds+'/'+bb:24} {fmt(gm, gs, gn):>14} {fmt(fm, fs, fn):>14}  {len(seeds):>6}")
        summary[f"{ds}/{bb}"] = {"global_only": {"mean": gm, "std": gs, "n": gn},
                                 "maya_full": {"mean": fm, "std": fs, "n": fn},
                                 "seeds": sorted(seeds)}

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "sweep_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {OUT}/sweep_summary.json")
    print("Margins smaller than roughly 2x the reported std should not carry a claim.")


if __name__ == "__main__":
    main()
