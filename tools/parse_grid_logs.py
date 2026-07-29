"""
Parse the authoritative grid logs into navigable JSON.

Sources -- ONLY these two directories are read:
  results/grid/story/story_main_{dataset}_{backbone}_analytic_etf_etf.log
  results/grid/story/story_2c_{dataset}_{backbone}_analytic_etf_etf.log
  results/grid/ablations/ablation_{dataset}_{backbone}.log

Stage order and ablation order are taken from the scripts that produced the logs
(run_paper_story.py and run_ablations.py), not guessed from the text:

  run_paper_story.py:49  stages 0, 1, 2, 2b, 2c, 3, 4, 5, 5b
  run_ablations.py:146   component: NCM, +Analytic ETF (a=0), +alpha=1, Full MAYA (a=0.6)
  run_ablations.py:166   lambda sweep: 0.01, 0.1, 0.5
  run_ablations.py:175   k sweep: 1, 16, 64, 128

The ablation log contains exactly 10 run_experiment blocks (3 + 3 + 4); the
"Standard NCM" component row is not printed there, so it is taken from story
stage 1, which run_ablations.py:64 reimplements with identical logic.

Outputs:
  results/parsed/{dataset}_{backbone}.json   per configuration
  results/parsed/all_results.json            everything, keyed "dataset/backbone"
  results/parsed/paper_tables.json           the draft's Table 1 / 2 / 3 shapes
"""

import json
import os
import pathlib
import re
from collections import OrderedDict

# Resolve paths relative to the repo root so this runs from anywhere.
ROOT = pathlib.Path(__file__).resolve().parent.parent
os.chdir(ROOT)

STORY_DIR = "results/grid/story"
ABL_DIR = "results/grid/ablations"
OUT_DIR = "results/parsed"

DATASETS = ["imagenet_r", "tinyimagenet", "objectnet"]
BACKBONES = ["dinov3", "resnet50", "siglip2"]

STAGE_LABELS = OrderedDict([
    ("0", "Vanilla NCM"), ("1", "Standard NCM"), ("2", "Basic Replay"),
    ("2b", "ER+MLP (Optimized Replay)"), ("2c", "Memory-Matched Raw Vectors"),
    ("3", "Node-Replay Only"), ("4", "Node Sweeps"), ("5", "MAYA Full"),
    ("5b", "MAYA + Linear Head"),
])

# "✅ [<name>] AIA: x% | Mem: y MB"  ->  stage key
STAGE_NAME_TO_KEY = {
    "0. Vanilla NCM": "0",
    "1. Standard NCM": "1",
    "2. Basic Replay": "2",
    "2b. Optimized Replay": "2b",
    "2c. Memory-Matched Raw Vectors": "2c",
    "3. Node-Replay Only": "3",
    "5b. TQM + Linear Head": "5b",
}

RE_STAGE = re.compile(r"\[([^\]]+)\] AIA: ([\d.]+)% \| Mem: ([\d.]+) MB")
RE_SUMMARY = re.compile(r"^(0|1|2b|2c|2|3|4|5b|5)\.\s+(.+?)\s+\| AIA: ([\d.]+)% \| Mem: ([\d.]+) MB(?:\s+\[F=([\d.]+)\])?")
RE_PURECIL = re.compile(r"Pure CIL Metrics -> AIA: ([\d.]+)% \| Forgetting: ([\d.]+)%")
RE_FOOTPRINT = re.compile(r"Bio-Graph Memory Footprint: ([\d.]+) MB")
RE_ALPHA = re.compile(r"Selected alpha: ([\d.]+)")
RE_SECTION = re.compile(r"\[(\d)/3\]")
RE_SWEEPFLOOR = re.compile(r"\[Sweep Floor=([\d.]+)\] AIA: ([\d.]+)% \| Mem: ([\d.]+) MB")


def read(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.readlines()


def parse_story(dataset, backbone):
    """Stages 0-5b. story_main holds everything except 2c, which has its own log."""
    stages, floors, meta = {}, [], {}

    main = read(f"{STORY_DIR}/story_main_{dataset}_{backbone}_analytic_etf_etf.log")
    if main is None:
        return None, None, None

    for line in main:
        m = RE_STAGE.search(line)
        if m and m.group(1) in STAGE_NAME_TO_KEY:
            stages[STAGE_NAME_TO_KEY[m.group(1)]] = {
                "aia": round(float(m.group(2)), 2),
                "mem_mb": round(float(m.group(3)), 2),
            }
        m = RE_SWEEPFLOOR.search(line)
        if m:
            floors.append({"floor": float(m.group(1)),
                           "aia": round(float(m.group(2)), 2),
                           "mem_mb": round(float(m.group(3)), 2)})

    # Stage 5 prints no "✅ [..]" line: take the last Pure CIL / footprint / alpha.
    tail = "".join(main)
    pc = RE_PURECIL.findall(tail)
    fp = RE_FOOTPRINT.findall(tail)
    al = RE_ALPHA.findall(tail)
    if pc:
        stages["5"] = {"aia": round(float(pc[-1][0]), 2),
                       "forgetting": round(float(pc[-1][1]), 2),
                       "mem_mb": round(float(fp[-1]), 2) if fp else None}
        if al:
            meta["stage5_selected_beta"] = float(al[-1])

    # Stage 4 = best node sweep; the summary line records which floor won.
    for line in main:
        m = RE_SUMMARY.match(line.strip())
        if m and m.group(1) == "4":
            stages["4"] = {"aia": round(float(m.group(3)), 2),
                           "mem_mb": round(float(m.group(4)), 2),
                           "best_floor": float(m.group(5)) if m.group(5) else None}

    sec = read(f"{STORY_DIR}/story_2c_{dataset}_{backbone}_analytic_etf_etf.log")
    if sec:
        for line in sec:
            m = RE_STAGE.search(line)
            if m and m.group(1) in STAGE_NAME_TO_KEY:
                stages[STAGE_NAME_TO_KEY[m.group(1)]] = {
                    "aia": round(float(m.group(2)), 2),
                    "mem_mb": round(float(m.group(3)), 2),
                }
        budget = re.search(r"Stage 2c budget: ([\d.]+) MB", "".join(sec))
        if budget:
            meta["stage2c_budget_mb"] = float(budget.group(1))

    return stages, floors, meta


def parse_ablation(dataset, backbone, ncm_row):
    """10 run_experiment blocks, split by the [1/3] [2/3] [3/3] section markers."""
    lines = read(f"{ABL_DIR}/ablation_{dataset}_{backbone}.log")
    if lines is None:
        return None

    section, blocks = 0, {1: [], 2: [], 3: []}
    pending_mem = pending_beta = None
    restarts = 0
    for line in lines:
        m = RE_SECTION.search(line)
        if m:
            section = int(m.group(1))
            # A second "[1/3]" means the job aborted and re-ran; keep only the
            # last complete attempt (tinyimagenet/siglip2 restarts mid-section).
            if section == 1 and any(blocks.values()):
                blocks = {1: [], 2: [], 3: []}
                pending_mem = pending_beta = None
                restarts += 1
            continue
        m = RE_FOOTPRINT.search(line)
        if m:
            pending_mem = round(float(m.group(1)), 2)
            continue
        m = RE_ALPHA.search(line)
        if m:
            pending_beta = float(m.group(1))
            continue
        m = RE_PURECIL.search(line)
        if m and section:
            blocks[section].append({"aia": round(float(m.group(1)), 2),
                                    "forgetting": round(float(m.group(2)), 2),
                                    "mem_mb": pending_mem,
                                    "tuned_beta": pending_beta})
            pending_mem = pending_beta = None

    n = {k: len(v) for k, v in blocks.items()}
    if n != {1: 3, 2: 3, 3: 4}:
        raise ValueError(f"{dataset}/{backbone}: expected 3/3/4 blocks, got {n}")

    comp_steps = ["+Analytic ETF (beta=0, global only)", "+beta=1 (episodic only)",
                  "Full MAYA (beta=0.6)"]
    component = [dict(step="Standard NCM", **ncm_row)]
    component += [dict(step=s, **b) for s, b in zip(comp_steps, blocks[1])]

    lambda_sweep = [dict(**{"lambda": l}, **b) for l, b in zip([0.01, 0.1, 0.5], blocks[2])]
    k_sweep = [dict(k=k, **b) for k, b in zip([1, 16, 64, 128], blocks[3])]

    return {"component": component, "lambda_sweep": lambda_sweep, "k_sweep": k_sweep,
            "log_restarts_discarded": restarts}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    everything, missing = OrderedDict(), []

    for ds in DATASETS:
        for bb in BACKBONES:
            stages, floors, meta = parse_story(ds, bb)
            if stages is None:
                missing.append(f"{ds}/{bb} story")
                continue

            ncm = stages.get("1")
            ncm_row = ({"aia": ncm["aia"], "mem_mb": ncm["mem_mb"], "forgetting": None,
                        "source": "story_main stage 1 (not printed in ablation log)"}
                       if ncm else {"aia": None, "mem_mb": None, "forgetting": None})
            abl = parse_ablation(ds, bb, ncm_row)
            if abl is None:
                missing.append(f"{ds}/{bb} ablation")

            record = OrderedDict([
                ("dataset", ds), ("backbone", bb), ("seed", 42),
                ("sources", {
                    "story_main": f"{STORY_DIR}/story_main_{ds}_{bb}_analytic_etf_etf.log",
                    "story_2c": f"{STORY_DIR}/story_2c_{ds}_{bb}_analytic_etf_etf.log",
                    "ablation": f"{ABL_DIR}/ablation_{ds}_{bb}.log",
                }),
                ("notes", meta),
                ("stage_labels", dict(STAGE_LABELS)),
                ("story", OrderedDict((k, stages[k]) for k in STAGE_LABELS if k in stages)),
                ("story_node_sweep_floors", floors),
                ("ablation", abl),
            ])
            everything[f"{ds}/{bb}"] = record
            with open(f"{OUT_DIR}/{ds}_{bb}.json", "w") as f:
                json.dump(record, f, indent=2)

    with open(f"{OUT_DIR}/all_results.json", "w") as f:
        json.dump(everything, f, indent=2)

    # ── Paper table shapes ────────────────────────────────────────────────
    t1, t2, t3 = OrderedDict(), OrderedDict(), OrderedDict()
    for key, r in everything.items():
        s, a = r["story"], r["ablation"]
        g = lambda k, f="aia": (s[k][f] if k in s and s[k].get(f) is not None else None)
        t1[key] = {"NCM": g("1"), "ER+MLP": g("2b"), "Nodes Only": g("3"), "MAYA Full": g("5"),
                   "Basic Replay": g("2"), "Raw Vec (memory-matched)": g("2c"),
                   "memory_mb": {"NCM": g("1", "mem_mb"), "ER+MLP": g("2b", "mem_mb"),
                                 "Nodes Only": g("3", "mem_mb"), "MAYA Full": g("5", "mem_mb"),
                                 "Basic Replay": g("2", "mem_mb"),
                                 "Raw Vec (memory-matched)": g("2c", "mem_mb")}}
        t2[key] = {"MAYA Full": {"aia": g("5"), "mem_mb": g("5", "mem_mb")},
                   "Deep SLDA": None, "Exact ACL": None, "RanPAC": None,
                   "_note": "SLDA/ACL/RanPAC are not present in grid/ablations or "
                            "grid/story; only source found was results/comparisons/*.json "
                            "(1 Jul, from standalone run_comparisons.py). Confirm or re-run."}
        if a:
            c = {x["step"]: x for x in a["component"]}
            t3[key] = {"Standard NCM": c["Standard NCM"]["aia"],
                       "+Analytic ETF": c["+Analytic ETF (beta=0, global only)"]["aia"],
                       "+beta=1 (episodic only)": c["+beta=1 (episodic only)"]["aia"],
                       "Full MAYA (beta=0.6)": c["Full MAYA (beta=0.6)"]["aia"],
                       "forgetting": {k: v["forgetting"] for k, v in c.items()}}

    with open(f"{OUT_DIR}/paper_tables.json", "w") as f:
        json.dump({"table1_mechanics": t1, "table2_sota": t2, "table3_components": t3}, f, indent=2)

    print(f"Parsed {len(everything)}/9 configurations -> {OUT_DIR}/")
    if missing:
        print("MISSING:", ", ".join(missing))

    print(f"\n{'config':26} {'NCM':>6} {'Replay':>7} {'RawVec':>7} {'ER+MLP':>7} {'Nodes':>7} {'MAYA':>7} {'MAYAmem':>8}")
    print("-" * 84)
    for key, r in everything.items():
        s = r["story"]
        v = lambda k: f"{s[k]['aia']:.1f}" if k in s else "  -  "
        print(f"{key:26} {v('1'):>6} {v('2'):>7} {v('2c'):>7} {v('2b'):>7} {v('3'):>7} {v('5'):>7} "
              f"{(str(s['5']['mem_mb']) if '5' in s else '-'):>8}")

    print(f"\n{'config':26} {'NCM':>6} {'+ETF':>6} {'beta=1':>7} {'MAYA':>6} | {'K=1':>6} {'K=16':>6} {'K=64':>6} {'K=128':>6}")
    print("-" * 84)
    for key, r in everything.items():
        a = r["ablation"]
        if not a:
            continue
        c = [x["aia"] for x in a["component"]]
        k = [x["aia"] for x in a["k_sweep"]]
        cs = " ".join(f"{x:6.1f}" if x is not None else "     -" for x in c)
        ks = " ".join(f"{x:6.1f}" for x in k)
        print(f"{key:26} {cs} | {ks}")


if __name__ == "__main__":
    main()
