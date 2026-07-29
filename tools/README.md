# tools/

Analysis and figure scripts. The experiment pipeline itself stays at the repo
root (`run_paper_grid.py` → `run_paper_story.py` / `run_ablations.py` → `main.py`,
plus the standalone `run_comparisons.py`).

All scripts here resolve paths relative to the repo root, so they can be run from
anywhere: `python tools/<script>.py`.

## Live

| Script | Purpose |
|---|---|
| `parse_grid_logs.py` | Parses `results/grid/{story,ablations}/*.log` into `results/parsed/`. **Run this first** — everything else reads its output. |
| `make_paper_tables.py` | Emits LaTeX for the paper's Tables 1/2/3 from `results/parsed/` + `results/comparisons/`. |
| `plot_k_sweep.py` | Figure 3 (node-budget ablation), read from `results/parsed/`. |
| `generate_pareto_figure.py` | Accuracy–memory Pareto audit across all nine configs. |
| `generate_family_figure.py` | Two-family comparison (statistics-only vs data-storing). |

## legacy/

Superseded scripts, kept for reference. **Several hardcode result numbers inline**
— that is how the April values reached Table 3 and Figure 3 of `tmlr_draft_3.pdf`
and stayed there after the June re-run. Do not use them to generate anything that
goes into the paper; use the live scripts, which read `results/parsed/`.

Known offenders with inlined numbers: `plot_k_sweep.py` (original),
`plot_pareto.py`, `generate_results_plots.py`, `generate_inference_fig.py`,
`generate_paper_plots.py`.
