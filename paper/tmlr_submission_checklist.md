# MAYA / TMLR — submission checklist

Supersedes `tmlr_claim_calibration.md` (April 2026), which was written against
`tmlr_draft.pdf` and TinyImageNet-only results.

Status: 2026-07-28, against `tmlr_draft_3.pdf`.

**Data of record: `results/parsed/`** — parsed from `results/grid/{story,ablations}/`
by `parse_grid_logs.py`. Do **not** use `results/ablation_*.json` at the repo
root; those are 18 April and superseded by the 10 June grid run.

**Already verified, do not redo:** Table 1 matches the June story logs exactly
(two NCM cells round differently: 95.3→95.2, 47.8→47.9).

---

## Done on 2026-07-28

Codebase:
- Stale April results quarantined to `results/_deprecated_2026-04/` with a README.
- Repo reorganised: pipeline at root, analysis in `tools/`, superseded scripts in `tools/legacy/`, paper in `paper/`.
- `tools/parse_grid_logs.py`, `make_paper_tables.py`, `plot_k_sweep.py`, `generate_pareto_figure.py`,
  `generate_family_figure.py` all read `results/parsed/`. **No result number is hardcoded in any live script.**

Paper (`paper/sections/`, tables auto-generated into `paper/tables/`):
- Tables 1--4 regenerated from the June data; Table 4 (matched-memory) is new.
- Figure 3 regenerated; now shows the K=1 anomaly against both references instead of cropping it.
- Ablation, Results, Experiments sections rewritten against correct numbers.
- Method text corrected to match the implementation (see 5.2/5.3 below).
- Reproducibility table and Limitations section added.
- `paper_sections.tex` no longer duplicates the Results section (was a label collision).
- `algorithm_draft.tex` reduced to preamble; the algorithm bodies now live only in
  `methodology_template.tex` (they were defining `alg:maya_train`/`alg:maya_infer` twice).

**New finding, larger than the ones originally listed:** Algorithm 1 as published did not
describe the code at all. The paper described streaming k-means with a distance threshold
$\tau$ and a momentum update $\alpha$; the implementation buffers a block, runs **offline
k-means in the aligned space** (`model.py:1329-1345`), stores **raw backbone-space cluster
means**, and enforces $K$ by **pruning on an importance score** (`model.py:890-927`). There
is no $\tau$ and no $\alpha$ anywhere in the node path. Algorithm 1, Eq. 2 and the
hyperparameter table have been rewritten to match. Note this makes the §2.5
"clustering after projection" claim **correct** -- clustering is genuinely done in aligned
space; only node *storage* is in backbone space.

---

## 2026-07-29 — items 1 and 2 wired up, and a finding that reframes item 2

`tools/run_sweeps.sh geometry` (item 2, cheap) and `tools/run_sweeps.sh seeds`
(item 1, long pole); aggregate with `tools/parse_sweep.py`. Both write only to
`results/sweep/`, so `results/grid/` and `results/comparisons/` are untouched.
`--seed` and `--target-geometry` added to `run_ablations.py`, `run_paper_story.py`
and `run_comparisons.py`; `Config.BIO_TARGET_GEOMETRY` selects
`etf` / `onehot` / `random` via `_generate_targets` in `src/model.py`.

### The ETF targets were never an ETF

`python tools/parse_sweep.py --check-targets`:

| geometry | p | ‖v‖ range | max pairwise cos | verdict |
|---|---|---|---|---|
| etf | **256 (the setting every result used)** | 0.645–0.792 | **+0.0954** | **not an ETF** |
| etf | 512 | 1.000–1.000 | −0.0020 | true simplex ETF |
| onehot | 512 | 1.000–1.000 | +0.0000 | orthonormal |
| random | 512 | 1.000–1.000 | +0.1996 | random unit frame |

A simplex ETF over $k$ points spans $k-1$ dimensions. With
`BIO_ETF_MAX_CLASSES = 500` and `BIO_ALIGN_DIM = 256`, the construction takes its
`d < k-1` fallback branch — an SVD of a random matrix — producing targets that are
neither equinorm nor equiangular. The ideal ETF cosine is $-1/499 = -0.0020$; the
frame actually used has max pairwise cosine $+0.0954$.

**Every number in `tmlr_draft_3.pdf` was produced with a random projection frame,
not the Equiangular Tight Frame the paper is built around.** This affects the
abstract, §2.3, §3.2 and the method's name.

Consequences:

- Item 2 is no longer only "ETF vs one-hot". It is the first test of whether the
  paper's central mechanism does anything at all, since it has never been enabled.
  The published results sit close to the `random` arm.
- The sweep runs at `--align-dim 512` so all three arms share a dimension **and**
  the ETF arm is a genuine ETF. Expect the `etf @ p=512` numbers to differ from
  the current tables for this reason alone.
- Either fix `BIO_ETF_MAX_CLASSES` to the real class count and require
  `BIO_ALIGN_DIM ≥ C-1`, or keep p=512 as the default. Do not ship p=256 with an
  ETF claim.

---

## Blocking decisions

- [x] **B1. Source of record for SLDA / ACL / RanPAC.** They appear nowhere in
  `results/grid/`. The only file is `results/comparisons/*.json` (1 Jul, from the
  standalone `run_comparisons.py`). Confirm it is current, or re-run into the grid.
  **Table 2 and every accuracy-per-byte claim depend on this.**
- [ ] **B2. Framing.** The draft sells state-of-the-art accuracy; the evidence
  supports a decomposition + honest-tradeoff paper. TMLR's criteria are "claims
  supported by evidence" and "some audience would be interested" — not SOTA.
  Commit to one; it decides which experiments below matter.

---

## Phase 1 — Data integrity

Nothing downstream is trustworthy until this is done.

- [x] **1.1 Regenerate Table 3 and Figure 3 from `results/parsed/`.** `[core]`
  Both are built on April data. **All nine Table 3 rows are wrong** by 0.2–1.6 pp,
  always optimistic. Figure 3's K-sweep is stale too (draft 30.1/70.4/75.9/76.3 →
  actual 36.8/70.0/75.0/75.4).
- [ ] **1.2 Unify the evaluation protocol, or state it per table.** `[core]`
  Story uses an 80/20 per-task shuffle (`run_paper_story.py:200`); ablations use
  `Config.TRAIN_TEST_SPLIT`. Table 1 and Table 3 therefore cannot be read against
  each other — the 76.0 vs 75.4 gap on ObjectNet/DINOv3 is protocol, not method.
- [ ] **1.3 Run 3–5 seeds per configuration; report mean ± std.** `[core]`
  Analytic solves over cached features in `cache/`, so this is cheap. Without it
  no margin under ~2 pp in this paper is defensible. **Start first — long pole,
  runs unattended.**
- [ ] **1.4 Resolve the K=1 collapse.** `[core]` Systematic across all nine configs
  (5.8%–70.8%), always far below the `+Analytic ETF` row that uses one target per
  class. Either it is a bug in the episodic branch at K=1, or Figure 3's
  "representational bottleneck" reading is wrong. Both cannot hold.

---

## Phase 2 — New comparisons that test the paper's own claims

These come **before** the claims rewrite, because their outcomes decide what you
can claim. Items 2.1–2.3 share one code path (the target matrix `M`).

- [ ] **2.1 MAYA with one-hot targets instead of ETF targets.** `[core]`
  §2.4 states the distinction from ACIL explicitly: *"Existing analytic CL methods
  solve a standard least-squares regression from features to one-hot labels…
  MAYA instead solves for a projection onto pre-defined ETF targets."* **This is
  the paper's central methodological claim and it is never tested.** ACL-vs-MAYA
  does not isolate it — ACL differs in pipeline and episodic memory too. Keep
  everything in MAYA fixed, swap `M` for one-hot. **Highest priority experiment
  in this document.**
- [ ] **2.2 Random orthogonal targets, as a control.** `[core]` Same swap.
  Separates "ETF geometry specifically" from "any fixed target frame beats one-hot."
  2.1 + 2.2 together triangulate the claim properly.
- [ ] **2.3 Per-class covariance baseline (FeCAM / per-class Mahalanobis).** `[core]`
  The whole ObjectNet narrative rests on §3.3: SLDA/ACL *"implicitly assume
  homoscedasticity… Real-world vision datasets deeply violate this."* FeCAM
  (Goswami et al., NeurIPS 2023) relaxes exactly that assumption **without**
  episodic memory. Without it, "isn't the fix just per-class covariance rather
  than a node bank?" has no answer. Exemplar-free, so cheap.
- [ ] **2.4 Joint linear probe upper bound.** `[cheap]` Linear classifier trained
  offline on all classes at once from cached features. There is currently **no
  ceiling reference anywhere in the paper** — a reader cannot tell whether 76% on
  ObjectNet is near-optimal or leaving 15 points on the table.
- [ ] **2.5 Offline k-means centroids per class at matched K.** `[cheap]`
  Tests whether *streaming* node construction matters or offline clustering of the
  same budget does as well. This is the control for §2.5's design argument.
- [ ] **2.6 Herding exemplar selection at matched memory.** `[cheap]`
  Stage 2c uses raw vectors under a budget; iCaRL-style herding is the standard
  *strong* exemplar baseline. Makes the +16.0 pp result in 3.1 robust to
  "you only beat a weak random buffer."
- [ ] **2.7 Inference latency and training time.** `[cheap]` A second axis, and one
  that may not favour MAYA: inference cost grows with `K × C` nearest-node
  distances, while RanPAC pays a fixed projection. Better reported than found.
- [ ] **2.8 Class-order variation and task-count sensitivity.** `[robustness]`
  Different class orderings (not just seeds), and 5/10/20 tasks. Routine CIL
  expectations, cheap over cached features.
- [ ] **2.9 Scope out prompt-based methods in prose — do not implement.**
  L2P / DualPrompt / CODA-Prompt are cited and never compared. With a frozen
  DINOv3-7B they are not directly applicable (prompt insertion into a 7B backbone
  is a large lift, and they optimize by gradient, orthogonal to the gradient-free
  setting). One honest paragraph is the right move.

**If only two get done: 2.1 and 2.3.** The first defends the novelty claim, the
second defends the headline result's explanation.

---

## Phase 3 — Evidence already collected and unused

All parsed and sitting in `results/parsed/`. Highest value per hour in the repo.

- [ ] **3.1 Promote the memory-matched raw-vector baseline (story stage 2c).** `[core]`
  **9 of 9 configs, mean +16.0 pp, at 0.4–25% more memory** — ObjectNet/DINOv3 is
  28.8% @ 508MB vs MAYA 76.0% @ 550MB. This is the matched-budget experiment the
  April memo asked for, the cleanest result in the repo, and absent from the paper.
  It directly answers "is MAYA's memory well spent?"
- [x] **3.2 Report forgetting.** `[cheap]` Computed in every run, reported nowhere.
  Present for every ablation row in `results/parsed/*.json`.
- [x] **3.3 Report the λ sweep.** `[cheap]` §5.4 promises a λ sensitivity analysis
  that never appears. Data is in `ablation.lambda_sweep`. "Insensitive to λ" is a
  perfectly good finding.
- [ ] **3.4 Demonstrate unlearning.** `[core]` Claimed four times, never shown.
  Delete one class's nodes; show that class collapses, others don't move, note that
  SLDA/ACL need a full recompute. The one capability no baseline in the comparison
  set can match, and a small experiment.

---

## Phase 4 — Claims and framing

Do this **after** Phases 1–3, so it is written against corrected numbers and
known outcomes.

- [ ] **4.1 Rewrite abstract, intro, conclusion against the evidence.** `[core]`
  Remove "state-of-the-art" and the `+28.2%` headline. Whatever B1 resolves to,
  MAYA wins ~1 of 9 configurations outright, on fractions of a point, single-seed.
- [ ] **4.2 Fix the β story.** `[core]` Eq. 4 presents β as a fixed blending constant
  and never gives a value. It is tuned per task on a validation split, and
  degenerates to a single branch in 4 of 9 configs:

  | config | β | | config | β |
  |---|---|---|---|---|
  | ImageNet-R/DINOv3 | **0.0** (global only) | | TinyImageNet/ResNet50 | **1.0** (episodic only) |
  | TinyImageNet/DINOv3 | **0.0** | | ObjectNet/SigLIP2 | **1.0** |
  | ImageNet-R/ResNet50 | 0.5 | | TinyImageNet/SigLIP2 | 0.9 |
  | ImageNet-R/SigLIP2 | 0.5 | | ObjectNet/ResNet50 | 0.7 |
  | ObjectNet/DINOv3 | 0.3 | | | |

  Either report the tuning honestly and soften "dual-system", or fix a global β
  and re-run. Do not leave Eq. 4 describing something the code does not do.
- [ ] **4.3 State the accuracy-per-byte position explicitly** (pending B1). `[core]`
  Owning it in a "when is MAYA not the right choice?" subsection is far stronger
  than letting a reviewer construct it.
- [x] **4.4 Rebuild Figure 2 around the two-family view.** `[core]`
  Draft Fig 2b plots only the family MAYA loses to (statistics-only) and omits the
  family it was designed against (data-storing). Drafts exist:
  `figures/pareto_audit.pdf`, `figures/family_comparison.pdf` — both need
  regenerating once B1 and 1.3 land.

---

## Phase 5 — Method text vs. implementation

Each is a place where a reviewer who reads the code finds it disagrees with the paper.

- [x] **5.1 τ.** §3.3 spawns a node when `z_t` exceeds distance threshold τ from all
  nodes. Algorithm 1 never uses τ — it spawns whenever `#nodes < K`. τ appears
  nowhere else in the paper.
- [x] **5.2 Sherman-Morrison.** §3.2 claims the inverse is never recomputed;
  Algorithm 1 line 13 is `W* ← Σ⁻¹Φ`, a full inverse per batch.
- [x] **5.3 Nodes are not in the aligned space.** `model.py:587` allocates
  `self.nodes` at `input_dim` (4096 for DINOv3, not the 256-d aligned space), and
  matching runs `cdist` in raw backbone space (`model.py:974`, `model.py:1632`).
  §2.5 claims the opposite and calls it *"not a minor implementation choice."*
  Fix the text or the code — load-bearing for the stated distinction from prior
  episodic-memory work.
- [ ] **5.4 Citations.** You run `timm/vit_7b_patch16_dinov3.lvd1689m` but cite
  Oquab et al. 2024 = **DINOv2** (arXiv:2304.07193). SigLIP2 is cited as Zhai et
  al. 2023 = SigLIP v1. Both wrong.
- [ ] **5.5 Garbled sentence, p5:** *"…aligned feature space `z = W That every time
  step`"*.

---

## Phase 6 — Required for submission

- [ ] **6.1 Title.** Still the template placeholder: *"Formatting Instructions for
  TMLR Journal Submissions."*
- [ ] **6.2 Duplicate paragraph.** §2.1 "Regularization methods add penalty terms…"
  appears twice, nearly verbatim.
- [x] **6.3 Reproducibility table.** No hyperparameter values appear anywhere:
  K, α, λ, β, τ, tasks, classes per task, split ratio, seeds, compute.
- [x] **6.4 Limitations section.** Absent. With the honest framing it writes itself.
- [x] **6.5 Related-work paragraph** covering 2.9 (prompt methods) and 2.3 (FeCAM
  and per-class-covariance approaches).

---

## Suggested order

Two tracks run in parallel — compute, and writing.

**Compute track (start now, mostly unattended):**

1. **1.3 seeds** — longest pole, kick off immediately.
2. **2.1 + 2.2** — one-hot and random targets; shared code path with each other.
3. **2.3 FeCAM**, then **2.4 joint probe** (independent, trivial).
4. **3.4 unlearning**, then **2.5–2.7** as time allows.
5. **2.8** last; nice to have, not load-bearing.

**Writing track (start now, no compute dependency):**

1. Answer **B1** and **B2** — they gate Table 2 and the whole narrative.
2. **Phase 5** and **Phase 6** entirely — mechanical, independent of every result.
3. **1.1, 1.2** — regenerate the stale tables from `results/parsed/`.
4. **3.1, 3.2, 3.3** — already-collected evidence, straight into the paper.

**Then converge:** Phase 4 last, written against the corrected numbers and the
Phase 2 outcomes.

`1.4` can happen any time but must be settled before Figure 3 ships.
