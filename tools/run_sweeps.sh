#!/bin/bash
# Checklist items 1 (seeds) and 2 (target geometry).
#
#   ./tools/run_sweeps.sh geometry   # item 2 -- run this FIRST, it is cheap and decisive
#   ./tools/run_sweeps.sh seeds      # item 1 -- the long pole
#   ./tools/run_sweeps.sh all
#
# Everything writes to results/sweep/ and is aggregated by tools/parse_sweep.py.
# Existing results/grid/ and results/comparisons/ are never touched.

set -e
cd "$(dirname "$0")/.."

DATASETS=(imagenet_r tinyimagenet objectnet)
BACKBONES=(resnet50 dinov3 siglip2)
SEEDS=(42 1 2 3 4)

# The target-geometry ablation MUST run at a dimension that fits one-hot
# targets, so all three arms are compared at the same p. BIO_ETF_MAX_CLASSES
# is 500, so p must be >= 500. This also makes the ETF arm a real simplex ETF
# for the first time -- at the default p=256 the construction silently falls
# back to a random frame (see tools/parse_sweep.py --check-targets).
ALIGN_DIM=512

mkdir -p results/sweep logs/sweep

run_geometry () {
  echo "=== Item 2: target geometry (etf / onehot / random) at p=${ALIGN_DIM} ==="
  for ds in "${DATASETS[@]}"; do
    for bb in "${BACKBONES[@]}"; do
      for geom in etf onehot random; do
        tag="${ds}_${bb}_${geom}_p${ALIGN_DIM}"
        out="results/sweep/ablation_${tag}_seed42.json"
        if [ -f "$out" ]; then
          echo "  skip  ${tag} (already done)"; continue
        fi
        echo "  run   ${tag}"
        python run_ablations.py --dataset "$ds" --backbone "$bb" \
          --target-geometry "$geom" --align-dim "$ALIGN_DIM" --seed 42 \
          > "logs/sweep/${tag}.log" 2>&1 || echo "  FAILED ${tag} -- see logs/sweep/${tag}.log"
      done
    done
  done
}

run_seeds () {
  echo "=== Item 1: seed sweep (${SEEDS[*]}) ==="
  for seed in "${SEEDS[@]}"; do
    for ds in "${DATASETS[@]}"; do
      for bb in "${BACKBONES[@]}"; do
        tag="${ds}_${bb}_seed${seed}"
        if [ -f "results/sweep/ablation_${ds}_${bb}_seed${seed}.json" ]; then
          echo "  skip  ablation ${tag}"; else
          echo "  run   ablation ${tag}"
          python run_ablations.py --dataset "$ds" --backbone "$bb" --seed "$seed" \
            > "logs/sweep/ablation_${tag}.log" 2>&1 || echo "  FAILED ablation ${tag}"
        fi
        if [ "$seed" != "42" ] && [ ! -f "results/sweep/sota_${ds}_${bb}_seed${seed}.json" ]; then
          echo "  run   sota ${tag}"
          python run_comparisons.py --dataset "$ds" --backbone "$bb" --seed "$seed" \
            > "logs/sweep/sota_${tag}.log" 2>&1 || echo "  FAILED sota ${tag}"
        fi
      done
    done
  done
}

case "${1:-all}" in
  geometry) run_geometry ;;
  seeds)    run_seeds ;;
  all)      run_geometry; run_seeds ;;
  *) echo "usage: $0 {geometry|seeds|all}"; exit 1 ;;
esac

echo
echo "Done. Aggregate with:  python tools/parse_sweep.py"
