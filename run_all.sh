#!/bin/bash
# run_all.sh - Automated Paper Storyline Experiments

# Exit on first failure
set -e

echo "=================================================="
echo " 📥 STEP 1: DOWNLOADING DATASETS"
echo "=================================================="

# Ensure data and results directories exist
mkdir -p data
mkdir -p results

# Download ImageNet-R (ImageNet-Renditions)
if [ ! -d "data/imagenet-r" ]; then
    echo "[Data] Downloading ImageNet-R..."
    wget -nc https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar -O data/imagenet-r.tar
    echo "[Data] Extracting ImageNet-R..."
    tar -xvf data/imagenet-r.tar -C data/
else
    echo "✅ [Data] ImageNet-R is already extracted at data/imagenet-r"
fi

# Download DomainNet (Real Domain)
if [ ! -d "data/domainnet/real" ]; then
    echo "[Data] Downloading DomainNet (Real)..."
    mkdir -p data/domainnet
    wget -nc http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip -O data/domainnet/real.zip
    echo "[Data] Extracting DomainNet (Real)..."
    unzip -q -n data/domainnet/real.zip -d data/domainnet/real/
else
    echo "✅ [Data] DomainNet (Real) is already extracted at data/domainnet/real"
fi

echo "=================================================="
echo " 🚀 STEP 2: RUNNING EXPERIMENTS"
echo "=================================================="

# Grid Definition
DATASETS=("tinyimagenet" "imagenet_r" "objectnet")
BACKBONES=("resnet50" "dinov3" "siglip2")

export PYTHONUNBUFFERED=1

for dataset in "${DATASETS[@]}"; do
    for backbone in "${BACKBONES[@]}"; do
        LOG_FILE="results/ablation_${dataset}_${backbone}.log"
        
        echo "--------------------------------------------------"
        echo "▶️ Launching: Dataset=[$dataset] | Backbone=[$backbone]"
        echo "Logs will stream to: $LOG_FILE"
        echo "--------------------------------------------------"
        
        # We tee the output so you can watch it live while saving it!
        python run_ablations.py \
            --dataset "$dataset" \
            --backbone "$backbone" \
            | tee "$LOG_FILE"
            
        echo "✅ Finished: $dataset + $backbone"
        echo ""
    done
done

echo "=================================================="
echo " 🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo " Check the 'results/' folder for your 4 output logs."
echo "=================================================="
