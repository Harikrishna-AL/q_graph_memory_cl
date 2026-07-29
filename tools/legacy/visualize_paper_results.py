import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Professional Paper Aesthetics
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.titlesize": 16,
    "figure.autolayout": True
})
sns.set_context("paper")

RESULTS_DIR = "results"
OUTPUT_DIR = "paper_plots_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_results():
    all_data = []
    for filename in os.listdir(RESULTS_DIR):
        if filename.startswith("ablation_") and filename.endswith(".json"):
            parts = filename.replace("ablation_", "").replace(".json", "").split("_")
            if len(parts) >= 2:
                backbone = parts[0]
                dataset = "_".join(parts[1:])
                try:
                    with open(os.path.join(RESULTS_DIR, filename), "r") as f:
                        data = json.load(f)
                        if "component_ablation" in data and len(data["component_ablation"]) > 0:
                            if data["component_ablation"][0].get("aia", 0) > 0:
                                all_data.append({"backbone": backbone, "dataset": dataset, "data": data})
                except: continue
    return all_data

def plot_gain_heatmap(all_results):
    """Visualizes the 'Success Matrix' - Delta Gain over NCM."""
    matrix_data = []
    for res in all_results:
        steps = res["data"]["component_ablation"]
        ncm_aia = steps[0]["aia"] * 100
        maya_aia = steps[-1]["aia"] * 100
        gain = maya_aia - ncm_aia
        
        matrix_data.append({
            "Backbone": res["backbone"].upper(),
            "Dataset": res["dataset"].replace("_", "-").upper(),
            "Gain": gain
        })
    
    df = pd.DataFrame(matrix_data).pivot(index="Backbone", columns="Dataset", values="Gain")
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".1f", cbar_kws={'label': 'AIA Gain over NCM (%)'})
    plt.title("MAYA Performance Uplift Matrix")
    plt.savefig(f"{OUTPUT_DIR}/heatmap_uplift.png", dpi=300)
    plt.close()

def plot_stability_slopes(all_results):
    """Slope chart showing the drop in forgetting."""
    slope_data = []
    for res in all_results:
        steps = res["data"]["component_ablation"]
        if "forgetting" not in steps[0]: continue
        
        id_str = f"{res['backbone'].upper()}\n({res['dataset'].replace('_','-').upper()})"
        slope_data.append({"ID": id_str, "Method": "NCM", "Forgetting": steps[0]["forgetting"] * 100})
        slope_data.append({"ID": id_str, "Method": "MAYA", "Forgetting": steps[-1]["forgetting"] * 100})
    
    df = pd.DataFrame(slope_data)
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="Method", y="Forgetting", hue="ID", marker="o", palette="tab10")
    plt.title("Stability Improvement: Forgetting Reduction")
    plt.ylabel("Average Forgetting (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Backbone + Dataset")
    plt.grid(axis='y', ls='--', alpha=0.7)
    plt.savefig(f"{OUTPUT_DIR}/forgetting_slopes.png", dpi=300)
    plt.close()

def plot_bubble_efficiency(all_results):
    """Accuracy vs Memory Bubble Chart. Size = Stability."""
    bubble_data = []
    for res in all_results:
        steps = res["data"]["component_ablation"]
        maya = steps[-1]
        
        # Stability = 1 / Forgetting (capped for visual clarity)
        stability = 1.0 / max(maya.get("forgetting", 0.05), 0.01)
        
        bubble_data.append({
            "Backbone": res["backbone"].upper(),
            "Dataset": res["dataset"].replace("_", "-").upper(),
            "Memory (MB)": maya["mem"],
            "AIA (%)": maya["aia"] * 100,
            "Stability": stability
        })
    
    df = pd.DataFrame(bubble_data)
    plt.figure(figsize=(10, 7))
    scatter = sns.scatterplot(data=df, x="Memory (MB)", y="AIA (%)", 
                              hue="Backbone", style="Dataset", 
                              size="Stability", sizes=(100, 1000), alpha=0.7)
    
    plt.xscale('log')
    plt.title("Architectural Efficiency: Accuracy, Memory, and Stability")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/bubble_efficiency.png", dpi=300)
    plt.close()

def plot_k_scaling_ribbon(all_results):
    """Line plot with error/confidence style ribbon for K-Sweep."""
    plot_data = []
    for res in all_results:
        if "k_sweep" not in res["data"]: continue
        for sweep in res["data"]["k_sweep"]:
            plot_data.append({
                "Backbone": res["backbone"].upper(),
                "K": sweep["k"],
                "AIA (%)": sweep.get("aia", 0) * 100
            })
            
    df = pd.DataFrame(plot_data)
    plt.figure(figsize=(9, 6))
    sns.lineplot(data=df, x="K", y="AIA (%)", hue="Backbone", style="Backbone", markers=True, dashes=False)
    plt.xscale('log', base=2)
    plt.title("Manifold Fidelity Scaling (K-Sweep)")
    plt.xlabel("Nodes per Class (Episodic Capacity)")
    plt.savefig(f"{OUTPUT_DIR}/k_scaling_ribbon.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    results = load_results()
    if results:
        print(f"🎨 Generating high-signal visualizations for {len(results)} results...")
        plot_gain_heatmap(results)
        plot_stability_slopes(results)
        plot_bubble_efficiency(results)
        plot_k_scaling_ribbon(results)
        print(f"🚀 Paper-ready plots saved to '{OUTPUT_DIR}/'")
    else:
        print("❌ No valid results to plot.")
