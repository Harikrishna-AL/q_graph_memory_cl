import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_k_sweep_plot():
    # Setup aesthetics
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    
    # Data for ObjectNet + DINOv3
    k_values = [1, 16, 64, 128]
    x_positions = np.arange(len(k_values))  # 0, 1, 2, 3 for equal spacing
    accuracies = [30.12, 70.37, 75.94, 76.33]
    
    # Create figure
    plt.figure(figsize=(8, 5))
    
    # Plot line with markers
    plt.plot(x_positions, accuracies, marker='o', markersize=10, linewidth=3, 
             color='#2ca02c', markerfacecolor='white', markeredgewidth=2)
    
    # Customize axes
    plt.xticks(x_positions, [str(k) for k in k_values])
    plt.xlabel("Maximum Nodes per Class ($K$)", fontweight='bold', labelpad=10)
    plt.ylabel("Accuracy (AIA %)", fontweight='bold', labelpad=10)
    
    # Set y-axis limits to give some breathing room
    plt.ylim(20, 85)
    
    # Annotate points with their values
    for x, y in zip(x_positions, accuracies):
        plt.annotate(f"{y:.1f}%", 
                     (x, y), 
                     textcoords="offset points", 
                     xytext=(0, -20) if x == 0 else (0, 10), # First point text below, rest above
                     ha='center', 
                     fontsize=12,
                     fontweight='bold')

    plt.title("Impact of Node Budget on Complex Topologies (ObjectNet)", pad=15, fontweight='bold')
    
    # Add a horizontal line representing standard NCM for reference
    plt.axhline(y=47.85, color='red', linestyle='--', linewidth=2, alpha=0.7)
    plt.text(0.1, 49, 'Standard NCM Baseline (47.8%)', color='red', fontsize=12, fontweight='bold', alpha=0.7)
    
    plt.tight_layout()
    
    # Save as both SVG and PNG for easy latex inclusion
    plt.savefig('figures/k_sweep_ablation.svg', transparent=True)
    plt.savefig('figures/k_sweep_ablation.png', transparent=True, dpi=300)
    print("Saved K-sweep plot!")

if __name__ == "__main__":
    generate_k_sweep_plot()
