import matplotlib.pyplot as plt
import numpy as np

# Set standard academic style
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 1.5

def plot_objectnet_bar():
    # Data for ObjectNet + DINOv3
    methods = ['Deep SLDA', 'Exact ACL', 'RanPAC', 'MAYA (Ours)']
    accuracies = [73.66, 75.66, 75.36, 76.00]
    
    # Colors: Baselines in gray/blue, MAYA highlighted
    colors = ['#A0B3C6', '#A0B3C6', '#A0B3C6', '#E94A4A']
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    bars = ax.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=1.5, width=0.6)
    
    # Zoom in on the relevant accuracy range to show the difference clearly
    ax.set_ylim(70, 77)
    
    # Add text labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
                    
    ax.set_ylabel('Accuracy (AIA %)', fontsize=14, fontweight='bold')
    ax.set_title('Performance on Complex Topologies\n(ObjectNet with DINOv3)', fontsize=16, pad=20, fontweight='bold')
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('objectnet_accuracy_bar.svg', transparent=True, bbox_inches='tight')
    plt.close()

def plot_memory_accuracy_tradeoff():
    # Data for ObjectNet + DINOv3
    methods = ['Deep SLDA', 'Exact ACL', 'RanPAC', 'MAYA (Ours)']
    accuracies = [73.66, 75.66, 75.36, 76.00]
    memory_mb = [68.89, 68.89, 549.65, 550.00]
    
    colors = ['#A0B3C6', '#4A90E2', '#F5A623', '#E94A4A']
    markers = ['o', 's', '^', '*']
    sizes = [150, 150, 200, 400]
    
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    for i in range(len(methods)):
        ax.scatter(memory_mb[i], accuracies[i], color=colors[i], marker=markers[i], 
                   s=sizes[i], edgecolor='black', linewidth=1.5, label=methods[i], zorder=5)
                   
    # Add annotations next to the points
    for i in range(len(methods)):
        offset_x = 10 if i != 2 else -10
        offset_y = 0.1 if i != 2 else -0.3
        ha = 'left' if i != 2 else 'right'
        ax.annotate(methods[i], (memory_mb[i] + offset_x, accuracies[i] + offset_y), 
                    fontsize=12, fontweight='bold', ha=ha)
                    
    ax.set_xlabel('Memory Footprint (MB)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (AIA %)', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy vs. Memory Trade-off\n(ObjectNet with DINOv3)', fontsize=16, pad=20, fontweight='bold')
    
    # Use log scale for memory if the gap is huge, or just linear since they are clustered in two groups
    ax.set_xscale('log')
    from matplotlib.ticker import ScalarFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks([50, 100, 500, 1000])
    
    ax.set_ylim(73, 76.5)
    
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('memory_tradeoff_scatter.svg', transparent=True, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating results plots...")
    plot_objectnet_bar()
    plot_memory_accuracy_tradeoff()
    print("Done! SVGs saved.")
