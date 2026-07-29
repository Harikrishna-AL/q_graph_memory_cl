import matplotlib.pyplot as plt
import numpy as np
import os

# Create directory if it doesn't exist
os.makedirs('paper_plots', exist_ok=True)

# Data for ObjectNet + DINOv3
obj_methods = ['NCM', 'Raw Vec Budget', 'Nodes Only', 'ER+MLP', 'MAYA Full (Ours)']
obj_acc = [47.8, 28.8, 71.1, 73.0, 76.0]
obj_mem = [4.7, 507.6, 268.4, 605.0, 550.0]
obj_colors = ['#888888', '#ff7f0e', '#1f77b4', '#d62728', '#2ca02c']
obj_markers = ['o', 'X', 's', '^', '*']

# Data for TinyImageNet + DINOv3
tiny_methods = ['NCM', 'Raw Vec Budget', 'Nodes Only', 'ER+MLP', 'MAYA Full (Ours)']
tiny_acc = [89.5, 88.3, 90.2, 94.5, 93.6]
tiny_mem = [3.1, 553.1, 227.8, 1250.0, 555.5]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

def plot_pareto(ax, acc, mem, methods, colors, markers, title):
    # Plot points
    for i in range(len(methods)):
        # Make MAYA star larger
        ms = 18 if methods[i] == 'MAYA Full (Ours)' else 12
        ax.scatter(mem[i], acc[i], color=colors[i], marker=markers[i], s=ms**2, 
                   label=methods[i], edgecolor='black', linewidth=1.5, zorder=3)
        
    # Find Pareto frontier (minimize memory, maximize accuracy)
    # Sort by memory
    pts = sorted(zip(mem, acc, methods))
    pareto_mem = []
    pareto_acc = []
    max_acc = -1
    for m, a, name in pts:
        if a >= max_acc:
            pareto_mem.append(m)
            pareto_acc.append(a)
            max_acc = a
            
    # Draw Pareto line
    ax.plot(pareto_mem, pareto_acc, '--', color='gray', alpha=0.6, linewidth=2, zorder=1)
    
    # Formatting
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Memory Footprint (MB)', fontsize=14)
    ax.set_ylabel('Average Incremental Acc. (%)', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.7, zorder=0)
    ax.tick_params(labelsize=12)
    
    # Add subtle annotations for clarity
    for i, txt in enumerate(methods):
        y_offset = 1.0 if 'MAYA' in txt or 'NCM' in txt else -1.5
        ax.annotate(txt, (mem[i], acc[i]), xytext=(10, y_offset*10), 
                    textcoords='offset points', fontsize=11, fontweight='bold' if 'MAYA' in txt else 'normal')

# Plot both
plot_pareto(ax1, obj_acc, obj_mem, obj_methods, obj_colors, obj_markers, 'ObjectNet (DINOv3)')
plot_pareto(ax2, tiny_acc, tiny_mem, tiny_methods, obj_colors, obj_markers, 'TinyImageNet (DINOv3)')

# Adjust layout and legend
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=12, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)

# Save
save_path = os.path.join('paper_plots', 'accuracy_vs_memory_pareto.svg')
plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
print(f"Saved plot to {save_path}")
