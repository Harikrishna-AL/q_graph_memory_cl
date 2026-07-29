import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('paper_plots', exist_ok=True)

data = {
    'ImageNet-R': {
        'ResNet50': {'NCM': (40.7, 1.6), 'Raw Vec Budget': (37.6, 138.0), 'Nodes Only': (45.8, 82.2), 'ER+MLP': (46.8, 187.5), 'MAYA Full': (47.6, 159.9)},
        'DINOv3':   {'NCM': (85.7, 3.1), 'Raw Vec Budget': (69.8, 306.6), 'Nodes Only': (93.5, 227.8), 'ER+MLP': (94.5, 374.9), 'MAYA Full': (94.2, 383.2)},
        'SigLIP2':  {'NCM': (95.3, 1.2), 'Raw Vec Budget': (78.9, 100.1), 'Nodes Only': (92.6, 55.8),  'ER+MLP': (96.1, 140.6), 'MAYA Full': (95.6, 114.1)}
    },
    'TinyImageNet': {
        'ResNet50': {'NCM': (63.3, 1.6), 'Raw Vec Budget': (56.7, 245.3), 'Nodes Only': (64.5, 82.2), 'ER+MLP': (71.1, 625.0), 'MAYA Full': (69.0, 246.1)},
        'DINOv3':   {'NCM': (89.5, 3.1), 'Raw Vec Budget': (88.3, 553.1), 'Nodes Only': (90.2, 227.8), 'ER+MLP': (94.5, 1250.0),'MAYA Full': (93.6, 555.5)},
        'SigLIP2':  {'NCM': (86.1, 1.2), 'Raw Vec Budget': (83.2, 178.1), 'Nodes Only': (82.7, 55.8),  'ER+MLP': (90.2, 468.8), 'MAYA Full': (89.0, 178.8)}
    },
    'ObjectNet': {
        'ResNet50': {'NCM': (19.4, 2.3), 'Raw Vec Budget': (12.2, 229.3), 'Nodes Only': (18.2, 102.5), 'ER+MLP': (23.4, 302.5), 'MAYA Full': (20.0, 243.4)},
        'DINOv3':   {'NCM': (47.8, 4.7), 'Raw Vec Budget': (28.8, 507.6), 'Nodes Only': (71.1, 268.4), 'ER+MLP': (73.0, 605.0), 'MAYA Full': (76.0, 550.0)},
        'SigLIP2':  {'NCM': (76.6, 1.8), 'Raw Vec Budget': (66.0, 167.7), 'Nodes Only': (76.6, 71.0),  'ER+MLP': (80.2, 226.9), 'MAYA Full': (80.5, 176.7)}
    }
}

methods_order = ['NCM', 'Raw Vec Budget', 'Nodes Only', 'ER+MLP', 'MAYA Full']
method_colors = {'NCM': '#888888', 'Raw Vec Budget': '#ff7f0e', 'Nodes Only': '#1f77b4', 'ER+MLP': '#d62728', 'MAYA Full': '#2ca02c'}
method_markers = {'NCM': 'o', 'Raw Vec Budget': 'X', 'Nodes Only': 's', 'ER+MLP': '^', 'MAYA Full': '*'}

datasets = ['ImageNet-R', 'TinyImageNet', 'ObjectNet']
backbones = ['ResNet50', 'DINOv3', 'SigLIP2']

fig, axes = plt.subplots(3, 3, figsize=(16, 14))

def get_pareto(mem, acc):
    pts = sorted(zip(mem, acc))
    pareto_mem, pareto_acc = [], []
    max_acc = -1
    for m, a in pts:
        if a >= max_acc:
            pareto_mem.append(m)
            pareto_acc.append(a)
            max_acc = a
    return pareto_mem, pareto_acc

for i, ds in enumerate(datasets):
    for j, bb in enumerate(backbones):
        ax = axes[i, j]
        if bb not in data[ds]:
            ax.axis('off')
            continue
            
        res = data[ds][bb]
        accs, mems = [], []
        
        for m in methods_order:
            acc, mem = res[m]
            accs.append(acc)
            mems.append(mem)
            ms = 18 if m == 'MAYA Full' else 12
            ax.scatter(mem, acc, color=method_colors[m], marker=method_markers[m], s=ms**2, 
                       edgecolor='black', linewidth=1.5, zorder=3, label=m if (i==0 and j==0) else "")
            
        p_mem, p_acc = get_pareto(mems, accs)
        ax.plot(p_mem, p_acc, '--', color='gray', alpha=0.6, linewidth=2, zorder=1)
        
        ax.set_title(f'{ds} ({bb})', fontsize=14, fontweight='bold')
        if i == 2:
            ax.set_xlabel('Memory (MB)', fontsize=12)
        if j == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.7)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=14, bbox_to_anchor=(0.5, 0.02))
plt.tight_layout()
plt.subplots_adjust(bottom=0.08)
grid_path = os.path.join('paper_plots', 'pareto_grid_3x3.svg')
plt.savefig(grid_path, format='svg', bbox_inches='tight', dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 8))
dataset_colors = {'ImageNet-R': '#3498db', 'TinyImageNet': '#e67e22', 'ObjectNet': '#9b59b6'}
backbone_markers = {'ResNet50': 'o', 'DINOv3': 's', 'SigLIP2': '^'}

for ds in datasets:
    for bb in backbones:
        if bb not in data[ds]: continue
        res = data[ds][bb]
        acc_gain = res['MAYA Full'][0] - res['NCM'][0]
        mem_saved = res['ER+MLP'][1] - res['MAYA Full'][1]
        
        ax.scatter(mem_saved, acc_gain, color=dataset_colors[ds], marker=backbone_markers[bb], 
                   s=200, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.annotate(f"{ds[:3]} ({bb[:4]})", (mem_saved, acc_gain), xytext=(8, 8), textcoords='offset points', fontsize=9)

ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
ax.axvline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
ax.set_title("MAYA Efficiency: Accuracy Gains vs. Memory Savings", fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("Memory Saved vs. ER+MLP (MB)", fontsize=14)
ax.set_ylabel("Accuracy Gained vs. NCM (%)", fontsize=14)
ax.grid(True, linestyle=':', alpha=0.7)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='ImageNet-R', markerfacecolor='#3498db', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='TinyImageNet', markerfacecolor='#e67e22', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='ObjectNet', markerfacecolor='#9b59b6', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='ResNet50', markerfacecolor='gray', markersize=10),
    Line2D([0], [0], marker='s', color='w', label='DINOv3', markerfacecolor='gray', markersize=10),
    Line2D([0], [0], marker='^', color='w', label='SigLIP2', markerfacecolor='gray', markersize=10)
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=12, framealpha=0.9)

ax.set_ylim(bottom=-1, top=30) # Adds a little breathing room at the top

plt.tight_layout()
delta_path = os.path.join('paper_plots', 'delta_efficiency_plot.svg')
plt.savefig(delta_path, format='svg', bbox_inches='tight', dpi=300)
plt.close(fig)
print("Updated plots generated!")
