import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 2

def generate_node_dynamics():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Common settings
    radius = 1.0
    
    # --- Panel 1: Update Existing Node (d < tau) ---
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("1. Update Node ($d < \\tau$)", fontsize=16, pad=15, fontweight='bold')
    
    # Existing Node
    old_node_x, old_node_y = -0.5, -0.5
    ax1.scatter(old_node_x, old_node_y, c='#4A90E2', marker='*', s=600, edgecolors='black', lw=1, zorder=5, label='Old Node')
    
    # Threshold Circle
    circle1 = patches.Circle((old_node_x, old_node_y), radius, fill=True, color='#4A90E2', alpha=0.1, ls='--', lw=2, ec='black')
    ax1.add_patch(circle1)
    
    # New Point (Inside)
    new_pt_x, new_pt_y = 0.2, 0.2
    ax1.scatter(new_pt_x, new_pt_y, c='#F8E71C', s=200, edgecolors='black', lw=2, zorder=6, label='New Feature $z_t$')
    
    # Arrow showing movement
    shifted_x, shifted_y = -0.2, -0.2 # Moving average shift
    ax1.annotate('', xy=(shifted_x, shifted_y), xytext=(old_node_x, old_node_y), 
                arrowprops=dict(arrowstyle="->", color="black", lw=3))
    
    # New Node Position (faint)
    ax1.scatter(shifted_x, shifted_y, c='#4A90E2', marker='*', s=600, edgecolors='black', lw=2, ls='--', alpha=0.6, zorder=7)


    # --- Panel 2: Spawn New Node (d > tau) ---
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("2. Spawn Node ($d > \\tau$)", fontsize=16, pad=15, fontweight='bold')
    
    # Existing Node
    ax2.scatter(old_node_x, old_node_y, c='#4A90E2', marker='*', s=600, edgecolors='black', lw=1, zorder=5)
    
    # Threshold Circle
    circle2 = patches.Circle((old_node_x, old_node_y), radius, fill=True, color='#4A90E2', alpha=0.1, ls='--', lw=2, ec='black')
    ax2.add_patch(circle2)
    
    # New Point (Outside)
    far_pt_x, far_pt_y = 1.2, 1.2
    ax2.scatter(far_pt_x, far_pt_y, c='#F8E71C', s=200, edgecolors='black', lw=2, zorder=6)
    
    # Spawn action (turning into a node)
    ax2.annotate('Spawns', xy=(far_pt_x+0.1, far_pt_y-0.3), xytext=(far_pt_x+0.1, far_pt_y-0.8),
                 arrowprops=dict(arrowstyle="->", color="black", lw=2), fontsize=12, fontweight='bold', ha='center')
    ax2.scatter(far_pt_x, far_pt_y, c='#4A90E2', marker='*', s=600, edgecolors='black', lw=2, zorder=7, alpha=0.5) # Faint star behind it to show it becomes a node

    plt.tight_layout()
    plt.savefig('node_dynamics.svg', transparent=True)
    plt.close()

if __name__ == "__main__":
    generate_node_dynamics()
