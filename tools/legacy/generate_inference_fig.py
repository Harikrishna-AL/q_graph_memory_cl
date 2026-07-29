import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 2

def generate_inference_comparison():
    np.random.seed(42)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # --- Generate Data ---
    # Class 1: A crescent moon shape (Multi-modal)
    theta = np.linspace(0, np.pi, 100)
    r = 2.0
    x1 = r * np.cos(theta) + np.random.normal(0, 0.15, 100)
    y1 = r * np.sin(theta) + np.random.normal(0, 0.15, 100)
    
    # Class 2: A dense blob sitting "inside" the curve of the crescent
    x2 = np.random.normal(0, 0.3, 100)
    y2 = np.random.normal(0.8, 0.3, 100)
    
    mean1 = [np.mean(x1), np.mean(y1)]
    mean2 = [np.mean(x2), np.mean(y2)]
    
    # A tricky test point (yellow) that clearly belongs to Class 1 visually (it's on the edge of the crescent)
    # but is mathematically closer to the center of Class 2 than the center of Class 1.
    test_x, test_y = -1.5, 1.2
    
    # --- Plot 1: NCM Inference (Fails) ---
    ax1.scatter(x1, y1, alpha=0.3, c='#4A90E2', edgecolors='none', s=50)
    ax1.scatter(x2, y2, alpha=0.3, c='#E94A4A', edgecolors='none', s=50)
    
    # Plot NCM Means
    ax1.scatter(mean1[0], mean1[1], c='#4A90E2', marker='*', s=400, edgecolors='black', linewidth=1, zorder=5, label='Class 1 Mean')
    ax1.scatter(mean2[0], mean2[1], c='#E94A4A', marker='*', s=400, edgecolors='black', linewidth=1, zorder=5, label='Class 2 Mean')
    
    # Test Point
    ax1.scatter(test_x, test_y, c='#F8E71C', s=150, zorder=7, edgecolors='black', linewidth=2, label='Test Image')
    
    # Draw decision boundary roughly
    ax1.plot([-3, 3], [0, 2.5], 'k--', alpha=0.5, lw=2) 
    
    # Show wrong classification
    ax1.annotate('', xy=(mean2[0], mean2[1]), xytext=(test_x, test_y), 
                arrowprops=dict(arrowstyle="->", color="#E94A4A", lw=3))
    
    ax1.set_title("Inference via NCM (Misclassification)", fontsize=14, pad=15, fontweight='bold')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-1, 3)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # --- Plot 2: Episodic Graph Inference (Succeeds) ---
    ax2.scatter(x1, y1, alpha=0.3, c='#4A90E2', edgecolors='none', s=50)
    ax2.scatter(x2, y2, alpha=0.3, c='#E94A4A', edgecolors='none', s=50)
    
    # Define K-means nodes tracing the crescent
    node_angles = np.linspace(np.pi/8, 7*np.pi/8, 5)
    nx1 = r * np.cos(node_angles)
    ny1 = r * np.sin(node_angles)
    
    # Define K-means nodes for the blob
    nx2 = [0, 0.4, -0.4, 0, 0]
    ny2 = [0.8, 0.8, 0.8, 1.2, 0.4]
    
    # Plot nodes
    ax2.scatter(nx1, ny1, c='#4A90E2', marker='^', s=200, edgecolors='black', linewidth=1, zorder=5, label='Class 1 Nodes')
    ax2.scatter(nx2, ny2, c='#E94A4A', marker='^', s=200, edgecolors='black', linewidth=1, zorder=5, label='Class 2 Nodes')
    
    # Test Point
    ax2.scatter(test_x, test_y, c='#F8E71C', s=150, zorder=7, edgecolors='black', linewidth=2)
    
    # Show correct classification
    # Find nearest Class 1 node
    nearest_idx = np.argmin(np.sqrt((nx1 - test_x)**2 + (ny1 - test_y)**2))
    ax2.annotate('', xy=(nx1[nearest_idx], ny1[nearest_idx]), xytext=(test_x, test_y), 
                arrowprops=dict(arrowstyle="->", color="#4A90E2", lw=3))
    
    ax2.set_title("Inference via Episodic Graph (Correct)", fontsize=14, pad=15, fontweight='bold')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-1, 3)
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.tight_layout()
    plt.savefig('inference_comparison.svg', transparent=True)
    plt.close()

if __name__ == "__main__":
    print("Generating inference comparison SVG...")
    generate_inference_comparison()
