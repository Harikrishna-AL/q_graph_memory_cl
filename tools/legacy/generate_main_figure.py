import matplotlib.pyplot as plt
import numpy as np

# Set global styles for clean, academic SVGs
plt.rcParams['svg.fonttype'] = 'none' # Keep text as text in SVG
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 2

def generate_panel_a():
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Generate squished, overlapping data (Geometric Bias)
    cov1 = [[0.8, 0.6], [0.6, 0.5]]
    cov2 = [[0.6, 0.4], [0.4, 0.8]]
    class1 = np.random.multivariate_normal([1.0, 1.0], cov1, 100)
    class2 = np.random.multivariate_normal([2.0, 1.5], cov2, 100)
    
    ax.scatter(class1[:, 0], class1[:, 1], alpha=0.6, c='#4A90E2', edgecolors='white', s=80, label='Class 1')
    ax.scatter(class2[:, 0], class2[:, 1], alpha=0.6, c='#E94A4A', edgecolors='white', s=80, label='Class 2')
    
    # Means
    mean1 = np.mean(class1, axis=0)
    mean2 = np.mean(class2, axis=0)
    ax.scatter(mean1[0], mean1[1], c='black', marker='*', s=400, zorder=5)
    ax.scatter(mean1[0], mean1[1], c='#4A90E2', marker='*', s=200, zorder=6)
    ax.scatter(mean2[0], mean2[1], c='black', marker='*', s=400, zorder=5)
    ax.scatter(mean2[0], mean2[1], c='#E94A4A', marker='*', s=200, zorder=6)
    
    # Decision boundary (perpendicular bisector)
    midpoint = (mean1 + mean2) / 2
    slope = (mean2[1] - mean1[1]) / (mean2[0] - mean1[0])
    perp_slope = -1 / slope
    x_vals = np.array([-1, 4])
    y_vals = perp_slope * (x_vals - midpoint[0]) + midpoint[1]
    ax.plot(x_vals, y_vals, 'k--', linewidth=2, alpha=0.8, zorder=4)
    
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("A: Native Space (Geometric Bias)", fontsize=16, pad=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('panel_a_native.svg', transparent=True)
    plt.close()

def generate_panel_b():
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Generate perfectly separated data (ETF Space)
    # Class 1 at top left, Class 2 at bottom right
    cov = [[0.15, 0], [0, 0.15]]
    class1 = np.random.multivariate_normal([-2.0, 2.0], cov, 100)
    class2 = np.random.multivariate_normal([2.0, -2.0], cov, 100)
    
    # Draw ETF target vectors from origin
    ax.annotate('', xy=(-2.0, 2.0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="black", lw=2, ls='--'))
    ax.annotate('', xy=(2.0, -2.0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="black", lw=2, ls='--'))
    ax.scatter(0, 0, c='black', s=50, zorder=3) # Origin
    
    ax.scatter(class1[:, 0], class1[:, 1], alpha=0.6, c='#4A90E2', edgecolors='white', s=80, zorder=4)
    ax.scatter(class2[:, 0], class2[:, 1], alpha=0.6, c='#E94A4A', edgecolors='white', s=80, zorder=4)
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("B: Global Alignment (ETF Space)", fontsize=16, pad=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('panel_b_etf.svg', transparent=True)
    plt.close()

def generate_panel_c():
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Generate a complex, multi-modal shape (like a crescent)
    theta = np.linspace(0, np.pi, 100)
    r = 2.0
    x = r * np.cos(theta) + np.random.normal(0, 0.15, 100)
    y = r * np.sin(theta) + np.random.normal(0, 0.15, 100)
    
    ax.scatter(x, y, alpha=0.4, c='#4A90E2', edgecolors='white', s=80)
    
    # Define K-means nodes tracing the crescent
    node_angles = np.linspace(np.pi/8, 7*np.pi/8, 5)
    nx = r * np.cos(node_angles)
    ny = r * np.sin(node_angles)
    
    # Plot nodes
    ax.scatter(nx, ny, c='black', marker='*', s=500, zorder=5)
    ax.scatter(nx, ny, c='#F8E71C', marker='*', s=250, zorder=6, edgecolors='black', linewidth=1)
    
    # Draw Voronoi-like decision boundaries for the local nodes (just connecting them to show graph structure)
    for i in range(len(nx)-1):
        ax.plot([nx[i], nx[i+1]], [ny[i], ny[i+1]], color='black', lw=2, ls='--', alpha=0.5)
        
    # Draw a test image mapping to the nearest node
    test_x, test_y = 1.0, 1.8
    nearest_idx = np.argmin(np.sqrt((nx - test_x)**2 + (ny - test_y)**2))
    ax.scatter(test_x, test_y, c='#50E3C2', s=150, zorder=7, edgecolors='black', linewidth=2)
    ax.annotate('', xy=(nx[nearest_idx], ny[nearest_idx]), xytext=(test_x, test_y), 
                arrowprops=dict(arrowstyle="->", color="#50E3C2", lw=3))
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("C: Episodic Graph (Local Tracking)", fontsize=16, pad=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('panel_c_episodic.svg', transparent=True)
    plt.close()

if __name__ == "__main__":
    print("Generating SVG panels...")
    generate_panel_a()
    generate_panel_b()
    generate_panel_c()
    print("Done! SVGs saved in current directory.")
