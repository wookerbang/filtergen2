import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

# ===========================
# 0. Plot Settings (Standard English Fonts)
# ===========================
plt.rcParams['font.family'] = 'sans-serif'
# Ensure negative signs display correctly
plt.rcParams['axes.unicode_minus'] = True

# ===========================
# 1. Data Preparation
# ===========================

# --- A. Define Macro List (English Names) ---
macros = [
    'Ser L', 'Ser C',         # Basic Series
    'Shunt L', 'Shunt C',     # Basic Shunt
    'Ser Reso', 'Ser Tank',   # Complex Series
    'Shunt Reso', 'Shunt Notch', # Complex Shunt
    'SKIP (Padding)'          # Padding
]
n_macros = len(macros)
skip_idx = n_macros - 1

# --- B. Generate Static Topology Matrix (Connectivity Mask, K=10) ---
# For a cascade structure, connection goes from Node i to Node i+1.
# Values appear on the "super-diagonal".
K = 10
num_nodes = K + 1
topology_matrix = np.zeros((num_nodes, num_nodes))
for i in range(num_nodes - 1):
    topology_matrix[i, i+1] = 1.0

# --- C. Generate Macro Interaction Matrix ---
PENALTY_HARD = 100.0 # Hard Ban
PENALTY_SOFT = 50.0  # Soft Redundancy Penalty
ALLOWED = 0.0

interaction_matrix = np.full((n_macros, n_macros), ALLOWED)

# Index range for basic macros (the first 4)
basic_macros_end_idx = 4 

for i in range(n_macros):
    for j in range(n_macros):
        # Rule 1: Enforce Padding Structure (SKIP -> Real Macro banned)
        # If current is SKIP, and next is NOT SKIP
        if i == skip_idx and j != skip_idx:
            interaction_matrix[i, j] = PENALTY_HARD
        
        # Rule 2: Basic Macro Self-Redundancy (Soft Penalty)
        # Only for self-transitions of the first 4 basic components
        if i == j and i < basic_macros_end_idx: 
             interaction_matrix[i, j] = PENALTY_SOFT

# ===========================
# 2. Plotting Setup (Harmonious Contrast)
# ===========================

# Create canvas
fig, axes = plt.subplots(1, 2, figsize=(18, 9))
# Adjust space between subplots to accommodate labels
plt.subplots_adjust(wspace=0.3) 

# --- Left Palette (Cold Tone: White/Deep Blue) ---
colors_topology = ["#F8F9FA", "#004080"] 
cmap_topology = LinearSegmentedColormap.from_list("custom_cold", colors_topology, N=2)

# --- Right Palette (Warm/Warning Tone: White/Orange/Red) ---
colors_interaction = ["#FFFFFF", "#FFC300", "#C70039"]
cmap_interaction = LinearSegmentedColormap.from_list("custom_warm", colors_interaction, N=100)


# ===========================
# 3. Draw Left Plot: Static Topology Backbone
# ===========================
ax1 = axes[0]
# Use mask to only show connections, emphasizing sparsity
sns.heatmap(
    topology_matrix, 
    ax=ax1, 
    cmap=cmap_topology, 
    cbar=False,
    linewidths=1, 
    linecolor='#D0D0D0',
    square=True,
    annot=True, 
    fmt='.0f',
    annot_kws={"size": 12, "weight": "bold", "color": "white"},
    mask=topology_matrix < 0.5 # Mask zero values
)

# Set English Labels
ax1.set_title("(A) Static Topology Backbone (K=10 Cascade)\nFixed Unidirectional Physical Flow", fontsize=16, pad=20, weight='bold')
ax1.set_xlabel("To Node ID", labelpad=12, fontsize=12, weight='bold')
ax1.set_ylabel("From Node ID", labelpad=12, fontsize=12, weight='bold')
# Ticks
node_labels = [f"Node {i}" for i in range(num_nodes)]
ax1.set_xticklabels(node_labels, rotation=45, ha='right')
ax1.set_yticklabels(node_labels, rotation=0)

# Add Annotation Box
ax1.text(num_nodes/2, num_nodes + 2, "Feature: Rigid Cascade Structure (Strictly $N_i \\to N_{i+1}$)", 
         ha='center', va='center', fontsize=12, style='italic', color='#333333',
         bbox=dict(facecolor='#F8F9FA', edgecolor='#D0D0D0', boxstyle='round,pad=0.5'))


# ===========================
# 4. Draw Right Plot: Macro Interaction Matrix
# ===========================
ax2 = axes[1]
sns.heatmap(
    interaction_matrix, 
    ax=ax2, 
    cmap=cmap_interaction, 
    cbar=True, 
    cbar_kws={"label": "Transition Penalty Score (Soft $\\to$ Hard)", "shrink": 0.8, "pad":0.02},
    linewidths=0.5, 
    linecolor='#F0F0F0',
    square=True,
    annot=False,
    vmin=0, vmax=PENALTY_HARD
)

# Set English Labels
ax2.set_title("(B) Learnable Macro Interaction Rules\nDifferentiable Logic & Physics Constraints", fontsize=16, pad=20, weight='bold')
ax2.set_xlabel("Next Macro ($M_{t+1}$)", labelpad=12, fontsize=12, weight='bold')
ax2.set_ylabel("Current Macro ($M_t$)", labelpad=12, fontsize=12, weight='bold')
# Ticks using macro names
ax2.set_xticklabels(macros, rotation=45, ha='right', fontsize=10)
ax2.set_yticklabels(macros, rotation=0, fontsize=10)

# Highlight SKIP -> Real area (Hard Ban Zone)
# Create a rectangle bounding the last row (except the last element)
rect = patches.Rectangle((0, skip_idx), skip_idx, 1, fill=False, edgecolor='#C70039', lw=3, linestyle='--')
ax2.add_patch(rect)
# Add English Annotation
ax2.text(skip_idx/2, skip_idx+0.5, "Padding Rule Violation\n(SKIP $\\to$ Real)", 
         ha='center', va='center', color='#C70039', weight='bold', fontsize=10, 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2))

# Add Diagonal Redundancy Annotation
ax2.text(1.5, 1.5, "Basic Component\nSelf-Redundancy", 
         ha='center', va='center', color='#FFC300', weight='bold', fontsize=9, rotation=45)

# ===========================
# 5. Overall Title and Save
# ===========================
plt.suptitle("Figure 1: The Neuro-Symbolic Foundation of FilterGen2\nCombining Rigid Physical Topology with Differentiable Logical Constraints", 
             fontsize=20, weight='bold', y=1.02)

# Save plot
output_filename = "contrast_matrices_en.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"English comparison plot saved as {output_filename}")
plt.show()