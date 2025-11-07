import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
file_names = [
    'rollout_filter_1200.csv',
    'rollout_filter_1600.csv',
    'rollout_filter_2000.csv'
]

labels = [
    "Filter K=1200",
    "Filter K=1600",
    "Filter K=2000"
]

colors = ['tab:blue', 'tab:orange', 'tab:green']  # distinct, readable palette

# --- Plot setup ---
plt.figure(figsize=(10, 6))
plt.title("Computation Time vs. Number of Optimization Problems per Step")

for file, label, color in zip(file_names, labels, colors):
    df = pd.read_csv('./data/experiment/' + file)
    
    # Convert to numeric just in case
    df['comp_time'] = pd.to_numeric(df['comp_time'], errors='coerce')
    df['num_opts'] = pd.to_numeric(df['num_opts'], errors='coerce')
    df = df.dropna(subset=['comp_time', 'num_opts'])
    
    plt.scatter(df['num_opts'], df['comp_time'], 
                label=label, color=color, alpha=0.6, edgecolors='k', s=50)

# --- Axis labels, grid, legend ---
plt.xlabel("Number of Optimization Problems per Step", fontsize=12)
plt.ylabel("Computation Time (s)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Rollout Configurations")
plt.tight_layout()

# --- Save figure ---
plt.savefig("./data/experiment/scatter_numopts_vs_time.png", dpi=300, bbox_inches='tight')
plt.show()