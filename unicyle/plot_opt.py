import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

comp_time_list = []
opt_list = []

file_names = [
    'rollout_filter_1200_count_opt.csv', 'rollout_filter_1600_count_opt.csv', 'rollout_filter_2000_count_opt.csv'
]

labels = [
    "Filter K=1200", "Filter K=1600", "Filter K=2000"
]

# Read data
for file in file_names:
    df = pd.read_csv('./data/experiment/' + file)
    comp_time_list.append(df['comp_time'])
    opt_list.append(df['num_opts'])

# --- Plot setup ---
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(hspace=0.4)

def annotate_medians(ax, data_list):
    """Annotate median values on each box."""
    for i, data in enumerate(data_list):
        median = np.median(data)
        ax.text(i + 1.25, median, f"{median:.3f}",
                va='center', ha='left', fontsize=14, color='black')

# --- Computation time boxplot ---
axs[0].boxplot(comp_time_list, labels=labels, patch_artist=True, sym='')
axs[0].set_title("Computation Time for MPPI Without and With Safety-Filtered Rollouts")
axs[0].set_ylabel("Time (s)")
axs[0].grid(True, axis='y', linestyle='--', alpha=0.6)
annotate_medians(axs[0], comp_time_list)

# --- Distance boxplot ---
axs[1].boxplot(opt_list, labels=labels, patch_artist=True, sym='')
axs[1].set_title("Number of Rollout Optimization Problems per Step")
axs[1].set_ylabel("# Optimizations per Rollout")
axs[1].grid(True, axis='y', linestyle='--', alpha=0.6)
annotate_medians(axs[1], opt_list)

plt.tight_layout()
plt.savefig("./data/experiment/number_opts.png", dpi=300, bbox_inches='tight')