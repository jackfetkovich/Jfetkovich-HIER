import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


comp_time_list = []
dist_list = []

file_names = ['no_rollout_filter_1200.csv', 'no_rollout_filter_1600.csv', 'no_rollout_filter_2000.csv',
              'rollout_filter_1200.csv', 'rollout_filter_1600.csv', 'rollout_filter_2000.csv']

labels = ["No Filter K=1200", "No Filter K=1600", "No Filter K=2000", "Filter K=1200", "Filter K=1600", "Filter K=2000"]

for file in file_names:
    df = pd.read_csv('./data/experiment/' + file)
    comp_time_list.append(df['comp_time'])

plt.boxplot(comp_time_list, labels=labels, patch_artist=True, sym='')
plt.show()