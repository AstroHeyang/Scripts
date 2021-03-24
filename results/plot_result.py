import pandas as pd
import numpy as np

from matplotlib import pyplot as plt


file = '../lc/obs_id.txt'
df_infos = pd.read_csv(file, dtype={"obs_id": str})
obj_names = np.array(df_infos['obj_name'])
rs_list = np.array(df_infos['redshift'])
rs_list_no_random = np.array([2.44, 0.7, 0.46, 0.35, 0.91, 0.25, 0.21, 0.72, 0.58, 0.46, 1.01, 0.34])

df = pd.read_csv('redshift.csv', names=obj_names)
box_data = []
for x in obj_names:
    box_data.append(df[x])
    print(np.median(df[x]))

fig, ax = plt.subplots()
box_plot = ax.boxplot(box_data, vert=False, patch_artist=False,
                      whis=1000, showfliers=False, labels=obj_names,
                      boxprops={'color': 'black'})

ax.plot(rs_list, np.arange(1, 13), 'r.', label='Observed redshift')
ax.plot(rs_list_no_random, np.arange(1, 13), 'b*', label='Simulated redshift using original parameters')
plt.grid(True, ls='--')
handles, labels = ax.get_legend_handles_labels()
handles.append(box_plot["boxes"][0])
# handles.append(box_plot["medians"][0])
labels.append('Simulated redshift using random parameters')
# labels.append('Median simulated redshift using random parameters')
plt.legend(handles, labels)
ax.set_xlabel('redshift')

plt.savefig('result.jpg', dpi=1000, bbox_inches='tight')
