import glob
import numpy as np

from matplotlib import pyplot as plt
from astropy.io import ascii as asc


paths = glob.glob('*.txt')
paths.sort()

for file in paths:
    print(file)
    lc = asc.read(file)
    time, count = lc['end'], lc['rate']
    index_good = np.where(count > 0)
    time, flux = time[index_good], count[index_good]

    font0 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    plt.figure(figsize=(9, 6))
    plt.grid(True, ls='--')
    plt.plot(time, flux, '-o', alpha=0.7, mfc='black', linewidth=3)
    plt.xlabel('Time (s)', font0)
    plt.ylabel('Count Rate (count s$^{-1}$ )', font0)
    plt.title(file[:-4], font0)
    plt.savefig(file[:-4] + '.jpg', dpi=1000, bbox_inches='tight')
