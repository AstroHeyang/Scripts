import numpy as np
import pandas as pd

from cal_sens import get_obs_infos, lc_rebin
from xraysim.astroio.read_data import read_lc

df = get_obs_infos()
obs_ids = df['obs_id']
bin_sizes = df['bin_size']

for obs_id, bin_size in zip(obs_ids, bin_sizes):
    lc_raw_file = './lc/' + obs_id + '_lc.txt'
    lc_data = read_lc(lc_raw_file)
    # set negative flux to zero
    negative_index = np.where(lc_data.flux < 0.0)[0]
    lc_data.flux[negative_index] = 0.0
    lc_data.counts[negative_index] = 0.0
    n_pixels = len(lc_data.time)
    lc_raw = {'time': lc_data.time,
              'timedel': lc_data.timedel,
              'counts': lc_data.counts,
              'flux': lc_data.flux,
              'rate': lc_data.counts / lc_data.timedel,
              'bkg_counts': np.zeros(n_pixels)}
    lc_raw_rebin = lc_rebin(lc_raw, bin_method='fixed', bin_size=bin_size)
    out_put = {'Time': lc_raw_rebin['time'],
               'Flux': lc_raw_rebin['flux']}
    path_output = './lc/lc_rebin/' + obs_id + '_lc_rebin.txt'
    pd.DataFrame(out_put).to_csv(path_output, sep=' ', index=False)
