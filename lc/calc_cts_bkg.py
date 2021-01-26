import numpy as np
import pandas as pd


df = pd.read_csv('net_cts.txt', names=['id', 'ctr_net', 'fraction'])
ctr_bkg = df['ctr_net'] * (100-df['fraction'])/df['fraction']
df['ctr_bkg'] = ctr_bkg
df.to_csv('bkg_cts.txt', index=False)

