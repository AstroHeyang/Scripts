import numpy as np
import glob

from astropy.io import ascii as asc
from astropy.table import Table

paths = glob.glob('*.txt')
paths.sort()

for file in paths:
    print(file)
    lc = asc.read(file)
    lc_col1 = lc[1]
    lc_col0 = [0.0] + list(lc_col1[:-1])
    data = Table([lc_col0, lc_col1, lc[2], lc[3]],
                 names=['start', 'end', 'rate', 'flux'])
    asc.write(data, file, overwrite=True)
