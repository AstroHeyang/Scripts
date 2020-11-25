import numpy as np
import re

from astropy.io import ascii as asc


find_txt = re.compile(r'[0-9a-zA-Z]]+\.txt')

for file in find_txt:
    with asc.read(file)


