import numpy as np
from xraysim.io import read_pha, read_arf, read_rmf


test_dir = "/home/Isolt/Documents/Work/1H1934-063/spectra/0761870201/"

phas = ['PN_S003_spec.fits', 'M1_S001_spec.fits', 'nu60101003002A01_sr.pha']

grpphas = ['PN_S003_spec.grp', 'M1_S001_spec.grp', 'nu60101003002A01_sr.grp']

rmfs = ['PN_S003.rmf', 'M1_S001.rmf', 'nu60101003002A01_sr.rmf']

arfs = ['PN_S003.arf', 'M1_S001.arf', 'nu60101003002A01_sr.arf']

def test_datapha(phas):

    data_set = []

    if not np.iterable(phas):
        phas = [phas]

    for pha in phas:
        pha = test_dir + pha
        res = read_pha(pha)
        data_set.append(res)

    return data_set


def test_datarmf(rmfs=rmfs):

    rmf_set = []

    if np.size(rmfs) == 1:
        rmfs = [rmfs]

    for rmf in rmfs:
        rmf = test_dir + rmf
        print(rmf)
        res = read_rmf(rmf)
        rmf_set.append(res)

    return rmf_set


def test_dataarf(arfs):

    arf_set = []

    if not np.iterable(arfs):
        arfs = [arfs]

    for arf in arfs:
        arf = test_dir + arf
        res = read_arf(arf)
        arf_set.append(res)

    return arf_set


def run_test():
    pha_sets = test_datapha(phas)
    rmf_sets = test_datarmf(rmfs)
    arf_sets = test_dataarf(arfs)
    return [pha_sets, rmf_sets, arf_sets]
