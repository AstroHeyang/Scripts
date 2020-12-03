import argparse
import numpy as np
import os
import glob
import random
import time

from scipy.stats import poisson
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u
from xraysim import models as mo
from xraysim.astroio.read_data import read_arf, read_rmf, read_pha, read_lc
from xraysim.utils.simtools import fakespec
from xraysim.utils import simtools
from astropy.io import ascii as asc
from astropy.coordinates import SkyCoord
from gdpyc import GasMap
from tqdm import tqdm
from astropy.stats import bayesian_blocks
from matplotlib import pyplot as plt

""" The script is used to calculate the maximum redshift that one transient could be detected.

    Input： 
        The original light curve of the observed source.
    Output:
        The maximum redshift that the object can be detected using XMM-Newton.
         
    The procedures are described as follows.
    1. generate fake light curves with different xmm newton calibration files at 
    different redshifts.
    2. calculate the maximum redshift where the transients can be detected.

"""


class GetResp(object):
    """To get the x-ray calibration matrix, such as arf, rmf, pha and background """

    # Define the base directory for the calibration files.
    caldb_dir = '/Users/liuheyang/xmm_newton/ccf'

    def __init__(self, satellite: str, instrument: str, filter_name: str, ao: int):
        """

        Args:
            satellite: Name of the satellite, e.g., xmm_newton
            instrument: Name of the instrument, e.g., pn
            filter_name: Name of the filter, e.g., thick
            ao: Name of the Announcement of Opportunity, e.g, 19

        """
        self._satellite = satellite
        self._instrument = instrument
        self._filter_name = filter_name
        self._ao = ao

    def _get_resp_files(self) -> dict:
        """

        Notes:
            This function should be modified along with the file path.

        """
        base_name = self.caldb_dir + '/' + self._satellite.lower() + '/' + self._instrument.lower()
        if self._filter_name:
            base_name += '-' + self._filter_name.lower()

        if self._ao:
            base_name += '-' + 'ao' + str(self._ao)

        res = dict()

        res['pha'] = base_name + '.pi'
        res['arf'] = base_name + '.arf'
        res['rmf'] = base_name + '.rmf'
        res['bkg'] = base_name + '.pi'

        for key, value in res.items():
            if not os.path.exists(value):

                # Note: this is written particularly for files with names like those in the 'caldb_dir',
                #       so if the names are changed, these codes also should be modified accordingly.
                if value.find('.pi') != -1:
                    files_bkg = value[:value.index('ao')] + '*.pi'
                    substitutes = glob.glob(files_bkg)
                    substitute = np.random.choice(substitutes)
                    res['pha'] = substitute
                    res['bkg'] = substitute
                    print(f"{value} not found! This 'pi' file would be replaced by "
                          f"{substitute}!")
                else:
                    print(f"{value} not found! Please check it again!")
                    raise FileNotFoundError

        return res

    @property
    def response(self) -> tuple:
        resp_dict = self._get_resp_files()
        pha = read_pha(resp_dict['pha'])
        arf = read_arf(resp_dict['arf'])
        rmf = read_rmf(resp_dict['rmf'])
        bkg = read_pha(resp_dict['bkg'])
        return pha, arf, rmf, bkg


def generate_lc(satellite: str, instrument: str, filter_name: str, ao: int,
                nh_gal=None, nh_host=None, coordinates=None, rs_in=0.0, rs_out=0.0,
                lc_file=None) -> dict:
    """

    Args:
        satellite: Name of the satellite, e.g., xmm_newton
        instrument: Name of the instrument, e.g., pn, mos1, mos2
        filter_name: Name of the filter, e.g., thick, thin, med
        ao: Name of the Announcement of Opportunity, e.g, 19
        nh: column density, in unit of 1e22 cm^-2
        coordinates: Galactic coordinates, (l, b), in units of degrees
        rs_in: input redshift
        rs_out: output redshift
        lc_file: original light curve file

    Returns:
        lc: A fake lightcurve, in format of dict: {'time':ndarray, 'timedel':ndarray,
        'bkg_counts': ndarray, 'counts': ndarray, 'rate':ndarray}

    """
    pha, arf, rmf, bkg = GetResp(satellite=satellite, instrument=instrument,
                                 filter_name=filter_name, ao=ao).response

    # define the path of the original light curve file
    if not lc_file:
        lc_file = './lc/0149780101_lc.txt'

    if not os.path.exists(lc_file):
        print(f"Error! The light curve file is {lc_file}!")
        raise FileNotFoundError("LC file not found! Please check the path!")

    # read the data from the light curve files. It will return a DataLc class. For
    # the detail of DataLC class, see xraysim/astrodata.py
    snb_lc = read_lc(lc_file)

    # set negative flux to zero
    negative_index = np.where(snb_lc.flux < 0.0)[0]
    snb_lc.flux[negative_index] = 0.0

    # Define model and set up the parameter values
    if not nh_gal:
        if coordinates:
            galactic_coordinates = SkyCoord(l=coordinates[0], b=coordinates[1],
                                            unit="deg", frame='galactic')
            nh_gal = GasMap.nh(galactic_coordinates, nhmap="LAB").value / 1e22
        else:
            nh_gal = (1.9 + np.random.rand() * 9.3) / 100  # in the range of 1.9e20-11.2e20

    if not nh_host:
        nh_host = (0.5 + np.random.rand() * 2.9) / 100  # in the range of 0.5e20-3.4e20

    if not rs_out:
        rs_out = 0.01 + np.random.rand() * 2

    alpha = np.random.rand() * 1.6 + 1.9  # alpha in the range of [1.9, 3.5]
    mo_spec = mo.PLSpec(nh_gal=nh_gal, nh_host=nh_host, rs=rs_in, alpha=alpha)
    # mo_bb = mo.BBSpec(nh_gal=nh_gal, nh_host=nh_host, rs=rs_in, temperature=100.0)

    lc_pn = simtools.fakelc(snb_lc, mo_spec.model, rs_in=rs_in, rs_out=rs_out, input_pha=pha,
                            input_arf=arf, input_rmf=rmf, input_bkg=bkg, pha=pha,
                            rmf=rmf, arf=arf, bkg=bkg)

    return lc_pn


def lc_rebin(lc: dict) -> dict:
    """The generated light curve are binned such that each bin contains at least 25 counts.

    Args:
        lc: see the args in 'generate_lc'

    Returns:
        lc_binned: the structure are the same as lc

    """
    counts_series = lc['counts']
    time_delta_series = lc['timedel']
    counts_bkg_series = lc['bkg_counts']

    time_series_new = []
    time_delta_series_new = []
    counts_series_new = []
    counts_bkg_series_new = []

    counts_current = 0
    index_list = [0]

    for i, counts in enumerate(counts_series):
        if counts_current >= 25:
            index_list.append(i+1)
            counts_current = 0
        else:
            counts_current += counts_series[i]

    # remove the last index if counts_series[index_last:] < 25
    if np.sum(counts_series[index_list[-1]:]) < 25:
        index_list.pop()

    index_list.append(-1)

    for j in range(len(index_list)-1):
        time_series_new.append(np.sum(time_delta_series[:index_list[j+1]]))
        time_delta_series_new.append(np.sum(time_delta_series[index_list[j]:index_list[j+1]]))
        counts_series_new.append(np.sum(counts_series[index_list[j]:index_list[j+1]]))
        counts_bkg_series_new.append(np.sum(counts_bkg_series[index_list[j]:index_list[j+1]]))

    lc_binned = {'time': np.array(time_series_new),
                 'timedel': np.array(time_delta_series_new),
                 'bkg_counts': np.array(counts_bkg_series_new),
                 'counts': np.array(counts_series_new),
                 'rate': np.array(counts_series_new)/np.array(time_delta_series_new)}

    return lc_binned


def plot_lc(lc: dict):
    """Plot the generated light curve

    Args:
        lc: see the args in 'generate_lc'.

    Returns:
        None
    """
    time = lc['time']
    flux = lc['rate']
    font0 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    plt.figure(figsize=(9, 6))
    plt.grid(True, ls='--')
    plt.plot(time, flux, '-o', alpha=0.7, mfc='black', linewidth=3)
    plt.xlabel('Time (s)', font0)
    plt.ylabel('Count Rate (count s$^{-1}$ )', font0)
    plt.show()


def cal_flux(flux_s: float, flux_e: float, exposure: float, model: classmethod,
             pha=None, arf=None, rmf=None, bkg=None, mo_en_low=0.5, mo_en_hi=2.0,
             LiMa=True) -> float:
    """Find the detection flux limit for a given exposure and model

    Args:
        flux_s (float): starting flux
        flux_e (float): ending flux
        exposure (float): exposure time
        model (astropy model): the spectral model used to calculate flux

    Kwargs:
        pha (DATAPHA/None): the input source spectrum
        arf (DATAARF/None): effective area file of the instrument
        rmf (DATARMF/None): response matrix of the instrument
        bkg (DATAPHA/None): the background spectrum of the instrument
        mo_en_low (float): the lower energy boundary to calculate the sensitivity
        mo_en_hi (float): the upper energy boundary to calculate the sensitivity
        LiMa: use Lima formula  to estimate the ratio of signal to noise or not

    Returns:
        The flux limit that a source can be detected.

    References:
        Lima formula: Li & Ma, 1983 https://ui.adsabs.harvard.edu/abs/1983ApJ...272..317L/abstract
    """

    flux_del = np.log10(flux_e) - np.log10(flux_s)
    flux = 0.5 * (flux_s + flux_e)

    while flux_del > 0.001:
        # Set up the model and calculate the photon counts using the fakespec
        # function
        model.set_flux(flux, en_low=mo_en_low, en_hi=mo_en_hi)
        res = fakespec(pha=pha, arf=arf, rmf=rmf, bkg=bkg,
                       modelfunc=model.model, model_en_lo=mo_en_low,
                       model_en_hi=mo_en_hi, exposure=exposure)

        tot_cts = np.sum(res['counts'])
        bkg_cts = np.sum(res['bkg_counts'])
        src_cts = tot_cts - bkg_cts
        cts_5sig = poisson.ppf(0.999999, bkg_cts)

        # Two methods to calculate the detection
        # 1. Li-Ma formula
        # 2. Poisson statistic
        if LiMa:
            sn = (tot_cts - bkg_cts) / np.sqrt(tot_cts + bkg_cts)
            if sn >= 5.0:
                flux_e = 0.5 * (flux_s + flux_e)
            else:
                flux_s = 0.5 * (flux_s + flux_e)
        else:
            if src_cts <= 5.0:
                flux_s = 0.5 * (flux_s + flux_e)
            elif tot_cts > cts_5sig:
                flux_e = 0.5 * (flux_s + flux_e)
            else:
                flux_s = 0.5 * (flux_s + flux_e)

        flux = 0.5 * (flux_s + flux_e)
        flux_del = np.log10(flux_e) - np.log10(flux_s)

    return flux


def get_random_lb(threshold: float) -> tuple:
    """

    Args:
        threshold: to avoid the Galactic plane, generally we select objects with |b|> threshold

    Returns:
        coordinates randomly generated ([l, b])
    """
    if np.abs(threshold) >= 90:
        raise ValueError("The threshold should be less than 90!")
    l = random.uniform(0, 360)
    b = np.random.choice([random.uniform(90-threshold, 90),
                          random.uniform(-90, -90+threshold)])
    return l, b


def cal_redshift(lc_file: str, rs_in: float, satellite=None, instrument=None,
                 filter_name=None, ao=None, LiMa=True) -> float:
    """Roughly find the maximum redshift where the source just could be detected.

    Notes:
        Here we use both of these two criteria described in Alp & Larsson (2020), Appendix A.2
        1. The ratio of the maximum background-subtracted flux bin over the 50th flux
        percentile (i.e., percentile of the bins weighted by time for this individual
        light curve) is larger than 3, while the signal-to-background ratio (S/B) is
        higher than 10 at the time of peak flux.
        2. Same as above, but with a peak flux a factor of 5 above the 50th percentile
        and an S/B of at least 3.

    Args:
        lc_file: observed or faked light curve
        rs_in (float): redshift of this object
        satellite (str): the name of satellite, e.g., 'xmm_newton'
        instrument (str): the name of instrument, e.g, 'pn', 'mos1', 'mos2'
        filter_name (str): e.g., 'thick-5'
        ao (int): e.g., 7, 8, 10, 13, 14 , 18, 19
        in_en_low (float): the lower energy boundary to calculate the luminosity
        in_en_hi (float): the upper energy boundary to calculate the luminosity
        mo_en_low (float): the lower energy boundary to calculate the sensitivity
        mo_en_hi (float): the upper energy boundary to calculate the sensitivity

    Returns:
        redshift: the redshift when the signal to noise ratio only just greater than 5
        and the counts greater than 100.
    """
    rs_min = 0.001
    rs_max = 2.0

    rs_list = np.linspace(rs_min, rs_max, num=200, endpoint=True)

    l, b = get_random_lb(15)  # |b| > 15 degree

    for rs in tqdm(rs_list):

        fake_lc = generate_lc(satellite, instrument, filter_name, ao, lc_file=lc_file,
                              coordinates=(l, b), rs_in=rs_in, rs_out=rs)
        fake_lc_binned = lc_rebin(fake_lc)
        cts = fake_lc_binned['counts']
        cts_bkg = fake_lc_binned['bkg_counts']
        cts_net = cts - cts_bkg
        time_delta_series = fake_lc_binned['timedel']
        time_series = fake_lc_binned['time']
        count_rate_net = cts_net / time_delta_series

        if len(cts_net) < 3:
            return rs

        count_rate_50_percentile = np.percentile(count_rate_net, 50, interpolation='midpoint')
        index_max = np.argmax(count_rate_net)
        snr_maximum_flux_bin = cts[index_max] / cts_bkg[index_max]
        criterion1 = (np.max(count_rate_net) >= 3*count_rate_50_percentile) and \
            snr_maximum_flux_bin >= 10
        criterion2 = (np.max(count_rate_net) >= 5*count_rate_50_percentile) and \
            snr_maximum_flux_bin >= 3

        cts_bkg = np.sum(fake_lc['bkg_counts'])
        cts_total = np.sum(fake_lc['counts'])
        sn = (cts_total - cts_bkg) / np.sqrt(cts_total + cts_bkg)
        print(f"Z: {rs}, SNR: {sn}，bkg counts:{cts_bkg}, total counts: {cts_total}")
        print(f"SNR of max flux bin: {snr_maximum_flux_bin}, "
              f"max/median count rate: {np.max(count_rate_net)/count_rate_50_percentile}, "
              f"number of bins: {len(count_rate_net)}")

        if not (criterion1 or criterion2):
            print('=========================================================')
            print(cts_net)
            return rs

        """
        res = fakespec(pha=pha, arf=arf, rmf=rmf, bkg=bkg,
                       modelfunc=model.model, model_en_lo=mo_en_low,
                       model_en_hi=mo_en_hi, exposure=exposure)

        tot_cts = np.sum(res['counts'])
        bkg_cts = np.sum(res['bkg_counts'])
        cts_5sig = poisson.ppf(0.999999, bkg_cts)
        # print("%.2f %.2E %.3f %.1f %.1f" % (exposure, flux, rs, tot_cts,
        # bkg_cts))
        """

        # Two methods to calculate the detection
        # 1. Li-Ma equation
        # 2. Poisson statistic
        """
        if LiMa:
            sn = (cts_total - cts_bkg) / np.sqrt(cts_total + cts_bkg)
            print(f"Z: {rs}, SNR: {sn}，bkg counts:{cts_bkg}, total counts: {cts_total}")
            if sn <= 5.0 or cts_total <= 100:
                return rs
        else:
            print(f"Z: {rs}, total counts: {cts_total}")
            if cts_total < cts_5sig or cts_total <= 100:
                return rs
        """


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Define command line arguments
    parser.add_argument("--lc_file", type=str, default='./lc/0149780101_lc.txt', help="The path of light curve file")
    parser.add_argument("--rs_in", type=float, default=1.17, help="The redshift of this object")
    parser.add_argument("--satellite", type=str, default='xmm_newton', help="Name of the satellite. "
                                                                            "Default is xmm_newton")
    parser.add_argument("--instrument", type=str, default='pn', help="Name of the instrument, e.g., pn, "
                                                                     "mos1, or mos2. Default is pn.")
    parser.add_argument("--filter_name", type=str, default='thick-5', help="Filter in use. e.g., thin, "
                                                                           "med, or thick. "
                                                                           "Default is thick-5.")
    parser.add_argument("--ao", type=int, default=19, help="Name of the Announcement of Opportunity, "
                                                           "e.g., 17, 18, 19, 20. Default is 19.")
    parser.add_argument("--en_low", type=float, default=0.3, help="Lower limit for the energy range. "
                                                                  "Default is 0.3 keV.")
    parser.add_argument("--en_hi", type=float, default=10.0, help="Upper limit for the energy range. "
                                                                  "Default is 10.0 keV")
    parser.add_argument("--mo_en_low", type=float, default=0.3, help="Lower limit for the energy range. "
                                                                     "Default is 0.3 keV.")
    parser.add_argument("--mo_en_hi", type=float, default=10.0, help="Upper limit for the energy range. "
                                                                     "Default is 10.0 keV.")
    parser.add_argument("--mo_lumi", type=float, default=1.E45, help="Luminosity over the model energy "
                                                                     "range. Default is 1e45 erg/s.")

    args = parser.parse_args()

    lc_file = args.lc_file
    rs_in = args.rs_in
    satellite = args.satellite
    instrument = args.instrument
    filter_name = args.filter_name
    ao = args.ao
    en_hi = args.en_hi
    en_low = args.en_low
    mo_en_hi = args.mo_en_hi
    mo_en_low = args.mo_en_low
    mo_lumi = args.mo_lumi

    time_start = time.time()
    # load the calibration matrix
    pha, arf, rmf, bkg = GetResp(satellite=satellite, instrument=instrument,
                                 filter_name=filter_name, ao=ao).response

    lc = generate_lc(satellite, instrument, filter_name, ao, rs_in=rs_in,
                     rs_out=0.5, lc_file=lc_file)
    lc_binned = lc_rebin(lc)
    # plot_lc(lc_binned)

    rs = cal_redshift(lc_file, rs_in, satellite=satellite, instrument=instrument,
                      filter_name=filter_name, ao=ao)
    time_end = time.time()
    print(rs)
    time_delta = time_end - time_start
    print(f'Times Used: {time_delta} s')

    """
    # a list of exposures
    exps = np.logspace(2.0, 5.0, 100)

    # Define model and set up the parameter values
    mo_pl = mo.PLSpec(nh=0.03, rs=0.0, alpha=1.7)
    mo_bb = mo.BBSpec(nh=0.03, rs=0.0, temperature=100.0)
    
    # loop through a list of exposure time
    for exp in exps:
        flux_min = 1.0E-20
        flux_max = 5.0E-4

        # Calculate the detection flux limit for a give exposure time exp and
        # instrument
        flux = cal_flux(flux_min, flux_max, exp, mo_pl, pha=pha, arf=arf,
                        rmf=rmf, bkg=bkg, mo_en_low=mo_en_low,
                        mo_en_hi=mo_en_hi, LiMa=False)

        print("Exposure: %.3f, detection flux limit: %.3e" % (exp, flux))

        # Calculate the maximum redshift at a given luminosity for a particular
        # model and instrument
        rs = cal_redshift(mo_lumi, exp, mo_bb, pha=pha, arf=arf, rmf=rmf,
                          bkg=bkg, in_en_low=en_low, in_en_hi=en_hi,
                          mo_en_low=mo_en_low, mo_en_hi=mo_en_hi)

        print("Exposure: %.3f, detection flux limit: %.3e, luminosity: %.3e, "
              "maximum redshift: %.3f" % (exp, flux, mo_lumi, rs))
    """


