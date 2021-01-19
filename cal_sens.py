import argparse
import numpy as np
import pandas as pd
import os
import glob
import random
import time
import csv

from scipy.stats import poisson
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u
from astropy.io import ascii as asc
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.stats import bayesian_blocks
from astropy.table import Table
from xraysim import models as mo
from xraysim.astroio.read_data import read_arf, read_rmf, read_pha, read_lc
from xraysim.utils.simtools import fakespec
from xraysim.utils import simtools
from gdpyc import GasMap
from tqdm import tqdm
from matplotlib import pyplot as plt
from multiprocessing import Process

""" The script is used to calculate the maximum redshift that one transient could be detected.

    Input： 
        The original light curve of the observed source (12 SN SBOs in Alp & Larsson 2020).
    Output:
        The maximum redshift that the object can be detected using XMM-Newton.
         
    The procedures are described as follows.
    1. generate fake light curves with different xmm newton calibration files at 
    different redshifts.
    2. calculate the maximum redshift where the transients can be detected.
    3. after plenty of simulations, we record all the redshifts and plot their distribution. 
    
"""


class GetResp(object):
    """To get the x-ray calibration matrix, including arf, rmf, pha and background """

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

    # if use bkg file like '.pi' to get the src2bkg rate; else if use the observed event files then use (40/80)^2
    @property
    def src_bkg_area_ratio(self, radius_src=1/60) -> float:
        bkg_file = self._get_resp_files()['bkg']
        with fits.open(bkg_file) as hdul:
            bkg_header = hdul[0].header
            deg2pixel_y = np.abs(bkg_header['REFYCDLT'])
            deg2pixel_x = np.abs(bkg_header['REFXCDLT'])
            radius_bkg = hdul[3].data[0][3]
            area_bkg = np.square(radius_bkg) * deg2pixel_y * deg2pixel_x
            area_src = np.square(radius_src)
        return area_src/area_bkg


def generate_lc(satellite: str, instrument: str, filter_name: str, ao: int,
                nh_gal=None, nh_host=None, alpha=None, temperature=None,
                coordinates=None, rs_in=0.0, rs_out=0.0, lc_file=None,
                bkg_rate=0.0, poisson=True, src2bkg=None) -> dict:
    """

    Args:
        satellite: Name of the satellite, e.g., xmm_newton
        instrument: Name of the instrument, e.g., pn, mos1, mos2
        filter_name: Name of the filter, e.g., thick, thin, med
        ao: Name of the Announcement of Opportunity, e.g, 19
        nh_gal: The Galactic column density, in unit of 1e22 cm^-2
        nh_host: The host galaxy column density, in unit of 1e22 cm^-2
        alpha: powerlaw index (X-ray spectrum model)
        coordinates: Galactic coordinates, (l, b), in units of degrees
        rs_in: input redshift
        rs_out: output redshift
        lc_file: original light curve file

    Returns:
        lc: A fake lightcurve, in format of dict: {'time':ndarray, 'timedel':ndarray,
        'bkg_counts': ndarray, 'counts': ndarray, 'rate':ndarray}

    """
    resp = GetResp(satellite=satellite, instrument=instrument, filter_name=filter_name, ao=ao)
    pha, arf, rmf, bkg = resp.response
    if not src2bkg:
        src2bkg = resp.src_bkg_area_ratio

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
        nh_host_1 = 0.01 + np.random.rand() * 4.7
        nh_host_2 = 1e-5
        # in the range of 0.01e22-4.7e22 or 1e17
        nh_host = np.random.choice((nh_host_1, nh_host_2))
    if not rs_out:
        rs_out = 0.01 + np.random.rand() * 2
    if not alpha:
        alpha = np.random.rand() * 1.6 + 1.9  # alpha in the range of [1.9, 3.5]
    if not temperature:
        temperature = np.random.rand() * 130 + 800  # T in the range of [0.13, 0.93] * u.keV

    # mo_spec = mo.PLSpec(nh_gal=nh_gal, nh_host=nh_host, rs=rs_in, alpha=alpha)
    mo_spec = mo.BBSpec(nh_gal=nh_gal, nh_host=nh_host, rs=rs_in, temperature=temperature)

    lc_pn = simtools.fakelc(snb_lc, mo_spec.model, rs_in=rs_in, rs_out=rs_out, input_pha=pha,
                            input_arf=arf, input_rmf=rmf, input_bkg=bkg, pha=pha, poisson=poisson,
                            rmf=rmf, arf=arf, bkg=bkg, bkg_rate=bkg_rate, src2bkg=src2bkg)

    return lc_pn


def lc_rebin(lc: dict, bin_method=None, bin_size=1) -> dict:
    """The generated light curve are binned such that each bin contains m seconds or n counts.
        n = [60, 800, 12, 160, 8, 400, 48, 400, 160, 160, 24, 8], see Figure 1 in
        Alp & Larsson (2020)

    Args:
        lc: see the args in 'generate_lc'
        bin_method (str): fixed or dynamic bin_size; if fixed, to keep each bin contains
            the same (default 25) counts; if dynamic, the bin size are the same as in the
            article (Alp & Larsson 2020).
        bin_size (int): fixed bin size

    Returns:
        lc_binned: the structure are the same as lc

    """
    counts_series = lc['counts']
    time_series = lc['time']
    time_delta_series = lc['timedel']
    counts_bkg_series = lc['bkg_counts']

    time_series_new = []
    time_delta_series_new = []
    counts_series_new = []
    counts_bkg_series_new = []

    if bin_method == 'dynamic':
        counts_current = 0
        index_list = [0]
        counts_threshold = 25

        for i, counts in enumerate(counts_series):
            if counts_current >= 25:
                index_list.append(i + 1)
                counts_current = 0
            else:
                counts_current += counts_series[i]

        # remove the last index if counts_series[index_last:] < counts_threshold (default 25)
        if np.sum(counts_series[index_list[-1]:]) < counts_threshold:
            index_list.pop()

        index_list.append(-1)

        # TODO: the time series need to be checked later.
        for j in range(len(index_list) - 1):
            time_series_new.append(np.sum(time_delta_series[:index_list[j + 1]]))
            time_delta_series_new.append(np.sum(time_delta_series[index_list[j]:index_list[j + 1]]))
            counts_series_new.append(np.sum(counts_series[index_list[j]:index_list[j + 1]]))
            counts_bkg_series_new.append(np.sum(counts_bkg_series[index_list[j]:index_list[j + 1]]))

    if bin_method == 'fixed':

        i, time_in_bin, index_pre = 0, 0, 0
        index_pre_list = [index_pre]
        time_current = 0

        while i < len(time_series):
            if time_in_bin < bin_size:
                time_in_bin += time_delta_series[i]
                time_current += time_delta_series[i]
            else:
                if not time_series_new:
                    time_series_new.append(time_in_bin/2)
                else:
                    time_series_pre = time_series_new[-1]
                    time_series_new.append(time_series_pre + time_in_bin)
                time_delta_series_new.append(time_in_bin)
                counts_series_new.append(np.sum(counts_series[index_pre:i]))
                counts_bkg_series_new.append(np.sum(counts_bkg_series[index_pre:i]))
                index_pre = i
                time_in_bin = 4
                index_pre_list.append(index_pre)
                time_current += time_delta_series[i]

            if i == len(time_series)-1 and time_in_bin != 4:
                time_series_pre = time_series_new[-1]
                time_series_new.append(time_series_pre + 30 + time_in_bin/2)
                time_delta_series_new.append(time_in_bin)
                counts_series_new.append(np.sum(counts_series[index_pre:]))
                counts_bkg_series_new.append(np.sum(counts_bkg_series[index_pre:]))
            i += 1

    lc_binned = {'time': np.array(time_series_new),
                 'timedel': np.array(time_delta_series_new),
                 'bkg_counts': np.array(counts_bkg_series_new),
                 'counts': np.array(counts_series_new),
                 'rate': (np.array(counts_series_new)-np.array(counts_bkg_series_new))
                         / np.array(time_delta_series_new)}

    return lc_binned


def combine_lc(lc_list):
    """

    Args:
        lc_list: lc generated using pn, mos1, and mos2

    Returns:
        the combined lc：the format is the same as input lc

    """
    if not lc_list:
        raise TypeError("The input lc list is empty!")
    if len(lc_list) == 1:
        return lc_list[0]

    lc_combined = lc_list[0].copy()
    net_counts = lc_combined['counts'] * 0.0
    bkg_counts = lc_combined['bkg_counts'] * 0.0
    for lc in lc_list:
        net_counts += (lc['counts'] - lc['bkg_counts'])
        bkg_counts += lc['bkg_counts']
    lc_combined['bkg_counts'] = bkg_counts
    lc_combined['counts'] = net_counts + bkg_counts
    lc_combined['rate'] = net_counts/lc_combined['timedel']
    return lc_combined


def plot_lc(lc: dict, save_fig=False, file_name=None):
    """Plot the generated light curve

    Notes:
        plot the raw light curve and a new one that were binned to five data points.

    Args:
        lc: see the args in 'generate_lc'.
        save_fig: whether save the figure or not.
        file_name: the name of the figure to be saved.

    Returns:
        None
    """
    time_raw = lc['time']
    flux_raw = lc['rate']

    font0 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    plt.figure(figsize=(9, 6))
    plt.grid(True, ls='--')
    plt.plot(time_raw, flux_raw, '-o', alpha=0.7, mfc='black', linewidth=3)
    plt.xlabel('Time (s)', font0)
    plt.ylabel('Count Rate (count s$^{-1}$ )', font0)
    if save_fig:
        plt.savefig(file_name, dpi=1200, bbox_inch='tight')
    plt.close("all")


def plot_lc_total(obs_id=None, filter_type='thin-5', ao=19, title=None, mos1=True, bin_size=0):
    """Plot the generated combined light curve (pn+mos1+mos2), used to compare with the raw light curve in the paper

    Notes:
        plot the raw light curve and a new one that were binned to five data points.

    Args:
        obs_id: xmm_newton obs_id
        filter_type: thick, med, or thin
        ao: e.g., 19

    Returns:
        None
    """
    lc_dir = './fake_lc/'
    lc_pn = lc_dir + obs_id + '_lc_pn_' + filter_type + '_ao_' + str(ao) + '.csv'
    lc_mos1 = lc_dir + obs_id + '_lc_mos1_' + filter_type + '_ao_' + str(ao) + '.csv'
    lc_mos2 = lc_dir + obs_id + '_lc_mos2_' + filter_type + '_ao_' + str(ao) + '.csv'

    df_pn = pd.read_csv(lc_pn)
    df_mos1 = pd.read_csv(lc_mos1)
    df_mos2 = pd.read_csv(lc_mos2)

    time = df_pn['time']
    if mos1:
        flux = df_pn['rate'] + df_mos1['rate'] + df_mos2['rate']
    else:
        flux = df_pn['rate'] + df_mos2['rate']

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
              'rate': lc_data.counts/lc_data.timedel,
              'bkg_counts': np.zeros(n_pixels)}
    lc_raw_rebin = lc_rebin(lc_raw, bin_method='fixed', bin_size=bin_size)

    time_raw = lc_raw_rebin['time']
    flux_raw = lc_raw_rebin['rate']
    peak = np.max(flux_raw)
    mean = np.mean(flux_raw)
    print("peak/mean: ", peak/mean)
    print("net counts: ", np.sum(lc_raw_rebin['counts']) - np.sum(lc_raw_rebin['bkg_counts']))

    font0 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    plt.figure(figsize=(9, 6))
    plt.grid(True, ls='--')
    plt.plot(time, flux, '-o', alpha=0.7, mfc='black', linewidth=3, label='generated')
    plt.plot(time_raw, flux_raw, '-o', alpha=0.7, linewidth=3, label='raw')
    plt.xlabel('Time (s)', font0)
    plt.ylabel('Count Rate (count s$^{-1}$ )', font0)
    plt.title(title)
    plt.legend()
    # file_name = lc_dir + obs_id + '_total.jpg'
    file_name = obs_id + '_total.jpg'
    plt.savefig(file_name, dpi=1200, bbox_inch='tight')


def get_obs_infos(file=None) -> dict:
    """To get the basic information of the 12 SN SBOs in Alp & Larsson (2020)

    Args:
        file: the csv file containing the basic information of the 12 sources

    Returns:
        pandas.DataFrame

    """
    if not file:
        file = './lc/obs_id.txt'
    df = pd.read_csv(file, dtype={"obs_id": str})
    return df


def get_random_lb(threshold: float) -> tuple:
    """

    Args:
        threshold: sometimes we want to select objects out of the Galactic plane, here we select
        sky regions with |b|> threshold

    Returns:
        coordinates randomly generated ([l, b])
    """
    if np.abs(threshold) >= 90:
        raise ValueError("The threshold should be less than 90!")
    l = random.uniform(0, 360)
    b = np.random.choice([random.uniform(90 - threshold, 90),
                          random.uniform(-90, -90 + threshold)])
    return l, b


def get_bkg_count_rate(instrument: str, filter_name=None, threshold=15, radius=40):
    """

    Args:
        instrument: pn, mos1, or mos2.
        filter_name: thin, thick, or medium.
        threshold: see function 'get_random_lb', default is 15 degree.
        radius: the radius of source region, default is 40".

    Returns:
        mean, sigma of the background count rates
    """

    dir_bkg = './background/'
    if instrument == 'pn':
        path_bkg = dir_bkg + 'PN_bkg_info.txt'
    elif instrument == 'mos1':
        path_bkg = dir_bkg + 'MOS1_bkg_info.txt'
    elif instrument == 'mos2':
        path_bkg = dir_bkg + 'MOS2_bkg_info.txt'
    else:
        print('Error! The instrument should be pn, mos1, or mos2!')
        raise NameError

    bkg = Table.read(path_bkg, format='ascii')
    column_names = bkg.colnames
    bkg_dict = {}
    for i in column_names:
        bkg_dict[i] = np.array(bkg[i])
    df_bkg = pd.DataFrame(bkg_dict)
    df_bkg_full_window = df_bkg[df_bkg['SUBMODE'] != 'PrimeSmallWindow'].reset_index()

    # remove those with |b| < threshold, in unit of degree.
    lb = SkyCoord(df_bkg_full_window['ra_Deg']*u.degree,
                  df_bkg_full_window['dec_Deg']*u.degree,
                  frame='icrs').galactic
    l, b = lb.l.value, lb.b.value
    df_bkg_full_window_new = pd.concat([df_bkg_full_window,
                                        pd.DataFrame({"l_deg": l,
                                                      "b_deg": b})],
                                       axis=1, join='outer')
    df_bkg_full_window_new2 = df_bkg_full_window_new.drop(
        df_bkg_full_window_new[np.abs(df_bkg_full_window_new['b_deg']) <= threshold].index)

    # depend on filter_name
    if filter_name == 'medium':
        count_rate_bkg_raw = df_bkg_full_window_new2[df_bkg_full_window_new['FILTER'] == 'Medium']
    elif filter_name == 'thin':
        count_rate_bkg_raw = df_bkg_full_window_new2[df_bkg_full_window_new['FILTER'] == 'Thin1']
    else:
        count_rate_bkg_raw = df_bkg_full_window_new2

    radius_scaling_factor = np.square(count_rate_bkg_raw['radius_arcmin'] / radius)
    count_rate_bkg_mean = np.mean(count_rate_bkg_raw['count_rate'] / radius_scaling_factor)
    count_rate_bkg_sigma = np.std(count_rate_bkg_raw['count_rate'] / radius_scaling_factor)

    return count_rate_bkg_mean, count_rate_bkg_sigma


def calc_snr(count_total, count_bkg, radius_src=40, radius_bkg=80, LiMa=False):
    """
    Notes:
        As for so-called LiMa equation, Details see Ti-Pei Li & Yu-Qian Ma 1983, ApJ, 272, 317
        http://articles.adsabs.harvard.edu/pdf/1983ApJ...272..317L
    Args:
        count_total: total count (rate)
        count_bkg: background count (rate)
        radius_src: the radius of source region, default 40"
        radius_bkg:  the radius of background region, default 80"
        LiMa: Using LiMa equation or not

    Returns:
        S/N: signal to noise ratio

    """
    alpha = np.square(radius_src) / np.square(radius_bkg)
    count_bkg /= alpha
    if LiMa:
        snr = np.sqrt(2) * np.sqrt(
            count_total * np.log((1 + alpha) / alpha * (count_total / (count_total + count_bkg)))
            + count_bkg * np.log((1 + alpha) * (count_bkg / (count_total + count_bkg))))
    else:
        count_bkg_scaled = count_bkg * alpha
        snr = count_total * 5 / poisson.ppf(0.999999, count_bkg_scaled)

    return snr


def cal_redshift(lc_file: str, rs_in: float, satellite=None, instrument=None, filter_name=None,
                 ao=None, nh_gal=None, nh_host=None, alpha=None, temperature=None,
                 LiMa=False, bin_size=None, poisson=False,
                 save_res=False, plot_res=False, index_obj=None) -> float:
    """Roughly find the maximum redshift where the source just could be detected.

    Notes:
        Here we use two criteria:
        1. The ratio of the maximum background-subtracted flux bin over the mean flux
           is larger than 2.
        2. S/N > 5.

    Args:
        lc_file: observed or faked light curve
        rs_in (float): redshift of this object
        satellite (str): the name of satellite, e.g., 'xmm_newton'
        instrument (str): the name of instrument, e.g, 'pn', 'mos1', 'mos2'
        filter_name (str): e.g., 'thick-5'
        ao (int): e.g., 7, 8, 10, 13, 14 , 18, 19
        LiMa: use LiMa formula to calculate the S/N or not
        bin_size: the bin size when rebinning the light curve
        save_res: save the light curve at the maximum redshift or not
        plot_res: plot the light curve at the maximum redshift or not

    Returns:
        redshift: the redshift when the signal to noise ratio only just greater than 5
        and the peak flux greater than two times of the mean.
    """

    rs_min = rs_in*0.8
    rs_max = rs_in*3.0
    rs_list = np.linspace(rs_min, rs_max, num=50, endpoint=True)

    if not instrument:
        instrument = ('pn', 'mos1', 'mos2')
    l, b = get_random_lb(15)  # |b| > 15 degree

    tqdm_range = tqdm(rs_list)
    for i, rs in enumerate(tqdm_range):
        fake_lc_list = []
        for ins in instrument:
            cts_bkg = get_bkg_count_rate_observation(instrument=ins)
            if index_obj is not None:
                count_rate_bkg = cts_bkg[index_obj]
            else:
                cts_bkg_mean, cts_bkg_sigma = np.mean(cts_bkg), np.std(cts_bkg)
                count_rate_bkg = max(np.random.normal(cts_bkg_mean, cts_bkg_sigma), 0.01)

            fake_lc = generate_lc(satellite, ins, filter_name, ao, lc_file=lc_file,
                                  coordinates=(l, b), rs_in=rs_in, rs_out=rs, nh_gal=nh_gal,
                                  nh_host=nh_host, alpha=alpha, temperature=temperature,
                                  bkg_rate=count_rate_bkg, poisson=poisson)

            fake_lc_binned = lc_rebin(fake_lc, bin_method='fixed', bin_size=bin_size)
            fake_lc_list.append(fake_lc_binned)

        fake_lc_combined = combine_lc(fake_lc_list)
        cts = fake_lc_combined['counts']
        cts_bkg = fake_lc_combined['bkg_counts']
        cts_net = cts - cts_bkg
        time_delta_series = fake_lc_combined['timedel']
        count_rate_net = cts_net / time_delta_series

        cts_bkg_observed = np.sum(cts_bkg)
        cts_total = np.sum(cts_net) + cts_bkg_observed
        if LiMa:
            sn = (cts_total - cts_bkg_observed) / np.sqrt(cts_total + cts_bkg_observed)
        else:
            # here times 5, in order to be consistent with LiMa
            sn = cts_total*5/poisson.ppf(0.999999, cts_bkg_observed)

        flux_scale = np.max(count_rate_net) / np.mean(count_rate_net)
        criterion1 = (flux_scale < 1.92)
        criterion2 = (sn < 5)
        criterion3 = (np.sum(cts_net) < 62)

        # the threshold of terminating the loop
        if criterion1 or criterion2 or rs == rs_list[-1]:
            # res = np.sum(cts_net)
            print(f"Z: {rs}, SNR: {sn}，bkg counts:{cts_bkg_observed}, total counts: {cts_total}, "
                  f"net counts: {cts_total - cts_bkg_observed}")
            print(f"flux_peak/flux_mean: {flux_scale}")
            print("-------------------------------------------")
            path = './fake_lc'
            path_csv = path + '/' + lc_file[lc_file.index('0'):-4] + '-' + ''.join(instrument) + '_' + \
                       f'{filter_name}_' + 'ao_' + str(ao) + '.csv'

            path_img = path + '/' + lc_file[lc_file.index('0'):-4] + '-' + ''.join(instrument) + '_' + \
                       f'{filter_name}_' + 'ao_' + str(ao) + '.jpg'
            if save_res:
                pd.DataFrame(fake_lc_combined).to_csv(path_csv, index=False)
            if plot_res:
                plot_lc(fake_lc_combined, save_fig=True, file_name=path_img)

            tqdm_range.close()
            return rs


"""This part is used to test this program (start)"""


def get_bkg_count_rate_observation(bkg_file=None, instrument=None, src2bkg=0.25) -> tuple:
    """
    Notes:
        The radius of the background region is 80", while the source radius is around 40".
    Args:
        bkg_file: csv file recording the background count rate of the 12 SBO in Alp & Larsson (2020)
        instrument: pn, mos1 or mos2
        src2bkg: the ratio between the areas of source and background

    Returns:
        count_rate_bkg (tuple)
    """
    if not bkg_file:
        bkg_file = './background/bkg_rate_expo_cutGTI.txt'
    if not instrument:
        instrument = 'PN'
    else:
        if isinstance(instrument, str):
            instrument = instrument.upper()
        else:
            raise TypeError('Illegal instrument!')
    df = pd.read_csv(bkg_file, skipinitialspace=True)
    df_this = df[(df['instrument'] == instrument)].reset_index(drop=True)
    res = np.array(df_this['count_rate']) * src2bkg
    return res


def generate_light_curve(satellite=None, instrument=None, filter_name=None, ao=None):
    """This function is used to generate light curve at the redshift given by the paper, namely,
    here we want to compare these generated light curve with those in the paper (Alp & Larsson
    2020).

    Args:
        satellite: see the cal_redshift
        instrument: see the cal_redshift
        filter_name: see the cal_redshift
        ao: see the cal_redshift

    Returns:
        None; but it will generate a series of csv file and plots. These plots can be used to
        compare the light curves in the paper (see their Figure 1) and those generated by our
        simulations.
    """

    if not satellite:
        satellite = 'xmm_newton'
    if not instrument:
        instrument = 'pn'
    if not filter_name:
        filter_name = 'thin-5'
    if not ao:
        ao = 19

    paths = glob.glob('./lc/*.txt')
    paths.sort()
    df_info = get_obs_infos()

    rs_list = np.array(df_info['redshift'])
    bin_size_list = np.array(df_info['bin_size'])
    nh_gal_list = np.array(df_info['nh_gal']) * 0.01
    nh_host_list = np.array(df_info['nh_host'])
    alpha_list = np.array(df_info['pl_index'])
    temperature_list = np.array(df_info['bb_T']) * 1e3

    count_rate_bkg_list = get_bkg_count_rate_observation()
    tqdm_range = tqdm(rs_list)

    for i, rs in enumerate(tqdm_range):
        fake_lc = generate_lc(satellite, instrument, filter_name, ao, lc_file=paths[i],
                              rs_in=rs, rs_out=rs, nh_gal=nh_gal_list[i], poisson=False,
                              nh_host=nh_host_list[i], alpha=alpha_list[i], temperature=temperature_list[i],
                              bkg_rate=count_rate_bkg_list[i])
        fake_lc_binned = lc_rebin(fake_lc, bin_method='fixed', bin_size=bin_size_list[i])

        lc_file = paths[i]
        path = './fake_lc'
        path_csv = path + '/' + lc_file[lc_file.index('0'):-4] + f'_{instrument}_' + \
                   f'{filter_name}_' + 'ao_' + str(ao) + '.csv'
        path_csv2 = path + '/' + lc_file[lc_file.index('0'):-4] + f'_{instrument}_' + \
                   f'{filter_name}_' + 'ao_' + str(ao) + '_raw.csv'
        pd.DataFrame(fake_lc_binned).to_csv(path_csv, index=False)
        pd.DataFrame(fake_lc).to_csv(path_csv2, index=False)

        path_img = path + '/' + lc_file[lc_file.index('0'):-4] + f'_{instrument}_' + \
                   f'{filter_name}_' + 'ao_' + str(ao) + '.jpg'
        plot_lc(fake_lc_binned, save_fig=True, file_name=path_img)
    return None


"""This part is used to test this program (end)"""


def main(res_file):
    """The main function of this script"""
    satellite = 'xmm_newton'
    filter_name = np.random.choice(['thin-5', 'med-5', 'thick-5'])
    ao = np.random.choice([13, 14, 16, 17, 18, 19])
    lc_file_list = glob.glob('./lc/*.txt')
    lc_file_list.sort()

    df_infos = get_obs_infos()
    bin_sizes = np.array(df_infos['bin_size'])
    rs_list = np.array(df_infos['redshift'])
    temperature_list = np.array(df_infos['bb_T']) * 1e3

    rs_out = []
    for i, lc_file, rs_in, bin_size, temperature in zip(range(len(lc_file_list)), lc_file_list,
                                                        rs_list, bin_sizes, temperature_list):
        rs = cal_redshift(lc_file, rs_in, satellite=satellite,
                          filter_name=filter_name, ao=ao, bin_size=bin_size,
                          temperature=temperature, LiMa=True, poisson=True)
        rs_out.append(rs)

    with open(res_file, res_file, newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(rs_out)

    return None


if __name__ == '__main__':
    time_start = time.time()
    res_file = './results/redshift.csv'
    number_simulation = 5
    subprocesses = []
    for _ in range(number_simulation):
        p = Process(target=main, args=(res_file,))
        subprocesses.append(p)
    for p in subprocesses:
        p.start()
    for p in subprocesses:
        p.join()
    time_end = time.time()
    print(f"Time consumed {time_end - time_start} s.")
