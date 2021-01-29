#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import os
import glob
import random
import time
import csv
import math

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
from threading import Thread

__author__ = 'He-Yang Liu'

""" The script is used to calculate the maximum redshift that one transient could be detected.

    Input:
        The original light curves of the observed sources (12 SN SBOs in Alp & Larsson 2020).
    Output:
        The maximum redshifts that these object could be detected using XMM-Newton.
         
    The procedures are described as follows.
    1. Generate fake light curves with different xmm newton calibration files at different 
       redshifts using xmm_newton filters 'pn', 'mos1', and 'mos2'. See class 'GetResp', 
       functions 'get_bkg_count_rate_observation' and 'generate_lc'.
       
    2. Rebin these three light curves (pn, mos1 and mos2) using the same bin sizes as in the 
       paper (in fact, these time bin sizes vary along with redshift because of time dilation 
       <cosmic effect>), then we combined the three light curves into one for each source.
       See functions 'lc_rebin' and 'combine_lc'.
     
    3. Calculate the maximum redshift where the transients just could be detected. In a loop 
       for redshift, we calculate the S/N, net counts and flux_peak/flux_mean for each light 
       curve. The loop would be terminated if the light curve failed to meet these criteria: 
       (1) S/N >= 5; here S/N is the signal to noise ratio for the whole duration of the light
           curve.
       (2) net counts > 62; 62 is the minimum observed counts of the SN SBO candidates in the 
           paper.
       (3) flux_peak/flux_mean >= 1.92; flux_peak and flux_mean are the peak and the mean count
           rate of the light curves after rebinning; similarly, 1.92 is also the minimum obtained
           from the paper.
       See functions 'calc_redshift'.   
    
    4. After plenty of simulations, we record all the redshifts and plot their distribution. See
       './results/plot_result.py' 
    
    Notes:
    1. Calibration database
        The XMM Newton calibration files are got from NASA webspec; details see the following link:
        https://heasarc.gsfc.nasa.gov/FTP/webspec/
        
    2. Background count rate
        The background count rate is crucial when calculating the S/N. In this program, there are 
        two options, one is derived from the calibration database like '*.pi' (see ./xraysim/
        simtools.fakelc 'if not bkg_rate' and GetResp.src_bkg_area_ratio); the other one is derived
        from the XMM Newton observations (the 12 SN SBO candidates ), see the function 
        'get_bkg_count_rate_observation', and we appreciate Jingwei Hu for her awesome job. Here we 
        choose the second one.  
        
    3. Parameter randomization
        X-ray model (Blackbody): 
           (1) Temperature, fixed to the values given by the paper; 
           (2) Galactic N_H: a random Galactic (l,b) with |b| > 15 degree is generated, then we 
               calculate corresponding N_H;
           (3) host N_H: this is difficult to estimate. According to the private communication with the 
               author and the test we made, we decided that one half have no host absorption (1e17) and 
               the other half are in the range of (1e20-4.7e22).
               
        Background count rate:
           we got the background count rates for these 12 objects from the XMM Newton pn, mos1, and mos2 
           observations. A circle with a radius of 80" are used. As a rule of thumb, the radius of the 
           circle for a source is 40", then we scale the background count rate using this ratio. To expand
           the range, we use a random rate in the range of [minimum-sigma, maximum+sigma]. 
           
    4. redshift
        When moving a lightcurve from the original redshift z0 to z1, we have conducted the following 
        corrections.
        (1). energy band: from [0.3, 10]*(1+z0) to [0.3, 10]*(1+z1)  (in the rest frame); 
        (2). flux: the corresponding luminosity distance are used to calculate the flux;
        (3). time dilation: details see Fig. 9.4 in <An introduction to active galactic nuclei> written
             by Bradley M. Peterson.           
            
"""


class GetResp(object):
    """To get the x-ray calibration matrix, including arf, rmf, pha and background """

    # Define the base directory for the calibration files.
    caldb_dir = './ccf'

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
            This function should be modified along with the file name.

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
                #       so if the names are changed, these code also should be modified accordingly.
                if value.find('.pi') != -1:
                    substitute = value[:value.index('ao')] + 'ao19.pi'
                    res['pha'] = substitute
                    res['bkg'] = substitute
                    # print(f"{value} not found! This 'pi' file would be replaced by "
                    #       f"{substitute}!")
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

    # this method is useful if using bkg file like '.pi' to get the src2bkg rate; else we would use
    # the observed event files, then (40/80)^2 would be adopted, see function 'get_bkg_count_rate_observation'.
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


def get_bkg_count_rate_observation(bkg_file=None, instrument=None, src2bkg=0.25) -> tuple:
    """ get the background count rate from the original observed events

    Notes:
        The radius of the background region is 80", while the source radius is around 40".
    Args:
        bkg_file: csv file recording the background count rate of the 12 SBO in Alp & Larsson (2020)
        instrument (str): pn, mos1 or mos2
        src2bkg: the ratio between the areas of source and background

    Returns:
        count_rate_bkg

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


def generate_lc(satellite: str, instrument: str, filter_name: str, ao: int,
                nh_gal=None, nh_host=None, alpha=None, temperature=None,
                coordinates=None, rs_in=0.0, rs_out=0.0, lc_file=None,
                bkg_rate=0, poisson=True, src2bkg=None) -> dict:
    """generate fake light curve at rs_out

    Args:
        satellite: Name of the satellite, e.g., xmm_newton
        instrument: Name of the instrument, e.g., pn, mos1, mos2
        filter_name: Name of the filter, e.g., thick, thin, med
        ao: Name of the Announcement of Opportunity, e.g, 19
        nh_gal: The Galactic column density, in unit of 1e22 cm^-2
        nh_host: The host galaxy column density, in unit of 1e22 cm^-2
        alpha: powerlaw index (X-ray spectrum model, optional)
        temperature: temperature of black body model (X-ray spectrum model, default)
        coordinates: Galactic coordinates, (l, b), in units of degrees
        rs_in: input redshift
        rs_out: output redshift
        lc_file: original light curve file
        bkg_rate: background count rate
        poisson: randomize counts using poisson distribution or not
        src2bkg: the ratio between source area and background area

    Returns:
        lc: A fake lightcurve, in format of dict: {'time':ndarray, 'timedel':ndarray,
        'bkg_counts': ndarray, 'counts': ndarray, 'rate':ndarray}

    """
    resp = GetResp(satellite=satellite, instrument=instrument, filter_name=filter_name, ao=ao)
    pha, arf, rmf, bkg = resp.response

    if not bkg_rate:
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

    # default: blackbody
    # mo_spec = mo.PLSpec(nh_gal=nh_gal, nh_host=nh_host, rs=rs_in, alpha=alpha)
    mo_spec = mo.BBSpec(nh_gal=nh_gal, nh_host=nh_host, rs=rs_in, temperature=temperature)

    fake_lc = simtools.fakelc(snb_lc, mo_spec.model, rs_in=rs_in, rs_out=rs_out, input_pha=pha,
                            input_arf=arf, input_rmf=rmf, input_bkg=bkg, pha=pha, poisson=poisson,
                            rmf=rmf, arf=arf, bkg=bkg, bkg_rate=bkg_rate, src2bkg=src2bkg)

    return fake_lc


def lc_rebin(lc: dict, bin_method=None, bin_size=60, bin_size_raw=4) -> dict:
    """The generated light curve are binned such that each bin contains m seconds or n counts.
        m = [60, 800, 12, 160, 8, 400, 48, 400, 160, 160, 24, 8] * time dilation factor, see
        Figure 1 in Alp & Larsson (2020); time dilation factor see ./xraysim/utils/simtools.
        transform_redshift; default of n is 25.

    Args:
        lc: see the args in 'generate_lc'
        bin_method (str): fixed or dynamic bin_size; if dynamic, to keep each bin contains
            the same (default 25) counts; if fixed, the bin size are the same as in the
            article (Alp & Larsson 2020), note that this bin size would change along with
            time dilation.
        bin_size (int): fixed bin size
        bin_size_raw (int): raw bin size; default is 4

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

        scale = bin_size // bin_size_raw
        len_lc = len(time_delta_series)
        n_segments = math.floor(len_lc/scale)

        # first we build the first time bin; this may be not equal to bin_size, then we get the rest indexes.
        scale_first = len_lc - n_segments * scale
        if not scale_first:
            index_list = np.arange(n_segments) * scale
        else:
            index_list_first = np.array([0])
            index_list_rest = scale_first + np.arange(n_segments) * scale
            index_list = np.concatenate((index_list_first, index_list_rest), axis=None)

        for i in range(len(index_list)-1):
            time_delta_series_new.append(np.sum(time_delta_series[index_list[i]:index_list[i+1]]))
            counts_series_new.append(np.sum(counts_series[index_list[i]:index_list[i+1]]))
            counts_bkg_series_new.append(np.sum(counts_bkg_series[index_list[i]:index_list[i+1]]))

        time_delta_series_new.append(np.sum(time_delta_series[index_list[i]:]))
        counts_series_new.append(np.sum(counts_series[index_list[i]:]))
        counts_bkg_series_new.append(np.sum(counts_bkg_series[index_list[i]:]))
        time_series_new = np.cumsum([0] + time_delta_series_new)[:-1] + np.array(time_delta_series_new)/2

    lc_binned = {'time': np.array(time_series_new),
                 'timedel': np.array(time_delta_series_new),
                 'bkg_counts': np.array(counts_bkg_series_new),
                 'counts': np.array(counts_series_new),
                 'rate': (np.array(counts_series_new)-np.array(counts_bkg_series_new))
                         / np.array(time_delta_series_new)}

    return lc_binned


def combine_lc(lc_list: str) -> dict:
    """Combine the light curves in lc_list into one light curve

    Args:
        lc_list: lc generated using pn, mos1, and mos2

    Returns:
        the combined lc：the format is the same as input lc

    """
    if not lc_list:
        raise TypeError("The input lc list is empty!")
    if len(lc_list) == 1:
        return lc_list[0]

    # here copy is a combination of deep copy and shallow copy
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


def get_obs_infos(file=None):
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
        threshold: sometimes we want to select objects outside the Galactic plane, here we select
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


def cal_redshift(lc_file: str, rs_in: float, satellite=None, instrument=None, filter_name=None,
                 ao=None, nh_gal=None, nh_host=None, alpha=None, temperature=None,
                 LiMa=False, bin_size=None, poisson=False,
                 save_res=False, plot_res=False, index_obj=None) -> float:
    """Roughly find the maximum redshift where the source could be detected.

    Notes:
        Here we use three criteria:
        1. The ratio of the maximum background-subtracted flux bin over the mean flux
           is larger than 1.92.
        2. S/N >= 5.
        3. net counts >= 62

        The loop range [rs_in*0.8, rs_in*3.0] are obtained from the initial results.

    Args:
        lc_file: observed light curve of one SN SBO candidate
        rs_in (float): redshift of this object
        satellite (str): the name of satellite, e.g., 'xmm_newton'
        instrument (list): instrument list, e.g, ['pn', 'mos1', 'mos2']
        filter_name (str): the name of filter, e.g., 'thick-5'
        ao (int): e.g., 7, 8, 10, 13, 14, 18, 19
        nh_gal: The Galactic column density, in unit of 1e22 cm^-2
        nh_host: The host galaxy column density, in unit of 1e22 cm^-2
        alpha: powerlaw index (X-ray spectrum model, optional)
        temperature: temperature of black body model (X-ray spectrum model, default)
        LiMa: use LiMa formula to calculate the S/N or not
        bin_size: the bin size when rebinning the light curve
        poisson: randomize counts using poisson distribution or not
        save_res: save the light curve at the maximum redshift or not
        plot_res: plot the light curve at the maximum redshift or not
        index_obj: the index of this object in the object list; if not None, then this
            parameter is used to get the corresponding background count rate.

    Returns:
        redshift: maximum redshift

    """

    rs_min = rs_in*0.8
    rs_max = rs_in*3.0
    rs_list = np.linspace(rs_min, rs_max, num=100, endpoint=True)

    if not instrument:
        instrument = ('pn', 'mos1', 'mos2')
    l, b = get_random_lb(15)  # |b| > 15 degree

    tqdm_range = tqdm(rs_list)
    cts_bkg_list = []
    for ins in instrument:
        cts_bkg_list.append(get_bkg_count_rate_observation(instrument=ins))

    for i, rs in enumerate(tqdm_range):
        fake_lc_list = []
        for j, ins in enumerate(instrument):
            cts_bkg = cts_bkg_list[j]
            if index_obj is not None:
                count_rate_bkg = cts_bkg[index_obj]
            else:
                cts_bkg_mean, cts_bkg_sigma = np.mean(cts_bkg), np.std(cts_bkg)
                count_rate_bkg_raw = np.random.uniform(np.min(cts_bkg)-cts_bkg_sigma, np.max(cts_bkg)+cts_bkg_sigma)
                count_rate_bkg = max(count_rate_bkg_raw, 2e-3)

            fake_lc = generate_lc(satellite, ins, filter_name, ao, lc_file=lc_file,
                                  coordinates=(l, b), rs_in=rs_in, rs_out=rs, nh_gal=nh_gal,
                                  nh_host=nh_host, alpha=alpha, temperature=temperature,
                                  bkg_rate=count_rate_bkg, poisson=poisson)

            fake_lc_binned = lc_rebin(fake_lc, bin_method='fixed', bin_size=bin_size)
            fake_lc_list.append(fake_lc_binned)

        fake_lc_combined = combine_lc(fake_lc_list)
        counts = fake_lc_combined['counts']
        counts_bkg = fake_lc_combined['bkg_counts']
        counts_net = counts - counts_bkg
        time_delta_series = fake_lc_combined['timedel']
        count_rate_net = counts_net / time_delta_series

        cts_bkg_observed = np.sum(counts_bkg)
        cts_total = np.sum(counts_net) + cts_bkg_observed
        if LiMa:
            sn = (cts_total - cts_bkg_observed) / np.sqrt(cts_total + cts_bkg_observed)
        else:
            # here times 5, in order to be consistent with LiMa
            sn = cts_total*5/poisson.ppf(0.999999, cts_bkg_observed)

        flux_scale = np.max(count_rate_net) / np.mean(count_rate_net)
        criterion1 = (flux_scale < 1.92)
        criterion2 = (sn < 5)
        criterion3 = (np.sum(counts_net) < 62)

        # the threshold of terminating the loop
        if criterion1 or criterion2 or criterion3 or rs == rs_list[-1]:
            print(f"Z: {rs}, SNR: {sn}, bkg counts:{cts_bkg_observed}, total counts: {cts_total}, "
                  f"net counts: {cts_total - cts_bkg_observed}")
            print(f"flux_peak/flux_mean: {flux_scale}")
            print("-------------------------------------------")

            path = './fake_lc'
            if save_res:
                path_csv = path + '/' + lc_file[lc_file.index('0'):-4] + '_' + ''.join(instrument) + '_' + \
                           f'{filter_name}_' + 'ao_' + str(ao) + '.csv'
                pd.DataFrame(fake_lc_combined).to_csv(path_csv, index=False)
            if plot_res:
                path_img = path + '/' + lc_file[lc_file.index('0'):-4] + '_' + ''.join(instrument) + '_' + \
                           f'{filter_name}_' + 'ao_' + str(ao) + '.jpg'
                plot_lc(fake_lc_combined, save_fig=True, file_name=path_img)

            tqdm_range.close()
            return rs


"""This part is used to test this program (start)"""


def generate_light_curve(satellite=None, instrument=None, filter_name=None, ao=None):
    """This function is used to generate light curve at the redshift given by the paper, namely,
    here we want to compare these generated light curve with those in the paper (Alp & Larsson
    2020), then to check the program correctness.

    Args:
        satellite: see the cal_redshift
        instrument: see the cal_redshift
        filter_name: see the cal_redshift
        ao: see the cal_redshift

    Returns:
        None; but it will generate a series of csv file and plots. These plots can be used to
        compare the light curves in the paper (see their Figure 1) and those generated by our
        simulations, see the function "plot_lc_total".
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


def plot_lc_total(obs_id=None, filter_type='thin-5', ao=19, title=None, mos1=True, bin_size=0):
    """Plot the generated combined light curve (pn+mos1+mos2), used to compare with the raw light
    curve in the paper

    Args:
        obs_id: xmm_newton obs_id
        filter_type: thick, med, or thin
        ao: e.g., 19
        title: image title
        mos1: mos1 in the instrument list or not
        bin_size: bin sizes in the paper

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


"""This part is used to test this program (end)"""


def main(res_file):
    """The main function of this script

    Args:
        res_file: file used to record the result

    Returns:
        None

    """
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
        # Thread_current = Thread(target=cal_redshift, args=())
        rs = cal_redshift(lc_file, rs_in, satellite=satellite,
                          filter_name=filter_name, ao=ao, bin_size=bin_size,
                          temperature=temperature, LiMa=True, poisson=True)

        rs_out.append(rs)

    with open(res_file, 'a', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(rs_out)

    return None


if __name__ == '__main__':
    res_file = './results/redshift.csv'
    time_start = time.time()
    main(res_file)
    time_end = time.time()
    print(f"Time consumed {time_end - time_start} s.")

    """
    time_start = time.time()
    number_simulation = 2
    subprocesses = []
    for _ in range(number_simulation):
        p = Thread(target=main, args=(res_file,))
        subprocesses.append(p)
    for p in subprocesses:
        p.start()
    for p in subprocesses:
        p.join()
    time_end = time.time()
    print(f"Time consumed {time_end - time_start} s.")
    """

