import argparse
import numpy as np
import pandas as pd
import os
import glob
import random
import time

from scipy.stats import poisson
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u
from astropy.io import ascii as asc
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
                lc_file=None, bkg_rate=0.0) -> dict:
    """

    Args:
        satellite: Name of the satellite, e.g., xmm_newton
        instrument: Name of the instrument, e.g., pn, mos1, mos2
        filter_name: Name of the filter, e.g., thick, thin, med
        ao: Name of the Announcement of Opportunity, e.g, 19
        nh_gal: The Galactic column density, in unit of 1e22 cm^-2
        nh_host: The host galaxy column density, in unit of 1e22 cm^-2
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
                            rmf=rmf, arf=arf, bkg=bkg, bkg_rate=bkg_rate)

    return lc_pn


def lc_rebin(lc: dict, bin_method=None, bin_size=1, count_rate_bkg=0.) -> dict:
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

        for j in range(len(index_list) - 1):
            time_series_new.append(np.sum(time_delta_series[:index_list[j + 1]]))
            time_delta_series_new.append(np.sum(time_delta_series[index_list[j]:index_list[j + 1]]))
            counts_series_new.append(np.sum(counts_series[index_list[j]:index_list[j + 1]]))
            counts_bkg_series_new.append(np.sum(counts_bkg_series[index_list[j]:index_list[j + 1]]))

    if bin_method == 'fixed':
        i, time_in_bin, time_current, index_pre = 0, 0, 0, 0
        while i < len(time_series):
            if i < len(time_series):
                if time_in_bin < bin_size:
                    time_in_bin += time_delta_series[i]
                    time_current += time_delta_series[i]
                else:
                    time_series_new.append(time_current)
                    time_delta_series_new.append(time_in_bin)
                    counts_series_new.append(np.sum(counts_series[index_pre:i]))
                    counts_bkg_series_new.append(np.sum(np.sum(counts_bkg_series[index_pre:i])))
                    index_pre = i
                    time_in_bin = 0
            else:
                time_in_bin += time_delta_series[i]
                time_current += time_delta_series[i]
                time_series_new.append(time_current)
                time_delta_series_new.append(time_in_bin)
                counts_series_new.append(np.sum(counts_series[index_pre:]))
                counts_bkg_series_new.append(np.sum(np.sum(counts_bkg_series[index_pre:])))
            i += 1

    lc_binned = {'time': np.array(time_series_new),
                 'timedel': np.array(time_delta_series_new),
                 'bkg_counts': np.array(counts_bkg_series_new),
                 'counts': np.array(counts_series_new),
                 'rate': np.array(counts_series_new) / np.array(time_delta_series_new)}

    return lc_binned


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
    time_delta = lc['timedel']

    """
    bin_size = 5
    step = len(time_raw)//bin_size
    time_binned = []
    flux_binned = []
    for i in range(bin_size):
        if i < bin_size-1:
            time_binned.append((time_raw[step*i] + time_raw[step*(i+1)]) // 2)
            flux_binned.append(np.sum(time_delta[step*i:step*(i+1)] * flux_raw[step*i:step*(i+1)])
                               / np.sum(time_delta[step*i: step*(i+1)]))
        else:
            time_binned.append((time_raw[step*i] + time_raw[-1]) // 2)
            flux_binned.append(np.sum(time_delta[step*i:] * flux_raw[step*i:])
                               / np.sum(time_delta[step*i:]))
    """

    font0 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    plt.figure(figsize=(9, 6))
    plt.grid(True, ls='--')
    plt.plot(time_raw, flux_raw, '-o', alpha=0.7, mfc='black', linewidth=3)
    # plt.plot(time_binned, flux_binned, '-o', alpha=0.7, mfc='red', linewidth=3, label='Binned')
    plt.xlabel('Time (s)', font0)
    plt.ylabel('Count Rate (count s$^{-1}$ )', font0)
    # plt.legend(loc='best')
    if save_fig:
        plt.savefig(file_name, dpi=1200, bbox_inch='tight')


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


def get_bkg_count_rate(instrument: str, filter_name=None, threshold=15, radius=1):
    """

    Args:
        instrument: pn, mos1, or mos2.
        filter_name: thin, thick, or medium.
        threshold: see function 'get_random_lb', default is 15 degree.
        radius: the radius of source region, default is 1 arcmin.

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


def cal_redshift(lc_file: str, rs_in: float, satellite=None, instrument=None,
                 filter_name=None, ao=None, LiMa=True, bin_size=1,
                 save_res=False, plot_res=False) -> float:
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
        LiMa: use LiMa formula to calculate the S/N or not
        bin_size: the bin size when rebin the light curve
        save_res: save the light curve at the maximum redshift or not
        plot_res: plot the light curve at the maximum redshift or not

    Returns:
        redshift: the redshift when the signal to noise ratio only just greater than 5
        and the peak flux greater than two times of the mean.
    """
    rs_min = 0.01
    rs_max = 2.5

    rs_list = np.linspace(rs_min, rs_max, num=200, endpoint=True)

    l, b = get_random_lb(15)  # |b| > 15 degree
    cts_bkg_mean, cts_bkg_sigma = get_bkg_count_rate(instrument, filter_name)
    count_rate_bkg = np.random.normal(cts_bkg_mean, cts_bkg_sigma)

    for rs in tqdm(rs_list):
        fake_lc = generate_lc(satellite, instrument, filter_name, ao, lc_file=lc_file,
                              coordinates=(l, b), rs_in=rs_in, rs_out=rs,
                              bkg_rate=count_rate_bkg)
        fake_lc_binned = lc_rebin(fake_lc, bin_method='fixed', bin_size=bin_size,
                                  count_rate_bkg=count_rate_bkg)
        cts = fake_lc_binned['counts']
        cts_bkg = fake_lc_binned['bkg_counts']
        cts_net = cts - cts_bkg
        time_delta_series = fake_lc_binned['timedel']
        time_series = fake_lc_binned['time']
        count_rate_net = cts_net / time_delta_series

        # here we adopt background counts, but not bkg counts derived from the calibration files.
        cts_bkg_observed = count_rate_bkg * np.sum(time_delta_series)
        cts_total = np.sum(cts_net) + cts_bkg_observed
        if LiMa:
            sn = (cts_total - cts_bkg_observed) / np.sqrt(cts_total + cts_bkg_observed)
        else:
            # here times 5, in order to be consistent with LiMa
            sn = cts_total*5/poisson.ppf(0.999999, cts_bkg_observed)

        criterion1 = (np.max(cts_net/time_delta_series) - 2*np.mean(cts_net/time_delta_series) >= 0)
        criterion2 = (sn > 5)
        print(' ')
        print(f"Z: {rs}, SNR: {sn}，bkg counts:{cts_bkg_observed}, total counts: {cts_total}, "
              f"net counts: {cts_total-cts_bkg_observed}")
        print(f"number of bins: {len(count_rate_net)}")

        if not (criterion1 and criterion2):
            res = np.sum(cts_net)
            print(res)
            path = './fake_lc'
            path_csv = path + '/' + lc_file[lc_file.index('0'):-4] + f'_{instrument}_' + \
                       f'{filter_name}_' + 'ao_' + str(ao) + '.csv'

            path_img = path + '/' + lc_file[lc_file.index('0'):-4] + f'_{instrument}_' + \
                       f'{filter_name}_' + 'ao_' + str(ao) + '.jpg'
            if save_res:
                pd.DataFrame(fake_lc_binned).to_csv(path_csv)
            if plot_res:
                plot_lc(fake_lc_binned, save_fig=True, file_name=path_img)

            return rs, res

        """
        if not criterion1:
            print('=========================================================')
            print(count_net_net)
            return rs
        """

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
    parser.add_argument("--filter_name", type=str, default='thin-5', help="Filter in use. e.g., thin, "
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

    # lc_file = args.lc_file
    # rs_in = args.rs_in
    satellite = args.satellite
    instrument = args.instrument
    filter_name = args.filter_name
    ao = args.ao
    en_hi = args.en_hi
    en_low = args.en_low
    mo_en_hi = args.mo_en_hi
    mo_en_low = args.mo_en_low
    mo_lumi = args.mo_lumi

    paths = glob.glob('./lc/*.txt')
    paths.sort()
    rs_list = [1.17, 0.5, 0.3, 0.3, 0.37, 0.13, 0.095, 0.57, 0.48, 0.3, 0.62, 0.29]
    rs_out = []
    time_start = time.time()
    # z = get_bkg_count_rate(instrument, filter_name)

    """
    # check the data, to ensure the simulated counts are consistent with the original one
    for lc_file, rs_in in zip(paths, rs_list):
        # load the calibration matrix
        print('---------------------------')
        print(lc_file)
        lc_raw = read_lc(lc_file)
        counts_raw = np.sum(lc_raw.counts)
        # set negative flux to zero
        negative_index = np.where(lc_raw.counts < 0.0)[0]
        lc_raw.counts[negative_index] = 0.0
        tem = np.sum(lc_raw.counts)
        print(f"counts of raw lc: {np.sum(lc_raw.counts)}")

        pha, arf, rmf, bkg = GetResp(satellite=satellite, instrument=instrument,
                                     filter_name=filter_name, ao=ao).response

        lc = generate_lc(satellite, instrument, filter_name, ao, rs_in=rs_in,
                         rs_out=rs_in, lc_file=lc_file)
        lc_binned = lc_rebin(lc)
        rs = cal_redshift(lc_file, rs_in, satellite=satellite, instrument=instrument,
                          filter_name=filter_name, ao=ao)
        print(f"counts of generated lc: {rs}")

    """
    bin_sizes = [60, 800, 12, 160, 8, 400, 48, 400, 160, 160, 24, 8]
    for lc_file, rs_in, bin_size in zip(paths, rs_list, bin_sizes):

        # load the calibration matrix
        pha, arf, rmf, bkg = GetResp(satellite=satellite, instrument=instrument,
                                     filter_name=filter_name, ao=ao).response

        # lc = generate_lc(satellite, instrument, filter_name, ao, rs_in=rs_in,
        #                 rs_out=0.5, lc_file=lc_file)
        # lc_binned = lc_rebin(lc)

        rs = cal_redshift(lc_file, rs_in, satellite=satellite, instrument=instrument,
                          filter_name=filter_name, ao=ao, bin_size=bin_size,
                          save_res=True, plot_res=True)
        rs_out.append(rs)

    print('finished!')
    print(rs_out)
    time_end = time.time()
    time_delta = time_end - time_start
    print(f'Times Used: {time_delta} s')
