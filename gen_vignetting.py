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
from xraysim.models_astro import BBSpec
from xraysim.astroio.read_data import read_arf, read_rmf, read_pha, read_lc
from xraysim.utils.simtools import fakespec
from xraysim.utils import simtools
from gdpyc import GasMap
from tqdm import tqdm
from matplotlib import pyplot as plt

caldb_dir = './ccf/arf_rmf_random/'
obs_ids = ['0149780101',
           '0203560201',
           '0300240501',
           '0300930301',
           '0502020101',
           '0604740101',
           '0675010401',
           '0743650701',
           '0760380201',
           '0765041301',
           '0770380401',
           '0781890401']

fig_dir = './vignetting/'


class GetVignetting(object):

    def __init__(self, obs_id='0149780101', xmm_newton_camera='PN'):
        self.obs_id = obs_id
        self.xmm_newton_camera = xmm_newton_camera

    def get_arf_path(self, file_id=1):
        path = caldb_dir + self.obs_id + '/' + self.xmm_newton_camera.upper() + \
               '_' + str(file_id) + '.arf'
        return path

    def get_arf_infos(self):
        info_path = caldb_dir + 'info/info_' + self.obs_id + '.txt'
        info = pd.read_csv(info_path, delim_whitespace=True)
        info_this = info[info['EPIC'] == self.xmm_newton_camera]
        return info_this

    @property
    def energy(self):
        file_path = self.get_arf_path()
        with fits.open(file_path) as hdul:
            energy_low = hdul[1].data['energ_lo']
            energy_high = hdul[1].data['energ_hi']
            energy = (energy_high + energy_low)/2
        return energy

    def get_effective_area(self, file_id=0):
        file_path = self.get_arf_path(file_id)
        with fits.open(file_path) as hdul:
            effective_area = hdul[1].data['specresp']
        return effective_area

    @property
    def effective_area_on_axis(self):
        effective_area_ox_axis = self.get_effective_area()
        return effective_area_ox_axis

    @property
    def mean_effective_area(self):
        infos = self.get_arf_infos()
        area = np.zeros(len(self.effective_area_on_axis))
        weights_total = np.sum(np.square(infos['distance_to0_deg']))
        for num, deg in zip(infos['number'], infos['distance_to0_deg']):
            area_weighted = self.get_effective_area(file_id=num) * np.square(deg)
            area += area_weighted
        return area / weights_total


def plot_area(obs_id='0203560201', xmm_newton_camera='PN', plot=True):
    vignet = GetVignetting(obs_id=obs_id, xmm_newton_camera=xmm_newton_camera)
    energies = vignet.energy
    effective_area_on_axis = vignet.effective_area_on_axis
    effective_area_off_axis = vignet.mean_effective_area

    if plot:
        plt.figure(figsize=(9, 6))
        plt.plot(energies, effective_area_on_axis, label='on axis')
        plt.plot(energies, effective_area_off_axis, label='off axis (mean)')
        plt.xlabel('energy (keV)')
        plt.ylabel('effective area (cm$^2$)')
        plt.title('XMM Newton: ' + xmm_newton_camera)
        plt.xscale('log')
        plt.grid(ls='--')
        plt.legend()
        plt.savefig(fig_dir + 'effective_area_' + xmm_newton_camera + '.png', dpi=1000, bbox_inches='tight')
        plt.close()

    if xmm_newton_camera in ('MOS1', 'MOS2'):
        index = np.where(energies == 0.1025)
    else:
        index = np.where(energies == 0.1005)
    vignetting_factor = effective_area_off_axis[index]/effective_area_on_axis[index]
    print(vignetting_factor)

    return vignetting_factor


def plot_area_off_axis(obs_id='0149780101', xmm_newton_camera='PN'):

    vignet = GetVignetting(obs_id=obs_id, xmm_newton_camera=xmm_newton_camera)
    infos = vignet.get_arf_infos()
    energies = vignet.energy
    np.savetxt('energy.csv', energies)
    colors = ['black', 'dimgrey', 'rosybrown', 'lightcoral', 'firebrick',
              'red', 'darkorange', 'tan', 'gold', 'olive',
              'yellow', 'darkseagreen', 'green', 'teal', 'deepskyblue',
              'steelblue', 'navy', 'blue', 'indigo', 'deeppink',
              'cyan']

    colors = colors[:len(infos['number'])]

    # calculate the vignetting factor using effective area @1.0keV, @1.5 keV, @4.5keV, @6.4keV, @8.0keV
    if xmm_newton_camera == 'PN':
        index1 = 504  # 1.0 keV
        index2 = 671  # 1.5 keV
        index3 = 1301  # 4.5 keV
        index4 = 1426  # 6.4 keV
        index5 = 1533  # 8.0 keV
    else:
        index1 = 200  # 1.0 keV
        index2 = 300  # 1.5 keV
        index3 = 900  # 4.5 keV
        index4 = 1280  # 6.4 keV
        index5 = 1600  # 8.0 keV

    effective_area_on_axis = vignet.effective_area_on_axis
    vf1, vf2, vf3, vf4, vf5 = [], [], [], [], []
    vignetting_factors = {}

    plt.figure(figsize=(9, 6))
    for num, deg, color in zip(infos['number'], infos['distance_to0_deg'], colors):
        effective_area = vignet.get_effective_area(file_id=num)
        plt.plot(energies, effective_area, color=color, label=f'angle={deg*60}')
        vf1.append(effective_area[index1]/effective_area_on_axis[index1])
        vf2.append(effective_area[index2]/effective_area_on_axis[index2])
        vf3.append(effective_area[index3]/effective_area_on_axis[index3])
        vf4.append(effective_area[index4]/effective_area_on_axis[index4])
        vf5.append(effective_area[index5]/effective_area_on_axis[index5])

    vignetting_factors['1.0 keV'] = vf1
    vignetting_factors['1.5 keV'] = vf2
    vignetting_factors['4.5 keV'] = vf3
    vignetting_factors['6.4 keV'] = vf4
    vignetting_factors['8.0 keV'] = vf5

    plt.xlabel('energy (keV)')
    plt.ylabel('effective area (cm$^2$)')
    plt.title('XMM Newton: ' + xmm_newton_camera)
    plt.xscale('log')
    plt.grid(ls='--')
    plt.legend()
    plt.savefig(fig_dir + 'effective_area_' + xmm_newton_camera + '_off_axis.png', dpi=1000, bbox_inches='tight')
    plt.close()

    # plot the vignetting factors
    plt.figure(figsize=(9, 6))
    angles = infos['distance_to0_deg'] * 60
    df_angle = pd.DataFrame({'angles': angles,
                             'vf_1.0keV': vf1,
                             'vf_1.5keV': vf2,
                             'vf_4.5keV': vf3,
                             'vf_6.4keV': vf4,
                             'vf_8.0keV': vf5
                             })
    df_angle = df_angle.sort_values(by='angles', ignore_index=True)
    plt.plot(df_angle['angles'], df_angle['vf_1.0keV'], '-o', label='@1.0keV')
    plt.plot(df_angle['angles'], df_angle['vf_1.5keV'], '-o', label='@1.5keV')
    plt.plot(df_angle['angles'], df_angle['vf_4.5keV'], '-o', label='@4.5keV')
    plt.plot(df_angle['angles'], df_angle['vf_6.4keV'], '-o', label='@6.4keV')
    plt.plot(df_angle['angles'], df_angle['vf_8.0keV'], '-o', label='@8.0keV')
    plt.xlabel('Off-axis angle (arcmin)')
    plt.ylabel('Vignetting factor')
    plt.title('XMM Newton vignetting factor: ' + xmm_newton_camera)
    plt.grid(ls='--')
    plt.legend()
    plt.savefig(fig_dir + obs_id + '_' + xmm_newton_camera + '_vf.png', dpi=1000, bbox_inches='tight')
    plt.close()


def compare_filters(camera='PN'):
    dir = './ccf/xmm_newton/'
    file1 = dir + camera.lower() + '-thick-5-ao19.arf'
    file2 = dir + camera.lower() + '-med-5-ao19.arf'
    file3 = dir + camera.lower() + '-thin-5-ao19.arf'
    with fits.open(file1) as hdul_thick:
        energy = hdul_thick[1].data['energ_lo']
        specresp_thick = hdul_thick[1].data['specresp']

    with fits.open(file2) as hdul_med:
        specresp_med = hdul_med[1].data['specresp']

    with fits.open(file3) as hdul_thin:
        specresp_thin = hdul_thin[1].data['specresp']

    plt.figure(figsize=(9, 6))
    plt.plot(energy, specresp_thick, label='thick')
    plt.plot(energy, specresp_med, label='med')
    plt.plot(energy, specresp_thin, label='thin')

    plt.xlabel('Energy (keV)')
    plt.ylabel(r'Effective area (cm$^2$)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title('XMM Newton: ' + camera.upper())
    plt.grid(ls='--')
    plt.savefig('effective_area_' + camera.lower() + '_filters.png', dpi=1000, bbox_inches='tight')
    plt.close()


def get_filter_exposure():
    file = '~/Alp2020/filters_exposure.csv'
    df = pd.read_csv(file)
    df_thin = df[df['pn_filter'] == 'Thin1']
    df_med = df[df['pn_filter'] == 'Medium']
    df_thick = df[df['pn_filter'] == 'Thick']
    df_undef = df[df['pn_filter'] == 'UNDEF']

    time_thin = df_thin['pn_time'].sum()
    time_med = df_med['pn_time'].sum()
    time_thick = df_thick['pn_time'].sum()
    time_undef = df_undef['pn_time'].sum()
    print(time_thin/time_thick, time_med/time_thick)

    bins = [time_thin, time_med, time_thick, time_undef]

    return bins


if __name__ == '__main__':
    get_filter_exposure()



