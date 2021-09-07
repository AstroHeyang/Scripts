import numpy as np
from matplotlib import pyplot as plt
from astropy.modeling.models import RedshiftScaleFactor, Scale
import astropy.units as u

from xraysim.models_astro import MDBlackBody, BBSpec, Phabs

x = np.linspace(0.01, 6, 10000)
bb = MDBlackBody(temperature=500*u.eV)
bb2 = MDBlackBody(temperature=1000*u.eV)

c = bb(1)
c2 = bb2(1)
nh_gal = 0.1
nh_host = 0.1
rs = 1


def plot_transform(nh_gal=0.1, nh_host=0.1, rs=1):
    phabs_gal = Phabs(nh_gal)
    phabs_host = Phabs(nh_host)
    redshift = RedshiftScaleFactor(rs)
    scale_factor = Scale(1.0 / (1.0 + rs))

    g0 = RedshiftScaleFactor(0) | bb
    g1 = phabs_host * bb
    g2 = redshift | bb
    g3 = phabs_gal * (redshift | bb)
    g4 = phabs_gal * (redshift | g2)

    plt.figure(figsize=(9, 6))
    plt.plot(x, g0(x), label='Rest Frame (nh_host=0)')
    plt.plot(x, g1(x), label='Rest Frame (nh_host=1e21)')
    plt.plot(x, g2(x), label='Observed Frame (nh_host=0, nh_gal=0)')
    plt.plot(x, g3(x), label='Observed Frame (nh_host=0, nh_gal=1e21)')
    plt.plot(x, g4(x), label='Observed Frame (nh_host=1e21, nh_gal=1e21)')

    plt.xlabel('Energy (keV)')
    plt.ylabel('Flux (ph cm$^{-2}$ keV$^{-2}$ s$^{-1}$)')
    plt.legend()
    plt.title(f'Blackbody (kT = 0.5 keV, z={rs})')
    plt.savefig(f'Blackbody_transformation_z={rs}.png', dpi=1000, bbox_inches='tight')


def plot_nh():
    plt.figure(figsize=(9, 6))
    plt.plot(x, bb(x), label='nh=0')
    for nh in [0.01, 0.05, 0.1, 0.5]:
        g = Phabs(nh=nh) * bb
        plt.plot(x, g(x), label=f'nh={nh}e22')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Flux (ph cm$^{-2}$ keV$^{-2}$ s$^{-1}$)')
    plt.title('Blackbody (kT = 0.5 keV)')
    plt.legend()
    plt.savefig('Blackbody_nh.png', dpi=1000, bbox_inches='tight')


def plot_z(nh=None):
    plt.figure(figsize=(9, 6))
    for rs in [0, 0.1, 0.3, 0.5, 1]:
        redshift = RedshiftScaleFactor(rs)
        scale_factor = Scale(1.0 / (1.0 + rs))
        g = Phabs(nh) * (redshift | bb )
        plt.plot(x, g(x), label=f'z={rs}')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Flux (ph cm$^{-2}$ keV$^{-1}$ s$^{-1}$)')
    plt.title(f'Blackbody (kT = 0.5 keV, nh={nh}e22)')
    plt.legend()
    plt.savefig(f'Blackbody_nh={nh}.png', dpi=1000, bbox_inches='tight')


plot_z(0.01)
plot_z(0.05)
plot_z(0.1)
plot_z(0.5)
plot_z(0)


