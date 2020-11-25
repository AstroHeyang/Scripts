#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy.modeling import Fittable1DModel, Parameter


__all__ = ('Phabs', 'Powerlaw', 'CompGRB', 'Blackbody', 'SFlaring')

# Define your models below
class Phabs(Fittable1DModel):
    """
    Photoelectric absorption model based on Morrison & McCammon 1983.

    Parameters
    ----------
    nh : float
        Column density in units of 1E22 atom/cm^2

    Notes
    -----
    Model formula (with :math:`nh` for ``nh``):
        .. math:: please check Morrison & McCammon 1983 for detials

    """
    _model_type = "multiplicative"
    nh = Parameter(default=0.01)

    @staticmethod
    def evaluate(x, nh):
        """
        Photoelectric absorption model based on Morrison & McCammon 1983.

        Variables
        ---------
        mat: matrix of the coefficients in the cross-section functions

        """
        mat = np.zeros((14, 3))
        mat[0, :] = [17.3, 608.1, -2150.0]
        mat[1, :] = [34.6, 267.9, -476.1]
        mat[2, :] = [78.1, 18.8, 4.3]
        mat[3, :] = [71.4, 66.8, -51.4]
        mat[4, :] = [95.5, 145.8, -61.1]
        mat[5, :] = [308.9, -380.6, 294.0]
        mat[6, :] = [120.6, 169.3, -47.7]
        mat[7, :] = [141.3, 146.8, -31.5]
        mat[8, :] = [202.7, 104.7, -17.0]
        mat[9, :] = [342.7, 18.7, 0.0]
        mat[10, :] = [352.2, 18.7, 0.0]
        mat[11, :] = [433.9, -2.4, 0.75]
        mat[12, :] = [629.0, 30.9, 0.0]
        mat[13, :] = [701.2, 25.2, 0.0]
        res = np.ones_like(x)
        x_limit = [0.001, 0.1, 0.284, 0.4, 0.532, 0.707, 0.867, 1.303, 1.84,
                   2.471, 3.21, 4.038, 7.111, 8.331, 10.0]
        for i in range(len(x_limit)-1):
            mask = np.logical_and(x > x_limit[i], x <= x_limit[i+1])
            if len(res[mask]) > 0:
                res[mask] = (mat[i, 0] + mat[i, 1]*x[mask] + mat[i, 2]*x[mask]**2
                             ) / x[mask]**3
                res[mask] = np.exp(-1.0 * res[mask] * nh * 1.0E-2)

        return res


class Powerlaw(Fittable1DModel):
    """
    A simple power-law model

    Parameters
    ----------
    Amplitude: float, normalization at 1keV
    Alpha    : float, photon index
    Redshift : float, redshift

    """
    _model_type = "additive"

    alpha = Parameter(default=2.0)
    redshift = Parameter(default=0.0)
    amplitude = Parameter(default=1.0)

    @staticmethod
    def evaluate(x, alpha, redshift, amplitude):
        return amplitude*np.power(x*(1 + redshift), alpha)


class CompGRB(Fittable1DModel):

    """
    Band function for GRB

    Parameters
    ----------
    Amplitude: float, normalization at 1keV
    Alpha    : float
    Epeak    : float, peak energy

    """
    _model_type = "additive"

    alpha = Parameter(default=1.0)
    Epeak = Parameter(default=100.0)
    amplitude = Parameter(default=1.0)

    @staticmethod
    def evaluate(x, alpha, Epeak, amplitude):
        return amplitude*np.power(x/100.0, alpha)*np.exp(-(alpha+2)*x/Epeak)


class Blackbody(Fittable1DModel):

    """
    A 1 dimensional black body model

    Parameters
    ----------
    Normalization: normalization of the bb model
    Temperature  : temperature of the blackbody
    Redshift     : redshift

    """
    _model_type = "additive"

    temperature = Parameter(default=0.1)
    redshift = Parameter(default=0.0)
    normalization = Parameter(default=1.0)

    @staticmethod
    def evaluate(x, temperature, redshift, normalization):
        rs = 1.0 + redshift
        res = normalization*8.0525*np.power(x*rs,2)/(rs*temperature**4*(np.exp(x/temperature)-1.0))
        return res

class SFlaring(Fittable1DModel):
    """
    A model describes the X-ray lightcurve for stellar flaring. The rapid rise
    is modeled with a half Gaussian profile, while the decay is described with
    an exponential function.

    See reference: Pitkin, M. et al 2014, MNRAS, 445, 2268

    Parameters
    ----------
    Tpeak    : the peak time of the flare
    Taugau   : standard deviation of the Gaussian rise
    Tauexp   : exponential decay time constant
    Amplitude: amplitude of the flare at peak time T0
    """
    _model_type = "additive"

    tpeak = Parameter(default=1.0)
    taugau = Parameter(default=0.1)
    tauexp = Parameter(default=1.0)
    amplitude = Parameter(default=1.0)

    @staticmethod
    def evaluate(x, tpeak, taugau, tauexp, amplitude):
        res = np.ones_like(x)

        mark = (x <= tpeak)
        if len(res[mark]) > 0:
            res[mark] = amplitude*np.exp(-0.5*((x[mark] - tpeak)/taugau)**2)

        mark = (x > tpeak)
        if len(res[mark]) > 0:
            res[mark] = amplitude*np.exp(-(x[mark] - tpeak)/tauexp)

        return res
