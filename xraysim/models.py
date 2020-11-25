# -*- coding: utf-8 -*-

import numpy as np
from astropy.modeling.core import Fittable1DModel
from astropy.modeling.parameters import Parameter
from astropy.modeling import models
from astropy import units as u
from astropy import constants as const

from .utils.simtools import calculate_model_flux


class MDBlackBody(Fittable1DModel):
    """
    Modified Blackbody model using the Planck function. The units is modified
    to compatible with the norm units in high energy physics, example: Photons
    per cm^2 per Hz.

    Arguments:
        temperature (astropy.units.Quantity or float) : Blackbody temperature
        scale (astropy.units.Quantity or float) : Scale factor

    Notes:
        Model formula:
        .. math:: B_{\\nu}(T) = A \\frac{2 h \\nu^{3} / c^{2}}{exp(h \\nu / k
        T) - 1} / h\\nu / erg2keV

    """

    # We parametrize this model with a temperature and a scale.
    temperature = Parameter(default=5000.0, min=20.0, unit=u.K)
    scale = Parameter(default=1.0, min=0)

    # We allow values without units to be passed when evaluating the model, and
    # in this case the input x values are assumed to be frequencies in Hz.
    _input_units_allow_dimensionless = True

    # We enable the spectral equivalency by default for the spectral axis
    input_units_equivalencies = {"x": u.spectral()}

    def evaluate(self, x, temperature, scale):
        """Evaluate the model

        Args:
            x (float, numpy.ndarray, or astropy.unit.Quantity): Frequency at
                which to compute the blackbody. If no units are given, this
                defaults to eV.
            temperature (float, numpy.ndarray, or astropy.units.Quantity):
                Temperature of the blackbody. If no units are given,
                this defaults to eV.

            scale (float, numpy.ndarray, or astropy.units.Quantity): Desired
                scale for the blackbody.

        Raises:
            ValueError: Invalid temperature.

            ZeroDivisionError: Wavelength is zero (when converting to
                frequency).

        """

        if not isinstance(temperature, u.Quantity):
            in_temp = u.Quantity(temperature, u.K)
        else:
            in_temp = temperature

        # Convert to units for calculations, also force double precision
        with u.add_enabled_equivalencies(u.spectral() + u.temperature_energy()):
            en = u.Quantity(x, u.keV, dtype=np.float64)
            temp = u.Quantity(in_temp, u.K)

        # Check if input values are physically possible
        if np.any(temp < 0):
            raise ValueError(f"Temperature should be positive: {temp}")

        log_boltz = en / (const.k_B.cgs * temp).to(u.keV)
        boltzm1 = np.expm1(log_boltz)

        # Calculate blackbody flux in unit of photon / s / cm^2 / Hz / sr,
        # bb(nu)/hnu
        cons = ((const.h.cgs * const.c).to(u.keV * u.cm))**2
        bb_nu = 2.0 * en ** 2 / cons / boltzm1 * u.ph
        tmp_unit = u.ph / (u.cm ** 2 * u.s * u.Hz)  # default unit

        y = scale * bb_nu.to(tmp_unit, u.spectral()) * np.pi
        y *= (u.Hz / u.keV * u.keV.to(u.Hz, equivalencies=u.spectral()))

        # If the temperature parameter has no unit, we should return a unitless
        # value. This occurs for instance during fitting, since we drop the
        # units temporarily.
        if hasattr(temperature, "unit"):
            return y.value
        else:
            return y.value

    @property
    def input_units(self):
        # The input units are those of the 'x' value, which should always be
        # keV. Because we do this, and because input_units_allow_dimensionless
        # is set to True, dimensionless values are assumed to be in keV.
        return {"x": u.keV}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {"temperature": u.K}

    @property
    def bolometric_flux(self):
        """Bolometric flux."""
        # bolometric flux in the native units of the planck function
        native_bolflux = (
            self.scale.value * const.sigma_sb * self.temperature ** 4
        )
        # return in more "astro" units
        return native_bolflux.to(u.erg / (u.cm ** 2 * u.s))


class BandGRB(Fittable1DModel):
    """Band function for GRBs

    Args:
        Amplitude (float): normalization at 1keV.
        Alpha (float): index in the band function.
        Beta (float): index in the band function.
        Epeak (float): peak energy in unit of keV.

    """

    _model_type = "additive"

    alpha = Parameter(default=-0.5, min=-2.0)
    beta = Parameter(default=-2.3)
    Epeak = Parameter(default=500.0, min=0.5)
    amplitude = Parameter(default=1.0)

    @staticmethod
    def evaluate(x, alpha, beta, Epeak, amplitude):
        """Evaluate the model

        Args:
            x (float or numpy.ndarray): energies at which the model will be
                calculated. The value should be in units of keV.
            alpha (float): the index in the band function.
            beta (float): the index in the band function.
            Epeak (float): the peak energy in unit of keV.
            amplitude (float): normalization at 1keV.

        """
        y = np.zeros_like(x)
        E0 = Epeak / (2. + alpha)
        e_break = (alpha - beta) * E0
        index1 = np.where(x < e_break)
        y[index1] = (x[index1] / 100.0)**alpha * np.exp(-x[index1] / E0)
        index2 = np.where(x >= e_break)
        y[index2] = (e_break/100.)**(alpha-beta) * (
            np.exp(beta-alpha) * (x[index2]/100.0)**beta)
        return amplitude * y


class Phabs(Fittable1DModel):

    """
    Photoelectric absorption model based on Morrison & McCammon 1983.

    Args:
        nh (float): column density of the absorption material, in units of
            1.0E22 cm^2.

    """

    # Parametrize the model with a nh
    nh = Parameter(default=0.03, min=0)

    # We allow values without units to be passed when evaluating the model, and
    # in this case the input x values are assumed to be energy in keV.
    _input_units_allow_dimensionless = True

    # We enable the spectral equivalency by default for the spectral axis
    input_units_equivalencies = {"x": u.spectral()}

    # The default unit is keV. Due to the conflict with the astropy Blackbody
    # model which has a default unit of Hz, this parameter is allowed to be
    # changed to 'Hz'.
    _input_units = {"x": u.keV}

    def evaluate(self, x, nh):
        """
        Evlulate the model.

        Args:
            x (float, numpy.ndarray, or astropy.units.Quantity): at which value
                will the absorption be calculated. If no units are given, this
                defaults to keV.
            nh (float): column density of the absorption material, in units of
                1.0E22 cm^2.

        """
        # check the unit of input x. If it is an astropy quantity, it will then
        # be convert to keV. If it is not, then it is assumed that the input
        # values could be in keV or Hz, which is determined by the value of the
        # input_units variable.
        if not isinstance(x, u.Quantity):
            en = u.Quantity(x, u.keV)
        else:
            with u.add_enabled_equivalencies(u.spectral()):
                en = u.Quantity(x, u.keV)

        # the matrix used to calculate the absorption factor
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

        y = np.ones_like(en.value)
        en_limit = [0.001, 0.1, 0.284, 0.4, 0.532, 0.707, 0.867, 1.303, 1.84,
                    2.471, 3.21, 4.038, 7.111, 8.331, 10.0]

        for i in range(len(en_limit)-1):
            mask = np.logical_and(en.value > en_limit[i], en.value <=
                                  en_limit[i+1])
            y[mask] = (mat[i, 0] + mat[i, 1]*en.value[mask]
                       + mat[i, 2]*en.value[mask]**2) / en.value[mask]**3
            y[mask] = np.exp(-1.0 * y[mask] * nh * 1.0E-2)

        return y

    @property
    def input_units(self):
        """Return the unit of 'x' value, which should be 'keV' or 'Hz'.
        """
        return {"x": u.keV}


class SpecModel(object):
    """
    This is a base model class for the spectral model of different type of
    transients. It consists of an absorption component, a redshift scale
    parameter, and a main model. The main model should be user defined. All the
    parameters in each single model should also be defined during initiation.

    Attributes:
        MODEL_EN_HI (float): global variable. The upper energy boundry over
            which the flux will be calculated for the model.
        MODEL_EN_LOW (float): global variable. The lower energy boundry over
            which the flux will be calculated for the model.
        EN_HI (float): global variable. The upper energy boundry over which
            the flux will be calculated.
        EN_LOW (float): global variable. The lower energy boundry over which
            the flux will be calculated.
        _norm_par (str): the name of the normalization parameter in the
            main spectral model.
        phabs (astropy.model): the absorption model which is named as 'phabs'.
        redshift (astropy.model): the redshift model which is named as 'rs'.
        main (astropy.model): the main spectral model which is named as 'main'.
        model (astropy.model): the compounds model consists of the phabs,
            redshit, and main model.
        _init_norm_val (float, astropy.units.Quantanty: initial value of the
            normalization parameter in the main model.
        parameter_valid: To check if the value of a parameter(s) is valid.
        set_flux: set the flux over a given energy range to a given value.
        cal_flux: calcaulte the flux (absobed) over a given energy range.
        get_model_name: return the name of the main spectral model.

    """
    MODEL_EN_HI = 10.0
    MODEL_EN_LOW = 0.3
    EN_HI = 10.0
    EN_LOW = 0.3

    def __init__(self, nh_gal, nh_host, rs, norm_par, main_mo):
        """
        Initial the class

        Args:
            nh (float): the column density in unit of 1.0E22cm^2.
            rs (float): the redshift.
            norm_par (str): the name of the normalization parameter in the
                the main spectral model.
            main_mo (astropy.modelling.model): the main spectral model

        """
        self._norm_par = norm_par
        self.phabs_gal = Phabs(nh_gal, name='phabs')
        self.phabs_host = Phabs(nh_host, name='phabs')
        self.redshift = models.RedshiftScaleFactor(rs, name='rs')
        self.main = main_mo
        if not nh_host:
            self.model = self.phabs_gal * (self.redshift | self.main)
        else:
            self.model = self.phabs_gal * (self.redshift | (self.main * self.phabs_host))

        self._init_norm_val = self.main.__getattribute__(self._norm_par).value

    def parameter_valid(self, param=None, error=True):
        """
        Check if the value of the parameter is valid.

        Args:
            param (None or tuple): if None, then all the parameters in the
                model will be checked. If a tuple (param_name, param_val) is
                provided, the param_name should be the parameter wants to be
                validated.
            error (boolean): if True, the function will raise a ValueError and
                stopped when the validation failed. If False, the function will
                return True if the validation succeed, and False if failed.

        Returns:
            is_valid (boolean): True if the parameter valus is valid,
                otherwise False.

        """
        is_valid = True

        if param is None:
            param_names = self.main.param_names
            param_vals = self.main.parameters
        elif isinstance(param, tuple):
            param_names = [param[0]]
            param_vals = [param[1]]
        else:
            raise TypeError(
                'A tuple (param_name, param_val) is quired for parameter')

        for name, val in zip(param_names, param_vals):
            bounds = self.main.__getattribute__(name).bounds
            bounds_lo = -np.inf if bounds[0] is None else bounds[0]
            bounds_hi = np.inf if bounds[1] is None else bounds[1]

        if val < bounds_lo or val > bounds_hi:
            if error:
                raise ValueError('Parameter ' + name + ' is out of range!')
            else:
                is_valid = False

        if not error:
            return is_valid

    def set_flux(self, flux, en_low=None, en_hi=None, rest=True,
                 k_const=None):
        """
        Renorm the model so that the flux over the [en_low, en_hi] energy
        range equals the given flux. Note that if k_const not equals to zero,
        which suggests that the kcorrection is done by multiple a constant. The
        en_low and en_hi should be the energy band after the kcorrection. The
        same applies to the MODEL_EN_LOW and MODEL_EN_HI parameters.

        Args:
            flux (float): a flux value provided by the user
            en_low (None or float): lower energy limit in units of keV. If
                None the class attribute will be used
            en_hi (None or float), upper energy limit in units of keV. If None
                the class attribute will be used
            rest (boolean): if True, the flux is in the rest frame. Otherwise
                it will be in the observed frame. Default is True.
            k_const (None/float): if not None, it than assums that the flux
                over the give energy range [MODEL_EN_LOW, MODEL_EN_HI] should
                be divided by k_const. This is useful when the given flux is
                the bolometric flux, while a constant correction factor is
                needed to calculate the flux over the [MODEL_EN_LOW,
                MODEL_EN_HI] energy range.

        """

        en_hi = self.MODEL_EN_HI if en_hi is None else en_hi
        en_low = self.MODEL_EN_LOW if en_low is None else en_low

        self.check_energy_range(en_low, en_hi)

        if k_const is not None:
            flux = flux / k_const

        # First set the value of the normal parameter to the initial value to
        # avoid errors caused by occationally setting this value to inf or nan.
        self.main.__getattribute__(self._norm_par).value = self._init_norm_val
        f = self.cal_flux(en_low=en_low, en_hi=en_hi, rest=rest)
        param_val = self._init_norm_val * flux / f['flux']
        self.main.__getattribute__(self._norm_par).value = param_val

    def cal_flux(self, en_low=None, en_hi=None, rest=False, unabs=True):
        """Calculate the flux over the [en_low, en_hi] energy range.

        Args:
            en_low (None or float): lower energy limit in units of keV. If
                None, the class attribute will be used.
            en_hi (None or float): upper energy limit in units of keV. If None,
                the class attribute will be used.
            rest (boolean): if True, the flux is in the rest frame. Otherwise
                it will be in the observed frame. Default is True.
            unabs (boolean): if True, the unabsorbed flux is calculated.
                Otherwise the absorbed flux will be calculated.

        Returns:
            flux (float): the flux over the given energy range.

        """
        en_hi = self.EN_HI if en_hi is None else en_hi
        en_low = self.EN_LOW if en_low is None else en_low

        self.check_energy_range(en_low, en_hi)

        if rest:
            flux = calculate_model_flux(self.main, en_low, en_hi)
        elif unabs:
            flux = calculate_model_flux((self.redshift | self.main), en_low, en_hi)
        else:
            flux = calculate_model_flux(self.model, en_low, en_hi)
        return flux

    def get_model_name(self):
        """This is a function to return the name of the main model.

        """
        return self.main.__class__.name

    @staticmethod
    def check_energy_range(en_low, en_hi):
        msg = "The upper limit of the enery is smaller than lower limit!"
        assert en_low < en_hi, msg


class BBSpec(SpecModel):
    """A class for absobred blackbody spectral model.

    The model consists of three components: the absorption model, the redshift
    scale factor, and the modified blackbody. The parameters in each of the
    component can be set independently (recommended). For instance:
        # define the model
        mo = BBSpec()
        # set the values for the parameters in the model
        mo.phabs.nh_gal = 0.5
        mo.phabs.nh_host = 0.3
        mo.redshift.z = 2.0
        mo.main.temperature = 50.0 * u.eV
        # or set the model parameter directly (not recommended due to the name
        # rules in Astropy model class
        mo.model.nh_0 = 0.5
        mo.model.z_1 = 2.0
        mo.model.temperature_2 = 50.0 * u.eV

    """

    def __init__(self, nh_gal=0.3, nh_host=None, rs=0.0, temperature=70.):
        """Initial the function

        Args:
            nh_gal (float): Galactic column density, in unit of 1E22 cm^-2. Default to 0.3.
            nh_host (float): column density of host galaxies, in unit of 1E22 cm^-2. Default to None.
            rs (float): redshift. Default to 0.0.
            temperature (float or astropy.units.Quantity): if it is not an
                astropy Quantity, than it is assumed that the temperature is
                in units of eV. Default to 70 eV.

        """
        if not isinstance(temperature, u.Quantity):
            temperature = temperature * u.eV
        norm_parameter = 'scale'
        main_model = MDBlackBody(temperature=temperature, scale=1.E-23,
                                 name='main')
        SpecModel.__init__(self, nh_gal, nh_host, rs, norm_parameter, main_model)


class PLSpec(SpecModel):
    """A class for absobred power-law spectral model.

    The model consists of three components: the absorption model, the redshift
    scale factor, and the astropy powerlaw model. The parameters in each of the
    component can be set independently (recommended). For instance:
        # define the model
        mo = PLSpec()
        # set the values for the parameters in the model
        mo.phabs.nh_gal = 0.5
        mo.phabs.nh_host = 0.5
        mo.redshift.z = 2.0
        mo.main.alpha = 1.9
        # or set the model parameter directly (not recommended due to the name
        # rules in Astropy model class
        mo.model.nh_0 = 0.5
        mo.model.z_1 = 2.0
        mo.model.alpha_2= 1.9

    """

    def __init__(self, nh_gal=None, nh_host=None, rs=0.0, alpha=1.8):
        """Initial the function

        Args:
            nh (float): column density, in unit of 1E22 cm^-2. Default to 0.3.
            rs (float): redshift. Default to 0.0.
            alpha (float): the index of the powerlaw model. Default to 1.8.
            peak_lumi (float or None): this parameter may be provided if the
                other parameters depend on the peak luminosity,
                e.g. peak energy in GRB. Default to None.

        """
        norm_parameter = 'amplitude'
        main_model = models.PowerLaw1D(x_0=1.0, alpha=alpha,
                                       amplitude=1.0, name='main')
        SpecModel.__init__(self, nh_gal, nh_host, rs, norm_parameter, main_model)


class GRBSpec(SpecModel):
    """A class for the absobred GRB spectral model. A absorbed band function is
    used to describe the GRB spectrum.

    The model consists of three components: the absorption model, the redshift
    scale factor, and the band function. The parameters in each of the
    component can be set independently (recommended). For instance:
        # define the model
        mo = GRBSpec()
        # set the values for the parameters in the model
        mo.phabs.nh_gal = 0.5
        mo.phabs.nh_host = 0.5
        mo.redshift.z = 2.0
        mo.main.Epeak = 320.0
        # or set the model parameter directly (not recommended due to the name
        # rules in Astropy model class
        mo.model.nh_0 = 0.5
        mo.model.z_1 = 2.0
        mo.model.Epeak_2 = 320.0

    """

    def __init__(self, nh_gal=0.3, nh_host=None, rs=0.0, alpha=-0.5, beta=-2.3, Epeak=100.0,
                 peak_lumi=None, norm=None):
        """Initial the model parameters.

        Args:
            nh_gal (float): column density, in unit of 1E22 cm^-2. Default to 0.3.
            nh_host (float): column density, in unit of 1E22 cm^-2. Default to 0.3.
            rs (float): redshift. Deafult to 0.0
            alpha (float): value of the alpha parameter in the band function.
                Default to -0.5.
            beta (float): value of the beta parameter in the band function.
                Default to -2.3.
            Epeak (float): value of the peak energy in the band function,
                in units of keV. Default to 100.0 keV.
            peak_lumi (float or None): if not none, than the Epeak is
                calculated based on the peak lumi - peak energy relation.
                Default to None.
            norm (float or None), the normlization parameter in the Epeak-Lpeak
                realtion. If the peak_lumi is not None, than a norm factor
                should also be provided for calculating the peak energy.
                Default to None.

        """
        norm_parameter = 'amplitude'
        self.peak_lumi = peak_lumi
        self.norm = norm
        if self.peak_lumi is not None and self.norm is not None:
            Epeak = self.set_epeak(self.peak_lumi, self.norm)

        main_model = BandGRB(alpha=alpha, beta=beta, Epeak=Epeak, amplitude=1.,
                             name='main')
        SpecModel.__init__(self, nh_gal, nh_host, rs, norm_parameter, main_model)

    def set_epeak(self, peak_lumi, norm):
        """
        Calculate the peak energy using the peak_lumi - peak_energy relation

        Args:
            peak_lumi (float): the peak luminosity
            norm (float): the normalization factor in the relation

        Returns:
            Epeak (float): the peak energy in unit of keV

        """
        Epeak = (peak_lumi / 1.E47 / norm)**(1./1.72)
        return Epeak
