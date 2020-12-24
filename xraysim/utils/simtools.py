import numpy as np
import scipy.integrate as integrate

from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy.modeling import models


def flux_to_rate(flux, modelfunc, pha=None, arf=None, rmf=None,
                 input_en_lo=0.3, input_en_hi=10.0,
                 output_en_lo=None, output_en_hi=None,
                 photon_flux=False):
    """
    Convert a given flux to counts rate. Instrument response file must be
    provided.

    Input parameters:
    ===========
    flux: scalar, list or numpy ndarray
    modelfunc: function class
    pha: can be optional. If None, a valid arf file must be provided
    arf: response file. It can be optional, if a valid pha file which has the
        AURSFILE keyword is provided
    rmf: response file. It can be optional. If None and no valid keyword in
        the pha file is found, rmf will be 1.0.
    input_en_lo: lower energy boundry that the flux is calculated
    input_en_hi: higher energy boundry that the flux is calculated
    output_en_lo: lower energy boundry that the rate is calculated. If None,
        then assuming is the same as the input_en_lo
    output_en_hi: higher energy boundry that the rate is calculated. If None,
        then assuming is the same as the input_en_hi
    photon_flux: if True, then the input flux is in units of photon/s/cm^2/keV,
        otherwise, the unit will be erg/s/cm^2. Default is False

    """
    if not isinstance(flux, np.ndarray):
        flux = np.asarray(flux)

    if modelfunc is None:
        raise Exception('No model is defeined. A model must be provided!')

    if output_en_lo is None: output_en_lo = input_en_lo
    if output_en_hi is None: output_en_hi = input_en_hi

    mo_flux = calculate_model_flux(modelfunc, low=input_en_lo, hi=input_en_hi)
    if photon_flux:
        norm_scale = flux / mo_flux['photon']
    else:
        norm_scale = flux / mo_flux['flux']

    if not np.iterable(norm_scale):
        norm_scale = [norm_scale]

    res = []
    # TODO: this way all the normalization has to be called as amplitude!!!
    #       maybe could check each model
    for scale in norm_scale:
        rate = calculate_model_rate(modelfunc=modelfunc,
                                    pha=pha, arf=arf, rmf=rmf,
                                    model_en_lo=output_en_lo,
                                    model_en_hi=output_en_hi)[1]
        rate *= scale
        res.append(rate)
    return res


def rate_to_flux(rate, modelfunc, pha=None, arf=None, rmf=None,
                 input_en_lo=0.3, input_en_hi=10.0,
                 output_en_lo=None, output_en_hi=None):
    """
    For a given instrument and counts rate, return the flux
        calculated using the user defined model.

    rate: counts rate
    modelfunc: function class
    pha: can be optional. If None, a valid arf file must be provided
    arf: response file. It can be optional, if a valid pha file which has the
        AURSFILE keyword is provided
    rmf: response file. It can be optional. If None and no valid keyword in
        the pha file is found, rmf will be 1.0.
    input_en_lo: lower energy boundry that the rate is estimated
    input_en_hi: higher energy boundry that the rate is estimated
    output_en_lo: lower energy boundry that the flux is calculated. If None,
        then assuming it is the same as the input_en_lo
    output_en_hi: higher energy boundry that the flux is calculated. If None,
        then assuming it is the same as the input_en_hi

    """
    # TODO: arf, rmf can also be file names, so that
    #       these functions can be used independently.
    # TODO: do we need a rate_to_rate function?
    init_flux = 1.0E-11
    if not isinstance(rate, np.ndarray):
        rate = np.asarray(rate)

    if output_en_lo is None: output_en_lo = input_en_lo
    if output_en_hi is None: output_en_hi = input_en_hi

    init_rate = flux_to_rate(init_flux, modelfunc=modelfunc,
                             arf=arf, rmf=rmf,
                             input_en_lo=output_en_lo,
                             input_en_hi=output_en_hi,
                             output_en_lo=input_en_lo,
                             output_en_hi=input_en_hi,
                             photon_flux=False)
    fluxes = rate / init_rate * init_flux
    return fluxes


def calculate_model_rate(modelfunc, pha=None, arf=None, rmf=None,
                         model_en_lo=0.3, model_en_hi=10.0):
    if pha is None and arf is None:
        raise Exception("No response file is defined!")
    elif arf is None:
        arf = pha.get_arf()
        if arf is None:
            raise Exception("No response file is found!")

    matrix = 1.0
    en_rmf = (arf.energ_lo + arf.energ_hi) / 2.0

    if pha is None and rmf is None:
        print('Warning: No rmf data loaded, an unity matrix is assumed!')
    elif rmf is None:
        rmf = pha.get_rmf()
        if rmf is None:
            print('Warning: No rmf data loaded, an unity matrix is assumed!')
        else:
            matrix = rmf.matrix
            en_rmf = (rmf.e_max + rmf.e_min) / 2.0
    else:
        matrix = rmf.matrix
        en_rmf = (rmf.e_max + rmf.e_min) / 2.0

    en_arf = (arf.energ_lo + arf.energ_hi) / 2.0
    en_del = arf.energ_hi - arf.energ_lo
    rate = np.dot(modelfunc(en_arf) * arf.specresp * en_del, matrix)
    mask = np.where(np.logical_and(en_rmf >= model_en_lo,
                                   en_rmf <= model_en_hi))
    rate = rate[mask]
    total_rate = np.sum(rate)
    return (rate, total_rate)


def calculate_model_flux(model, low, hi, num=10000, log=False,
                         scipy_inte=False):
    """[summary]

    Args:
        model (astropy.modeling.models): the model that will be integrated.
        low (float): the lower energy limit in unit of keV.
        hi (float): the upper energy limit in unit of keV.
        num (int, optional): the number of input x in the sampling.
            Defaults to 10000.
        log (bool, optional): If True, the input x will be sampled in
            logorithm space. Defaults to False.
        scipy_inte (bool, optional): If True the scipy's integrate function
            will be used. Defaults to False.

    Returns:
        float: the integrated flux over the given energy range [en_low, en_hi]
            for the given model.
    """
    # TODO: currently, the integrate function in scipy is not very efficient in
    # integrating the absobed model, likely due to the complexity of the Phabs
    # model. Try to optimise the Phabs model may help improve the efficiency.
    # So if the absorbed flux is calculated and nh i not zero, then numerical
    # solution will be used rather than using the scipy integrate method.
    keV2erg = u.keV.to(u.erg)

    if scipy_inte:
        res_flux = integrate.quad(lambda x: x * model(x), low, hi)[0]
        res_photon = integrate.quad(lambda x: model(x), low, hi)[0]
    else:
        if log:
            x = np.logspace(np.log10(low), np.log10(hi), num=num,
                            endpoint=True)
            step = x[1:] - x[:-1]
            x = (x[1:] + x[:-1]) / 2.0
        else:
            x, step = np.linspace(low, hi, num=num, endpoint=True,
                                  retstep=True)

        res_flux = np.sum(model(x + step / 2.0) * x * step)
        res_photon = np.sum(model(x + step / 2.0) * step)

    res = {'flux': res_flux * keV2erg, 'photon_flux': res_photon}

    return res


def fakespec(pha=None, arf=None, rmf=None, bkg=None, add_background=True,
             modelfunc=None, poisson=True, model_en_lo=0.3, model_en_hi=10.0,
             exposure=1000.0):
    """
    The fakespec function will generate simulated spectrum based on the
    user defined models, response files and background spectrum. If the
    boolean paramter poisson is False, then the statistical uncertainty
    will not applied. If the background spectrum is not provided, then only
    the source spectrum will be generated.

    """
    # load all the necssary files using the loaddata function
    if arf is None and pha is None:
        raise Exception("Valid response files are not defined!")
    elif arf is None:
        arf = pha.get_arf()
        if arf is None:
            raise Exception("PHA file does not has a valid response file!")
    if bkg is None and pha is not None:
        bkg = pha.get_background()

    if bkg is None and add_background:
        raise Exception("Background file is not defined!")

    # define a dictionary for the faked spectrum
    model_spec = {}

    if pha is not None:
        model_spec['src_backscale'] = pha.get_backscal()
    else:
        model_spec['src_backscale'] = 1.0

    if pha is not None and exposure <= 0.0:
        exposure = pha.exposure

    model_spec['exposure'] = exposure

    if rmf is not None:
        model_spec['en'] = (rmf.e_min + rmf.e_max) / 2.0
        model_spec['en_del'] = rmf.e_max - rmf.e_min
    else:
        model_spec['en'] = (arf.energ_lo + arf.energ_hi) / 2.0
        model_spec['en_del'] = arf.energ_hi - arf.energ_lo

    mask = np.where(np.logical_and(model_spec['en'] >= model_en_lo, model_spec['en'] <= model_en_hi))
    model_spec['en'] = model_spec['en'][mask]
    model_spec['en_del'] = model_spec['en_del'][mask]

    if add_background:
        # Note: to increase the S/N of background spectrum, here a very long exposure time of
        # the background is assumed! Poisson statistic is then added, and the background spectrum
        # is then downscaled to the desired exposure time
        model_spec['bkg_counts'] = bkg.get_dep() * 1.0E6 / bkg.exposure

        if poisson:
            model_spec['bkg_counts'] = np.random.poisson(model_spec['bkg_counts'])

        model_spec['bkg_counts'] = model_spec['bkg_counts'] * (
                exposure * 1.0E-6) / bkg.get_backscal() * model_spec['src_backscale']

        model_spec['bkg_counts'] = model_spec['bkg_counts'][mask]
        model_spec['bkg_backscale'] = bkg.get_backscal()
    else:
        model_spec['bkg_counts'] = 0.0
        model_spec['bkg_backscale'] = 1.0

    model_src_rate = calculate_model_rate(modelfunc, pha, arf, rmf,
                                          model_en_lo, model_en_hi)[0]

    model_spec['counts'] = model_src_rate * exposure + model_spec['bkg_counts']

    if poisson:
        model_keys = [key for key in list(model_spec.keys())
                      if 'counts' in key]
        for key in model_keys:
            model_spec[key] = np.random.poisson(model_spec[key])

    return model_spec


def transform_redshift(flux, spec_model, rs_in, rs_out,
                       en_low=0.3, en_high=10.0):
    """calculate the flux when one object was moved from z1 to z2

    Notes:
        Here we assume that the spectrum model and absorption are the same in rs_in and rs_out.

        1. first, we obtain the luminosity integrated from v1(1+rs_in) to v2(1+rs_in) in
        the rest frame, namely, L1 = 4 * pi * Ld1^2
        2. second, the luminosity L2, integrated from v1(1+rs_out) to v2(1+rs_out) are
        calculated, namely,
        L2 = L1 * [model(v1*(1+rs_out), v2*(1+rs_out)) / model(v1*(1+rs_in), v2*(1+rs_in))]
        3. finally, we get the flux in the output redshift, namely,
        flux = L2/(4 * pi * Ld2^2)

        Concretely, here we use two correction factors, one is (Ld1/Ld2)^2, the other is
        [model(v1*(1+rs_out), v2*(1+rs_out)) / model(v1*(1+rs_in), v2*(1+rs_in))].

    Args:
        flux (np.ndarray or float):  flux or flux list;
        spec_model (astropy.model): the spectrum model;
        rs_in (float): input redshift;
        rs_out (float): output redshift;
        en_low (float): energy lower boundary;
        en_high (float): energy upper boundary;

    Returns:
        flux in the output redshift (np.ndarray).

    Todo:
        Code needs modification if the input and output energy ranges are different.

    """
    if not isinstance(flux, np.ndarray):
        flux = np.asarray(flux)

    # first, corrected for redshift
    scale_distance = np.square(cosmo.luminosity_distance(rs_in) / cosmo.luminosity_distance(rs_out))
    flux_redshifted = flux * scale_distance.value

    # second, corrected for energy band
    scale_energy = calculate_model_flux(spec_model, en_low * (1 + rs_out), en_high * (1 + rs_out))['flux'] / \
        calculate_model_flux(spec_model, en_low * (1 + rs_in), en_high * (1 + rs_in))['flux']
    flux_redshifted *= scale_energy

    return flux_redshifted


def fakelc(lc_data, spec_model, rs_in=None, rs_out=None, input_pha=None, input_arf=None,
           input_rmf=None, input_bkg=None, pha=None, arf=None, rmf=None, bkg=None,
           input_en_lo=0.3, input_en_hi=10.0, output_en_lo=0.3, output_en_hi=10.0,
           add_background=True, poisson=True, exposure=0.0, bkg_rate=0.0):
    """ Generate faked light curves.

    Notes:
        input_pha, input_arf, input_rmf, input_bkg are necessary only when
        the input is in units of counts or count rate.

    Args:
        lc_data (class): original light curve data, see DataLC in astrodata.py
        spec_model (astropy.models, or models likewise): spectrum model, e.g. powerlaw or blackbody
        rs_in (float): redshift of input lc_data
        rs_out (float): redshift of output lc_data
        input_pha: calibration files of input lc_data
        input_arf: calibration files of input lc_data
        input_rmf: calibration files of input lc_data
        input_bkg: calibration files of input lc_data
        pha: calibration files of output lc_data
        arf: calibration files of output lc_data
        rmf: calibration files of output lc_data
        bkg: calibration files of output lc_data
        input_en_lo (float): energy lower boundary of input lc_data
        input_en_hi (float): energy upper boundary of input lc_data
        output_en_lo (float): energy lower boundary of output lc_data
        output_en_hi (float): energy upper boundary of output lc_data
        spec_model_origin (astropy.models, or models likewise): spectral model with no absorption
        add_background (bool): add background or not
        poisson (bool): using poisson or not
        exposure (np.ndarray or float): exposure time of this light curve

    Returns:
        faked light curve data (dict): {'time':ndarray, 'timedel':ndarray,
        'bkg_counts': ndarray, 'counts': ndarray, 'rate':ndarray}

    """
    if arf is None and pha is None:
        raise Exception("Valid response files are not defined!")
    elif arf is None:
        arf = pha.get_arf()
        if arf is None:
            raise Exception("PHA file does not has a valid response file!")
    if bkg is None and pha is not None:
        bkg = pha.get_background()

    if bkg is None and add_background:
        raise Exception("Background file is not defined!")

    res = {}
    if lc_data.flux is not None:
        flux_raw = lc_data.flux
        flux = transform_redshift(flux_raw, spec_model, rs_in, rs_out,
                                  en_low=input_en_lo, en_high=input_en_hi)

    # TODO: if lc_data.counts, these should be modified for redshift correction.
    elif lc_data.counts is not None:
        flux = rate_to_flux(lc_data.counts / lc_data.get_timedel(), modelfunc,
                            pha=input_pha, arf=input_arf, rmf=input_rmf,
                            input_en_lo=input_en_lo,
                            input_en_hi=input_en_hi)
    else:
        print("Error: No counts/rate/flux column is found in the lc file")

    ctr = flux_to_rate(flux, spec_model,
                       pha=pha, arf=arf, rmf=rmf,
                       input_en_lo=input_en_lo,
                       input_en_hi=input_en_hi,
                       output_en_lo=output_en_lo,
                       output_en_hi=output_en_hi,
                       photon_flux=False)

    res['time'] = lc_data.time
    if exposure > 0:
        res['timedel'] = exposure
    else:
        res['timedel'] = lc_data.get_timedel()

    # TODO: check if the following procedures are correct
    if add_background:
        # Note: to increase the S/N of background spectrum, here a very long
        # exposure time of the background is assumed! Poisson statistic is then
        # added, and the background spectrum is then downscaled to the desired
        # exposure time
        if not bkg_rate:
            bkg_rate = bkg.get_dep() * 1.0E6 / bkg.exposure

            if poisson:
                bkg_rate = np.random.poisson(bkg_rate)
                bkg_rate = bkg_rate * 1.0E-6  # if pha and bkg are not the same, then /bkg.get_backscal()

            if rmf is not None:
                en = (rmf.e_min + rmf.e_max) / 2.0
            else:
                en = (arf.energ_lo + arf.energ_hi) / 2.0

            mask = np.where(np.logical_and(en >= output_en_lo, en <= output_en_hi))
            bkg_rate = bkg_rate[mask]

            res['bkg_counts'] = np.ones_like(res['timedel'])
            res['bkg_counts'] = res['bkg_counts'] * res['timedel'] * np.sum(bkg_rate)

            if poisson:
                res['bkg_counts'] = np.random.poisson(res['bkg_counts'])
                res['counts'] = np.random.poisson(np.asarray(ctr) * res['timedel']) + res['bkg_counts']
            else:
                res['counts'] = ctr * res['timedel'] + res['bkg_counts']
        else:
            res['bkg_counts'] = np.random.poisson(res['timedel'] * bkg_rate)
            res['counts'] = np.random.poisson(np.asarray(ctr) * res['timedel']) + res['bkg_counts']

    res['rate'] = (res['counts'] - res['bkg_counts']) / res['timedel']

    return res
