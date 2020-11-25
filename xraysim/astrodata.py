#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .utils.essential import bool_cast, create_expr, filter_bins
from .utils.essential import astroFloat
from .data import BaseData, Data1DInt


def _notice_resp(chans, arf, rmf):
    bin_mask = None

    if rmf is not None and arf is not None:
        bin_mask = rmf.notice(chans)
        if len(rmf.energ_lo) == len(arf.energ_lo):
            arf.notice(bin_mask)
        elif len(rmf.energ_lo) < len(arf.energ_lo):
            arf_mask = None
            if bin_mask is not None:
                arf_mask = np.zeros(len(arf.energ_lo), dtype=bool)
                for ii, val in enumerate(bin_mask):
                    if val:
                        en_los = (rmf.energ_lo[ii],)
                        en_his = (rmf.energ_hi[ii],)
                        grid = (arf.energ_lo, arf.energ_hi)
                        idx = filter_bins(en_los, en_his, grid).nonzero()[0]
                        arf_mask[idx] = True
            arf.notice(arf_mask)
    else:
        if rmf is not None:
            bin_mask = rmf.notice(chans)
        if arf is not None:
            arf.notice(bin_mask)


class DataARF(Data1DInt):

    """ARF Data Set"""

    mask = property(BaseData._get_mask, BaseData._set_mask,
                    doc=BaseData.mask.__doc__)

    @property
    def specresp(self):
        return self._specresp

    @specresp.setter
    def specresp(self, val):
        self._specresp = val
        self._rsp = val

    def __init__(self, name, energ_lo, energ_hi, specresp, bin_lo=None,
                 bin_hi=None, exposure=None, header=None):
        self._lo = energ_lo
        self._hi = energ_hi
        BaseData.__init__(self)

    def __str__(self):
        # Print the metadata first
        old = self._fields
        ss = old
        try:
            self._fields = tuple(filter((lambda x: x != 'header'),
                                        self._fields))
            ss = BaseData.__str__(self)
        finally:
            self._fields = old
        return ss

    def __setstate__(self, state):
        if 'header' not in state:
            self.header = None
        self.__dict__.update(state)

        if '_specresp' not in state:
            self.__dict__['_specresp'] = state.get('specresp', None)
            self.__dict__['_rsp'] = state.get('specresp', None)

    #  def apply_arf(self, src, *args, **kwargs):
    #      model = arf_fold(src, self._rsp)

    #      # Rebin the sourcec model folded through ARF down t the size of the
    #      # PHA/RMF.
    #      if args != ():
    #          (arf, rmf) = args
    #          if rmf != () and len(arf[0]) > len(rmf[0]):
    #              model = rebin(model, arf[0], arf[1], rmf[0], rmf[1])

    #      return model

    def notice(self, bin_mask=None):
        self._rsp = self.specresp
        self._lo = self.energ_lo
        self._hi = self.energ_hi
        if bin_mask is not None:
            self._rsp = self.specresp[bin_mask]
            self._lo = self.energ_lo[bin_mask]
            self._hi = self.energ_hi[bin_mask]

    def get_indep(self, filter=False):
        filter = bool_cast(filter)
        return (self._lo, self._hi)

    def get_dep(self, filter=False):
        filter = bool_cast(filter)
        return self._rsp

    def get_xlabel(self):
        return 'Energy (keV)'

    def get_ylabel(self):
        return 'cm^2'


class DataRMF(Data1DInt):

    """RMF Data Set"""

    mask = property(BaseData._get_mask, BaseData._set_mask,
                    doc=BaseData.mask.__doc__)

    def __init__(self, name, detchans, energ_lo, energ_hi, n_grp, f_chan,
                 n_chan, matrix, offset=1, e_min=None, e_max=None,
                 header=None):
        self._fch = f_chan
        self._nch = n_chan
        self._grp = n_grp
        self._rsp = matrix
        self._lo = energ_lo
        self._hi = energ_hi
        BaseData.__init__(self)

    def __str__(self):
        # Print the metadata first
        old = self._fields
        ss = old
        try:
            self._fields = tuple(filter((lambda x: x != 'header'),
                                        self._fields))
            ss = BaseData.__str__(self)
        finally:
            self._fields = old
        return ss

    def __setstate__(self, state):
        if 'header' not in state:
            self.header = None
        self.__dict__.update(state)

    #  def apply_rmf(self, src, *args, **kwargs):
    #      # Rebin the source model from the PHA down to the size of the RMF
    #      if args != ():
    #          (rmf, pha) = args
    #          if pha != () and len(pha[0]) > len(rmf[0]):
    #              src = rebin(src, pha[0], pha[1], rmf[0], rmf[1])

    #      if len(src) != len(self._lo):
    #          raise TypeError("Mismatched filter between ARF and RMF or PHA and RMF")

    #      return rmf_fold(src, self._grp, self._fch, self._nch, self._rsp,
    #                      self.detchans, self.offset)

    #  def notice(self, noticed_chans=None):
    #      bin_mask = None
    #      self._fch = self.f_chan
    #      self._nch = self.n_chan
    #      self._grp = self.n_grp
    #      self._rsp = self.matrix
    #      self._lo = self.energ_lo
    #      self._hi = self.energ_hi
    #      if noticed_chans is not None:
    #          (self._grp, self._fch, self._nch, self._rsp,
    #           bin_mask) = filter_resp(noticed_chans, self.n_grp, self.f_chan,
    #                                   self.n_chan, self.matrix, self.offset)
    #          self._lo = self.energ_lo[bin_mask]
    #          self._hi = self.energ_hi[bin_mask]

    #      return bin_mask

    def get_indep(self, filter=False):
        filter = bool_cast(filter)
        return (self._lo, self._hi)

    def get_dep(self, filter=False):
        filter = bool_cast(filter)
        return self.apply_rmf(np.ones(self.energ_lo.shape, astroFloat))

    def get_xlabel(self):
        if (self.e_min is not None) and (self.e_max is not None):
            return 'Energy (keV)'
        return 'Channel'

    def get_ylabel(self):
        return 'Counts'


class DataPHA(Data1DInt):

    """PHA Data Set, including any associated instrument and background data.
    """

    mask = property(BaseData._get_mask, BaseData._set_mask,
                    doc=BaseData.mask.__doc__)

    @property
    def grouped(self):
        """Are the data grouped?"""
        return self._grouped

    @grouped.setter
    def grouped(self, val):
        val = bool(val)

        if val and (self.grouping is None):
            raise Exception("No grouping information", self.name)

        if self._grouped != val:
            do_notice = np.iterable(self.mask)
            if do_notice:
                old_filter = self.get_filter(val)
                self._grouped = val
                self.ignore()
                for vals in parse_expr(old_filter):
                    self.notice(*vals)
        self._grouped = val

    @property
    def subtracted(self):
        return self._subtracted

    @subtracted.setter
    def subtracted(self, val):
        val = bool(val)
        if len(self._backgrounds) == 0:
            raise Exception('No background!')
        self._subtracted = val

    @property
    def units(self):
        """
        Units of the independent axis
        """
        return self._units

    @units.setter
    def units(self, val):
        units = str(val).strip().lower()

        if units == 'bin':
            units = 'channel'

        if units.startswith('chan'):
            self._to_channel = (lambda x, group=True, response_id=None: x)
            self._from_channel = (lambda x, group=True, response_id=None: x)
            units = 'channel'
        elif units.startswith('ener'):
            self._to_channel = self._energy_to_channel
            self._from_channel = self._channel_to_energy
            units = 'energy'
        elif units.startswith('wave'):
            self._to_channel = self._wavelength_to_channel
            self._from_channel = self._channel_to_wavelength
            units = 'wavelength'
        else:
            raise Exception('Data has bad quantity')

        for id in self._background_ids:
            bkg = self.get_background(id)
            if bkg.get_response() != (None, None) or \
               (bkg.bin_lo is not None and bkg.bin_hi is not None):
                bkg.units = units

        self._units = units

    @property
    def rate(self):
        """
        Quantity of y-axis: counts or counts/sec.
        """
        return self._rate

    @rate.setter
    def rate(self, val):
        self._rate = bool_cast(val)
        for id in self.background_ids:
            self.get_background(id).rate = val

    # TODO: may be add plot function later
    # def _get_plot_fac(self)
    # def _set_plot_fac(self, val)
    # ...

    @property
    def response_ids(self):
        """
        IDs of defined instrument responses ARF/RMF pairs.
        """
        return self._response_ids

    @response_ids.setter
    def response_ids(self, ids):
        if not np.iterable(ids):
            # TODO: Error handling
            raise Exception('Response ids is not an array')
        keys = self._responses.keys()
        for id in ids:
            if id not in keys:
                raise Exception('Response id is bad id')
        ids = list(ids)
        self._response_ids = ids

    @property
    def background_ids(self):
        """
        IDs of defined background data sets.
        """
        return self._background_ids

    @background_ids.setter
    def background_ids(self, ids):
        if not np.iterable(ids):
            raise Exception('Backgound ids is not an array')
        keys = self._backgrounds.keys()
        for id in ids:
            if id not in keys:
                raise Exception('Backgound id is bad id')
        ids = list(ids)
        self._background_ids = ids

    _extra_fileds = ('grouped', 'subtracted', 'units', 'rate', 'response_ids',
                     'backgound_ids')

    def __init__(self, name, channel, counts, staterror=None, syserror=None,
                 bin_lo=None, bin_hi=None, grouping=None, quality=None,
                 exposure=None, backscal=None, areascal=None, header=None):
        self._grouped = (grouping is not None)
        self._original_groups = True
        self._subtracted = False
        self._response_ids = []
        self._background_ids = []
        self._responses = {}
        self._backgrounds = {}
        self._rate = True
        # self._plot_fac = 0
        self.units = 'channel'
        self.quality_filter = None
        BaseData.__init__(self)

    def __str__(self):
        # Print the metadata first
        old = self._fields
        ss = old
        try:
            self._fields = tuple(filter((lambda x: x != 'header'),
                                        self._fields))
            ss = BaseData.__str__(self)
        finally:
            self._fields = old
        return ss

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_to_channel']
        del state['_from_channel']
        return state

    def __setstate__(self, state):
        self._background_ids = state['_background_ids']
        self._backgrounds = state['_backgrounds']
        self._set_units(state['_units'])

        if 'header' not in state:
            self.header = None
        self.__dict__.update(state)

    primary_response_id = 1

    # NB: type ---> unit_type
    def set_analysis(self, quantity, unit_type='rate', factor=0):
        unit_type = str(unit_type).strip().lower()
        if not (unit_type.startswith('counts') or
                unit_type.startswith('rate')):
            raise Exception('Type should be either rate or counts')

        self.rate = (unit_type == 'rate')

        arf, rmf = self.get_response()
        if rmf is not None and rmf.detchans != len(self.channel):
            raise Exception('Incompatible response matrix!')

        if quantity != 'channel' and ((rmf is None and arf is None) and
                                      (self.bin_lo is None and
                                       self.bin_hi is None)):
            raise Exception('No instrument response')

        if quantity != 'channel' and ((rmf is None and arf is not None) and
                                      len(arf.energ_lo) != len(self.channel)):
            raise Exception('Incomplete response matrix')

        self.units = quantity

    def get_analysis(self):
        return self.units

    def _fix_response_id(self, id):
        if id is None:
            id = self.primary_response_id
        return id

    def get_response(self, id=None):
        id = self._fix_response_id(id)
        return self._responses.get(id, (None, None))

    def set_response(self, arf=None, rmf=None, id=None):
        if (arf is None) and (rmf is None):
            return
        id = self._fix_response_id(id)
        self._responses[id] = (arf, rmf)
        ids = self.response_ids[:]
        if id not in ids:
            ids.append(id)
        self.response_ids = ids

    def delete_response(self, id=None):
        id = self._fix_response_id(id)
        self._responses.pop(id, None)
        ids = self.response_ids[:]
        ids.remove(id)
        self.response_ids = ids

    def get_arf(self, id=None):
        return self.get_response(id)[0]

    def get_rmf(self, id=None):
        return self.get_response(id)[1]

    def set_arf(self, arf, id=None):
        self.set_response(arf, self.get_rmf(id), id)

    def set_rmf(self, rmf, id=None):
        self.set_response(self.get_arf(id), rmf, id)

    def get_specresp(self, filter=False):
        filter = bool_cast(filter)
        self.notice_response(False)
        arf, rmf = self.get_response()
        newarf = None

        if arf is not None and rmf is not None:
            specresp = arf.get_dep()
            elo, ehi = arf.get_indep()
            lo, hi = self._get_ebins(group=False)

            newarf = interpolate(lo, elo, specresp)
            newarf[newarf <= 0] = 1.

            if filter:
                newarf = self.apply_filter(newarf, self._middle)
        return newarf

    def _get_ebins(self, response_id=None, group=True):
        group = bool_cast(group)
        arf, rmf = self.get_response(response_id)
        if (self.bin_lo is not None) and (self.bin_hi is not None):
            elo = self.bin_lo
            ehi = self.bin_hi
            if (elo[0] > elo[-1]) and (ehi[0] > ehi[-1]):
                elo = self._hc / self.bin_hi
                ehi = self._hc / self.bin_lo
        elif rmf is not None:
            if (rmf.e_min is None) or (rmf.e_max is None):
                raise Exception("RMF has no energy bins")
            elo = rmf.e_min
            ehi = rmf.e_max
        elif arf is not None:
            elo = arf.energ_lo
            ehi = arf.energ_hi
        else:
            elo = self.channel - 0.5
            ehi = self.channel + 0.5

        if self.units == 'channel':
            elo = self.channel - 0.5
            ehi = self.channel + 0.5

        if (self.grouped and group):
            elo = self.apply_grouping(elo, self._min)
            ehi = self.apply_grouping(ehi, self._max)

        return (elo, ehi)

    def get_indep(self, filter=True):
        if filter:
            return (self.get_noticed_channels(),)

        return (self.channel,)

    def _get_indep(self, filter=False):
        if (self.bin_lo is not None) and (self.bin_hi is not None):
            elo = self.bin_lo
            ehi = self.bin_hi
            if (elo[0] > elo[-1]) and (ehi[0] > ehi[-1]):
                if self.units == 'wavelength':
                    return (elo, ehi)

                elo = self._hc / self.bin_hi
                ehi = self._hc / self.bin_lo
        else:
            energylist = []
            for id in self.response_ids:
                arf, rmf = self.get_response(id)
                lo = None
                hi = None

                if rmf is not None:
                    lo = rmf.energ_lo
                    hi = rmf.energ_hi
                    if filter:
                        lo, hi = rmf.get_indep()
                elif arf is not None:
                    lo = arf.energ_lo
                    hi = arf.energ_hi
                    if filter:
                        lo, hi = arf.get_indep()

                energylist.append((lo, hi))

            if len(energylist) > 1:
                elo, ehi, lookuptable = compile_energy_grid(energylist)
            elif (not energylist or (len(energylist) == 1 and
                                     np.equal(energylist[0], None).any())):
                raise Exception('Response has no energy bins')
            else:
                elo, ehi = energylist[0]

        lo, hi = elo, ehi
        if self.units == 'wavelength':
            lo = self._hc / ehi
            hi = self._hc / elo

        return (lo, hi)

    def _channel_to_energy(self, val, group=True, response_id=None):
        elo, ehi = self._get_ebins(response_id=response_id, group=group)
        val = np.asarray(val).astype(np.int_) - 1
        try:
            return (elo[val] + ehi[val]) / 2.0
        except IndexError:
            raise Exception('Invalid channel')

    def _energy_to_channel(self, val):
        elo, ehi = self._get_ebins()

        val = np.asarray(val)
        res = []
        for v in val.flat:
            if tuple(np.flatnonzero(elo <= v)) == ():
                if elo[0] > elo[-1] and ehi[0] > ehi[-1]:
                    res.append(astroFloat(len(elo)))
                else:
                    res.append(astroFloat(1))
            elif tuple(np.flatnonzero(ehi > v)) == ():
                if elo[0] > elo[-1] and ehi[0] > ehi[-1]:
                    res.append(astroFloat(1))
                else:
                    res.append(astroFloat(len(ehi)))
            elif tuple(np.flatnonzero((elo <= v) & (ehi > v)) + 1) != ():
                res.append(astroFloat(
                    np.flatnonzero((elo <= v) & (ehi > v)) + 1))
            elif (elo <= v).argmin() == (ehi > v).argmax():
                res.append(astroFloat((elo <= v).argmin()))
            else:
                raise Exception('Something wrong: energy to channel')

        if val.shape == ():
            return res[0]

        return np.asarray(res, astroFloat)

    _hc = 12.39841874

    def _channel_to_wavelength(self, val, group=True, response_id=None):
        tiny = np.finfo(np.float32).tiny
        vals = np.asarray(self._channel_to_energy(val, group, response_id))
        if vals.shape == ():
            if vals == 0.0:
                vals = tiny
        else:
            vals[vals == 0.0] = tiny
        vals = self._hc / vals
        return vals

    def _wavelength_to_channel(self, val):
        tiny = np.finfo(np.float32).tiny
        vals = np.asarray(val)
        if vals.shape == ():
            if vals == 0.0:
                vals = tiny
        else:
            vals[vals == 0.0] = tiny
        vals = self._hc / vals
        return self._energy_to_channel(vals)

    default_background_id = 1

    def _fix_background_id(self, id):
        if id is None:
            id = self.default_background_id
        return id

    def get_background(self, id=None):
        id = self._fix_background_id(id)
        return self._backgrounds.get(id)

    def set_background(self, bkg, id=None):
        id = self._fix_background_id(id)
        self._backgrounds[id] = bkg
        ids = self.background_ids[:]
        if id not in ids:
            ids.append(id)
        self.background_ids = ids

    def delete_background(self, id=None):
        id = self._fix_background_id(id)
        self._backgrounds.pop(id, None)
        if len(self._backgrounds) == 0:
            self._subtracted = False
        ids = self.background_ids[:]
        if id in ids:
            ids.remove(id)
        self.background_ids = ids

    def get_background_scale(self):
        if len(self.background_ids) == 0:
            return None
        return self.sum_background_data(lambda key, bkg: 1.)

    def _check_scale(self, scale, group=True, filter=False):
        if np.isscalar(scale) and scale <= 0.0:
            scale = 1.0
        elif np.iterable(scale):
            scale = np.asarray(scale, dtype=astroFloat)
            if group:
                if filter:
                    scale = self.apply_filter(scale, self._middle)
                else:
                    scale = self.apply_grouping(scale, self._middle)
            scale[scale <= 0.0] = 1.0
        return scale

    def get_backscal(self, group=True, filter=False):
        backscal = self.backscal
        if backscal is not None:
            backscal = self._check_scale(backscal, group, filter)
        return backscal

    def get_areascal(self, group=True, filter=False):
        areascal = self.areascal
        if areascal is not None:
            areascal = self._check_scale(areascal, group, filter)
        return areascal

    def apply_filter(self, data, groupfunc=np.sum):
        """

        Filter the array data, first passing it through apply_grouping()
        (using groupfunc) and then applying the general filters

        """
        if (data is None):
            return data
        elif len(data) != len(self.counts):
            counts = np.zeros(len(self.counts), dtype=astroFloat)
            mask = self.get_mask()
            if mask is not None:
                counts[mask] = np.asarray(data, dtype=astroFloat)
                data = counts
            # else:
            #     raise DataErr('mismatch', "filter", "data array")
        return Data1DInt.apply_filter(self,
                                      self.apply_grouping(data, groupfunc))

    def ignore_bad(self):
        """Exclude channels marked as bad.

        Ignore any bin in the PHA data set which has a quality value
        that is larger than zero.

        Raises
        ------
        sherpa.utils.err.DataErr
           If the data set has no quality array.

        See Also
        --------
        ignore : Exclude data from the fit.
        notice : Include data in the fit.

        Notes
        -----
        Bins with a non-zero quality setting are not automatically
        excluded when a data set is created.

        If the data set has been grouped, then calling `ignore_bad`
        will remove any filter applied to the data set. If this
        happens a warning message will be displayed.

        """
        if self.quality is None:
            raise DataErr("noquality", self.name)

        qual_flags = np.asarray(self.quality, bool)

        if self.grouped and (self.mask is not True):
            self.notice()
            print('Warning: filtering grouped data with quality flags' +
                  'previous filters deleted')

        elif not self.grouped:
            # if ungrouped, create/combine with self.mask
            if self.mask is not True:
                self.mask = self.mask & qual_flags
                return
            else:
                self.mask = qual_flags
                return

        # self.quality_filter used for pre-grouping filter
        self.quality_filter = qual_flags

    # TODO: add group function later? def _dynamic_group()

    def sum_background_data(self,
                            get_bdata_func=(lambda key, bkg: bkg.counts)):
        bdata_list = []

        for key in self.background_ids:
            bkg = self.get_background(key)
            bdata = get_bdata_func(key, bkg)

            backscal = bkg.backscal
            if backscal is not None:
                backscal = self._check_scale(backscal, group=False)

            areascal = bkg.get_areascal(group=False)
            if areascal is not None:
                bdata = bdata / areascal

            if bkg.exposure is not None:
                bdata = bdata / bkg.exposure

            bdata_list.append(bdata)

        nbkg = len(bdata_list)
        assert (nbkg > 0)
        if nbkg == 1:
            bkgsum = bdata_list[0]
        else:
            bkgsum = sum(bdata_list)

        backscal = self.get_backscal
        if backscal is not None:
            backscal = self._check_scale(backscal, group=False)
            bkgsum = backscal * bkgsum

        areascal = self.areascal
        if areascal is not None:
            areascal = self._check_scale(areascal, group=False)
            bkgsum = areascal * bkgsum

        if self.exposure is not None:
            bkgsum = self.exposure * bkgsum

        return bkgsum / astroFloat(nbkg)

    def get_dep(self, filter=False):
        dep = self.counts
        filter = bool_cast(filter)

        if self.subtracted:
            bkg = self.sum_background_data()
            if len(dep) != len(bkg):
                raise Exception("Subtraction lenght error")
            dep = dep - bkg

        if filter:
            dep = self.apply_filter(dep)

        return dep

    def set_dep(self, val):
        dep = None
        if np.iterable(val):
            dep = np.asarray(val, astroFloat)
        else:
            val = astroFloat(val)
            dep = np.array([val] * len(self.get_indep()[0]))
        setattr(self, 'counts', dep)

    def get_x(self, filter=False, response_id=None):
        if self.units != 'channel':
            elo, ehi = self._get_ebins(group=False)
            if len(elo) != len(self.channel):
                raise Exception('Incomplete response')
            return self._from_channel(self.channel, group=False,
                                      response_id=response_id)
        else:
            return self._from_channel(self.channel)

    def get_xlabel(self):
        xlabel = self.units.capitalize()
        if self.units == 'energy':
            xlabel += ' (keV)'
        elif self.units == 'wavelenght':
            xlabel += ' (Angstrom)'

        return xlabel

    def _set_initial_quantity(self):
        arf, rmf = self.get_response()

        if arf is not None and rmf is None:
            if len(arf.energ_lo) == len(self.channel):
                self.units = 'energy'

        if rmf is not None:
            if len(self.channel) != len(rmf.e_min):
                raise Exception('Incomplete response rmf')
            self.units = 'energy'

    def _fix_y_units(self, val, filter=False, response_id=None):

        if val is None:
            return val

        filter = bool_cast(filter)
        val = np.array(val, dtype=astroFloat)

        if self.rate and self.exposure is not None:
            val /= self.exposure

        if self.areascal is not None:
            areascal = self._check_scale(self.areascal, filter=filter)
            val /= areascal

        if self.grouped or self.rate:
            if self.units != 'channel':
                elo, ehi = self._get_ebins(response_id, group=False)
            else:
                elo, ehi = (self.channel, self.channel + 1.)

            if filter:
                elo = self.apply_filter(elo, self._min)
                ehi = self.apply_filter(ehi, self._max)
            elif self.grouped:
                elo = self.apply_grouping(elo, self._min)
                ehi = self.apply_grouping(ehi, self._max)

            if self.units == 'energy':
                ebin = ehi - elo
            elif self.units == 'wavelength':
                ebin = self._hc / elo - self._hc / ehi
            elif self.units == 'channel':
                ebin = ehi - elo
            else:
                raise Exception('units has bad quantity')

            val /= np.abs(ebin)

            if self.plot_fac <= 0:
                return val

            scale = self.apply_filter(self.get_x(response_id=response_id),
                                      self._middle)
            for ii in range(self.plot_fac):
                val *= scale

            return val

    def get_y(self, filter=False, yfunc=None, response_id=None):
        vallist = Data1DInt.get_y(self, yfunc=yfunc)
        filter = bool_cast(filter)

        if not isinstance(vallist, tuple):
            vallist = (vallist,)

        newvallist = []

        for val in vallist:
            if filter:
                val = self.apply_filter(val)
            else:
                val = self.apply_grouping(val)
            val = self._fix_y_units(val, filter, response_id)
            newvallist.append(val)

        if len(vallist) == 1:
            vallist = newvallist[0]
        else:
            vallist = tuple(newvallist)

        return vallist

    def get_yerr(self, filter=False, staterrfunc=None, response_id=None):
        filter = bool_cast(filter)
        err = self.get_error(filter, staterrfunc)
        return self._fix_y_units(err, filter, response_id)

    def get_xerr(self, filter=False, response_id=None):
        elo, ehi = self._get_ebins(response_id=response_id)
        filter = bool_cast(filter)
        if filter:
            elo, ehi = self._get_ebins(response_id, group=False)
            elo = self.apply_filter(elo, self._min)
            ehi = self.apply_filter(ehi, self._max)

        return ehi - elo

    def get_ylabel(self):
        ylabel = 'Counts'

        if self.rate and self.exposure:
            ylabel += '/sec'

        if self.rate or self.grouped:
            if self.units == 'energy':
                ylabel += '/keV'
            elif self.units == 'wavelength':
                ylabel += '/Angstrom'
            elif self.units == 'channel':
                ylabel += '/channel'

        return ylabel

    # TODO: skip all the dynamic group

    @staticmethod
    def _make_groups(array):
        pass

    @staticmethod
    def _middle(array):
        array = np.asarray(array)
        return (array.min() + array.max()) / 2.0

    @staticmethod
    def _max(array):
        array = np.asarray(array)
        return array.max()

    @staticmethod
    def _min(array):
        array = np.asarray(array)
        return array.min()

    @staticmethod
    def _sum_sq(array):
        return np.sqrt(np.sum(array * array))

    def get_noticed_channels(self):
        chans = self.channel
        mask = self.get_mask()
        if mask is not None:
            chans = chans[mask]
        return chans

    def get_mask(self):
        groups = self.grouping
        if self.mask is False:
            return None

        if self.mask is True or not self.grouped:
            if self.quality_filter is not None:
                return self.quality_filter
            elif np.iterable(self.mask):
                return self.mask
            return None

        if self.quality_filter is not None:
            groups = groups[self.quality_filter]
        return expand_grouped_mask(self.mask, groups)

    def get_noticed_expr(self):
        chans = self.get_noticed_channels()
        if self.mask is False or len(chans) == 0:
            return 'No noticecd channels'
        return create_expr(chans, format='%i')

    def get_filter(self, group=True, format='%.12f', delim=':'):
        if self.mask is False:
            return 'No noticed bins'

        x = self.get_noticed_channels()
        if group:
            x = self.apply_filter(self.channel, self._make_groups)

        x = self._from_channel(x, group=group)

        if self.units in ('channel',):
            format = '%i'

        mask = np.ones(len(x), dtype=bool)
        if np.iterable(self.mask):
            mask = self.mask

        if self.units in ('wavelength',):
            x = x[::-1]
            mask = mask[::-1]
        return create_expr(x, mask, format, delim)

    def get_filter_expr(self):
        return (self.get_filter(format='%.4f', delim='-') +
                ' ' + self.get_xlabel())

    def notice_response(self, notice_resp=True, noticed_chans=None):
        notice_resp = bool_cast(notice_resp)

        if notice_resp and noticed_chans is None:
            noticed_chans = self.get_noticed_channels()

        for id in self.response_ids:
            arf, rmf = self.get_response(id)
            _notice_resp(noticed_chans, arf, rmf)

    def notice(self, lo=None, hi=None, ignore=False, bkg_id=None):
        filter_background_only = False
        if (bkg_id is not None):
            if (not(np.iterable(bkg_id))):
                bkg_id = [bkg_id]
            filter_background_only = True
        else:
            bkg_id = self.background_ids

        for bid in bkg_id:
            bkg = self.get_background(bid)
            old_bkg_units = bkg.units
            bkg.units = self.units
            bkg.notice(lo, hi, ignore)
            bkg.units = old_bkg_units

        if filter_background_only:
            return

        ignore = bool_cast(ignore)
        if lo is None and hi is None:
            self.quality_filter = None
            self.notice_response(False)

        elo, ehi = self._get_ebins()
        if lo is not None and type(lo) != str:
            lo = self._to_channel(lo)
        if hi is not None and type(hi) != str:
            hi = self._to_channel(hi)

        if ((self.units == 'wavelength' and
             elo[0] < elo[-1] and ehi[0] < ehi[-1]) or
            (self.units == 'energy' and
             elo[0] > elo[-1] and ehi[0] > ehi[-1])):
            lo, hi = hi, lo

        if self.units == 'channel' and self.grouped:
            if lo is not None and type(lo) != str and \
               not(lo < self.channel[0]):
                lo_index = np.where(self.channel >= lo)[0][0]
                lo = len(np.where(self.grouping[:lo_index] > -1)[0]) + 1

            if hi is not None and type(hi) != str and \
               not(hi > self.channel[-1]):
                hi_index = np.where(self.channel >= hi)[0][0]
                hi = len(np.where(self.grouping[:hi_index] > -1)[0])

                if (self.grouping[hi_index] > -1):
                    hi = hi + 1

                if (hi_index + 1 < len(self.grouping)):
                    if not(self.grouping[hi_index + 1] > -1):
                        hi = hi - 1

        BaseData.notice(self, (lo,), (hi,),
                        (self.apply_grouping(self.channel,
                                             self._make_groups),),
                        ignore)

    def to_guess(self):
        elo, ehi = self._get_ebins(group=False)
        elo = self.apply_filter(elo, self._min)
        ehi = self.apply_filter(ehi, self._max)
        if self.units == 'wavelength':
            lo = self._hc / ehi
            hi = self._hc / elo
            elo = lo
            ehi = hi
        cnt = self.get_dep(True)
        arf = self.get_specresp(filter=True)

        y = cnt / (ehi - elo)
        if self.exposure is not None:
            y /= self.exposure
        if arf is not None:
            y /= arf
        return (y, elo, ehi)

    def subtract(self):
        self.subtracted = True

    def unsubtract(self):
        self.subtracted = False


class DataLC(Data1DInt):
    """Light Curve Data Set"""

    mask = property(BaseData._get_mask, BaseData._set_mask,
                    doc=BaseData.mask.__doc__)

    def __init__(self, name, time, timedel=None, counts=None, error=None, flux=None,
                 flux_error=None, mjdref=None, tstart=None, tstop=None,
                 timezero=None, timesys=None, timeunit='s', timeref=None,
                 reftime=None, clockcor=None, header=None):
        self._time = time
        self._timedel = timedel
        self._counts = counts
        self._flux = flux
        self._error = error
        self._flux_error = flux_error
        self._timeunit = timeunit
        BaseData.__init__(self)

    def __str__(self):
        # Print the metadata first
        old = self._fields
        ss = old
        try:
            self._fields = tuple(filter((lambda x: x != 'header'), self._fields))
            ss = BaseData.__str__(self)
        finally:
            self._fields = old
        return ss

    def __setstate__(self, state):
        if 'header' not in state:
            self.header = None
        self.__dict__.update(state)

    def get_time(self):
        return self._time

    def get_timedel(self):
        return self._timedel
