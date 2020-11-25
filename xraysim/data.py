#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import inspect
import numpy as np
from .utils.essential import (bool_cast, create_expr, print_fields,
                             filter_bins, calc_total_error)
from .utils.essential import astroFloat


__all__ = ('Data', 'Data1D', 'Data1DInt')


class NoNewAttributesAfterInit(object):

    """
    Prevents attribute deletion and setting of new attributes after __init__
    has been called. Derived classes must call
    NoNewAttributesAfterInit.__init__ after all other initialization.
    """

    # Use name mangling
    __initilized = False

    def __init__(self):
        self.__initilized = True

    def __delattr__(self, name):
        if self.__initilized and hasattr(self, name):
            raise Exception("Attributes can not be deleted")
        object.__delattr__(self, name)

    def __setattr__(self, name, val):
        if self.__initilized and (not hasattr(self, name)):
            raise Exception("Object has no attribute name")

        if self.__initilized and hasattr(self, name):
            if callable(getattr(self, name)) and not callable(val):
                raise Exception("Objects attribute can not be replaced with" +
                                "a non callable attribute")
            elif not callable(getattr(self, name)) and callable(val):
                raise Exception("Objects attribute can not be replaced with" +
                                "a callable attribute")
        object.__setattr__(self, name, val)


class BaseData(NoNewAttributesAfterInit):

    """
    Base class for all data set types.
    """

    def _get_filter(self):
        return self._filter

    def _set_filter(self, val):
        self._filter = val
        self._mask = True

    filter = property(_get_filter, _set_filter,
                      doc='Filter for dependent variable')

    def _get_mask(self):
        return self._mask

    def _set_mask(self, val):
        if (val is True) or (val is False):
            self._mask = val
        elif (val is None) or np.isscalar(val):
            raise Exception('Data is mask')
        else:
            self._mask = np.asarray(val, np.bool_)
        self._filter = None

    mask = property(_get_mask, _set_mask,
                    doc='Mask array for dependent variable')

    def __init__(self):
        """Initialize a data object. This method can only be called from a
        derived class constructor. Attempts to create a BaseData instance
        will raise NotImplementedErr.

        Derived class constructors must call this method directly, and not
        indeirectly through a supercalss constructor. When thus invoked, this
        method will extract the argumetn names and values from the derived
        class constructor invocation and set corresponding attributes on the
        instance.  If the name of an argument matches the name of a
        DataProperty of the derived class, then the corresponding attribute
        name will have an underscore prepended.
        """

        if type(self) is BaseData:
            raise Exception('NotImplementedErr: BaseData instance is not\
                            allowed')

        frame = sys._getframe().f_back
        cond = (frame.f_code is self.__init__.__func__.__code__)
        assert cond, (('%s constructor must call BaseData constructor ' +
                       'directly') % type(self).__name__)
        args = inspect.getargvalues(frame)

        self._fields = tuple(args[0][1:])
        for f in self._fields:
            cond = (f not in vars(self))
            assert cond, (("'%s' object alreday has attributes '%s'") %
                          (type(self).__name, f))
            setattr(self, f, args[3][f])

        self.filter = None
        self.mask = True

        NoNewAttributesAfterInit.__init__(self)

    def __str__(self):
        """Return a list of the attributes listed in self._fields and,
        if present, self._extra_fields.

        """
        fields = self._fields + getattr(self, '_extra_fields', ())
        fdict = dict(zip(fields, [getattr(self, f) for f in fields]))
        return print_fields(fields, fdict)

    def apply_filter(self, data):
        if data is not None:
            if self.filter is not None:
                if callable(self.filter):
                    data = self.filter(data)
                else:
                    data = data[self.filter]
            elif self.mask is not True:
                if self.mask is False:
                    raise Exception('Not a mask')
                data = np.asarray(data)
                if data.shape != self.mask.shape:
                    raise Exception('The shape of the mask is mismatch with' +
                                    'data array')
                data = data[self.mask]
        return data

    def ignore(self, *args, **kwargs):
        kwargs['ignore'] = True
        self.notice(*args, **kwargs)

    def notice(self, mins, maxes, axislist, ignore=False):
        ignore = bool_cast(ignore)
        if str in [type(bound_min) for bound_min in mins]:
            raise Exception('DataErr: typecheck lower bound')
        elif str in [type(bound_max) for bound_max in maxes]:
            raise Exception('DataErr: typecheck upper bound')
        elif str in [type(axis) for axis in axislist]:
            raise Exception('DataErr: typecheck grid')

        mask = filter_bins(mins, maxes, axislist)

        if mask is None:
            self.mask = not ignore
        elif not ignore:
            if self.mask is True:
                self.mask = mask
            else:
                self.mask |= mask
        else:
            mask = ~mask
            if self.mask is False:
                self.mask = mask
            else:
                self.mask &= mask


class Data(BaseData):

    """Generic data set"""

    def __init__(self, name, indep, dep, staterror=None,
                 syserror=None):
        """Initialize a Data instance. indep should be a tuple of independent
        axis arrays, dep should be an array of dependent variable values, and
        staterror and syserror should be arrays of statistical and systematical
        errors, respectively, in the dependent variable (or None).

        :name:
        :indep: tuple, a tuple of independent axis arryas.
        :dep: array, an array of dependent variable values.
        :staterror: array or None, an array of statistical errors in the
            dependent variable.
        :syserror: array or None, an array of systematical errors in the
            dependent variable

        """
        BaseData.__init__(self)

    def __repr__(self):
        r = '<%s data set instance' % type(self).__name__
        if hasattr(self, 'name'):
            r += " '%s'" % self.name
        r += '>'
        return r

    def eval_model(self, modelfunc):
        return modelfunc(*self.get_indep())

    def eval_model_to_fit(self, modelfunc):
        return modelfunc(*self.get_indep(filter=True))

    def get_indep(self, filter=False):
        """Return the independent axes of a data set.

        :filter: bool, optional, should the filter attached to the data set be
            applied to the return value or not.
        :returns: axis: tuple of arrays. The independent axis values for the
            data set. This gives the coordinates of each point in the data set.

        """
        indep = getattr(self, 'indep', None)
        filter = bool_cast(filter)
        if filter:
            indep = tuple([self.apply_filter(x) for x in indep])

        return indep

    def get_dep(self, filter=False):
        """Return the dependent axes of a data set.

        :filter: bool, optional, should the filter attached to the data set be
            applied to the return value or not.
            :returns: axis: array.

        """
        dep = getattr(self, 'dep', None)
        filter = bool_cast(filter)
        if filter:
            dep = self.apply_filter(dep)

        return dep

    def get_staterror(self, filter=False, staterrfunc=None):
        staterror = getattr(self, 'staterror', None)
        filter = bool_cast(filter)
        if filter:
            staterror = self.apply_filter(staterror)

        if (staterror is None) and (staterrfunc is not None):
            dep = self.get_dep()
            if filter:
                dep = self.apply_filter(dep)
            staterror = staterrfunc(dep)

        return staterror

    def get_syserror(self, filter=False):
        syserror = getattr(self, 'syserror', None)
        filter = bool_cast(filter)
        if filter:
            syserror = self.apply_filter(syserror)

        return syserror

    # Utility methods

    def _wrong_dim_error(self, baddim):
        raise Exception('Wrong dimension!')

    def _no_image_error(self):
        raise Exception('Not an image')

    def _no_dim_error(self):
        raise Exception('no dimension!')

    # Secondary properties. To best support subclasses, these should depend
    # only on the primary properties whenever possible.

    def get_dims(self):
        self._no_dim_error()

    def get_error(self, filter=False, staterrfunc=None):
        return calc_total_error(self.get_staterror(filter, staterrfunc),
                                self.get_syserror(filter))

    def get_x(self, filter=False):
        self._wrong_dim_error(1)

    def get_xerr(self, filter=False):
        return None

    def get_xlabel(self):
        return 'x'

    def get_y(self, filter=False, yfunc=None):
        y = self.get_dep(filter)

        if yfunc is not None:
            if filter:
                yfunc = self.eval_model_to_fit(yfunc)
            else:
                yfunc = self.eval_model(yfunc)
            y = (y, yfunc)

        return y

    def get_yerr(self, filter=False, staterrfunc=None):
        return self.get_error(filter, staterrfunc)

    def get_ylabel(self, yfunc=None):
        return 'y'

    def get_x0(self, filter=False):
        self._wrong_dim_error(2)

    def get_x0label(self):
        return 'x0'

    def get_x1(self, filter=False):
        self._wrong_dim_error(2)

    def get_x1label(self):
        return 'x1'

    def get_img(self, yfunc=None):
        self._no_image_error()

    def get_imgerr(self, yfunc=None):
        self._no_image_error()

    def to_guess(self):
        arrays = [self.get_y(True)]
        arrays.extend(self.get_indep(True))
        return tuple(arrays)

    def to_fit(self, staterrfunc=None):
        return (self.get_dep(True),
                self.get_staterror(True, staterrfunc),
                self.get_syserror(True))

    def to_plot(self, yfunc=None, staterrfunc=None):
        return (self.get_x(True),
                self.get_y(True, yfunc),
                self.get_yerr(True, staterrfunc),
                self.get_xerr(True),
                self.get_xlabel(),
                self.get_ylabel())

    def to_contour(self, yfunc=None):
        return (self.get_x0(True),
                self.get_x1(True),
                self.get_y(True, yfunc),
                self.get_x0label(),
                self.get_x1label())


class DataND(Data):

    """Base class for Data1D, Data2D, etc"""

    def get_dep(self, filter=False):
        y = self.y
        filter = bool_cast(filter)
        if filter:
            y = self.apply_filter(y)

        return y

    def set_dep(self, values):
        dep = None
        if np.iterable(values):
            dep = np.asarray(values, astroFloat)
        else:
            values = astroFloat(values)
            dep = np.array([values] * len(self.get_indep()[0]))
        setattr(self, 'y', dep)


class Data1D(DataND):

    """1-D data set """

    def _set_mask(self, val):
        DataND._set_mask(self, val)
        try:
            self._x = self.apply_filter(self.x)
        except DataErr:
            self._x = self.x

    mask = property(DataND._get_mask, _set_mask,
                    doc='Mask array for dependent variable')

    def __init__(self, name, x, y, staterror=None, syserror=None):
        self._x = x
        BaseData.__init__(self)

    def get_indep(self, filter=False):
        filter = bool_cast(filter)
        if filter:
            return (self._x,)
        return (self.x,)

    def get_x(self, filter=False):
        return self.get_indep(filter)[0]

    def get_dims(self, filter=False):
        return (len(self.get_x(filter)),)

    def get_filter(self, format='%.4f', delim=':'):
        # for derived integrated classes, this will return values in center of
        # bin.
        x = self.get_x(filter=True)
        mask = np.ones(len(x), dtype=bool)
        if np.iterable(self.mask):
            mask = self.mask
        return create_expr(x, mask, format, delim)

    def get_filter_expr(self):
        return (self.get_filter(delim='-') + ' ' + self.get_xlabel())

    def get_bounding_mask(self):
        mask = self.mask
        size = None
        if np.iterable(self.mask):
            # create bounding box around noticed image regions
            mask = np.array(self.mask)
            size = (mask.size,)
        return mask, size

    #TODO: may add later def get_img(self, yfunc=None)

    #TODO: may add later def get_imgerr(self)

    def notice(self, xlo=None, xhi=None, ignore=False):
        BaseData.notice(self, (xlo,), (xhi,), self.get_indep(), ignore)


class Data1DInt(Data1D):

    """1-D integrated data set"""

    def _set_mask(self, val):
        DataND._set_mask(self, val)
        try:
            self._lo = self.apply_filter(self.xlo)
            self._hi = self.apply_filter(self.xhi)
        except DataErr:
            self._lo = self.xlo
            self._hi = self.xhi

    mask = property(DataND._get_mask, _set_mask,
                    doc='Mask array for dependent variable')

    def __init__(self, name, xlo, xhi, y, staterror=None, syserror=None):
        self._lo = xlo
        self._hi = xhi
        BaseData.__init__(self)

    def get_indep(self, filter=False):
        filter = bool_cast(filter)
        if filter:
            return (self._lo, self._hi)
        return (self.xlo, self.xhi)

    def get_x(self, filter=False):
        indep = self.get_indep(filter)
        return (indep[0] + indep[1]) / 2.0

    def get_xerr(self, filter=False):
        xlo, xhi = self.get_indep(filter)
        return xhi - xlo

    def notice(self, xlo=None, xhi=None, ignore=False):
        BaseData.notice(self, (None, xlo), (xhi, None), self.get_indep(),
                        ignore)


# TODO class Data2D(DataND):
