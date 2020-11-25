#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np


astroInt = np.intp
astroUint = np.uintp
astroFloat = np.float_


def print_fields(names, values, converters={}):
    """ Given a list of strings names and mapping values, where names is a subset
    of vals.keys(), return a listing of name/value pairs printed one per line
    in the format '<name> = <value>'.  If a value is a NumPy array, print it in
    the format '<data type name>[<array size>]'.  Otherwise, use str(value).

    :names: TODO
    :values: TODO
    :converters: TODO
    :returns: TODO

    """
    width = max(len(n) for n in names)
    fmt = '%%-%ds = %%s' % width
    # check the sherpa function
    pass


def bool_cast(value):
    if type(value) in (tuple, list, np.ndarray):
        return np.asarray([bool_cast(item) for item in value], bool)
    elif type(value) == str:
        vlo = value.lower()
        if vlo in ('false', 'off', 'no', '0', 'f', 'n'):
            return False
        elif vlo in ('true', 'on', 'yes', '1', 't', 'y'):
            return True

        raise TypeError("Unknown boolean value: '%s'" % str(value))
    else:
        return bool(value)


def filter_bins(mins, maxes, axislist):
    mask = None

    for lo, hi, axis in zip(mins, maxes, axislist):
        if (lo is None) and (hi is None):
            continue

        if lo is None:
            # axismask = axis <= hi
            # axismask = (sao_fcmp(hi, axis, eps) >= 0)
            axismask = np.less_equal(axis, hi)
        elif hi is None:
            # axismask = axis >= lo
            # axismask = (sao_fcmp(lo, axis, eps) <= 0)
            axismask = np.greater_equal(axis, lo)
        else:
            # axismask = (axis >= lo) & (axis <= hi)
            # axismask = ((sao_fcmp(lo, axis, eps) <= 0) &
            #             (sao_fcmp(hi, axis, eps) >= 0))
            axismask = (np.greater_equal(axis, lo) &
                        np.less_equal(axis, hi))

        if mask is None:
            mask = axismask
        else:
            mask &= axismask

    return mask


def create_expr(vals, mask=None, format='%s', delim='-'):
    """
    collapse a list of channels into an expression using hyphens
    and commas to indicate filtered intervals.
    """
    expr = []

    if len(vals) == 0:
        return ''
    elif len(vals) == 1:
        return format % vals[0]

    diffs = np.apply_along_axis(np.diff, 0, vals)
    if mask is not None:
        index = np.arange(len(mask))
        diffs = np.apply_along_axis(np.diff, 0, index[mask])

    for ii, delta in enumerate(diffs):
        if ii == 0:
            expr.append(format % vals[ii])
            if delta != 1 or len(diffs) == 1:
                expr.append(',')
            continue
        if delta == 1:
            if expr[-1] == ',':
                expr.append(format % vals[ii])
            if expr[-1] != delim:
                expr.append(delim)
        else:
            if not expr[-1] in (',', delim):
                expr.append(',')
            expr.append(format % vals[ii])
            expr.append(',')
    if len(expr) and expr[-1] in (',', delim):
        expr.append(format % vals[-1])

    return ''.join(expr)


def calc_total_error(staterror=None, syserror=None):
    """Add statistical and systematic errors in quadrature.

    Parameters
    ----------
    staterror : array, optional
       The statistical error, or ``None``.
    syserror : array, optional
       The systematic error, or ``None``.

    Returns
    -------
    error : array or ``None``
       The errors, added in quadrature. If both ``staterror`` and
       ``syserror`` are ``None`` then the return value is ``None``.

    """

    if (staterror is None) and (syserror is None):
        error = None
    elif (staterror is not None) and (syserror is None):
        error = staterror
    elif (staterror is None) and (syserror is not None):
        error = syserror
    else:
        error = np.sqrt(staterror * staterror + syserror * syserror)
    return error
