#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy.io import fits
from astropy.io import ascii as asc
import os
import string
import re


__all__ = ('get_image_data', 'get_arf_data', 'get_rmf_data', 'get_pha_data',
           'get_ascii_data', 'get_lc_data')

# dtype
astroInt = np.intp
astroUInt = np.uintp
astroFloat = np.float64

# the variable length field class
_VLF = fits.column._VLF

# Create a new header attribute
_new_header = fits.Header

# Create a new binary table HDU, for ASCII use fits.TableHDU.from_columns
_new_table = fits.BinTableHDU.from_columns


def _has_hdu(hdulist, ext_id):
    """
    Check if a hdu with index/name of ext_id exists in the hdulist object.
    If not return False

    :hdulist: astropy hdulist object
    :ext_id: index/name of the hdu
    :returns: False if ext_id is out of range/not found, otherwise True

    """
    try:
        hdulist[ext_id]
    except (KeyError, IndexError):
        return False
    return True


def _has_key(hdu, key_id):
    """
    Check if key_id exists in the hdu header

    :hdu: astropy hdu object
    :key_id: key words in header
    :returns: False if key_id is not in header key words, otherwise True

    """
    return key_id in hdu.header


def _find_tbl_col(tbl, col_name):
    """
    Find if a column with col_id is present in a table

    :tbl: an astropy Table object
    :col_name: TODO
    :returns: TODO

    """
    r = re.compile(col_name, flags=re.IGNORECASE)
    col_list = list(filter(r.match, tbl.colnames))

    if len(col_list) == 0:
        return None
    else:
        return col_list[0]


def _try_tbl_col(tbl, col_name, fix_type=False, dtype=astroFloat):
    """
    Read a column of astropy Table object, if the col does not exist, return
    None
    :tbl: astropy Table object
    :col_name: str, column name, or a pattern in the column name
    :fix: TODO
    :returns: TODO

    """
    try:
        col = tbl[col_name].quantity.value
    except:
        raise Exception('Column name is not found!')

    if fix_type:
        col = col.astype(dtype)

    return col


def is_binary_file(filename):
    """Estimate if a file is a binary file.

    Returns True if a non-printable character is found in the first
    1024 bytes of the file.
    """
    fname = _check_filename(filename)
    fd = open(fname, 'r')
    try:  # Python 2
        lines = fd.readlines(1024)
        fd.close()

        if len(lines) == 0:
            return False

        # If a non-printable character is found in first 1024 --> binary
        for line in lines:
            for char in line:
                if char not in string.printable:
                    return True

        return False
    except UnicodeDecodeError:  # Python 3
        return True
    finally:
        fd.close()


def _is_fits_hudlist(arg):
    """Check whether a valid fits file or hdulist object is opened

    :arg: str, name of the file to be checked; or astropy hdulist
    :returns: True if the file is a valid fits file.

    """
    if isinstance(arg, fits.HDUList):
        return True
    else:
        fname = _check_filename(arg)
        fd = open(fname, 'r')
        char = fd.read(30)
        fd.close()
        if char == 'SIMPLE  =                    T':
            return True
        else:
            return False


def _try_key(hdu, key_id, fix_type=False, dtype=astroFloat):
    """
    Read the value of key key_id in the header of hdu object.
    If the key is not found or the value of key is 'none', then return None.

    :hdu: astropy hdu object
    :key_id: key word
    :fix_type: whether the value is a fixed type
    :dtype: type of the key value
    :returns: value of the key if has, otherwise None

    """
    if _has_key(hdu, key_id):
        key_value = hdu.header[key_id]
        if str(key_value).find('none') != -1:
            return None

        if fix_type:
            key_value = dtype(key_value)

        return key_value

    return None


def _require_key(hdu, key_id, fix_type=False, dtype=astroFloat):
    """
    Check if the required key key_id exists

    :hdu: astropy hdu object
    :key_id: key word
    :fix_type: whether the value is a fixed type
    :dtype: type of the key value
    :returns: value of the key if has, otherwise raise error

    """
    key_value = _try_key(hdu, key_id, fix_type, dtype)
    if key_value is None:
        # TODO: define Err in a more elegant way
        raise Exception("file '%s' does not have a '%s' keyword",
                        hdu._file.name, key_id)

    return key_value


def _get_file_contents(arg, exptype='PrimaryHDU', nobinary=False):
    """
    Check the type of the input. It can either be a fits file or an astropy
    HDUList.

    :arg: FITS file name or HDUList object
    :exptype: one of the HDUs in the HDUList, default be PrimaryHDU
    :nobinary: set to True to avoid checking a binary file
    :returns: List of HDUs and the file name

    Notes: binary check does't be implemented (the is_binary_file function).

    """
    if isinstance(arg, np.compat.basestring) and (not nobinary
                                                  or is_binary_file(arg)):
        tbl = open_fits(arg)
        filename = arg
    elif isinstance(arg, fits.HDUList) and len(arg) > 0 and \
            isinstance(arg[0], fits.PrimaryHDU):
        tbl = arg
        filename = tbl[0]._file.name
    else:
        msg = "a binary FITS table or a {} list".format(exptype)
        # TODO: define Err in a more elegant way
        # raise IOErr('badfile', arg, msg)
        raise Exception('file or HUDList "%s"', msg)

    return (tbl, filename)


# Note: it is not really WCS specific, but leave the name alone for now.
def _get_wcs_key(hdu, key0, key1, fix_type=False, dtype=astroFloat):
    """Return the pair of keyword values as an array of values of
    the requested datatype. If either key is missing then return
    ().
    """

    if _has_key(hdu, key0) and _has_key(hdu, key1):
        return np.array([_try_key(hdu, key0, fix_type, dtype),
                         _try_key(hdu, key1, fix_type, dtype)], dtype)
    return ()


def _find_col_index(hdu, col_name):
    """
    Return the index of a column in a given hdu object

    :hdu: TODO
    :col_name: TODO
    :returns: TODO

    """
    col_names = hdu.columns.names
    try:
        return col_names.index(col_name)
    except:
        raise Exception('Column name is not found!')


def _find_binary_table(hdulist, filename, blockname=None):
    """
    Return the first binary table extension we find. If blockname
    is not None then the name of the block has to match (case-insensitive
    match), and any spaces are removed from blockname before checking.

    Throws an exception if there aren't any.

    :hdulist: an astropy HDUList object
    :filename: str, name of the file
    :blockname: str, name of the extension
    """

    if blockname is None:
        for hdu in hdulist:
            if isinstance(hdu, fits.BinTableHDU):
                return hdu

    else:
        blockname = str(blockname).strip().lower()
        for hdu in hdulist:
            if hdu.name.lower() == blockname \
               or isinstance(hdu, fits.BinTableHDU):
                return hdu

    raise Exception('Extension is not found in "s%"!', filename)


def _try_col(hdu, col_name, dtype=astroFloat, fix_type=False):
    """
    Check and read a column from a HDU object.

    :hdu: astropy hdu object
    :col_name: str, name of the column that will be checked/read
    :dtype: type of the values of the col_name
    :fix_type: if True, using the dtype as the finally output type
    :returns: if col_name is found in hdu, then return value of the column,
        otherwise return None

    """
    if col_name not in hdu.columns.names:
        return None

    col = hdu.data.field(col_name)

    if isinstance(col, _VLF):
        col = np.concatenate([np.asarray(row) for row in col])
    else:
        col = np.asarray(col).ravel()

    col_index = _find_col_index(hdu, col_name)
    col_index += 1
    col_form = 'TFORM' + str(col_index)
    t_form = _require_key(hdu, col_form)

    if t_form not in ['L', 'X', 'B', 'A']:
        col_scale = 'TSCAL' + str(col_index)
        col_zero = 'TZERO' + str(col_index)
        t_scale = _try_key(hdu, col_scale, True)
        t_zero = _try_key(hdu, col_zero, True)
        if t_scale is not None and t_zero is not None:
            col = col * t_scale + t_zero

    if fix_type:
        col = col.astype(dtype)

    return col


def _require_col(hdu, col_name, dtype=astroFloat, fix_type=False):
    """
    This function is to tell whether the required column name exists in the hdu

    :hdu: astropy hdu object
    :col_name: str, name of the column that required
    :dtype: type of the values of the col_name
    :fix_type: if True, using the dtype as the finally output type
    :returns: if col_name is found in hdu, then return value of the column read
        by the _try_col() function, otherwise raise an error

    """
    col = _try_col(hdu, col_name, dtype, fix_type)
    if col is None:
        # TODO: define error exception
        raise Exception('Required column "s%" does not exist in file "s%"',
                        col_name, hdu._file.name)
    return col


def _try_col_or_key(hdu, name, dtype=astroFloat, fix_type=False):
    """
    Check if name is a column of a key. If it's a column return the value
    of the column. Otherewise check if it's a key.

    :hdu: astropy hdu object
    :name: str, name of a column or key
    :dtype: type of the values of the col_name
    :fix_type: if True, using the dtype as the finally output type
    :return: if name is a column in hdu, then return value of that column,
        otherwise return the value of the key if name is a key or None
        if it is not.

    """
    col = _try_col(hdu, name, dtype, fix_type)

    if col is not None:
        return col

    return _try_key(hdu, name, fix_type, dtype)


def _try_vec(hdu, name, size=2, dtype=astroFloat, fix_type=False):
    if name not in hdu.columns.names:
        # return np.array([None] * size)
        return None

    col = hdu.data.field(name)

    # TODO: how to hand _VLF (variable length array) and its type?
    # if isinstance(col, _VLF):
    #     col = np.concatenate([np.asarray(row) for row in col])
    # else:
    #     col = np.asarray(col)
    if not isinstance(col, _VLF):
        col = np.asarray(col)
        if fix_type:
            col = col.astype(dtype)

    if col is None:
        # return np.array([None] * size)
        return None

    return col


def _require_vec(hdu, name, size=2, dtype=astroFloat, fix_type=False):
    col = _try_vec(hdu, name, size, dtype, fix_type)
    # if np.equal(col, None).any():
    if col is None:
        raise Exception('Required column name "s%" is not found in hdu "s%"!',
                        name, hdu._file.name)
    return col


def _try_vec_or_key(hdu, name, size, dtype=astroFloat, fix_type=False):
    col = _try_col(hdu, name, dtype, fix_type)
    if col is not None:
        return col
    return np.array([_try_key(hdu, name, fix_type, dtype)] * size)


def _is_ogip_type(hdulist, bltype, bltype2=None):
    """
    Return True if hdus[1] exists and has the given type (as determined by the
    HDUCLAS1 or HDUCLAS2 keywords). If bltype2 is None then bltype is used for
    both checks, otherwise bltype2 is used for HDUCLAS2 and bltype is for
    HDUCLAS1.

    :hdulist: astropy HDUList object
    :bltype: TODO
    :bltype2: TODO
    :return:

    """
    bnum = 1
    if bltype2 is None:
        bltype2 = bltype
    return _has_hdu(hdulist, bnum) and (
        _try_key(hdulist[bnum], 'HDUCLAS1') == bltype or
        _try_key(hdulist[bnum], 'HDUCLAS2') == bltype2)


def get_ascii_data(filename, make_copy=False):
    """Read an ASCII file using the astropy ascii.read() module.
        It will guess the file format as well as the file content.
        Returns an astropy table object, which can be used or filtered
        by other functions based on the general purpose.

    :filename: str, name of the ASCII file.
    :make_copy: TODO.
    :returns: astropy table object.

    """
    if is_binary_file(filename):
        raise Exception('File is not an ASCII file!')

    asc_content = asc.read(filename)

    return asc_content, filename


def get_image_data(arg, make_copy=False):
    """
    arg is a filename or a HDUList object
    """
    hdu, filename = _get_file_contents(arg)

    #   FITS uses logical-to-world where we use physical-to-world.
    #   For all transforms, update their physical-to-world
    #   values from their logical-to-world values.
    #   Find the matching physical transform
    #      (same axis no, but sub = 'P' )
    #   and use it for the update.
    #   Physical tfms themselves do not get updated.
    #
    #  Fill the physical-to-world transform given the
    #  logical-to-world and the associated logical-to-physical.
    #      W = wv + wd * ( P - wp )
    #      P = pv + pd * ( L - pp )
    #      W = lv + ld * ( L - lp )
    # Then
    #      L = pp + ( P - pv ) / pd
    # so   W = lv + ld * ( pp + (P-pv)/pd - lp )
    #        = lv + ( ld / pd ) * ( P - [ pv +  (lp-pp)*pd ] )
    # Hence
    #      wv = lv
    #      wd = ld / pd
    #      wp = pv + ( lp - pp ) * pd

    #  EG suppose phys-to-world is
    #         W =  1000 + 2.0 * ( P - 4.0 )
    #  and we bin and scale to generate a logical-to-phys of
    #         P =  20 + 4.0 * ( L - 10 )
    #  Then
    #         W = 1000 + 2.0 * ( (20-4) - 4 * 10 ) + 2 * 4 $
    #

    try:
        data = {}

        img = hdu[0]
        if hdu[0].data is None:
            img = hdu[1]
            if hdu[1].data is None:
                raise Exception('badimg')

        data['y'] = np.asarray(img.data)

        cdeltp = _get_wcs_key(img, 'CDELT1P', 'CDELT2P')
        crpixp = _get_wcs_key(img, 'CRPIX1P', 'CRPIX2P')
        crvalp = _get_wcs_key(img, 'CRVAL1P', 'CRVAL2P')
        cdeltw = _get_wcs_key(img, 'CDELT1', 'CDELT2')
        crpixw = _get_wcs_key(img, 'CRPIX1', 'CRPIX2')
        crvalw = _get_wcs_key(img, 'CRVAL1', 'CRVAL2')

        # proper calculation of cdelt wrt PHYSICAL coords
        if (isinstance(cdeltw, np.ndarray)
                and isinstance(cdeltp, np.ndarray)):
            cdeltw = cdeltw / cdeltp

        # proper calculation of crpix wrt PHYSICAL coords
        if (isinstance(crpixw, np.ndarray)
                and isinstance(crvalp, np.ndarray)
                and isinstance(cdeltp, np.ndarray)
                and isinstance(crpixp, np.ndarray)):
            crpixw = crvalp + (crpixw - crpixp) * cdeltp

        sky = None
        if (transformstatus and isinstance(cdeltp, np.ndarray)
                and isinstance(crpixp, np.ndarray)
                and isinstance(crvalp, np.ndarray)):
            sky = WCS('physical', 'LINEAR', crvalp, crpixp, cdeltp)

        eqpos = None
        if (transformstatus and isinstance(cdeltw, np.ndarray)
                and isinstance(crpixw, np.ndarray)
                and isinstance(crvalw, np.ndarray)):
            eqpos = WCS('world', 'WCS', crvalw, crpixw, cdeltw)

        data['sky'] = sky
        data['eqpos'] = eqpos
        data['header'] = img.header

        keys = ['MTYPE1', 'MFORM1', 'CTYPE1P', 'CTYPE2P', 'WCSNAMEP',
                'CDELT1P', 'CDELT2P', 'CRPIX1P', 'CRPIX2P',
                'CRVAL1P', 'CRVAL2P', 'MTYPE2', 'MFORM2', 'CTYPE1', 'CTYPE2',
                'CDELT1', 'CDELT2', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2',
                'CUNIT1', 'CUNIT2', 'EQUINOX']

        for key in keys:
            try:
                data['header'].pop(key)
            except KeyError:
                pass

    finally:
        hdu.close()

    return data, filename


def get_header_data(arg, blockname=None, hdrkeys=None):
    """Read in the header data."""

    tbl, filename = _get_file_contents(arg, exptype="BinTableHDU")

    hdr = {}
    try:
        hdu = _find_binary_table(tbl, filename, blockname)

        if hdrkeys is None:
            hdrkeys = hdu.header.keys()

        for key in hdrkeys:
            hdr[key] = _require_key(hdu, key, dtype=str)

    finally:
        tbl.close()

    return hdr


def get_pha_data(arg, make_copy=False, use_background=False):
    """
    This function will get the data from spectrum fits files.

    :arg: str, filename or a HDUList object
    :make_copy: boolean
    :use_background: boolean, whether a background will be used
    :returns: TODO
    """

    pha, filename = _get_file_contents(arg, exptype="BinTableHDU")

    try:
        if _has_hdu(pha, 'SPECTRUM'):
            hdu = pha['SPECTRUM']
        elif _is_ogip_type(pha, 'SPECTRUM'):
            hdu = pha[1]
        else:
            raise Exception('"s%" is not a PHA spectrum!', filename)
            # raise IOErr('notrsp', filename, "a PHA spectrum")

        if use_background:
            for block in pha:
                if _try_key(block, 'HDUCLAS2') == 'BKG':
                    hdu = block

        keys = ['BACKFILE', 'ANCRFILE', 'RESPFILE',
                'BACKSCAL', 'AREASCAL', 'EXPOSURE']
        datasets = []

        if _try_col(hdu, 'SPEC_NUM') is None:
            data = {}

            # Keywords
            data['exposure'] = _try_key(hdu, 'EXPOSURE', True, astroFloat)
            # data['poisserr'] = _try_key(hdu, 'POISSERR', True, bool)
            data['backfile'] = _try_key(hdu, 'BACKFILE')
            data['arffile'] = _try_key(hdu, 'ANCRFILE')
            data['rmffile'] = _try_key(hdu, 'RESPFILE')

            # Keywords or columns
            data['backscal'] = _try_col_or_key(hdu, 'BACKSCAL', fix_type=True)
            data['backscup'] = _try_col_or_key(hdu, 'BACKSCUP', fix_type=True)
            data['backscdn'] = _try_col_or_key(hdu, 'BACKSCDN', fix_type=True)
            data['areascal'] = _try_col_or_key(hdu, 'AREASCAL', fix_type=True)

            # Columns
            data['channel'] = _require_col(hdu, 'CHANNEL', fix_type=True)

            # Make sure channel numbers not indices
            chan = list(hdu.columns.names).index('CHANNEL') + 1
            tlmin = _try_key(hdu, 'TLMIN' + str(chan), True, astroUInt)
            if int(data['channel'][0]) == 0 or tlmin == 0:
                data['channel'] = data['channel'] + 1

            data['counts'] = _try_col(hdu, 'COUNTS', fix_type=True)
            if data['counts'] is None:
                data['counts'] = _require_col(hdu, 'RATE',
                                              fix_type=True) * data['exposure']
            data['staterror'] = _try_col(hdu, 'STAT_ERR')
            data['syserror'] = _try_col(hdu, 'SYS_ERR')
            data['background_up'] = _try_col(hdu, 'BACKGROUND_UP',
                                             fix_type=True)
            data['background_down'] = _try_col(hdu, 'BACKGROUND_DOWN',
                                               fix_type=True)
            data['bin_lo'] = _try_col(hdu, 'BIN_LO', fix_type=True)
            data['bin_hi'] = _try_col(hdu, 'BIN_HI', fix_type=True)
            data['grouping'] = _try_col(hdu, 'GROUPING', astroInt)
            data['quality'] = _try_col(hdu, 'QUALITY', astroInt)
            # Note that no _get_meta_data function in this script
            # data['header'] = _get_meta_data(hdu)
            data['header'] = hdu.header

            for key in keys:
                try:
                    data['header'].pop(key)
                except KeyError:
                    pass

            if data['syserror'] is not None:
                # SYS_ERR is the fractional systematic error
                data['syserror'] = data['syserror'] * data['counts']

            datasets.append(data)

        else:
            data = {}
            # Type 2 PHA file support

            specnum = _try_col_or_key(hdu, 'SPEC_NUM')
            num = len(specnum)

            # Keywords
            exposure = _try_key(hdu, 'EXPOSURE', True, astroFloat)
            # poisserr = _try_key(hdu, 'POISSERR', True, bool)
            backfile = _try_key(hdu, 'BACKFILE')
            arffile = _try_key(hdu, 'ANCRFILE')
            rmffile = _try_key(hdu, 'RESPFILE')

            # Keywords or columns
            backscal = _try_vec_or_key(hdu, 'BACKSCAL', num, fix_type=True)
            backscup = _try_vec_or_key(hdu, 'BACKSCUP', num, fix_type=True)
            backscdn = _try_vec_or_key(hdu, 'BACKSCDN', num, fix_type=True)
            areascal = _try_vec_or_key(hdu, 'AREASCAL', num, fix_type=True)

            # Columns
            channel = _require_vec(hdu, 'CHANNEL', num, fix_type=True)

            # Make sure channel numbers not indices
            chan = list(hdu.columns.names).index('CHANNEL') + 1
            tlmin = _try_key(hdu, 'TLMIN' + str(chan), True, astroUInt)

            for ii in range(num):
                if int(channel[ii][0]) == 0:
                    channel[ii] += 1

            # if ((tlmin is not None) and tlmin == 0) or int(channel[0]) == 0:
            #     channel += 1

            counts = _try_vec(hdu, 'COUNTS', num, fix_type=True)
            if None in counts:
                counts = _require_vec(hdu, 'RATE', num,
                                      fix_type=True) * data['exposure']
            staterror = _try_vec(hdu, 'STAT_ERR', num)
            syserror = _try_vec(hdu, 'SYS_ERR', num)
            background_up = _try_vec(hdu, 'BACKGROUND_UP', num, fix_type=True)
            background_down = _try_vec(hdu, 'BACKGROUND_DOWN', num,
                                       fix_type=True)
            bin_lo = _try_vec(hdu, 'BIN_LO', num, fix_type=True)
            bin_hi = _try_vec(hdu, 'BIN_HI', num, fix_type=True)
            grouping = _try_vec(hdu, 'GROUPING', num, astroInt)
            quality = _try_vec(hdu, 'QUALITY', num, astroInt)

            orders = _try_vec(hdu, 'TG_M', num, astroInt)
            parts = _try_vec(hdu, 'TG_PART', num, astroInt)
            specnums = _try_vec(hdu, 'SPEC_NUM', num, astroInt)
            srcids = _try_vec(hdu, 'TG_SRCID', num, astroInt)

            # Iterate over all rows of channels, counts, errors, etc
            # Populate a list of dictionaries containing
            # individual dataset info
            for (bscal, bscup, bscdn, arsc, chan, cnt, staterr, syserr,
                 backup, backdown, binlo, binhi, group, qual, ordr, prt,
                 specnum, srcid) in zip(backscal, backscup, backscdn, areascal,
                                        channel, counts, staterror, syserror,
                                        background_up, background_down, bin_lo,
                                        bin_hi, grouping, quality, orders,
                                        parts, specnums, srcids):
                data = {}

                data['exposure'] = exposure
                # data['poisserr'] = poisserr
                data['backfile'] = backfile
                data['arffile'] = arffile
                data['rmffile'] = rmffile

                data['backscal'] = bscal
                data['backscup'] = bscup
                data['backscdn'] = bscdn
                data['areascal'] = arsc

                data['channel'] = chan
                data['counts'] = cnt
                data['staterror'] = staterr
                data['syserror'] = syserr
                data['background_up'] = backup
                data['background_down'] = backdown
                data['bin_lo'] = binlo
                data['bin_hi'] = binhi
                data['grouping'] = group
                data['quality'] = qual
                # TODO
                # data['header'] = _get_meta_data(hdu)
                data['header'] = hdu.header
                data['header']['TG_M'] = ordr
                data['header']['TG_PART'] = prt
                data['header']['SPEC_NUM'] = specnum
                data['header']['TG_SRCID'] = srcid

                for key in keys:
                    try:
                        data['header'].pop(key)
                    except KeyError:
                        pass

                if syserr is not None:
                    # SYS_ERR is the fractional systematic error
                    data['syserror'] = syserr * cnt

                datasets.append(data)

    finally:
        pha.close()

    return datasets, filename


def get_lc_data(arg, make_copy=False, use_background=False):
    """This function will get the light curve data either from a HDUList/fits file
       or from a ascii file.

    :arg: str, filename of a HDUList object
    :make_copy:
    :use-backgrond:
    :returns:
        time: time of the light curve
        timedel: time interval/error
        counts: photo counts
        error: error of  photon counts
        flux: optional, if read from user defined ascii file
        flux_error: optional
        mjdref: mjd reference time
        tstart: start time
        tstop: stop time
        timezero: zero time
        timesys: time system
        timeunits: units of the time
        clockcor
        timeref: time reference system
        reftime: reference time for timeref
    """
    if not _is_fits_hudlist(arg):
        # Read user defined light curve data. The file should be in ascii
        # format. The content of the file is, however, very complicated. The
        # function should has the ability to guess the content of the file. At
        # best, the user should provide a file with headers which specifies the
        # columns of the data. In case the header is missing, the function will
        # try:
        #   t_start t_end    flux flux_err
        #   t_start t_end    cts
        #   t_start t_end    cts  cts_err
        #   t_start t_end    rate
        #   t_start t_end    rate rate_err
        #   time    time_err flux flux_err
        #   time    time_err cts
        #   time    time_err cts  cts_err
        #   time    time_err rate
        #   time    time_err rate rate_err
        lc, filename = get_ascii_data(arg)

        data = {}

        # Try to find the time related columns
        # try to find columns named with 'start' and 'end'
        tstart = _find_tbl_col(lc, 'start')
        tend = _find_tbl_col(lc, 'end')

        # try to find columns named as 'time' or 'date'
        if _find_tbl_col(lc, 'time') is not None:
            t_time = _find_tbl_col(lc, 'time')
        elif _find_tbl_col(lc, 'date') is not None:
            t_time = _find_tbl_col(lc, 'date')
        else:
            t_time = None

        if (tstart is not None) and (tend is not None):
            data['time'] = 0.5 * (_try_tbl_col(lc, tstart, True) +
                                  _try_tbl_col(lc, tend, True))
            data['timedel'] = (_try_tbl_col(lc, tend, True) -
                               _try_tbl_col(lc, tstart, True))
        elif t_time is not None:
            data['time'] = _try_tbl_col(lc, t_time, True)
            # find the time duration or exposure time
            if _find_tbl_col(lc, 'exposure') is not None:
                del_time = _find_tbl_col(lc, 'exposure')
            elif _find_tbl_col(lc, 'interval') is not None:
                del_time = _find_tbl_col(lc, 'interval')
            elif _find_tbl_col(lc, 'time_err') is not None:
                del_time = _find_tbl_col(lc, 'time_err')
            else:
                del_time = None
            if del_time is not None:
                data['timedel'] = _try_tbl_col(lc, del_time, True)
        else:
            print('Warning: can not find the time columns. The 1st and 2ed ' +
                  'columns will be used as the time and time interval!')
            data['time'] = _try_tbl_col(lc, 'col1', True)
            data['timedel'] = _try_tbl_col(lc, 'col2', True)

        # Try to find the intensity (flux/counts/rate) related columns
        col_counts = _find_tbl_col(lc, 'count')
        col_rate = _find_tbl_col(lc, 'rate')
        col_flux = _find_tbl_col(lc, 'flux')
        col_err = None

        if col_counts is not None:
            data['counts'] = _try_tbl_col(lc, col_counts, True)
            col_err = _find_tbl_col(lc, col_counts + '.err')
        elif col_rate is not None:
            data['counts'] = _try_tbl_col(lc, col_rate, True) * data['timedel']
            col_err = _find_tbl_col(lc, col_rate + '.err')
        else:
            data['counts'] = None
            data['error'] = None

        if col_err is not None:
            data['error'] = _try_tbl_col(lc, col_err, True)

        if col_flux is not None:
            data['flux'] = _try_tbl_col(lc, col_flux, True)
            col_flux_err = _find_tbl_col(lc, col_flux + '.err')
            if col_flux_err is not None:
                data['flux_error'] = _try_tbl_col(lc, col_flux_err, True)
        else:
            data['flux'] = None
            data['flux_error'] = None

        # if no columns are named 'counts', 'rate' or 'flux', then the 3rd and
        # 4th columns will be the flux and its error.
        if 'counts' not in data.keys() and 'flux' not in data.keys():
            print('Warning: can not find the counts/rate/flux columns. The ' +
                  '3rd and 4th columns will be used as the counts/rate/flux ' +
                  'and error!')
            intensity = _try_tbl_col(lc, 'col3', True)
            if max(intensity) < -6.0:
                data['flux'] = np.power(10, intensity)
            elif max(intensity) > 0.0 and max(intensity) < 1.0E-5:
                data['flux'] = intensity
            elif min(intensity) > 0:
                data['counts'] = intensity
            else:
                data['counts'] = np.power(10, intensity)
            data['error'] = _try_tbl_col(lc, 'col4', True)
    else:
        lc, filename = _get_file_contents(arg, exptype='BinTableHDU')
        try:
            if _has_hdu(lc, 'RATE'):
                hdu = lc['RATE']
            elif _has_hdu(lc, 'EVENTS'):
                hdu = lc['EVENTS']
            elif _is_ogip_type(lc, 'LIGHTCURVE'):
                hdu = lc[1]
            else:
                raise Exception('"s%" is not a valid light curve fits file!',
                                filename)

            if use_background:
                for block in lc:
                    if _try_key(block, 'HDUCLAS2') == 'BKG':
                        hdu = block

            data = {}

            # Time definition keywords, first search paired keywords, and then
            # single valued if paired keywords are not found
            mjdrefi = _try_key(hdu, 'MJDREFI', True, astroInt)
            mjdreff = _try_key(hdu, 'MJDREFF', True, astroFloat)
            tstarti = _try_key(hdu, 'TSTARTI', True, astroInt)
            tstartf = _try_key(hdu, 'TSTARTI', True, astroFloat)
            tstopi = _try_key(hdu, 'TSTOPI', True, astroInt)
            tstopf = _try_key(hdu, 'TSTOPF', True, astroFloat)
            timezeri = _try_key(hdu, 'TIMEZERI', True, astroInt)
            timezerf = _try_key(hdu, 'TIMEZERF', True, astroFloat)

            if mjdrefi is None or mjdreff is None:
                data['mjdref'] = _try_key(hdu, 'MJDREF', True, astroFloat)
            else:
                data['mjdref'] = mjdrefi + mjdreff

            if tstarti is None or tstartf is None:
                data['tstart'] = _try_key(hdu, 'TSTART', True, astroFloat)
            else:
                data['tstart'] = tstarti + tstartf

            if tstopi is None or tstopf is None:
                data['tstop'] = _try_key(hdu, 'TSTOP', True, astroFloat)
            else:
                data['tstop'] = tstopi + tstopf

            if timezeri is None or timezerf is None:
                data['timezero'] = _try_key(hdu, 'TIMEZERO', True, astroFloat)
            else:
                data['timezero'] = timezeri + timezerf

            # TODO: The following keywords are required. TIMESYS should be
            # 'MJD', 'JD' or 'TJD'. If it is defined from an arbitary UT time,
            # the value is given in decimal years. 'TIMEUNIT' is the unit for
            # 'TSTART', 'TSTOP' and 'TIMEZERO', can be 's' or 'd'. 'CLOCKCOR'
            # defines whether the time given has been corrected for any drift
            # in the spacecraft clock relative to UT, 'YES', 'NO', or'UNKNOWN'
            data['timesys'] = _require_key(hdu, 'TIMESYS')
            data['timeunit'] = _require_key(hdu, 'TIMEUNIT')
            data['clockcor'] = _try_key(hdu, 'CLOCKCOR')
            if data['clockcor'] is None:
                data['clockcor'] = _require_key(hdu, 'CLOCKAPP')

            # The 'TIME' column.
            # 'TIME' column is not necessary for equally binned data, but the
            # 'TIMEDEL' keyword must present.
            # 'COUNTS'/'RATE' not None:
            #   'TIME' None, 'TIMEDEL' not None: equally binned data
            #   'TIME' not None, 'TIMEDEL' not None: unequally binned data
            # 'COUNTS'/'RATE' None:
            #   'TIME' not None, 'TIMEDEL' None: EVENT list
            #   'TIME' not None, 'TIMEDEL' not None: Packet data, PHA required
            data['time'] = _try_col(hdu, 'TIME', fix_type=True)

            if data['time'] is None:
                data['timedel'] = _require_key(hdu, 'TIMEDEL', fix_type=True)
                n_row = _require_key(hdu, 'NAXIS2', True, astroInt)
                data['time'] = data['timezero'] + \
                    np.arange(n_row)*data['timedel']
            else:
                data['timedel'] = _try_col_or_key(hdu, 'TIMEDEL',
                                                  fix_type=True)

            # The intensity column, either 'RATE' or 'COUNTS'
            data['counts'] = _try_col(hdu, 'COUNTS', fix_type=True,
                                      dtype=astroInt)
            data['error'] = _try_col(hdu, 'ERROR', fix_type=True)

            if data['counts'] is None:
                rate = _try_col(hdu, 'RATE', fix_type=True)
                if rate is not None:
                    data['counts'] = rate * data['timedel']
                    if data['error'] is not None:
                        data['error'] = data['error'] * data['timedel']

            # The PHA column, in case of EVENTS or Packet data
            # data['pha'] = _try_col(hdu, 'PHA')

            # TODO
            # Barycentric corrections
            # The 'TIMEREF' keyword specifies in which reference frame the
            # times are calculated: 'TSTART', 'TSTOP', TIMEZERO', and times
            # given in the rate table. The values can be 'LOCAL',
            # 'SOLARSYSTEM', 'HELIOCENTRIC', 'GEOCENTRIC'.
            data['timeref'] = _require_key(hdu, 'TIMEREF')
            if data['timeref'] == 'LOCAL':
                if _try_col(hdu, 'REFSUN') is not None:
                    data['reftime'] = _try_col('REFSUN')
                elif _try_col(hdu, 'REFEARTH') is not None:
                    data['reftime'] = _try_col('REFEARTH')
                elif _try_col(hdu, 'REFEARTH') is not None:
                    data['reftime'] = _try_col('REFEARTH')

            # Background

            # GTI

            # Exposure

        finally:
            lc.close()

    return data, filename


def get_arf_data(arg, make_copy=False):
    """
    arg is a filename or a HDUList object
    """

    arf, filename = _get_file_contents(arg, exptype="BinTableHDU",
                                       nobinary=True)

    try:
        if _has_hdu(arf, 'SPECRESP'):
            hdu = arf['SPECRESP']
        elif _has_hdu(arf, 'AXAF_ARF'):
            hdu = arf['AXAF_ARF']
        elif _is_ogip_type(arf, 'SPECRESP'):
            hdu = arf[1]
        else:
            # TODO: define error exception
            raise Exception('"s%" is not an ARF file!', filename)

        data = {}
        data['exposure'] = _try_key(hdu, 'EXPOSURE', fix_type=True)
        data['energ_lo'] = _require_col(hdu, 'ENERG_LO', fix_type=True)
        data['energ_hi'] = _require_col(hdu, 'ENERG_HI', fix_type=True)
        data['specresp'] = _require_col(hdu, 'SPECRESP', fix_type=True)
        data['bin_lo'] = _try_col(hdu, 'BIN_LO', fix_type=True)
        data['bin_hi'] = _try_col(hdu, 'BIN_HI', fix_type=True)
        # data['header'] = _get_meta_data(hdu)
        data['header'] = hdu.header
        # data['header'].pop('EXPOSURE')
    finally:
        arf.close()

    return data, filename


def get_rmf_data(arg, make_copy=False):
    """arg is a filename or a HDUList object.

    Notes
    -----
    The RMF format is described in [1]_.

    References
    ----------

    .. [1] OGIP Calibration Memo CAL/GEN/92-002, "The Calibration
           Requirements for Spectral Analysis (Definition of RMF and
           ARF file formats)", Ian M. George1, Keith A. Arnaud,
           Bill Pence, Laddawan Ruamsuwan and Michael F. Corcoran,
           https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html

    """

    rmf, filename = _get_file_contents(arg, exptype="BinTableHDU",
                                       nobinary=True)

    try:
        if _has_hdu(rmf, 'MATRIX'):
            hdu = rmf['MATRIX']
        elif _has_hdu(rmf, 'SPECRESP MATRIX'):
            hdu = rmf['SPECRESP MATRIX']
        elif _has_hdu(rmf, 'AXAF_RMF'):
            hdu = rmf['AXAF_RMF']
        elif _is_ogip_type(rmf, 'RESPONSE', bltype2='RSP_MATRIX'):
            hdu = rmf[1]
        else:
            raise Exception('"s%" is not an RMF', filename)

        data = {}

        data['detchans'] = astroUInt(_require_key(hdu, 'DETCHANS'))
        data['energ_lo'] = _require_col(hdu, 'ENERG_LO', fix_type=True)
        data['energ_hi'] = _require_col(hdu, 'ENERG_HI', fix_type=True)
        data['n_grp'] = _try_key(hdu, 'N_GRP', fix_type=True,
                                 dtype=astroUInt)
        if data['n_grp'] is None:
            data['n_grp'] = _require_col(hdu, 'N_GRP', fix_type=True,
                                         dtype=astroUInt)
        # TODO: check why?
        data['f_chan'] = _require_vec(hdu, 'F_CHAN', fix_type=True,
                                      dtype=astroUInt)
        data['n_chan'] = _require_vec(hdu, 'N_CHAN', fix_type=True,
                                      dtype=astroUInt)
        # data['f_chan'] = hdu.data.field('F_CHAN')
        # data['n_chan'] = hdu.data.field('N_CHAN')
        # Read MATRIX as-is -- we will flatten it below, because
        # we need to remove all rows corresponding to n_grp[row] == 0
        data['matrix'] = None
        if 'MATRIX' not in hdu.columns.names:
            pass
        else:
            data['matrix'] = hdu.data.field('MATRIX')

        # data['header'] = _get_meta_data(hdu)
        data['header'] = hdu.header
        data['header'].pop('DETCHANS')

        # Beginning of non-Chandra RMF support
        fchan_col = list(hdu.columns.names).index('F_CHAN') + 1
        tlmin = _try_key(hdu, 'TLMIN' + str(fchan_col), True, astroUInt)
        naxis2 = _require_key(hdu, 'NAXIS2', fix_type=True, dtype=astroUInt)
        lo_thres = _try_key(hdu, 'LO_THRES', fix_type=True, dtype=astroFloat)
        if lo_thres is None:
            lo_thres = 0.0

        if tlmin is not None:
            data['offset'] = tlmin
        else:
            # QUS: should this actually be an error, rather than just
            #      something that is logged to screen?
            print("Failed to locate TLMIN keyword for F_CHAN" +
                  " column in RMF file '%s'" +
                  'Update the offset value in the RMF data set to' +
                  ' appropriate TLMIN value prior to fitting')

        if _has_hdu(rmf, 'EBOUNDS'):
            hdu = rmf['EBOUNDS']
            data['e_min'] = _try_col(hdu, 'E_MIN', fix_type=True)
            data['e_max'] = _try_col(hdu, 'E_MAX', fix_type=True)

            # Beginning of non-Chandra RMF support
            chan_col = list(hdu.columns.names).index('CHANNEL') + 1
            tlmin = _try_key(hdu, 'TLMIN' + str(chan_col), True, astroUInt)
            if tlmin is not None:
                data['offset'] = tlmin

        else:
            data['e_min'] = None
            data['e_max'] = None
    finally:
        rmf.close()

    # change the data['matrix'] from one dimensional to two dimension
    # numpy array
    # nonzero_grp = (data['n_grp'] > 0)
    nonzero_grp = np.nonzero(data['n_grp'])[0]
    n_grp = data['n_grp'][nonzero_grp]
    n_chan = data['n_chan'][nonzero_grp]
    f_chan = data['f_chan'][nonzero_grp]
    rm = np.full((naxis2, data['detchans']), lo_thres, dtype=astroFloat)
    if lo_thres > 0.0:
        for i, ngrp, fch_list, nch_list in zip(nonzero_grp, n_grp, f_chan,
                                               n_chan):
            # for a fixed-length f_chan and n_chan value, the true number of
            # elements is detemined by n_grp
            # print(i, fch_list, nch_list, ngrp)
            if np.size(fch_list) == 1:
               fch_list = [fch_list]
            if np.size(nch_list) == 1:
               nch_list = [nch_list]
            fch_list = fch_list[0:ngrp]
            nch_list = nch_list[0:ngrp]
            n_init = astroUInt(0)
            for fch, nch in zip(fch_list, nch_list):
                end_ch = int(fch + nch - data['offset'])
                start_ch = int(fch - data['offset'])
                n_end = int(n_init + nch)
                # print(i, fch, nch, data['offset'], start_ch, end_ch, n_init,
                #       n_end, np.shape(rm), np.shape(data['matrix'][i]))
                rm[i, start_ch:end_ch] = data['matrix'][i][n_init:n_end]
                n_init = n_end
    else:
        rm = data['matrix']
    data['matrix'] = rm.astype(astroFloat)

    if data['f_chan'].ndim > 1 and data['n_chan'].ndim > 1:
        f_chan = []
        n_chan = []
        for grp, fch, nch, in zip(data['n_grp'], data['f_chan'],
                                  data['n_chan']):
            for i in range(grp):
                f_chan.append(fch[i])
                n_chan.append(nch[i])

        data['f_chan'] = np.asarray(f_chan, astroUInt)
        data['n_chan'] = np.asarray(n_chan, astroUInt)
    else:
        if len(data['n_grp']) == len(data['f_chan']):
            # filter out groups with zeroes.
            good = (data['n_grp'] > 0)
            data['f_chan'] = data['f_chan'][good]
            data['n_chan'] = data['n_chan'][good]

    return data, filename


def _check_filename(filename):
    """
    Check if a file exists

    :filename: name of the file
    :returns: file name if exists, otherwise raise an error

    """
    f_name = filename
    if not os.path.exists(f_name):
        f_name += '.gz'
        if not os.path.exists(f_name):
            raise FileNotFoundError

    return f_name


def open_fits(filename):
    """
    Try and open filename as a FITS file.

    :filename : str, name of the FITS file to open. If the file
       can not be opened then '.gz' is appended to the
       name and the attempt is tried again.
    :returns: return value from the `astropy.io.fits.open`
       function.

    """
    f_name = _check_filename(filename)
    return fits.open(f_name)
