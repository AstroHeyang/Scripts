#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path
import numpy as np

from ..astroio import backend
from ..astrodata import DataARF, DataRMF, DataPHA, DataLC

__all__ = ('read_arf', 'read_rmf', 'read_pha', 'read_lc')


def read_arf(arg):
    """
    read_arf( filename )
    read_arf( ARFCrate )

    """
    data, filename = backend.get_arf_data(arg)
    return DataARF(filename, **data)


def read_rmf(arg):
    """
    read_rmf( filename )
    read_rmf( RMFCreate )

    """
    data, filename = backend.get_rmf_data(arg)
    return DataRMF(filename, **data)


def _read_ancillary(data, key, label, dname,
                    read_func, output_once=True):
    """
    Read the ancillary files, i.e. ARF, RMF, background.
    If the corresponding keys are defined in the PHA data
    set, those values will then be used for searching the
    files.
    """

    if not(data[key]) or data[key].lower() == 'none':
        return None

    ancil_content = None
    try:
        if os.path.dirname(data[key]) == '':
            data[key] = os.path.join(dname, data[key])

        ancil_content = read_func(data[key])
        if output_once:
            print('Read {0} file: {1}'.format(label, data[key]))
    except:
        if output_once:
            print("Warning: cannot find the file")

    return ancil_content


def read_pha(arg, use_errors=False, use_background=False):
    """Create a DataPHA object.

    read_pha( filename [, use_errors=False, [, use_background=False]] )
    read_pha( PHACrate [, use_errors=False, [, use_background=False]] )

    Parameters
    __________
    arg: The name of the file or a representation of the file (the type depends
        on the I/O backend) containing the PHA data.
    use_errors : bool, optional, If the PHA file contains statistical error
        values for the count (or count rate) column, should it be read in.
        This defaults to ``False``.
    use_background : bool, optional, Should the background PHA data (and
        optional responses) also be read in and associated with the data set?

    Returns
    -------
    data : sherpa.astro.data.DataPHA

    """

    datasets, filename = backend.get_pha_data(arg,
                                              use_background=use_background)
    phasets = []
    output_once = True
    for data in datasets:
        if not use_errors:
            if data['staterror'] is not None or data['syserror'] is not None:
                if data['staterror'] is None:
                    msg = 'systematic'
                elif data['syserror'] is None:
                    msg = 'statistical'
                    if output_once:
                        wmsg = "systematic errors were not found in " + \
                                "file '{}'".format(filename)
                        print("Warning:" + wmsg)
                else:
                    msg = 'statistical and systematic'
                if output_once:
                    imsg = msg + " errors were found in file " + \
                            "'{}' \nbut not used; ".format(filename) + \
                            "to use them, re-read with use_errors=True"
                    print("Info:" + imsg)
                data['staterror'] = None
                data['syserror'] = None

        dname = os.path.dirname(filename)
        albl = 'ARF'
        rlbl = 'RMF'
        if use_background:
            albl = albl + ' (background)'
            rlbl = rlbl + ' (background)'

        arf = _read_ancillary(data, 'arffile', albl, dname, read_arf,
                              output_once)
        rmf = _read_ancillary(data, 'rmffile', rlbl, dname, read_rmf,
                              output_once)

        backgrounds = []
        if data['backfile'] and data['backfile'].lower() != 'none':
            try:
                if os.path.dirname(data['backfile']) == '':
                    data['backfile'] = os.path.join(os.path.dirname(filename),
                                                    data['backfile'])

                bkg_datasets = []

                if not use_background:
                    bkg_datasets = read_pha(data['backfile'], use_errors, True)

                    if output_once:
                        print('Read background file')

                if np.iterable(bkg_datasets):
                    for bkg_dataset in bkg_datasets:
                        if bkg_dataset.get_response() == (None, None) and \
                           rmf is not None:
                            bkg_dataset.set_response(arf, rmf)
                        backgrounds.append(bkg_dataset)
                else:
                    if bkg_datasets.get_response() == (None, None) and \
                       rmf is not None:
                        bkg_datasets.set_response(arf, rmf)
                    backgrounds.append(bkg_datasets)
            except:
                if output_once:
                    print('Warning: read background wrong')

        for bkg_type, bscal_type in zip(('background_up', 'background_down'),
                                        ('backscup', 'backscdn')):
            if data[bkg_type] is not None:
                b = DataPHA(filename,
                            channel=data['channel'],
                            counts=data[bkg_type],
                            bin_lo=data['bin_lo'],
                            bin_hi=data['bin_hi'],
                            grouping=data['grouping'],
                            quality=data['quality'],
                            exposure=data['exposure'],
                            backscal=data[bscal_type],
                            header=data['header'])
                b.set_response(arf, rmf)
                if output_once:
                    print('Read backgrounds into a dataset from file')
                backgrounds.append(b)

        for k in ['backfile', 'arffile', 'rmffile', 'backscup', 'backscdn',
                  'background_up', 'background_down']:
            data.pop(k, None)

        pha = DataPHA(filename, **data)
        pha.set_response(arf, rmf)
        for id, b in enumerate(backgrounds):
            if b.grouping is None:
                b.grouping = pha.grouping
                b.grouped = (b.grouping is not None)
            if b.quality is None:
                b.quality = pha.quality
            pha.set_background(b, id + 1)

        pha._set_initial_quantity()
        phasets.append(pha)
        output_once = False

    if len(phasets) == 1:
        phasets = phasets[0]

    return phasets


def read_lc(arg):
    data, filename = backend.get_lc_data(arg)
    # print(data, filename)
    return DataLC(filename, **data)


