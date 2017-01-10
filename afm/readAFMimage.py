# -*- coding: utf-8 -*-
# created by Benedikt Ziegler
# date: 28 November 2012

from os import path
from filereader import readIBW, readSXM, readWSxM, readNanoscope


def load_file(filename, metaout=True):
    """
    Function to open afm images

    This function grabs the relevant afm image information, data,
    and the meta information saved in the corresponding filetype.
    
    Supported afm file types:
        - IGOR Binary Wave files (.ibw)
        - NANONIS files (.sxm)
        - WSxM text files (.txt)
        - Nanoscope files (.000, .001, etc)
    
    
    Parameters
    ----------
    filename : string
        Filename
    metaout : bool, optional
        if True, meta information are returned

    Returns
    -------
    main_values : AFMImage
        AFMImage class contains the relevent image information, the data,
        and the type sepcific meta data if asked for (metaout=True)    
    """

    # check if filename is an existing file
    filename = path.realpath(filename)
    if not path.isfile(filename):
        raise IOError('This is not a file!')
    fname, fext = path.splitext(filename)

    if fext == '.ibw':
        return readIBW.load_file(filename, metaout)
    elif fext == '.sxm':
        return readSXM.load_file(filename, metaout)
    elif fext == '.txt':
        return readWSxM.load_file(filename, metaout)
    elif int(fext[1:]) in xrange(1000):
        return readNanoscope.load_file(filename, metaout)
    else:
        raise IOError('File extention is not supported!')


def pick_channel(image):
    """
    A command line tool to show and pick a channel by name.

    Parameters
    ----------
    image : image class

    Returns
    -------
    data : 2d array
    """
    chnames = image.channelnames
    exit = False
    reps = 0
    while not exit and reps < 10:
        for i, chn in enumerate(chnames + ['Abort!']):
            print('[{}] {}'.format(i, chn))
        nb = int(raw_input('Pick a channel by number: '))
        if nb < len(chnames):
            data = image.data[nb]
            print('You picked: {}'.format(chnames[nb]))
            exit = True
        elif nb == len(chnames):
            exit = True
        else:
            if reps == 8:
                strout = 'Last try!!!'
            else:
                strout = 'Try again...'
            print('Number does not correspond to a channel! {}'.format(strout))
        reps += 1
    return data