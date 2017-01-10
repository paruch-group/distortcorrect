from numpy import genfromtxt, dstack, rollaxis
from stdAFMvals import AFMImage, convert_length2m
import numpy as np


def check_filetype(filename):
    """
    Check if file is a WSxM ASCII Matrix file
    
    
    Parameters
    ---------_
    filename : string
    
    Returns
    -------
    out : bool
    """
    
    with open(filename, 'r') as f:
        if f.readline().strip() != 'WSxM file copyright Nanotec Electronica':
            out = False
        if f.readline().strip() != 'WSxM ASCII Matrix file':
            out = False
        else:
            out = True
    return out


def load_file(filename, metaout=False):
    """
    Load a WSxM text files (.txt) and retrieve scan informations.
    
    
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
    
    if not check_filetype(filename):
        raise TypeError('File is not a WSxM file or not a WSxM ASCII Matrix file!')
    with open(filename, "r") as f:
        f.readline()  # skip first line
        f.readline()  # skip second line
        l = f.readline()
        scansize_x, unit_x = l.strip().split(" ")[2:]
        scansize_x = convert_length2m(np.double(scansize_x), unit_x)
        l = f.readline()
        scansize_y, unit_y = l.strip().split(" ")[2:]
        scansize_y = convert_length2m(np.double(scansize_y), unit_y)
        data = genfromtxt(f, skip_header=1)
        data = dstack(data)
        data = rollaxis(data, -1)
        lines, points = data.shape

    # RETURN SCAN PARAMETERS AND CHANNEL INFOS
    afm_image = AFMImage()
    afm_image.pixels_x = np.int(points)
    afm_image.pixels_y = np.int(lines)
    afm_image.scansize_x = np.double(scansize_x)
    afm_image.scansize_y = np.double(scansize_y)
#    afm_image.scanangle = None
    afm_image.channelnames = ["arbitrary"]
    afm_image.channelunits = ["arbitrary"]
    afm_image._calc_pixSize()
    afm_image.data = data
    afm_image.filename = filename
    afm_image.filetype = 'WSxM txt'
    if metaout:
        afm_image.metadata = []

    return afm_image