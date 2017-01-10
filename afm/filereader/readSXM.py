from nanonisFileReader import loadsxm
from stdAFMvals import AFMImage
import numpy as np


def load_file(filename, metaout=False):
    '''
    Load a Nanonis file (.sxm) and retrieve scan informations.
    
    
    Parameters
    ----------
    filename : string
        Filename
    metaout : bool, optional
        if True, meta information are returned

    Returns
    -------
    afm_image : AFMImage
        AFMImage class contains the relevent image information, the data,
        and the type sepcific meta data if asked for (metaout=True)    
    '''

    # IMPORT .sxm FILE WITH LOADIBW
    data, metadata = loadsxm(filename)
    # data contains the different channel data

    # create list of channelnames
    channels = metadata['data_info']['Name']
    units = metadata['data_info']['Unit']
    channel_names = []
    channel_units = []
    for channel, unit in zip(channels, units):
        channel_names.append(channel + '_Trace')
        channel_names.append(channel + '_Retrace')
        channel_units.append(unit)
        channel_units.append(unit)

    # RETRIEVE SCAN PARAMETERS FROM FILE NOTES
    afm_image = AFMImage()
    afm_image.pixels_x = np.int(metadata['scan_pixels'][0])
    afm_image.pixels_y = np.int(metadata['scan_pixels'][1])
    afm_image.scan_center_x = np.double(metadata['scan_offset'][0])
    afm_image.scan_center_y = np.double(metadata['scan_offset'][1])
    afm_image.scansize_x = np.double(metadata["scan_range"][0])
    afm_image.scansize_y = np.double(metadata['scan_range'][1])
    afm_image.scanangle = np.double(metadata['scan_angle'])
    afm_image.channelnames = channel_names
    afm_image.channelunits = channel_units
    afm_image.notes = metadata['comment']
    afm_image._calc_pixSize()
    afm_image.data = data
    afm_image.filename = filename
    afm_image.filetype = 'sxm'
    afm_image.imagingmode = metadata['Z-Controller']['Controller name']
    afm_image.date = metadata['rec_date']
    afm_image.time = metadata['rec_time']
    if metaout:
        afm_image.metadata = metadata

    return afm_image
