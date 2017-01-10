import numpy as np
from stdAFMvals import AFMImage


def convert_length_to_SI_unit(value, unit):
    unit_dic = {'am': 1e-18,
                'fm': 1e-15,
                'pm': 1e-12,
                'nm': 1e-9,
                '~m': 1e-6,
                'mm': 1e-3,
                'cm': 1e-2,
                'dm': 1e-1,
                'm': 1.,
                'km': 1e3}

    if unit not in unit_dic.keys():
        scale = 1
        print('The unit type is not supported!')
    else:
        scale = unit_dic[unit]
    return value * scale

def load_nanoscope(filename):
    """
    Load a Nanoscope afm measurement files (.000, .001, etc.) and 
    retrieve scan informations.
    
    Parameters
    ----------
    filename : string
        Filename
    
    Returns
    -------
    data : array-like
        image data
    scan_info : dictonary
        scan information
    image_infos : list
        image information
    header : dictonary
        information contained in the header    
    """

    fn = open(filename, 'rb')

    file_str = fn.read()
    fn.close()
    # SEPARATE HEADER FROM DATA
    header = file_str.split('File list end\r\n')[0]
    # RETRIEVE SCAN PARAMETERS FROM FILE HEADER
    scan_info = extract_scan_info_from_header(header)
    image_infos = extract_image_infos_from_header(header)

    # extract the image data
    data_offset = image_infos[0]['Data offset']
    data_orig = np.frombuffer(file_str, dtype='<h', offset=data_offset)
    pixels = scan_info['Samps/line']
    lines = scan_info['Lines']
    data = np.zeros((len(image_infos), pixels, lines))
    start = 0
    for chan in range(len(image_infos)):
        tempX = int(image_infos[chan]['Valid data len X'])
        tempY = int(image_infos[chan]['Valid data len Y'])
        tempD = data_orig[start:start+tempX*tempY].reshape((tempY, tempX))
        data[chan][0:tempY,0:tempX] = tempD
        start = start+tempX*tempY


    return data, scan_info, image_infos, header

def load_nanoscope_old(filename):
    """
    Load a Nanoscope afm measurement files (.000, .001, etc.) and 
    retrieve scan informations.
    
    Parameters
    ----------
    filename : string
        Filename
    
    Returns
    -------
    data : array-like
        image data
    scan_info : dictonary
        scan information
    image_infos : list
        image information
    header : dictonary
        information contained in the header    
    """

    fn = open(filename, 'rb')

    file_str = fn.read()
    fn.close()
    # SEPARATE HEADER FROM DATA
    header = file_str.split('File list end\r\n')[0]
    # RETRIEVE SCAN PARAMETERS FROM FILE HEADER
    scan_info = extract_scan_info_from_header(header)
    image_infos = extract_image_infos_from_header(header)

    # extract the image data
    data_offset = image_infos[0]['Data offset']
    data = np.frombuffer(file_str, dtype='<h', offset=data_offset)
    pixels = scan_info['Samps/line']
    lines = scan_info['Lines']
    print pixels, lines
    data = data.reshape((len(image_infos), pixels, lines))

    return data, scan_info, image_infos, header


def extract_scan_info_from_header(header):
    scan_header = header.split('\*Ciao scan list')[1].split('\*')[0]
    scan_header = scan_header.strip().split('\r\n')
    scan_dic = {}
    for line in scan_header:
        line = line.strip('\\')
        # discard lines starting with an '@' for now
        if line.startswith('@'):
            continue
        try:
            key, value = line.split(': ')
        except:
            continue
        if key in ['Samps/line', 'Lines']:
            value = int(value)
        elif key in ['Rotate Ang.', 'Scan rate', 'Tip velocity']:
            value = float(value)
        elif key in ['Scan Size', 'X Offset', 'Y Offset']:
            value, unit = value.split(' ')
            value = convert_length_to_SI_unit(float(value), unit)

        scan_dic[key] = value
    return scan_dic


def extract_image_infos_from_header(header):
    header_of_images = header.split('\*Ciao image list\r\n')[1:]
    image_infos = []
    for head in header_of_images:
        head = head.strip().split('\r\n')
        image_infos.append(extract_image_info_from_header(head))
    return image_infos

def extract_image_info_from_header(header):
    header_dic = {}
    for line in header:
        line = line.strip('\\')
        # discard lines starting with an '@' for now
        if line.startswith('@'):
            continue
        try:
            key, value = line.split(': ')
        except:
            continue

        if key in ['Data offset', 'Data length', 'Bytes/pixel', 'Samps/line',
                'Number of lines']:
            value = int(value)
        elif key in ['Scan Size']:
            value_x, value_y, unit = value.split(' ')
            value_x = convert_length_to_SI_unit(float(value_x), unit)
            value_y = convert_length_to_SI_unit(float(value_y), unit)
            value = np.array([value_x, value_y], dtype='float')

        header_dic[key] = value

    return header_dic


def load_file(filename, metaout=False):
    """
    Load a Nanoscope afm measurement files (.000, .001, etc.) and 
    retrieve scan informations.
    
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

    data, scan_info, image_infos, metadata = load_nanoscope(filename)

    channel_names = []
    channel_units = []
    for index, channel in enumerate(image_infos):
        if 'Image Data' in channel.keys():
            channel_names.append(channel['Image Data'])
        else:
            channel_names.append('Channel {:02}'.format(index))

    note = ''
    if 'Note' in image_infos[0].keys():
        note = image_infos[0]['Note']

    # RETRIEVE SCAN PARAMETERS FROM FILE NOTES
    afm_image = AFMImage()
    afm_image.pixels_x = np.int(scan_info['Samps/line'])
    afm_image.pixels_y = np.int(scan_info['Lines'])
    afm_image.scan_center_x = np.float(scan_info['X Offset'])
    afm_image.scan_center_y = np.float(scan_info['Y Offset'])
    afm_image.scansize_x = np.float(image_infos[0]['Scan Size'][0])
    afm_image.scansize_y = np.float(image_infos[0]['Scan Size'][1])
    afm_image.scanangle = np.float(scan_info['Rotate Ang.'])
    afm_image.channelnames = channel_names
    afm_image.channelunits = channel_units
    afm_image.notes = note
    afm_image._calc_pixSize()
    afm_image.data = data
    afm_image.filename = filename
    afm_image.filetype = 'Nanoscope'
    if metaout:
        afm_image.metadata = metadata

    return afm_image
