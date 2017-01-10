import os
import numpy as np
from StringIO import StringIO


def loadsxm(filename, readdata=True):
    """ Load a Nanonis file (.sxm)

    and retrieve scan parameters and channels informations.
    """

    #def variables
    header = {}

    #check if file exists and open if so
    if os.path.exists(filename) and os.path.isfile(filename):
        fn = open(filename, mode='rb')
    else:
        print('File does not exist.')
        return
    #check if file is a Nanonis data file
    s1 = fn.readline().strip()
    if s1 != ':NANONIS_VERSION:':
        print('File seems not to be a Nanonis file.')
        return
    header['version'] = np.int(fn.readline().strip())
    read_tag = 1

    #The header consists of key-value pairs. Usually the key is on one line,
    #embedded in colons e.g. :SCAN_PIXELS:, the next line contains the value.
    #Some special keys may have multi-line values (like :COMMENT:),
    #in this case read value until next key is detected (line starts with
    #a colon) and set read_tag to 0 (because key has been read already).
    while 1:
        if read_tag:
            s1 = fn.readline().strip()
        s1 = s1.strip(':')  # remove leading and trailing colon
        read_tag = 1
        #strings
        if s1 in ['SCANIT_TYPE', 'REC_DATE', 'REC_TIME',
                  'SCAN_FILE', 'SCAN_DIR']:
            s2 = fn.readline().strip()
            header[s1.lower()] = s2
        #comment
        elif s1 == 'COMMENT':
            s_com = ''
            s2 = fn.readline().strip()
            while not s2.startswith(':'):
                s_com = s_com + s2 + u'\n'
                s2 = fn.readline().strip()
            header[s1.lower()] = s_com.strip()
            s1 = s2
            read_tag = 0  # already read next key (tag)
        #Z-controller settings
        elif s1 == 'Z-CONTROLLER':
            header['z_ctrl_tags'] = fn.readline().strip().split('\t')
            header['z_ctrl_values'] = fn.readline().strip().split('\t')
        #numbers
        elif s1 in ['BIAS', 'REC_TEMP', 'ACQ_TIME', 'SCAN_ANGLE']:
            s2 = fn.readline()
            header[s1.lower()] = np.float(s2)
        #array to number
        elif s1 in ['SCAN_PIXELS', 'SCAN_TIME', 'SCAN_RANGE', 'SCAN_OFFSET']:
            s2 = fn.readline()
            header[s1.lower()] = [np.float(s) for s in s2.split()]
        #data info
        elif s1 == 'DATA_INFO':
            data_info_tags = fn.readline().strip().split('\t')
            s2 = fn.readline().strip()
            data_info = ''
            while len(s2) > 2:
                data_info += s2 + u'\n'
                s2 = fn.readline().strip()
            data_info = np.asarray(np.genfromtxt(StringIO(data_info),
                                                 dtype='i,S64,S8,S10,f8,f8',
                                                 delimiter="\t"))
            header['data_info'] = {}
            for index, tag in enumerate(data_info_tags):
                header['data_info'][tag] = [line[index] for line in data_info]
            s1 = s2
            read_tag = 0  # already read next key (tag)
        #end of scan
        elif s1 == 'SCANIT_END':
            # this sets the pointer where the data start
            pointer = fn.tell() + 4
            break
        else:  # treat as strings, but try to convert to float
            if '>' in s1:
                father, son = s1.split('>')
                if father not in header.keys():
                    header[father] = {}
                s2 = fn.readline().strip()
                if ';' in s2:
                    s2 = np.asarray(s2.split(';'))
                    try:
                        s2 = s2.astype(np.float)
                    except:
                        # not a nice way of doing it,
                        # but it does the job if s2 is a string
                        pass

                else:
                    try:
                        s2 = np.float(s2)
                    except:
                        pass
                        # same as above,
                        # try to convert to float if not possible
                        # leave it as a string
                header[father][son] = s2

    if 0:  # print header
        for key in header.keys():
            print key + ':', header[key]

    #calculate number of images
    direction = header['data_info']['Direction']
    # channel_nb takes into account if for some channels not both scan
    # directions are saved. BUT this is not taken into account for the
    # correction of the arrays further down -> sort data
    channel_nb = len(direction) + \
                 len(np.where(np.array(direction) == 'both')[0])

    pixs = header['scan_pixels']
    fn.seek(pointer)  # set file pointer at start position of data
    data = np.fromfile(fn, dtype='>f4')  # read data in a block
    fn.close()  # close file

    data = data.reshape((channel_nb, pixs[1], pixs[0]))  # reshape the data
    # sort data
    scn_dir = header['scan_dir']
    for ch in range(channel_nb)[1::2]:
        data[ch, :, :] = data[ch, :, ::-1]  # flip every (n+1) channel left right
    if scn_dir == 'down':
        data = data[:, ::-1, :]  # flip data up down

    return data.astype(np.double), header

if __name__ == "__main__":
    filename = '/Volumes/Data/Users/bziegler/Documents/phd/data/afm/MultiscalingPaper/stripe_data/b11165s_044.sxm'
    data, header = loadsxm(filename)

    from matplotlib import pyplot as plt
    plt.imshow(data[1, :, :] / np.max(data[1]) * 100)
    plt.show()
