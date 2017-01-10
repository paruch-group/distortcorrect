from igorbinarywave import loadibw
from stdAFMvals import AFMImage, ibwChannel2Unit
import numpy as np


def load_file(filename, metaout=False):
    """
    Load a Asylum Research Cypher (.ibw) measurement file
    and retrieve scan informations.


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
    """

    # IMPORT .IBW FILE WITH LOADIBW
    data, bin_info = loadibw(filename)[:2]
    # data contains the different channel data

    # RETRIEVE SCAN PARAMETERS FROM FILE NOTES
    note = bin_info['note'].splitlines()
    for i, line in enumerate(note):
        note[i] = line.split(':', 1)
    # create a dictionary from 'note'
    metadata = {}
    for line in note:
        if len(line) == 2:
            metadata[str(line[0])] = line[1]
        elif line == '':
            pass
        elif len(line) == 1:
            metadata[str(line[0])] = ''
        else:
            pass

    # Check if file is an image or force curve
    isforce = metadata.get('IsForce', 0)
    isimage = metadata.get('IsImage', 0)
    # OK, it is an image if we have 3 dimensions
    # or if it is >= 128 in dim 0
    # or if it is >= 16 in dim 1
    if isforce + isimage == 0:
        if (data.shape[0] > 128) and (data.shape[1] < 16) and (len(data.shape) == 2):
            isforce = 1
        else:
            isimage = 1

    # swap axes to fit format [channel, slowscan, fastscan]
    if isimage:
        data = data.swapaxes(2, 0) #.swapaxes(2, 1)

        try:
            metadata['Flatten Offsets 0'] = np.fromstring(metadata['Flatten Offset 0'], sep=',')
        except:
            metadata['Flatten Offsets 0'] = 0
        try:
            metadata['Flatten Slopes 0'] = np.fromstring(metadata['Flatten Slopes 0'], sep=',')
        except:
            metadata['Flatten Slopes 0'] = 1

        channel_names = bin_info['dimLabels'][2]

        # CHECK IF NUMBER OF channel_names = NUMBER OF channels
        if len(channel_names) != data.shape[0]:
            for item in channel_names:
                if item in ['e', 'ce', 'ace', 'race', 'trace', 'etrace']:
                    channel_names.remove(item)
            # print 'Error in channel_names list! Auto correct.'
        if len(channel_names) != data.shape[0]:
            raise Warning('Unknown error in channel_names list!')

        if '@' in metadata['ScanAngle']:
            metadata['ScanAngle'] = metadata['ScanAngle'].split('@')[0]

        # get imaging mode
        imagingmode = metadata['ImagingMode']
        # which type of DART mode
        if imagingmode == 3:
            infastinput = metadata['InFast']
            if 'ACDefl' in infastinput:
                imagingmode = 'Vertical DART'
            elif 'PogoIn0' in infastinput:
                imagingmode = 'Lateral DART'

        channel_units = []
        for chn in channel_names:
            channel_units.append(ibwChannel2Unit(chn))


        # RETURN SCAN PARAMETERS AND CHANNEL INFOS
        afm_image = AFMImage()
        afm_image.pixels_x = int(metadata['ScanPoints'])
        afm_image.pixels_y = int(metadata['ScanLines'])
        afm_image.scan_center_x = np.double(metadata['XOffset'])
        afm_image.scan_center_y = np.double(metadata['YOffset'])
        afm_image.scansize_x = float(metadata["FastScanSize"])
        afm_image.scansize_y = float(metadata['SlowScanSize'])
        afm_image.scanangle = float(metadata['ScanAngle'])
        afm_image.channelnames = channel_names
        afm_image.channelunits = channel_units
        afm_image.notes = metadata['ImageNote']
        afm_image._calc_pixSize()
        afm_image.data = data
        afm_image.filename = filename
        afm_image.filetype = 'ibw'
        afm_image.date = metadata['Date']
        afm_image.time = metadata['Time']
        afm_image.imagingmode = imagingmode
        # parsing of date and time to datetime object
        # time.strptime(metadata['Date'] + ' ' + metadata['Time'], '%Y-%m-%d %H:%M:%S %p')
        if metaout:
            afm_image.metadata = metadata


    if isforce:
        raise NotImplementedError('Force curves are not yet supported!')
        #TODO: work here
        #data = data.T

        #channel_names = metadata['ForceSaveList']  # bin_info['dimLabels'][1]
        return



    return afm_image

if __name__ == '__main__':
    # IBW -> ASCII conversion

    import optparse
    import sys

    p = optparse.OptionParser()

    p.add_option('-f', '--infile', dest='infile', metavar='FILE',
                 default='-', help='Input IGOR Binary Wave (.ibw) file.')
    #p.add_option('-o', '--outfile', dest='outfile', metavar='FILE',
    #             default='-', help='File for ASCII output.')
    p.add_option('-v', '--verbose', dest='verbose', default=0,
                 action='store', help='Increment verbosity')
    p.add_option('-t', '--test', dest='test', default=False,
                 action='store_true', help='Run internal tests and exit.')
    p.add_option('-c', '--channel', type='int', dest='ch_nb', default=0,
                 help='Choose channel which should be plotted and/or saved')

    options, args = p.parse_args()

    if options.test == True:
        import doctest
        num_failures, num_tests = doctest.testmod(verbose=options.verbose)
        sys.exit(min(num_failures, 127))

    if len(args) > 0 and options.infile == None:
        options.infile = args[0]
    if options.infile == '-':
        options.infile = sys.stdin
    #if options.outfile == '-':
    #    options.outfile = sys.stdout

    afm_image = load_file(options.infile, metaout=True)
    data = afm_image.data
#    np.savetxt(options.outfile, data, fmt='%g', delimiter='\t')

    if data.ndim < 3:
        pdata = data
    else:
        # check if picked channel is in range
        if options.ch_nb > (len(data) - 1):
            print options.ch_nb
            print len(data)
            options.ch_nb = len(data) - 1
            pdata = data[options.ch_nb]
            print('Chosen channel was out of range, last channel picked!')
    if options.verbose > 0:
        for key in afm_image.keys():
            print(key + ': ' + str(afm_image[key]))
        if options.verbose > 1:
            import matplotlib.pyplot as plt
            plt.imshow(pdata)
            plt.show()
