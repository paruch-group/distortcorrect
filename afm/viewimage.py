import numpy as np
import os
from os import path
from matplotlib import transforms as mtransforms
from matplotlib import pyplot as plt, cm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import (AnchoredOffsetbox, AuxTransformBox, TextArea,
                                  VPacker, DrawingArea)
from matplotlib.patches import FancyArrow

from afm.readAFMimage import load_file
from afm.filereader.stdAFMvals import find_correct_unit
from afm.tools import line_flatten_image


subunit = {12: 'T',
           9: 'G',
           6: 'M',
           3: 'k',
           2: 'h',
           0: '',
           -1: 'd',
           -2: 'c',
           -3: 'm',
           -6: 'u',
           -9: 'n',
           -12: 'p',
           -15: 'f'}


def find_power10(value, scaling_parameter=0.42):
    """

    """
    vmax = scaling_parameter * value
    try:
        power10 = 3 * np.int(np.floor(np.log10(vmax) / 3.0))
    except (RuntimeWarning, ValueError):
        power10 = 0

    return vmax, power10


def scalebar_auto_length(real_size, siunit, adjustbar=0):
    """ This is a copy of the scalebar_auto_length of gwyddion
    """
    sizes = [1.0, 2.0, 3.0, 4.0, 5.0,
             10.0, 20.0, 30.0, 40.0, 50.0,
             100.0, 200.0, 300.0, 400.0, 500.0]

    vmax, power10 = find_power10(real_size)
    base = 10 ** (power10 + 1e-15)
    x = vmax / base
    j = 0
    for i, size in enumerate(sizes):
        j = i
        if x < size:
            break

    if j - 1 + adjustbar >= len(sizes):
        power10 += 3
        pt = j - 1 + adjustbar - len(sizes)
    elif j - 1 + adjustbar < 0:
        power10 -= 3
        pt = len(sizes) - (j - 1 + adjustbar)
    else:
        pt = j - 1 + adjustbar
    psize = sizes[pt]
    base = 10 ** (power10 + 1e-15)
    x = psize * base

    # create string
    s = '{:.0f}$\,${:s}{:s}'.format(psize, subunit[power10], siunit)

    return s, x


class CustomAnchoredSizeBar(AnchoredOffsetbox):
    def __init__(self, transform, size, label, loc,
                 pad=0.1, borderpad=0.1, sep=2, prop=None, frameon=True,
                 **kwargs):
        """
        Draw a horizontal bar with the size in data coordinate of the give axes.
        A label will be drawn above (center-aligned).

        pad, borderpad in fraction of the legend font size (or prop)
        sep in points.
        """
        self.size_bar = AuxTransformBox(transform)
        self.bar = Rectangle((0, 0), size, 0.015)
        self.size_bar.add_artist(self.bar)

        self.txt_label = TextArea(label, minimumdescent=False)

        self._box = VPacker(children=[self.txt_label, self.size_bar],
                            align="center",
                            pad=0, sep=sep)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=self._box,
                                   prop=prop,
                                   frameon=frameon, **kwargs)

    def set_color(self, color):
        self.bar.set_color(color)
        self.get_text_artist().set_color(color)

    def get_text_artist(self):
        return self.txt_label.get_children()[0]

    def set_text_visiblity(self, bool):
        self.txt_label.set_visible(bool)

    def autocorrect_label_size(self):
        aim_width = self.bar.get_width() * 0.7
        label = self.get_text_artist()

        cont = True
        while cont:
            textwidth = label.get_window_extent().width
            if textwidth > aim_width:
                size = label.get_size()
                if size > 8:
                    size -= 1
                else:
                    cont = False
                label.set_size(size)
            else:
                cont = False

                # def draw(self, renderer):
                # #     self.bar.set_height(self.get_text_artist().get_window_extent(renderer).height)

                # #     barheight = self.bar.get_height()
                # #     textsize = self.get_text_artist().get_size()
                # #     print(barheight, textsize)
                # # self.get_text_artist().set_size(barheight)
                #     super(CustomAnchoredSizeBar, self).draw(renderer)


class AnchoredArrow(AnchoredOffsetbox):
    def __init__(self, size, loc, angle=0, pad=0., borderpad=0.5,
                 prop=None, frameon=False, color='k'):
        self.da = DrawingArea(size, size, 0, 0, clip=True)

        head_width = size / 3.
        head_length = size / 3.
        x0 = 0
        y0 = head_width / 2.
        width = size / 6.

        self.arrow = FancyArrow(x0, y0, size, 0,
                                width=width,
                                head_width=head_width,
                                head_length=head_length,
                                length_includes_head=True,
                                facecolor=color,
                                edgecolor=color)

        t = mtransforms.Affine2D().rotate_deg_around(y0, y0, angle)
        t += self.da.get_transform()
        self.arrow.set_transform(t)
        self.da.add_artist(self.arrow)

        super(AnchoredArrow, self).__init__(loc, pad=pad, borderpad=borderpad,
                                            child=self.da,
                                            prop=prop,
                                            frameon=frameon)


def show_scale(axes, imageparameter, siunit='m', location=3, color='auto',
               show_text=True, text_size=20, adjustbar=0,
               autocorrtextsize=True):
    x_bound = axes.get_xbound()
    x_size = np.abs(x_bound[1] - x_bound[0])
    real_size = x_size * imageparameter.pixsize_x
    stext, real_width = scalebar_auto_length(real_size, siunit,
                                             adjustbar=adjustbar)

    trans = mtransforms.blended_transform_factory(axes.transData,
                                                  axes.transAxes)

    asb = CustomAnchoredSizeBar(trans,
                                size=real_width / imageparameter.pixsize_x,
                                label=stext,
                                loc=location,
                                pad=0.1,
                                borderpad=0.5,
                                sep=1,
                                frameon=True)

    axes.add_artist(asb)
    # we need to draw the scalebar before calling "get_window_extent" functions
    asb.get_text_artist().set_size(text_size)  # set text size
    plt.draw()
    # FIXME: python complains that it cannot determine the window_extent without
    # renderer
    if autocorrtextsize:
        asb.autocorrect_label_size()  # shrink text if to wide
    plt.draw()
    # textheight = asb.get_text_artist().get_window_extent().height
    # print(textheight)
    # print axes.transData.transform((textheight, 0))

    # asb.bar.set_height(asb.get_text_artist().get_window_extent().height)

    # we need the frameon=True to get the window_extent, but color can be none
    asb.patch.set_color('none')
    bbox = asb.patch.get_window_extent()
    # transform to data coordinates
    bbox_data = axes.transData.inverted().transform(bbox)

    if color == 'auto':
        x, y = bbox_data[0, 0], bbox_data[0, 1]
        width = bbox_data[1, 0] - bbox_data[0, 0]
        height = bbox_data[1, 1] - bbox_data[0, 1]

        axim = axes.get_images()
        data = np.array(axim[-1].get_array())

        data_cut = data[int(y):int(y + height), int(x):int(x + width)]
        meanval = data_cut.mean()
        ccm = axim[-1].get_cmap()
        if ccm(meanval)[0] > 0.5:
            color = 'w'
        else:
            color = 'k'

            # test rectangle
            # testrec = Rectangle((x, y), width, height, color='b', fc='none')
            #         axes.add_artist(testrec)
    asb.set_color(color)

    asb.set_text_visiblity(show_text)  # text visibilty

    return asb


def adjust_text_size(texti, aimwidth):
    cont = True
    while cont:
        textwidth = texti.get_window_extent().width
        if textwidth > aimwidth * 0.7:
            oldsize = texti.get_size()
            if oldsize > 5:
                newsize = oldsize - 1
            else:
                newsize = oldsize
                cont = False
            texti.set_size(newsize)
        else:
            cont = False


def show_arrow(axes, imageparameter=None, tipprops=None):
    # calculate good size
    width = axes.bbox.width
    default = {
        'size': width * .1,
        'loc': 1,
        'angle': 0,
        'color': 'b'
    }
    if imageparameter is not None:
        default['angle'] = imageparameter.scanangle
    if tipprops is None:
        tipprops = default
    else:
        for k in default:
            if k not in tipprops.keys():
                tipprops[k] = default[k]
        if 'ratio' in tipprops.keys():
            ratio = tipprops.pop('ratio')
            assert 0. < ratio <= 1.
            tipprops['size'] = width * ratio
    size = tipprops.pop('size')
    ada = AnchoredArrow(size, **tipprops)
    axes.add_artist(ada)


def show_image(data, axes=None, show_axes=False, scalebar=False,
               imageparameter=None, text_size=20, cblabel=None, siunit='m',
               colorbar=False, cbloc='right', zunit='m', cmap='YlOrBr_r',
               crange=None, crangeabsolute=True, adjustzero=False, adjustbar=0,
               location=3, scalecolor='auto', tiparrow=False, tipprops=None):
    data[np.isnan(data)] = 0

    if adjustzero:
        idata = data - data.min()
    else:
        idata = data

    if idata.max() == 0:
        power10 = 0
    else:
        power10 = find_power10(idata.max())[1]
    if data.max() / (10 ** power10) > 1000:
        power10 += 3
    pdata = idata / (10 ** power10)
    unit = subunit[power10] + zunit

    if crange is not None:
        if crangeabsolute:
            crange = (crange[0] / (10 ** power10), crange[1] / (10 ** power10))
        else:
            crange = (crange[0], crange[1] * pdata.max())

    if axes is None or not isinstance(axes, plt.Axes):
        fig = plt.figure()
        axes = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(axes)

    if not show_axes:
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
        axes.set_axis_off()

    ret = axes.imshow(pdata, origin='lower', cmap=cmap, clim=crange)

    if colorbar:
        show_colorbar(ret, label=cblabel, unit=unit, loc=cbloc)

    if scalebar and imageparameter is not None:
        show_scale(axes, imageparameter, siunit=siunit, text_size=text_size,
                   adjustbar=adjustbar, location=location, color=scalecolor)

    if tiparrow:
        show_arrow(axes, imageparameter, tipprops)

    plt.draw()
    return ret


def show_colorbar(plotref, label=None, unit=None, loc='right'):
    assert loc in ['top', 'bottom', 'left', 'right']

    if loc in ['top', 'bottom']:
        orientation = 'horizontal'
    else:
        orientation = 'vertical'
    divider = make_axes_locatable(plotref.get_axes())
    cax = divider.append_axes(loc, "6%", pad="5%")
    cb = plt.colorbar(plotref, cax=cax, orientation=orientation)

    if loc in ['top', 'bottom']:
        cb.ax.xaxis.set_ticks_position(loc)
        cb.ax.xaxis.set_label_position(loc)
    else:
        cb.ax.yaxis.set_ticks_position(loc)
        cb.ax.yaxis.set_label_position(loc)

    glob_label = ''
    if label is not None and isinstance(label, str):
        glob_label += label
    if unit is not None and isinstance(unit, str):
        if glob_label != '':
            glob_label += ' '
        glob_label += '(' + unit + ')'
    if glob_label != '':
        cb.set_label(glob_label)


def save_fig(fig, filename, dpi=80, pad_inches=0, **kwargs):
    if 'figsize' in kwargs:
        w, h = kwargs.pop('figsize')
        fig.set_sizes_inches(w, h)  # set figsize if given
    fig.patch.set_alpha(0)
    fig.savefig(filename, bbox_inches='tight', pad_inches=pad_inches, dpi=dpi)


def make_image(filename, channel=0, savename=None, scalebar=False,
               colorbar=False, title=False, dpi=300, text_size=20,
               fileformat='pdf'):
    filename = path.realpath(filename)

    image = load_file(filename)
    data = image.data

    if channel == 'all':
        channels = range(len(data))
    elif isinstance(channel, list):
        channels = channel
    else:
        channels = [channel, ]

    savenames = []
    for ch in channels:
        plt.close('all')
        pdata = data[ch]
        channelname = image.channelnames[ch]

        if savename is None and channel != 'all':
            savename = path.splitext(filename)[0] + '_{}.{}'.format(channelname,
                                                                    fileformat)
        elif channel == 'all':
            filepath, fn = path.split(filename)
            savename = path.join(filepath, path.splitext(fn)[0],
                                 path.splitext(fn)[0] + '_{}.{}'.format(
                                     channelname,
                                     fileformat))

        im_ret = show_image(pdata, scalebar=scalebar, colorbar=colorbar,
                            imageparameter=image, text_size=text_size)
        if title:
            im_ret.get_axes().set_title(channelname)
        save_fig(plt.gcf(), savename, dpi=dpi)
        savenames.append(savename)
    return savenames


def save_pure_image(filename, channel=0, savename=None, fileformat='png',
                    cmap='Blues_r',
                    saveinfo=True, zipfiles=False):
    from PIL import Image

    filename = path.realpath(filename)

    image = load_file(filename)
    data = image.data

    if channel == 'all':
        channels = range(len(data))
    elif isinstance(channel, list):
        channels = []
        for ch in channel:
            if isinstance(ch, int) and int < len(image.channelnames):
                channels.append(ch)
            elif isinstance(ch, str):
                pos = [i for i, k in enumerate(image.channelnames) if ch in k]
                channels += pos
    elif isinstance(channel, int):
        channels = [channel, ]
    else:
        raise ValueError('channel is not defined correctly!')
    colmap = cm.get_cmap(cmap)  # get colormap from matplotlib

    savenames = []
    for ch in channels:
        pdata = data[ch].copy()
        pdata -= pdata.min()
        pdata /= pdata.max()
        pdata = colmap(pdata)  # converting grayscale to RGBA
        pdata *= 255
        im = Image.fromarray(np.uint8(pdata))

        channelname = image.channelnames[ch]

        if savename is None and channel != 'all':
            savename = path.splitext(filename)[0] + '_{}.{}'.format(channelname,
                                                                    fileformat)
        elif channel == 'all' or isinstance(channel, list):
            filepath, fn = path.split(filename)
            savename = path.join(filepath,
                                 path.splitext(fn)[0] + '_{}.{}'.format(
                                     channelname,
                                     fileformat))
        im.save(savename)
        savenames.append(savename)

    if saveinfo:
        savenames.append(_image_info2txt(image))

    if zipfiles:
        zip_files(savenames, image.filename)

    return savenames


def zip_files(files, filename, remove_org=True):
    import zipfile

    zf = zipfile.ZipFile(path.splitext(path.basename(filename))[0] + '.zip',
                         'w')
    for fn in files:
        zf.write(path.basename(fn))
    zf.close()
    if remove_org:
        for fn in files:
            os.remove(path.abspath(fn))


def save_info2txt(filename):
    image = load_file(filename)
    return _image_info2txt(image)


def _image_info2txt(image):
    savename = path.splitext(image.filename)[0] + '_{}.{}'.format('Info', 'txt')

    keys2save = ['filename',
                 'date',
                 'time',
                 'scansize_x',
                 'scansize_y',
                 'pixels_x',
                 'pixels_y',
                 'scanangle',
                 'imagingmode',
                 'notes',
                 'channelnames']

    with open(savename, 'w') as f:
        for key in keys2save:
            value = image.__dict__[key]
            f.write(key + ': %s\n' % str(value).replace("'", ""))
    return savename


def image_overview(filename, ncols=4, show_table=True, fileformat='pdf',
                   cmap='YlOrBr_r'):
    try:
        image = load_file(filename)
    except NotImplementedError:
        return
    nrows = (len(image.channelnames) + int(show_table)) / ncols + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4))
    axes = axes.ravel()
    for ax in axes:
        ax.set_axis_off()

    table_vals = []
    if show_table:
        row_labels = ['Filename', 'Date', 'Time', 'Scansize', 'Pixels',
                      'ScanAngle', 'Mode']

        table_vals.append([os.path.basename(filename)])
        table_vals.append([image.date])
        table_vals.append([image.time])
        table_vals.append(['%.2e x %.2e m' % (image.scansize_x,
                                              image.scansize_y)])
        table_vals.append(['%d x %d' % (image.pixels_x, image.pixels_y)])
        table_vals.append(['%.2f$^\circ$' % image.scanangle])
        table_vals.append([image.imagingmode])

        table = axes[0].table(cellText=table_vals, rowLabels=row_labels,
                              colWidths=[0.5], loc='best')

        table.set_fontsize(35)
        table.scale(1.3, 1.3)

    for idx, ax in enumerate(
            axes[int(show_table):len(image.data) + int(show_table)]):
        channelname = image.channelnames[idx]
        if image.channelunits is not None:
            unit = image.channelunits[idx]
        else:
            unit = find_correct_unit(channelname)
        data = image.data[idx]
        # TODO: define in some ini file which flattening is used for which channel
        if 'height' in channelname.lower():
            data = line_flatten_image(data, 1)
        im_ret = show_image(data, ax, scalebar=True, colorbar=True,
                            imageparameter=image, zunit=unit, cmap=cmap)
        im_ret.get_axes().set_title(channelname)
    if len(axes) > len(image.data) + int(show_table):
        for ax in axes[len(image.data) + int(show_table):]:
            fig.delaxes(ax)

    plt.tight_layout()
    save_fig(plt.gcf(), os.path.splitext(filename)[0] + '.' + fileformat)
    plt.close()
    del axes, fig, image, table_vals

# if __name__ == '__main__':
# this is broken at the moment, needs to load/create a proper image
# container
# show_image(np.ones((30,30)), scalebar=True, pixsize=(10,10))
# plt.show()
