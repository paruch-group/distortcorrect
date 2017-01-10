# collection of useful tools for AFM image analysis
# author: zieglerb
# date: 29.11.2012

import sys
import numpy as np
import itertools
import functools

import uncertainties as unc
import uncertainties.unumpy as unp

from scipy.ndimage import binary_dilation, binary_erosion
from scipy.stats import linregress, skew, kurtosis
from scipy.ndimage import measurements
from scipy.optimize import leastsq, minimize
from scipy.ndimage.filters import gaussian_filter, sobel
# from scipy.signal import find_peaks_cwt
from skimage.filter import threshold_otsu

from afm.feature_detection import find_features, delete_features_all
from peakdetect import peakdetect


def binarize(data, threshold=None):
    """
    Binarization of an numpy array

    Threshold can be given or if None, it is estimated

    Parameter:
    ----------
    data: 2d numpy array
    threshold: float

    Return:
    -------
    bin_data: 2d numpy array of type bool
    """

    # check if data is already of type bool and return if so
    if np.all(data == np.bool_(data)):
        print('The data array is already binarized!')
        return data

    if threshold is None:
        hist_data = data.ravel()  # flatten the data matrix into a 1D array
        elements_per_bin, bin_edges = np.histogram(hist_data, bins=100)

        [bin_max, _] = peakdetect(elements_per_bin, 100, bin_edges[1:])
        bin_max_left = bin_max[0][0]
        bin_max_right = bin_max_left + 180

        if len(bin_max) > 2:
            print len(bin_max), 'maxima were found!'
            for i in bin_max[1:]:
                if i[0] > bin_max_left + 90:
                    bin_max_right = i[0]
                    break
        elif len(bin_max) == 2:
            print 'Two maxima were found'
            bin_max_right = bin_max[1][0]
        elif len(bin_max) == 1:  # just a guess were the other max should be
            print('Only one maximum was found!')
            if bin_max[0][0] < 90:
                bin_max_left = bin_max[0][0]
                bin_max_right = bin_max[0][0] + 180
            else:
                bin_max_left = bin_max[0][0] - 180
                bin_max_right = bin_max[0][0]
        elif len(bin_max) == 0:
            sys.exit('No maximum was found!')

        # threshold is defined as the value in the middle between both maxima
        threshold = abs((bin_max_right - bin_max_left) / 2) + bin_max_left

    return data > threshold  # binarization of data


def phase_correction(data, phaseshift=0, limits=(-90, 270)):
    """
    Function to correct the phase for AFM image files

    Parameter:
    ----------
    data: 2d numpy array
    phaseshift: double
    limits: tuple of length 2

    Return:
    -------
    shifted data: 2d numpy array
    """
    loc_data = np.copy(data)
    # check if data is already of type bool and return if so
    if np.all(loc_data == np.bool_(loc_data)):
        print('The data array is already binarized!')
        return loc_data

    # unpack limits
    low_lim = limits[0]
    up_lim = limits[1]
    rng = np.abs(low_lim) + np.abs(up_lim)  # range
    # phase shift correction
    if phaseshift != 0:
        loc_data += phaseshift
        mask = np.where(loc_data < low_lim, 1, 0)
        vals = np.extract(loc_data < low_lim, loc_data) + rng
        np.place(loc_data, mask, vals)
        mask = np.where(loc_data > up_lim, 1, 0)
        vals = np.extract(loc_data > up_lim, loc_data) - rng
        np.place(loc_data, mask, vals)
    return loc_data


def get_contour(bin_arr, its=1):
    """
    Returns an array with contours of features in a binarized array

    Parameter:
    ----------
    bin_arr: 2d numpy array

    Return:
    -------
    array: 2d numpy array of type int
    """

    inv_arr = np.invert(bin_arr.astype(bool)).astype(int)
    dil_arr = binary_dilation(inv_arr, iterations=its).astype(int)
    return (dil_arr - inv_arr).astype(int)


def interface_orient(data):
    """
    Determine the orientation of the interface by FFT

    Parameter:
    ----------
    data: 2d numpy array

    Return:
    -------
    angle: float
    """
    # check if data is already of type bool and if not, binarize image
    data = binarize(data)
    # if image is not squared, use square subimage to compute the orientation
    dim = data.shape
    if dim[0] != dim[1]:
        dmin = np.min(dim)
        data = data[:dmin, :dmin]

    data_fft = np.fft.fft2(data)
    data_fft = np.abs(np.fft.fftshift(data_fft))
    data_fft = np.log(data_fft)
    mask = (data_fft > np.max(data_fft) / 2)

    # make a list of x,y coordinates
    xlist, ylist = np.argwhere(mask).T

    fit_params = linregress(ylist, xlist)
    return np.arctan(fit_params[0]) * -1.


def line_fit(line, order=1,box=[0]):
    """
    Do a nth order polynomial line flattening

    Parameters
    ----------
    line : 1d array-like
    order : integer

    Returns
    -------
    result : 1d array-like
        same shape as data
    """
    if order < 0:
        raise ValueError('expected deg >= 0')
    newline=line
    if len(box)==2:
        newline = line[box[0]:box[1]]
    x = np.arange(len(newline))
    k = np.isfinite((newline))
    if not np.isfinite(newline).any():
        return line
    coefficients = np.polyfit(x[k], newline[k], order)

    return line - np.polyval(coefficients, np.arange(len(line)))


def line_flatten_image(data, order=1, axis=0, box=[0]):
    """
    Do a line flattening

    Parameters
    ----------
    data : 2d array
    order : integer
    axis : integer
        axis perpendicular to lines

    Returns
    -------
    result : array-like
        same shape as data
    """

    if axis == 1:
        data = data.T

    ndata = np.zeros_like(data)

    for i, line in enumerate(data):
        ndata[i, :] = line_fit(line, order, box)

    if axis == 1:
        ndata = ndata.T

    return ndata


def plane_flatten_image(data, order=1,box=[]):
    """
    Do a plane flattening

    Parameters
    ----------
    data : 2d array
    order : integer

    Returns
    -------
    result : array-like
        same shape as data
    """
    fitdata = data
    if len(box)==4:
        fitdata = data[box[0]:box[1],box[2]:box[3]]
    xx, yy = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    xxfit, yyfit = np.meshgrid(np.arange(fitdata.shape[1]), np.arange(fitdata.shape[0]))
    m = polyfit2d(xxfit.ravel(), yyfit.ravel(), fitdata.ravel(), order=order)
    return data - polyval2d(xx, yy, m)


def polyfit2d(x, y, z, order=1):
    ncols = (order + 1) ** 2
    g = np.zeros((x.size, ncols))
    ij = itertools.product(range(order + 1), range(order + 1))
    for k, (i, j) in enumerate(ij):
        g[:, k] = x ** i * y ** j
    k = np.isfinite(z)
    m, _, _, _ = np.linalg.lstsq(g[k], z[k])
    return m


def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order + 1), range(order + 1))
    z = np.zeros_like(x, dtype=float)
    for a, (i, j) in zip(m, ij):
        z += a * x ** i * y ** j
    return z


def _f_min(x, p):
    plane_xyz = p[:3]
    distance = (plane_xyz * x.T).sum(axis=1) + p[3]
    return distance / np.linalg.norm(plane_xyz)


def _residuals(params, x):
    return _f_min(x, params)


def flatten_steps(data):
    """
    Flatten terraces

    Parameters
    ----------
    data : array-like

    Returns
    -------
    result : array-like
        same shape as data
    """

    topo = data

    topo_filter = gaussian_filter(topo, 2)  # 2 seams to be a good value
    topo_sobel = sobel(topo_filter)  # find pos and neg slope

    neg = np.zeros_like(topo_sobel)  # create arrays for negative and positive slope
    neg[topo_sobel < 0] = 1  # asign data to arrays

    thresh = threshold_otsu(topo_sobel)

    # choose the relevant regions
    if neg.sum() < topo_sobel.size / 2:
        im_bin = topo_sobel < thresh
    else:
        im_bin = topo_sobel > thresh

    im_bin = binary_dilation(im_bin)
    im_bin = 1. - im_bin.astype(float)

    label_im, nb_labels = measurements.label(im_bin)
    sizes = measurements.sum(im_bin, label_im, range(1, nb_labels + 1))

    labels = [k + 1 for k in range(nb_labels) if sizes[k] > 100]

    sols = []
    for j in labels:
        x, y = np.argwhere(label_im == j).T
        # list of coordinates of region of interest
        roi = topo[x, y]

        xyz = np.vstack((x, y, roi))

        # guess inital parameters
        rmin = roi.argmin()
        rmax = roi.argmax()
        xs = (roi[rmax] - roi[rmin]) / (x[rmax] - x[rmin])
        ys = (roi[rmax] - roi[rmin]) / (y[rmax] - y[rmin])

        p0 = [xs, ys, roi.mean(), 0]

        sol = leastsq(_residuals, p0, args=(xyz,))[0]
        sols.append(sol)

    sol = np.mean(sols, axis=0)

    x, y = np.meshgrid(np.arange(topo_sobel.shape[1]),
                       np.arange(topo_sobel.shape[0]))

    correction_plane = (-1 * sol[3] - sol[1] * x - sol[0] * y) / sol[2]
    newtopo = topo - correction_plane
    return newtopo - newtopo.min()


def determine_step_height(data):
    data = data - data.min()

    # here should be smart way included to estimate a good value for bins
    #     counts, bins = np.histogram(data, bins=1000)
    #     x = bins[:-1] + np.diff(bins)
    #
    #     peaks = find_peaks_cwt(counts, np.arange(3,40))
    #
    #     peaks_x = x[peaks]
    #     peaks_y = counts[peaks]
    #     return np.diff(peaks_x).mean()

    label_steps, nb_steps, sizes_steps = determine_steps(data)
    means = measurements.mean(data, label_steps, np.unique(label_steps)[1:])
    means = np.sort(means)
    return np.diff(means).mean(), np.diff(means).std()


def determine_steps(data):
    data_g = gaussian_filter(data, 2)
    data_s = sobel(data_g)
    thresh = threshold_otsu(data_s)
    bin_data = data_s < thresh
    bin_data = binary_dilation(bin_data)
    bin_data = binary_dilation(1. - bin_data)

    bin_c = delete_features_all(bin_data, np.sqrt(bin_data.size) / 2)
    label_steps, nb_steps, sizes_steps = find_features(bin_c)
    return label_steps, nb_steps, sizes_steps


def determine_edges(data):
    data_g = gaussian_filter(data, 2)
    data_s = sobel(data_g)
    thresh = threshold_otsu(data_s)
    bin_data = data_s < thresh

    bin_c = delete_features_all(bin_data, np.sqrt(bin_data.size) / 2)
    label_edges, nb_edges, sizes_edges = find_features(bin_c)
    return label_edges, nb_edges, sizes_edges


def determine_step_orientation(data):
    label_edges, nb_edges, sizes_edges = determine_edges(data)

    edge_fitparams = []
    for i in np.arange(1, nb_edges + 1):
        xcoords, ycoords = np.argwhere(label_edges == i).T
        fit_params = linregress(xcoords, ycoords)
        edge_fitparams.append(fit_params)
    edge_fitparams = np.asarray(edge_fitparams)
    # avg_fitparams = edge_fitparams.mean(axis=0)

    return edge_fitparams, label_edges


def determine_step_width(data, imageparameter):
    label_steps, nb_steps, sizes_steps = determine_steps(data)

    step_width = []
    for i in np.arange(1, nb_steps + 1):
        xycoords = np.argwhere(label_steps == i).T
        xcoords, ycoords = xycoords
        fit_params = linregress(xcoords, ycoords)

        x1 = (ycoords.min() - fit_params[1]) / fit_params[0]
        x2 = (ycoords.max() - fit_params[1]) / fit_params[0]
        y1, y2 = np.asarray([x1, x2]) * fit_params[0] + fit_params[1]
        length = np.sqrt(np.abs(x1 - x2) ** 2 + np.abs(y1 - y2) ** 2)
        width = imageparameter.pixSize_x * sizes_steps[i] / (length * np.cos(np.arctan(fit_params[0])))
        if np.abs(sizes_steps[i] - np.mean(sizes_steps[1:])) < np.mean(sizes_steps[1:]) / 2:
            step_width.append(width * data.size / np.sum(sizes_steps))

    return np.mean(step_width), np.std(step_width), step_width


def determine_miscut(data, imageparameter):
    width = unp.uarray(determine_step_width(data, imageparameter)[:2])
    height = unp.uarray(determine_step_height(data))
    miscut = unp.degrees(unp.arctan(height / width))
    return float(unp.nominal_values(miscut)), float(unp.std_devs(miscut))


class Roughness(object):
    def __init__(self, data):
        self.data = data

        self.avg = self.data.mean()
        self.sdev = self.rq = self.data.std()
        self.rms = np.sqrt(self.avg ** 2 + self.sdev ** 2)
        self.max = self.data.max()
        self.min = self.data.min()
        self.max_height = self.rt = self.max - self.min
        self.adev = self.ra = np.abs(self.data).sum() / float(self.data.size)
        self.skew = skew(self.data.ravel())
        self.kurt = kurtosis(self.data.ravel())