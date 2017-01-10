# THIS FUNCTIONS LABEL FEATURES IN ARRAYS AND CAN DELETE FEATURES OF SIZE size
# author: zieglerb
# date: 30.11.2012

import numpy as np
from scipy.ndimage.measurements import label, sum


def find_features(array):
    label_im, nb_labels = label(array)
    sizes = sum(array, label_im, range(nb_labels + 1))
    return label_im, nb_labels, sizes


def delete_features(array, size):
    label_im, nb_labels, sizes = find_features(array)
    mask_size = sizes < size
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0
    return label_im


def delete_features_inv(array, size):
    narr = np.logical_not(array)
    narr = delete_features(narr, size)
    narr = np.int_(np.logical_not(narr))
    return narr


def delete_features_all(array, size):
    narr = delete_features(array, size)
    narr = delete_features_inv(narr, size)
    return narr
