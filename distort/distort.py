from __future__ import print_function
import numpy as np
import cv2
from skimage import img_as_ubyte
from scipy import interpolate
import afm
import alignImagesRansac as air
from skimage import transform as tf
from scipy.ndimage import map_coordinates

def distort_correct(fn, fn2, topochan1, topochan2, corrchannel, flipud=1, progress=0, debug=0, islog=False):
    if progress == 1 : print('[',end='')
    image1 = afm.readAFMimage.load_file(fn)
    image2 = afm.readAFMimage.load_file(fn2)
    if progress == 1 : print('.',end='')
    
    if flipud == 1:
        channel = np.flipud(image2.data[corrchannel])
    else:
        channel = image2.data[corrchannel]

    retrace1 = afm.tools.line_flatten_image(image1.data[topochan1]).astype(
        np.float32)
    retrace2 = afm.tools.line_flatten_image(image2.data[topochan2]).astype(
        np.float32)
    retrace1 = np.flipud(retrace1)
    if flipud == 1:
        retrace2 = np.flipud(retrace2)
    
    if progress == 1 : print('.',end='')
        
    retrace1 -= retrace1.min()
    retrace2 -= retrace2.min()

    if islog == True:
        retrace1 = np.log(retrace1+1e-9)
        retrace2 = np.log(retrace2+1e-9)
        retrace1 -= retrace1.min()
        retrace2 -= retrace2.min()
    retrace1 /= retrace1.max()

    retrace2 /= retrace2.max()

    img1 = img_as_ubyte(retrace1)
    img2 = img_as_ubyte(retrace2)

    img1 = cv2.GaussianBlur(img1, (5, 5), 0)
    img2 = cv2.GaussianBlur(img2, (5, 5), 0)

    if progress == 1 : print('.',end='')
        
    # find key points
    kp1i, des1, kp2i, des2 = air.find_keypoints(img1, img2, None, None, 'SURF',
                                                thresh=50, debug=debug)
    if progress == 1 : print('.',end='')    
    # match key points
    matches_subset = air.match_keypoints(kp1i, des1, kp2i, des2, 'Flann', debug=debug)
    if progress == 1 : print('.',end='')
    if debug == 1 : print("\t Filtered Match Count: ", len(matches_subset))

    distance = air.image_distance(matches_subset)
    if progress == 1 : print('.',end='')
    if debug == 1 : print("\t Distance from Key Image: ", distance)

    if len(matches_subset) == 0:
        averagePointDistance = 0
    else:
        averagePointDistance = distance / float(len(matches_subset))
    if progress == 1 : print('.',end='')
    if debug == 1 : print("\t Average Distance: ", averagePointDistance)

    kp1, kp2 = air.matched_keypoints(kp1i, kp2i, matches_subset)

    p1 = air.getrealpoints(kp1)
    p2 = air.getrealpoints(kp2)

    H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 10.0)
    if progress == 1 : print('.',end='')
    if debug == 1 : print('%d / %d  inliers/matched' % (np.sum(status), len(status)))

    # show the points and a line between the points on both images
    good_pts = np.where(status)[0]
    inlinep1 = np.int_(p1[tuple(good_pts), :])
    inlinep2 = np.int_(p2[tuple(good_pts), :])

    xs = inlinep2[:, 0]
    ys = inlinep2[:, 1]
    delta = inlinep2 - inlinep1
    deltax = delta[:, 0]
    deltay = delta[:, 1]

    coefficientsx = np.polyfit(xs, deltax, 0)
    coefficientsy = np.polyfit(ys, deltay, 0)
    if progress == 1 : print('.',end='')
        
    x = np.arange(
        image1.pixels_x)  # use numpy's arange instead of range, see below
    y = np.arange(image1.pixels_y)
    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    if progress == 1 : print('.',end='')
        
    x_corr = x - np.polyval(coefficientsx, x)
    y_corr = y - np.polyval(coefficientsy, y)
    if progress == 1 : print('.',end='')
        
    Xcorr, Ycorr = np.meshgrid(x_corr, y_corr)
    if progress == 1 : print('.',end='')
        
    corrected_channel = interpolate.griddata((Xcorr.ravel(), Ycorr.ravel()),
                                             channel.ravel(), (X, Y))
    if progress == 1 : print('.',end='')                                             
    if progress == 1 : print(']')
    return corrected_channel

def distort_correct_warp(fn, fn2, topochan1, topochan2, corrchannel, flipud=1, progress=0, debug=0):
    if progress == 1 : print('[',end='')
    image1 = afm.readAFMimage.load_file(fn)
    image2 = afm.readAFMimage.load_file(fn2)
    if progress == 1 : print('.',end='')
    
    if flipud == 1:
        channel = np.flipud(image2.data[corrchannel])
    else:
        channel = image2.data[corrchannel]

    retrace1 = afm.tools.line_flatten_image(image1.data[topochan1]).astype(
        np.float32)
    retrace2 = afm.tools.line_flatten_image(image2.data[topochan2]).astype(
        np.float32)
    retrace1 = np.flipud(retrace1)
    if flipud == 1:
        retrace2 = np.flipud(retrace2)
    
    if progress == 1 : print('.',end='')
        
    retrace1 -= retrace1.min()
    retrace1 /= retrace1.max()

    retrace2 -= retrace2.min()
    retrace2 /= retrace2.max()

    img1 = img_as_ubyte(retrace1)
    img2 = img_as_ubyte(retrace2)

    img1 = cv2.GaussianBlur(img1, (5, 5), 0)
    img2 = cv2.GaussianBlur(img2, (5, 5), 0)

    if progress == 1 : print('.',end='')
        
    # find key points
    kp1i, des1, kp2i, des2 = air.find_keypoints(img1, img2, None, None, 'SURF',
                                                thresh=50, debug=debug)
    if progress == 1 : print('.',end='')    
    # match key points
    matches_subset = air.match_keypoints(kp1i, des1, kp2i, des2, 'Flann', debug=debug)
    if progress == 1 : print('.',end='')
    if debug == 1 : print("\t Filtered Match Count: ", len(matches_subset))

    distance = air.image_distance(matches_subset)
    if progress == 1 : print('.',end='')
    if debug == 1 : print("\t Distance from Key Image: ", distance)

    if len(matches_subset) == 0:
        averagePointDistance = 0
    else:
        averagePointDistance = distance / float(len(matches_subset))
    if progress == 1 : print('.',end='')
    if debug == 1 : print("\t Average Distance: ", averagePointDistance)

    kp1, kp2 = air.matched_keypoints(kp1i, kp2i, matches_subset)

    p1 = air.getrealpoints(kp1)
    p2 = air.getrealpoints(kp2)

    H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 10.0)
    if progress == 1 : print('.',end='')
    if debug == 1 : print('%d / %d  inliers/matched' % (np.sum(status), len(status)))

    # show the points and a line between the points on both images
    good_pts = np.where(status)[0]
    inlinep1 = np.int_(p1[tuple(good_pts), :])
    inlinep2 = np.int_(p2[tuple(good_pts), :])

    src = inlinep2
    dst = inlinep1

    tform = tf.estimate_transform('polynomial', src, dst)
    res=tf.warp_coords(tform, channel.shape) 
    
    Ycorr=res[0]
    Xcorr=res[1]
    
    x = np.arange(
        image1.pixels_x)  # use numpy's arange instead of range, see below
    y = np.arange(image1.pixels_y)
    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
        
    corrected_channel = interpolate.griddata((Xcorr.ravel(), Ycorr.ravel()),
                                             channel.ravel(), (X, Y))
    if progress == 1 : print('.',end='')                                             
    if progress == 1 : print(']')
    return corrected_channel    
    
def gen_warp_transform(fn, fn2, topochan1, topochan2,flipud=1, progress=0, debug=0, tftype='polynomial', islog=False):
    if progress == 1 : print('[',end='')
    image1 = afm.readAFMimage.load_file(fn)
    image2 = afm.readAFMimage.load_file(fn2)
    if progress == 1 : print('.',end='')
    

    retrace1 = afm.tools.line_flatten_image(image1.data[topochan1]).astype(
        np.float32)
    retrace2 = afm.tools.line_flatten_image(image2.data[topochan2]).astype(
        np.float32)
    retrace1 = np.flipud(retrace1)
    if flipud == 1:
        retrace2 = np.flipud(retrace2)
    
    if progress == 1 : print('.',end='')
        
    retrace1 -= retrace1.min()
    retrace2 -= retrace2.min()

    if islog == True:
        retrace1 = np.log(retrace1+1e-9)
        retrace2 = np.log(retrace2+1e-9)
        retrace1 -= retrace1.min()
        retrace2 -= retrace2.min()
    retrace1 /= retrace1.max()

    retrace2 /= retrace2.max()

    img1 = img_as_ubyte(retrace1)
    img2 = img_as_ubyte(retrace2)

    img1 = cv2.GaussianBlur(img1, (5, 5), 0)
    img2 = cv2.GaussianBlur(img2, (5, 5), 0)

    if progress == 1 : print('.',end='')
        
    # find key points
    kp1i, des1, kp2i, des2 = air.find_keypoints(img1, img2, None, None, 'SURF',
                                                thresh=50, debug=debug)
    if progress == 1 : print('.',end='')    
    # match key points
    matches_subset = air.match_keypoints(kp1i, des1, kp2i, des2, 'Flann', debug=debug)
    if progress == 1 : print('.',end='')
    if debug == 1 : print("\t Filtered Match Count: ", len(matches_subset))

    distance = air.image_distance(matches_subset)
    if progress == 1 : print('.',end='')
    if debug == 1 : print("\t Distance from Key Image: ", distance)

    if len(matches_subset) == 0:
        averagePointDistance = 0
    else:
        averagePointDistance = distance / float(len(matches_subset))
    if progress == 1 : print('.',end='')
    if debug == 1 : print("\t Average Distance: ", averagePointDistance)

    kp1, kp2 = air.matched_keypoints(kp1i, kp2i, matches_subset)

    p1 = air.getrealpoints(kp1)
    p2 = air.getrealpoints(kp2)

    H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 10.0)
    if progress == 1 : print('.',end='')
    if debug == 1 : print('%d / %d  inliers/matched' % (np.sum(status), len(status)))

    # show the points and a line between the points on both images
    good_pts = np.where(status)[0]
    inlinep1 = np.int_(p1[tuple(good_pts), :])
    inlinep2 = np.int_(p2[tuple(good_pts), :])

    src = inlinep2
    dst = inlinep1

    tform = tf.estimate_transform(tftype, src, dst)

    return tform      
    
def get_channel_for_warp(fn2, corrchannel, flipud=1, progress=0, debug=0):
    if progress == 1 : print('[',end='')

    image2 = afm.readAFMimage.load_file(fn2)
    if progress == 1 : print('.',end='')
    
    if flipud == 1:
        channel = np.flipud(image2.data[corrchannel])
    else:
        channel = image2.data[corrchannel]


    if progress == 1 : print('.',end='')                                             
    if progress == 1 : print(']')
    return channel

def apply_warp_transform(channel, tform, flipud=1, progress=0, debug=0):
    if progress == 1 : print('[',end='')


    res=tf.warp_coords(tform, channel.shape) 
    
        
    corrected_channel = map_coordinates(channel, res)

    if progress == 1 : print('.',end='')                                             
    if progress == 1 : print(']')
    return corrected_channel        