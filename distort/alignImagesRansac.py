#!/usr/bin/python

import os
import os.path as osp
import sys
import re
import cv2
import math
import numpy as np
from glob import glob
import utils

from numpy import linalg


def filter_matches(matches, ratio=0.75):
    filtered_matches = []
    for m in matches:
        if m[0].distance < m[1].distance * ratio:
            filtered_matches.append(m[0])
    return filtered_matches


def image_distance(matches):
    sumDistance = 0.0

    for match in matches:
        sumDistance += match.distance

    return sumDistance


def find_dimensions(image, homography):
    base_p1 = np.ones(3, np.float32)
    base_p2 = np.ones(3, np.float32)
    base_p3 = np.ones(3, np.float32)
    base_p4 = np.ones(3, np.float32)

    (y, x) = image.shape[:2]

    base_p1[:2] = [0, 0]
    base_p2[:2] = [x, 0]
    base_p3[:2] = [0, y]
    base_p4[:2] = [x, y]

    max_x = None
    max_y = None
    min_x = None
    min_y = None

    for pt in [base_p1, base_p2, base_p3, base_p4]:

        hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T

        hp_arr = np.array(hp, np.float32)

        normal_pt = np.array([hp_arr[0] / hp_arr[2], hp_arr[1] / hp_arr[2]], np.float32)

        if max_x is None or normal_pt[0, 0] > max_x:
            max_x = normal_pt[0, 0]

        if max_y is None or normal_pt[1, 0] > max_y:
            max_y = normal_pt[1, 0]

        if min_x is None or normal_pt[0, 0] < min_x:
            min_x = normal_pt[0, 0]

        if min_y is None or normal_pt[1, 0] < min_y:
            min_y = normal_pt[1, 0]

    min_x = min(0, min_x)
    min_y = min(0, min_y)

    return min_x, min_y, max_x, max_y


def find_keypoints(img1, img2, mask1=None, mask2=None, detectorstr='SIFT', thresh=500, debug=0):

    if detectorstr in ['SIFT', 'ORB', 'SURF']:
        # Define detector
        detector = eval('cv2.{}()'.format(detectorstr))
        if detectorstr == 'SURF':
            detector.hessianThreshold = thresh
    else:
        if debug == 1 : print('Detector method %s is not implemented. Use SIFT.' % detectorstr)
        detector = cv2.SIFT()

    # Find key points in base image for motion estimation
    kp1i, des1 = detector.detectAndCompute(img1, mask1)
    kp2i, des2 = detector.detectAndCompute(img2, mask2)

    return kp1i, des1, kp2i, des2


def match_keypoints(kp1i, des1, kp2i, des2, matcherstr='Flann', debug=0):
    # Parameters for nearest-neighbor matching
    FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
    flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    if matcherstr not in ['Flann', 'BF']:
        if debug == 1 : print('Matcher method %s is not implemented. Use Flann.' % matcherstr)
        matcherstr = 'Flann'

    if matcherstr in ['Flann', 'BF']:
        if matcherstr == 'Flann':
            matcher = cv2.FlannBasedMatcher(flann_params, dict(checks=500))
            matches = matcher.knnMatch(des2, trainDescriptors=des1, k=2)

            if debug == 1 : print "\t Match Count: ", len(matches)
            matches_subset = filter_matches(matches)
        elif matcherstr == 'BF':
            # create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            matches = bf.match(des1, des2)
            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)
            matches_subset = matches

    return matches_subset


def matched_keypoints(kp1i, kp2i, matches_subset):
    kp1 = []
    kp2 = []

    for match in matches_subset:
        kp1.append(kp1i[match.trainIdx])
        kp2.append(kp2i[match.queryIdx])
    return kp1, kp2


def find_translation(kp1, kp2):
    p1 = getrealpoints(kp1)
    p2 = getrealpoints(kp2)

    H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
    idx = np.where(status.T[0])[0]

    poff = p1[idx] - p2[idx]
    return poff.mean(axis=0)


def find_translation_offset(file1, file2, mask1=None, mask2=None, detectorstr='SIFT', matcherstr='Flann'):
    img1rgb = cv2.imread(file1)
    img2rgb = cv2.imread(file2)

    img1 = cv2.GaussianBlur(cv2.cvtColor(img1rgb, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    img2 = cv2.GaussianBlur(cv2.cvtColor(img2rgb, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    # find key points
    kp1i, des1, kp2i, des2 = find_keypoints(img1, img2, mask1, mask2, detectorstr)
    # match key points
    matches_subset = match_keypoints(kp1i, des1, kp2i, des2, matcherstr)

    kp1, kp2 = matched_keypoints(kp1i, kp2i, matches_subset)

    return find_translation(kp1, kp2)


def getrealpoints(kps):
    return np.array([k.pt for k in kps])


def findoffset(file1, file2, mask1=None, mask2=None, index=-1, ratioThres=0.1, detectorstr='SIFT',
               matcherstr='Flann', only_translate=True):
    img1rgb = cv2.imread(file1)
    img2rgb = cv2.imread(file2)

    img1 = cv2.GaussianBlur(cv2.cvtColor(img1rgb, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    img2 = cv2.GaussianBlur(cv2.cvtColor(img2rgb, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    # find key points
    kp1i, des1, kp2i, des2 = find_keypoints(img1, img2, mask1, mask2, detectorstr)
    # match key points
    matches_subset = match_keypoints(kp1i, des1, kp2i, des2, matcherstr)

    print "\t Filtered Match Count: ", len(matches_subset)

    distance = image_distance(matches_subset)
    print "\t Distance from Key Image: ", distance

    averagePointDistance = distance / float(len(matches_subset))
    print "\t Average Distance: ", averagePointDistance

    kp1, kp2 = matched_keypoints(kp1i, kp2i, matches_subset)

    p1 = getrealpoints(kp1)
    p2 = getrealpoints(kp2)

    H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
    print '%d / %d  inliers/matched' % (np.sum(status), len(status))

    # show the points and a line between the points on both images
    if 0:
        good_pts = np.where(status)[0]
        inlinep1 = np.int_(p1[tuple(good_pts), :])
        inlinep2 = np.int_(p2[tuple(good_pts), :])
        utils.showComposite(img1, img2, pts1=inlinep1, pts2=inlinep2, scale=(0.5, 0.5), timeout=5000)

    inlierRatio = float(np.sum(status)) / float(len(status))

    H = H / H[2, 2]
    H_inv = linalg.inv(H)

    if only_translate:
        H[(0, 1, 2, 2), (1, 0, 0, 1)] = 0
        H[(0, 1), (0, 1)] = 1
        H_inv[(0, 1, 2, 2), (1, 0, 0, 1)] = 0
        H_inv[(0, 1), (0, 1)] = 1

    if inlierRatio > ratioThres:

        (min_x, min_y, max_x, max_y) = find_dimensions(img2, H_inv)

        # Adjust max_x and max_y by base img size
        max_x = max(max_x, img1.shape[1])
        max_y = max(max_y, img1.shape[0])

        move_h = np.matrix(np.identity(3), np.float32)

        if ( min_x < 0 ):
            move_h[0, 2] += -min_x
            max_x += -min_x

        if ( min_y < 0 ):
            move_h[1, 2] += -min_y
            max_y += -min_y

        print "Homography: \n", H
        print "Inverse Homography: \n", H_inv
        print "Min Points: ", (min_x, min_y)

        mod_inv_h = move_h * H_inv

        print "move_h: ", move_h
        print "mod_inv_h: ", mod_inv_h

        img_w = int(math.ceil(max_x))
        img_h = int(math.ceil(max_y))

        print "New Dimensions: ", (img_w, img_h)

        # Warp the new image given the homography from the old image
        img1_warp = cv2.warpPerspective(img1rgb, move_h, (img_w, img_h))
        print "Warped base image"

        img2_warp = cv2.warpPerspective(img2rgb, mod_inv_h, (img_w, img_h))
        print "Warped next image"

        # Put the base image on an enlarged palette
        enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)

        print "Enlarged Image Shape: ", enlarged_base_img.shape
        print "Base Image Shape: ", img1.shape
        print "Base Image Warp Shape: ", img1_warp.shape

        # Create a mask from the warped image for constructing masked composite
        ret, data_map = cv2.threshold(cv2.cvtColor(img2_warp, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)

        enlarged_base_img = cv2.add(enlarged_base_img, img1_warp, mask=np.bitwise_not(data_map), dtype=cv2.CV_8U)

        # Now add the warped image
        final_img = cv2.add(enlarged_base_img, img2_warp, dtype=cv2.CV_8U)

        # Crop off the black edges
        final_gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(final_gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print "Found %d contours..." % (len(contours))

        # max_area = 0
        # best_rect = (0, 0, 0, 0)
        #
        # for cnt in contours:
        #     x, y, w, h = cv2.boundingRect(cnt)
        #     # print "Bounding Rectangle: ", (x,y,w,h)
        #
        #     deltaHeight = h - y
        #     deltaWidth = w - x
        #
        #     area = deltaHeight * deltaWidth
        #
        #     if area > max_area and deltaHeight > 0 and deltaWidth > 0:
        #         max_area = area
        #         best_rect = (x, y, w, h)

        if 0:
            # Write out the current round
            path, basename = osp.split(file1)
            ext = osp.splitext(basename)[1]
            newpath = osp.join(path, 'stitched')
            if not osp.exists(newpath):
                os.makedirs(newpath)
            final_filename = osp.join(newpath, basename.replace(ext, '_{}-{}'.format(index, index+1) + ext))
            cv2.imwrite(final_filename, final_img)

        return final_img

    else:
        raise ValueError('Error during stitching!!!')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print >> sys.stderr, ("Usage: %s <image_dir> <search_pattern>" % sys.argv[0])
        sys.exit(-1)


    # Open the directory given in the arguments
    dirname = sys.argv[1]
    searchpattern = sys.argv[2]
    filenames = glob(osp.join(dirname, searchpattern))

    # sorting filenames
    filenames.sort(key=lambda fn: int(re.findall(r'\d+', osp.splitext(osp.basename(fn))[0])[0]))

    for i in range(len(filenames) - 1)[1:2]:
        fimage = findoffset(filenames[i], filenames[i + 1], None, None, index=i, detectorstr='SIFT', matcherstr='Flann')
        cv2.imshow('test', fimage)
        cv2.waitKey(5000)
