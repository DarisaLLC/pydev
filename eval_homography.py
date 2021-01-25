import alignment_utils

import numpy as np
import cv2 as cv


# built-in modules
import itertools as it
from multiprocessing.pool import ThreadPool
from contextlib import contextmanager
from alignment_utils import init_feature, filter_matches, explore_match

feature_name = 'orb'
def clock():
    return cv.getTickCount() / cv.getTickFrequency()

@contextmanager
def Timer(msg):
    print(msg, '...',)
    start = clock()
    try:
        yield
    finally:
        print("%.2f ms" % ((clock()-start)*1000))

def affine_skew(tilt, phi, img, mask=None):
    '''
    affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai

    Ai - is an affine transform matrix from skew_img to img
    '''
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A = np.float32([[1, 0, 0], [0, 1, 0]])
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c, -s], [s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32(np.dot(corners, A.T))
        x, y, w, h = cv.boundingRect(tcorners.reshape(1, -1, 2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv.warpAffine(img, A, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    if tilt != 1.0:
        s = 0.8 * np.sqrt(tilt * tilt - 1)
        img = cv.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv.resize(img, (0, 0), fx=1.0 / tilt, fy=1.0, interpolation=cv.INTER_NEAREST)
        A[0] /= tilt
    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv.warpAffine(mask, A, (w, h), flags=cv.INTER_NEAREST)
    Ai = cv.invertAffineTransform(A)
    return img, mask, Ai


def affine_detect(detector, img, mask=None, pool=None):
    '''
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs

    Apply a set of affine transformations to the image, detect keypoints and
    reproject them into initial image coordinates.
    See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.

    ThreadPool object may be passed to speedup the computation.
    '''
    params = [(1.0, 0.0)]
    for t in 2 ** (0.5 * np.arange(1, 6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))

    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)
        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple(np.dot(Ai, (x, y, 1)))
        if descrs is None:
            descrs = []
        return keypoints, descrs

    keypoints, descrs = [], []
    if pool is None:
        ires = it.imap(f, params)
    else:
        ires = pool.imap(f, params)

    for i, (k, d) in enumerate(ires):
        print('affine sampling: %d / %d\r' % (i + 1, len(params)), end='')
        keypoints.extend(k)
        descrs.extend(d)

    print()
    return keypoints, np.array(descrs)


def image_change(img1, img2, output=False):
    rows_1, cols_1 = map(int, img1.shape)
    rows_2, cols_2 = map(int, img2.shape)
    img1 = cv.pyrDown(img1, dstsize=(cols_1 // 2, rows_1 // 2))
    img2 = cv.pyrDown(img2, dstsize=(cols_2 // 2, rows_2 // 2))
    cols_1 = cols_1 // 2
    rows_1 = rows_1 // 2

    # img1 = cv.pyrDown(img1, dstsize=(cols_1 // 2, rows_1 // 2))
    # img2 = cv.pyrDown(img2, dstsize=(cols_2 // 2, rows_2 // 2))
    # cols_1 = cols_1 // 2
    # rows_1 = rows_1 // 2

    detector, matcher = init_feature(feature_name)

    if img1 is None:
        print('Failed to load fn1:', fn1)
        sys.exit(1)

    if img2 is None:
        print('Failed to load fn2:', fn2)
        sys.exit(1)

    if detector is None:
        print('unknown feature:', feature_name)
        sys.exit(1)

    print('using', feature_name)

    #pool = ThreadPool(processes=cv.getNumberOfCPUs())
    kp1, desc1 = affine_detect(detector, img1)
    kp2, desc2 = affine_detect(detector, img2)
    print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))

    img_matches = np.empty((max(img1.shape[0], img1.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)

    def match_and_draw(win):
        with Timer('matching'):
            raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2
        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)

        cv.drawMatches(img1, kp1, img2, kp2, [], img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        H = None
        if len(p1) >= 4:
            H, status = cv.findHomography(p1, p2, cv.RANSAC, 5.0)
            print('%d / %d  inliers/matched' % (np.sum(status), len(status)))

            return H

    HH = match_and_draw('affine find_obj')
    if HH is None:
        return None
    
    sums = np.sum(HH, axis=1)
    if output:
        print(HH)
        print(sums)
        print(sums[0] / sums[2])
        print(sums[1] / sums[2])

    img1_aligned = cv.warpPerspective(img1, HH, (cols_1, rows_1), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
    img2_aligned = cv.warpPerspective(img2, HH, (cols_2, rows_2), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

    motion = None
    motion = cv.absdiff(img1_aligned, img1)
    return {'image_1_aligned': img1_aligned, 'image_2_aligned': img2_aligned, 'change': motion}


from pathlib import Path


def image_change_from_path(image_path1, image_path2):
    img1 = cv.imread(image_path1, 0)
    img2 = cv.imread(image_path2, 0)
    result = image_change(img1, img2)
    return result

def image_change_store(image_path1, image_path2):
    result = image_change_from_path(image_path1, image_path2)
    cv.namedWindow('Display', cv.WINDOW_NORMAL)
    cv.imshow("Display", result['change'])
    cv.imwrite('/Users/arman/tmp/change.png', result['change'])
    cv.waitKey(0)
    return result



import sys

if __name__ == '__main__':
   image_change_store(sys.argv[1], sys.argv[2])
