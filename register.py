# !/usr/bin/python3
import argparse
import os
import sys
from pathlib import Path

import cv2
from matplotlib import pyplot as plt

def cv_size(img):
    return tuple(img.shape[1::-1])


# @memorize.Memorize
def fetch_reduce_image_from_file(filename, reduce):
    ## Note image read by skimage and therefore RGB
    bgr = cv2.imread(filename)
    h, w, channels = bgr.shape

    if reduce == 1:
        return bgr
    frame = (int(w / reduce), int(h / reduce))
    bgr_clone = cv2.resize(bgr, frame)
    return bgr_clone


def load_reduce_convert(image_file, reduce):
    bgr_image = fetch_reduce_image_from_file(image_file, reduce)
    h, w, channels = bgr_image.shape
    if channels == 3:
        lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        # Split LAB channels
        L, a, b = cv2.split(lab_image)
        return (L, a, b)
    if channels == 4:
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGRA2BGR)
        lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        # Split LAB channels
        L, a, b = cv2.split(lab_image)
        return (L, a, b)
    if channels == 1:
        return bgr_image


# @todo: add rect roi to limit search area
def find(image, templates, debug):
    result = {}
    result['score'] = 0
    ih, iw = image.shape
    lum = image
    best_loc = None
    best_val = 0
    best_roi = None
    best_res = None

    positions = []
    for template in templates:
        h, w = template.shape

        if w > iw or h > ih:
            continue

        roi = lum[0:ih, 0:iw]

        res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        tl = max_loc
        br = (tl[0] + w, tl[1] + h)
        positions.append([max_loc, tl, br])

        if best_val is None or max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_roi = roi
            best_res = res

        if debug:
            cv2.rectangle(roi, tl, br, 255, 2)
            score = int(max_val * 1000)
            plt.subplot(121), plt.imshow(res, cmap='gray')
            plt.title('Matching Result' + str(score)), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(best_roi, cmap='gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            plt.show()

    result['score'] = int(best_val * 1000)
    result['location'] = best_loc
    result['positions'] = positions
    result['space'] = res
    return result


def main():
    parser = argparse.ArgumentParser(description='Find Template')
    parser.add_argument('--image', '-i', required=True,
                        help='Top Directory or pickle file')
    parser.add_argument('--templates', nargs='*', help='some templates')
    parser.add_argument('--reduction', '-r', type=int)

    parser.add_argument('--out', '-a', required=False,
                        help='valid path for output and models')
    parser.add_argument('--pdf', '-p',
                        required=False,
                        default=False, action="store_true",
                        help='Output in PDF '
                        )

    args = parser.parse_args()
    print(args.templates)

    if not Path(args.image).exists():
        print(args.image + ' Does not exist ')
        return

    templates = []
    for template in args.templates:
        if not Path(template).exists:
            print(template + ' Does not exist ')
            return

    for template in args.templates:
        L, a, b = opencv_utils.import_reduce_convert2LAB(template, args.reduction)
        templates.append(L)



    L, a, b = opencv_utils.import_reduce_convert2LAB(args.image, args.reduction)
    if len(templates) > 0:
        find(L, templates, True)

    # montages = worker.gen_montage(worker.IDFiles)
    # # iterate through montages and display
    # for montage in montages:
    #     cv2.imshow('montage image', montage)
    #     cv2.waitKey(0)


if __name__ == '__main__':
    main()
