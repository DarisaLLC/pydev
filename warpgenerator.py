import os
import cv2
import colorsys
import numpy as np
from math import sqrt
import random
import sys
from glob import glob
from itertools import chain
import shutil

# Generates synthetic data from a dataset and small logo images

bgrpath = '/home/andras/data/misc/Tripadvisor'
logopath = '/home/andras/data/misc/logopull'
output = '/home/andras/data/datasets/SYNLOGO'

imagesetf = 'synmetu_ta_train_all.txt'


def transparentBgr(im):
    whiteim = (im > 220).all(axis=2)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)
    im[whiteim, :] = [255, 255, 255, 0]
    return im


def coloredBgr(im, bgrbgrmean):
    bgrhsv = colorsys.rgb_to_hsv(bgrbgrmean[2], bgrbgrmean[1], bgrbgrmean[0])
    bgrhsv = list(bgrhsv)
    logo = (im < 220).all(axis=2)
    logobackground = (im > 220).all(axis=2)
    # random color change
    if random.uniform(0, 1) < float(1) / 2:
        logopos = (im < 250).all(axis=2)
        meanr = np.mean(im[logopos, 2])
        meang = np.mean(im[logopos, 1])
        meanb = np.mean(im[logopos, 0])

        hsv = colorsys.rgb_to_hsv(meanr, meang, meanb)
        hsv = list(hsv)
        hsv[0] += 0.5

        rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2])
        im[logobackground] = rgb
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    im[logobackground, 2] = min(255, bgrhsv[2] + 50)
    im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)

    hsv[2] += (255 - hsv[2]) * 0.7

    return im


images = 'Images'
imagesoutput = os.path.join(output, images)
bboxes = 'Annotations'
bboxesoutput = os.path.join(output, bboxes)
imagesets = 'ImageSets'
imagesetsoutput = os.path.join(output, imagesets)

if os.path.exists(imagesoutput):
    shutil.rmtree(imagesoutput)
os.makedirs(imagesoutput)
if os.path.exists(bboxesoutput):
    shutil.rmtree(bboxesoutput)
os.makedirs(bboxesoutput)
if os.path.exists(imagesetsoutput):
    shutil.rmtree(imagesetsoutput)
os.makedirs(imagesetsoutput)

bgrs = [os.path.join(dp, f) for dp, dn, filenames in os.walk(bgrpath) for f in filenames if
        os.path.splitext(f)[1] == '.jpg']
logos = [os.path.join(dp, f) for dp, dn, filenames in os.walk(logopath) for f in filenames if
         os.path.splitext(f)[1] == '.jpg']

print
len(bgrs)
print
len(logos)


def generate():
    imageset = ''
    fileindir = 20000
    actdir = -1
    success = 0
    for i, logofile in enumerate(logos):
        f = logofile.split('/')[-1].split('.')[0]
        fwithext = f + '.jpg'

        if i < len(bgrs):
            bgrfile = bgrs[i]
        else:
            return imageset
        bgr = cv2.imread(bgrfile)
        if bgr is None:
            print
            bgrfile
            continue
        bgrheight, bgrwidth, bgrchannel = bgr.shape
        ratio = float(bgrheight) / bgrwidth
        if ratio < 0.25 or ratio > 4:
            continue

        logo = cv2.imread(logofile)
        logoheight, logowidth, logochannel = logo.shape
        if bgrwidth < logowidth or bgrheight < logoheight:
            continue

        x = int(random.uniform(0, bgrwidth - logowidth))
        y = int(random.uniform(0, bgrheight - logoheight))

        tlx = float(random.uniform(0, 0.1))
        tlyp = float(random.uniform(0, 0.1))
        trx = float(random.uniform(-0.1, 0.1))
        tryp = float(random.uniform(0, 0.1))
        brx = float(random.uniform(-0.1, 0.1))
        bryp = float(random.uniform(-0.1, 0.1))
        blx = float(random.uniform(0, 0.1))
        blyp = float(random.uniform(-0.1, 0.1))

        scalehighH = float(bgrheight) / (logoheight * 4)
        scalehighW = float(bgrwidth) / (logowidth * 4)
        scalelowH = float(bgrheight) / (logoheight * 12)
        scalelowW = float(bgrwidth) / (logowidth * 12)
        scale = float(random.uniform(min(scalelowH, scalelowW), min(scalehighH, scalehighW)))

        logo = cv2.resize(logo, (0, 0), fx=scale, fy=scale)
        logoheight, logowidth, logochannel = logo.shape
        logo = transparentBgr(logo)
        if random.uniform(0, 1) < float(2) / 3:
            bgrmeanr = np.mean(bgr[:, :, 2])
            bgrmeang = np.mean(bgr[:, :, 1])
            bgrmeanb = np.mean(bgr[:, :, 0])
            logo = coloredBgr(logo, [bgrmeanb, bgrmeang, bgrmeanr])
        np.copyto(bgr[y:y + logoheight, x:x + logowidth], logo)

        inp = np.float32([[0, 0], [logowidth - 1, 0], [logowidth - 1, logoheight - 1], [0, logoheight - 1]])
        out = np.float32(
            [[int(logowidth * (tlx)), int(logoheight * (tlyp))], [int(logowidth * (1 + trx)), int(logoheight * (tryp))], \
             [int(logowidth * (1 + brx)), int(logoheight * (1 + bryp))],
             [int(logowidth * (blx)), int(logoheight * (1 + blyp))]])

        warp = np.zeros((int(max(out[1, 0], out[2, 0])), int(max(out[2, 1], out[3, 1])), 4), np.uint8)

        warp[:, :, :] = [255, 255, 255, 0]

        # calculate the perspective transform matrix and warp
        # the perspective to grab the screen
        M = cv2.getPerspectiveTransform(inp, out)
        logo = cv2.cvtColor(logo, cv2.COLOR_BGR2BGRA)
        warp = cv2.warpPerspective(logo, M, (int(max(out[1, 0], out[2, 0])), int(max(out[2, 1], out[3, 1]))))

        rows, cols, channels = warp.shape

        shiftrows = random.randint(5, bgrheight - rows)
        shiftcols = random.randint(5, bgrwidth - cols)

        roi = bgr[shiftrows:shiftrows + rows, shiftcols:shiftcols + cols]

        overlay_img = warp[:, :, :3]  # Grab the BRG planes
        overlay_mask = warp[:, :, 3:]  # And the alpha plane

        overlay_mask = cv2.erode(overlay_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))
        overlay_mask = cv2.blur(overlay_mask, (1, 1))

        # Again calculate the inverse mask
        background_mask = 255 - overlay_mask

        # Turn the masks into three channel, so we can use them as weights
        overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
        background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

        # Create a masked out face image, and masked out overlay
        # We convert the images to floating point in range 0.0 - 1.0
        face_part = (roi * (1 / 255.0)) * (background_mask * (1 / 255.0))
        overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

        # And finally just add them together, and rescale it back to an 8bit integer image
        dst = np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

        bgr[shiftrows:shiftrows + rows, shiftcols:shiftcols + cols] = dst

        blurredlogo = cv2.blur(bgr[shiftrows - 5:shiftrows + rows + 5, shiftcols + 5:shiftcols + cols + 5], (3, 3))
        np.copyto(bgr[shiftrows - 5:shiftrows + rows + 5, shiftcols + 5:shiftcols + cols + 5], blurredlogo)

        if i % fileindir == 0:
            actdir += 1
            os.makedirs(os.path.join(imagesoutput, str(actdir)))
            os.makedirs(os.path.join(bboxesoutput, str(actdir)))

        cv2.imwrite(os.path.join(imagesoutput, str(actdir), fwithext), bgr)
        warp[:, :, :] = [255, 255, 255, 0]

        imageset += str(actdir) + '/' + f + '\n'

        with open(os.path.join(bboxesoutput, str(actdir), fwithext + '.bboxes.txt'), 'w') as bbox:
            bbox.write(str(shiftcols) + ' ' + str(shiftrows) + ' ' + str(shiftcols + cols) + ' ' + str(
                shiftrows + rows) + ' logo')
        success += 1

        if i % 1000 == 0:
            percentage = float(i) / len(bgrs) * 100
            print
            str(i) + '/' + str(len(bgrs)) + ' --> ' + str(int(percentage)) + '%'
    return imageset


imageset = generate()

with open(os.path.join(imagesetsoutput, imagesetf), 'w') as ims:
    ims.write(imageset)