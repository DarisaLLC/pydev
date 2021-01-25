
import sys
sys.path.append('..')

import cv2

import ellipse_detection as ed


def main():
    image = cv2.imread(sys.argv[1])
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    ellipse_detector = ed.EllipseDetector()
    ellipses = ellipse_detector.detect(image_gray)

    for ellipse in ellipses:
        image_ellipse = image.copy()
        ellipse.draw(image_ellipse)
        cv2.imshow('ellipse', image_ellipse)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()