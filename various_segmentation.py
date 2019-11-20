import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
import cv2
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from imutils import auto_canny

def build_lappyr(img, leveln=6, dtype=np.uint8):
    img = dtype(img)
    levels = []
    for i in range(leveln - 1):
        next_img = cv2.pyrDown(img)
        levels.append(next_img)
        img = next_img
    levels.append(img)
    return levels


if __name__ == "__main__":

    if len(sys.argv) < 2:
        exit(1)
    print(sys.argv[1])

    if not Path(sys.argv[1]).is_file(): exit(1)
    img = cv2.imread(sys.argv[1])
#    plevels = build_lappyr(img)
    image = cv2.pyrDown(img)
    img = cv2.pyrUp(image)
    #img = plevels[1]
    ac = auto_canny(img)

#    img = img_as_float(astronaut()[::2, ::2])

    segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=20)
    segments_slic = slic(img, n_segments=1250, compactness=10, sigma=2)
    segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    gradient = sobel(rgb2gray(img))
    segments_watershed = watershed(gradient, markers=1250, compactness=0.001)

    print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")
    print(f"SLIC number of segments: {len(np.unique(segments_slic))}")
    print(f"Quickshift number of segments: {len(np.unique(segments_quick))}")

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    ax[0, 0].imshow(mark_boundaries(img, segments_fz))
    ax[0, 0].set_title("Felzenszwalbs's method")
    ax[0, 1].imshow(ac)
    ax[0, 1].set_title('Auto Canny')
    ax[1, 0].imshow(mark_boundaries(img, segments_quick))
    ax[1, 0].set_title('Quickshift')
    ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
    ax[1, 1].set_title('Compact watershed')

    for a in ax.ravel():
        a.set_axis_off()

    plt.tight_layout()
    plt.show()