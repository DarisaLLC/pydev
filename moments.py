import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import color, img_as_ubyte, img_as_float
from skimage import io


def raw_moment(data, i_order, j_order):
    nrows, ncols = data.shape
    y_indices, x_indicies = np.mgrid[:nrows, :ncols]
    return (data * x_indicies ** i_order * y_indices ** j_order).sum()


def moments_cov(data):
    data_sum = data.sum()
    m10 = raw_moment(data, 1, 0)
    m01 = raw_moment(data, 0, 1)
    x_centroid = m10 / data_sum
    y_centroid = m01 / data_sum
    u11 = (raw_moment(data, 1, 1) - x_centroid * m01) / data_sum
    u20 = (raw_moment(data, 2, 0) - x_centroid * m10) / data_sum
    u02 = (raw_moment(data, 0, 2) - y_centroid * m01) / data_sum
    cov = np.array([[u20, u11], [u11, u02]])
    return cov


def moment_e(img):
    cov = moments_cov(img)
    evals, evecs = np.linalg.eig(cov)
    sort_indices = np.argsort(evals)[::-1]
    x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    x_v2, y_v2 = evecs[:, sort_indices[1]]
    return evals, evecs


if __name__ == '__main__':
    img = io.imread('./projects/pupil/images/blob_axes.png')
    y, x = np.nonzero(img)
    x = x - np.mean(x)
    y = y - np.mean(y)
    coords = np.vstack([x, y])

    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)

    sort_indices = np.argsort(evals)[::-1]
    x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    x_v2, y_v2 = evecs[:, sort_indices[1]]

    scale = 20
    plt.plot([x_v1 * -scale * 2, x_v1 * scale * 2],
             [y_v1 * -scale * 2, y_v1 * scale * 2], color='red')
    plt.plot([x_v2 * -scale, x_v2 * scale],
             [y_v2 * -scale, y_v2 * scale], color='blue')
    plt.plot(x, y, 'k.')
    plt.axis('equal')
    plt.gca().invert_yaxis()  # Match the image system with origin at top left
    plt.show()
