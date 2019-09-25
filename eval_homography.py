import sys

import cv2
import numpy as np
from skimage import color
from skimage.util import img_as_float
from skimage.util import img_as_ubyte
from matplotlib import pyplot as plt

def mutual_information(hgram):
    """ Mutual information for joint histogram
    """
    # Convert bins counts to probability values
    pxy = hgram / float ( np.sum ( hgram ) )
    px = np.sum ( pxy, axis=1 )  # marginal for x over y
    py = np.sum ( pxy, axis=0 )  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    return np.sum ( pxy[nzs] * np.log ( pxy[nzs] / px_py[nzs] ) )


# Use the keypoints to stitch the images
def get_stitched_image(img1, img2, M):
    # Get width and height of input images
    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]

    # Get the canvas dimesions
    img1_dims = np.float32 ( [[0, 0], [0, w1], [h1, w1], [h1, 0]] ).reshape ( -1, 1, 2 )
    img2_dims_temp = np.float32 ( [[0, 0], [0, w2], [h2, w2], [h2, 0]] ).reshape ( -1, 1, 2 )

    # Get relative perspective of second image
    img2_dims = cv2.perspectiveTransform ( img2_dims_temp, M )

    # Resulting dimensions
    result_dims = np.concatenate ( (img1_dims, img2_dims), axis=0 )

    # Getting images together
    # Calculate dimensions of match points
    [x_min, y_min] = np.int32 ( result_dims.min ( axis=0 ).ravel () - 0.5 )
    [x_max, y_max] = np.int32 ( result_dims.max ( axis=0 ).ravel () + 0.5 )

    # Create output array after affine transformation
    transform_dist = [-x_min, -y_min]
    transform_array = np.array ( [[1, 0, transform_dist[0]],
                                  [0, 1, transform_dist[1]],
                                  [0, 0, 1]] )

    # Warp images to get the resulting image
    result_img = cv2.warpPerspective ( img2, transform_array.dot ( M ),
                                       (x_max - x_min, y_max - y_min) )
    result_img[transform_dist[1]:w1 + transform_dist[1],
    transform_dist[0]:h1 + transform_dist[0]] = img1

    # Return the result
    return result_img


# Find SIFT and return Homography Matrix
def get_sift_homography(img1, img2):
    # Initialize SIFT
    sift = cv2.xfeatures2d.SIFT_create ()

    # Extract keypoints and descriptors
    k1, d1 = sift.detectAndCompute ( img1, None )
    k2, d2 = sift.detectAndCompute ( img2, None )

    # Bruteforce matcher on the descriptors
    bf = cv2.BFMatcher ()
    matches = bf.knnMatch ( d1, d2, k=2 )

    # Make sure that the matches are good
    verify_ratio = 0.8  # Source: stackoverflow
    verified_matches = []
    for m1, m2 in matches:
        # Add to array only if it's a good match
        if m1.distance < 0.8 * m2.distance:
            verified_matches.append ( m1 )

    # Mimnum number of matches
    min_matches = 8
    if len ( verified_matches ) > min_matches:

        # Array to store matching points
        img1_pts = []
        img2_pts = []

        # Add matching points to array
        for match in verified_matches:
            img1_pts.append ( k1[match.queryIdx].pt )
            img2_pts.append ( k2[match.trainIdx].pt )
        img1_pts = np.float32 ( img1_pts ).reshape ( -1, 1, 2 )
        img2_pts = np.float32 ( img2_pts ).reshape ( -1, 1, 2 )

        # Compute homography matrix
        M, mask = cv2.findHomography ( img1_pts, img2_pts, cv2.RANSAC, 5.0 )
        return M
    else:
        print ( 'Error: Not enough matches' )
        exit ()


# Equalize Histogram of Color Images
def equalize_histogram_color(img):
    img_yuv = cv2.cvtColor ( img, cv2.COLOR_BGR2YUV )
    img_yuv[:, :, 0] = cv2.equalizeHist ( img_yuv[:, :, 0] )
    img = cv2.cvtColor ( img_yuv, cv2.COLOR_YUV2BGR )
    return img


def MutualInformation(a, b, plotter=None):
    hist_2d, x_edges, y_edges = np.histogram2d ( a.ravel (), b.ravel (), bins=20 )
    mu = mutual_information ( hist_2d )
    if plotter != None:
        hist_2d_log = np.zeros ( hist_2d.shape )
        non_zeros = hist_2d != 0
        hist_2d_log[non_zeros] = np.log ( hist_2d[non_zeros] )
        plotter.imshow ( hist_2d_log.T, origin='lower', cmap=plt.cm.gray )
    return mu


def skiimage_display(img):
    return cv2.cvtColor ( img, cv2.COLOR_BGR2RGB )


def main_mu ():
    # Get input set of images
    img1 = cv2.imread ( sys.argv[1] )
    img2 = cv2.imread ( sys.argv[2] )
    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]

    img1 = cv2.resize ( img1, (int ( w1 / 3 ), int ( h1 / 3 )), interpolation=cv2.INTER_AREA )
    img2 = cv2.resize ( img2, (int ( w2 / 3 ), int ( h2 / 3 )), interpolation=cv2.INTER_AREA )
    lab1 = cv2.cvtColor ( img1, cv2.COLOR_BGR2LAB )
    lab2 = cv2.cvtColor ( img2, cv2.COLOR_BGR2LAB )
    # Split LAB channels
    L1, a1, b1 = cv2.split ( lab1 )
    L2, a2, b2 = cv2.split ( lab2 )


    f, axs = plt.subplots ( 2, 3, figsize=(20, 10), frameon=False,
                            subplot_kw={'xticks': [], 'yticks': []} )
    axs[0, 0].imshow ( skiimage_display(img1) )
    axs[0, 1].imshow ( skiimage_display(img2) )

    MutualInformation ( img_as_ubyte ( a1 ), img_as_ubyte ( a2 ), axs[1, 0] )
    MutualInformation ( img_as_ubyte ( b1 ), img_as_ubyte ( b2 ), axs[1, 1] )
    MutualInformation ( img_as_ubyte ( L1 ), img_as_ubyte ( L2 ), axs[1, 2] )

    plt.show()

# Main function definition
def main():
    # Get input set of images
    img1 = cv2.imread ( sys.argv[1] )
    img2 = cv2.imread ( sys.argv[2] )
    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]

    img1 = cv2.resize ( img1, (int ( w1 / 3 ), int ( h1 / 3 )), interpolation=cv2.INTER_AREA )
    img2 = cv2.resize ( img2, (int ( w2 / 3 ), int ( h2 / 3 )), interpolation=cv2.INTER_AREA )

    # Equalize histogram
    img1 = equalize_histogram_color ( img1 )
    img2 = equalize_histogram_color ( img2 )

    # Show input images
    input_images = np.hstack( (img1, img2) )
    cv2.imshow ('Input Images', input_images)
    input_image_name = '/Volumes/medvedev/_SP/results/inputs.png'
    cv2.imwrite(input_image_name, input_images)

    # Use SIFT to find keypoints and return homography matrix
    M = get_sift_homography ( img1, img2 )

    # Stitch the images together using homography matrix
    result_image = get_stitched_image ( img2, img1, M )

    # Write the result to the same directory
    result_image_name = '/Volumes/medvedev/_SP/results/homography.png'
    cv2.imwrite(result_image_name, result_image)

    # Show the resulting image
    cv2.imshow ( 'Result', result_image )
    cv2.waitKey ()


# Call main function
if __name__ == '__main__':
    main ()
