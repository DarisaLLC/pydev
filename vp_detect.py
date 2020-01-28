"""
Based on Python + OpenCV Implementation of the vanishing point algorithm by Xiaohu Lu
et al. - http://xiaohulugo.github.io/papers/Vanishing_Point_Detection_WACV2017.pdf.


"""

import sys
import cv2
import numpy as np
from itertools import combinations
from pathlib import Path
from rectangle import contains
import scipy
from scipy import linalg
from utils import get_line_angle

class vp_detection(object):
    """
    VP Detection Object

    Args:
        length_thresh: Line segment detector threshold (default=30)
        principal_point: Principal point of the image (in pixels)
        focal_length: Focal length of the camera (in pixels)
        seed: Seed for reproducibility due to RANSAC
    """

    def __init__(self, length_thresh=30, principal_point=None,
                 focal_length=20, seed=None):
        self._length_thresh = length_thresh
        self._principal_point = principal_point
        self._estimated_focal_length = focal_length
        self._focal_length = None
        self._angle_thresh = np.pi / 30  # For displaying debug image
        self._vps = None  # For storing the VPs in 3D space
        self._vps_2D = None  # For storing the VPs in 2D space
        self.__img = None  # Stores the image locally
        self.__clusters = None  # Stores which line index corresponds to what VP
        self.__tol = 1e-8  # Tolerance for floating point comparison
        self.__angle_tol = np.pi / 2.5  # (pi / 180 * (60 degrees)) --> +/- 30 deg
        self.__lines = None  # Stores the line detections internally
        self.__zero_value = 0.001  # Threshold to check augmented coordinate
        # Anything less than __tol gets set to this
        self.__seed = seed  # Set seed for reproducibility
        noise_ratio = 0.5  # Outlier/inlier ratio for RANSAC estimation
        # Probability of all samples being inliers
        p = (1.0 / 3.0) * ((1.0 - noise_ratio) ** 2.0)

        # Total number of iterations for RANSAC
        conf = 0.9999
        self.__ransac_iter = int(np.log(1 - conf) / np.log(1.0 - p))

    @property
    def length_thresh(self):
        """
        Length threshold for line segment detector
        Returns:
            The minimum length required for a line
        """
        return self._length_thresh

    @length_thresh.setter
    def length_thresh(self, value):
        """
        Length threshold for line segment detector
        Args:
            value: The minimum length required for a line
        Raises:
            ValueError: If the threshold is 0 or negative
        """
        if value <= 0:
            raise ValueError('Invalid threshold: {}'.format(value))

        self._length_thresh = value

    @property
    def principal_point(self):
        """
        Principal point for VP Detection algorithm
        Returns:
            The minimum length required for a line
        """
        return self._principal_point

    @principal_point.setter
    def principal_point(self, value):
        """
        Principal point for VP Detection algorithm
        Args:
            value: A list or tuple of two elements denoting the x and y coordinates

        Raises:
            ValueError: If the input is not a list or tuple and there aren't two elements
        """
        try:
            assert isinstance(value, (list, tuple)) and not isinstance(value, str)
            assert len(value) == 2
        except AssertionError:
            raise ValueError('Invalid principal point: {}'.format(value))

        self._length_thresh = value

    @property
    def focal_length(self):
        """
        Focal length for VP detection algorithm
        Returns:
            The focal length in pixels
        """
        return self._focal_length

    @property
    def vps(self):
        """
        Vanishing points of the image in 3D space.
        Returns:
            A np array where each row is a point and each column is a component / coordinate
        """
        return self._vps

    @property
    def vps_2D(self):
        """
        Vanishing points of the image in 2D image coordinates.
        Returns:
            A np array where each row is a point and each column is a component / coordinate
        """
        return self._vps_2D

    @property
    def lines(self):
        """
        Vanishing points of the image in 2D image coordinates.
        Returns:
            A np array where each row is a point and each column is a component / coordinate
        """
        return self.__lines

    @property
    def clusters(self):
        """
        Cluster of based on which VP they contributed to.

        Returns:
        # For each VP
        # Returns a list of 3 elements
        # Each element contains which line index corresponds to which VP
    """

        return self.__clusters

    def __detect_lines(self, img):
        """
        Detects lines using OpenCV LSD Detector
        """
        # Convert to grayscale if required
        if len(img.shape) == 3:
            img_copy = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_copy = img

        # Create LSD detector with default parameters
        lsd = cv2.createLineSegmentDetector(0)

        # Detect lines in the image
        # Returns a NumPy array of type N x 1 x 4 of float32
        # such that the 4 numbers in the last dimension are (x1, y1, x2, y2)
        # These denote the start and end positions of a line
        lines = lsd.detect(img_copy)[0]

        # Remove singleton dimension
        lines = lines[:, 0]

        # Filter out the lines whose length is lower than the threshold
        dx = lines[:, 2] - lines[:, 0]
        dy = lines[:, 3] - lines[:, 1]
        lengths = np.sqrt(dx * dx + dy * dy)
        mask = lengths >= self._length_thresh
        lines = lines[mask]

        # Store the lines internally
        self.__lines = lines

        # Return the lines
        return lines

    def __find_vp_hypotheses_two_lines(self):
        """
        Finds the VP hypotheses using pairs of lines
        """
        # Number of detected lines
        N = self.__lines.shape[0]

        # Number of bins for longitude - 360 bins so 1 deg. per bin
        # For estimating second VP along the great circle distance of the
        # first VP
        num_bins_vp2 = 360
        vp2_step = np.pi / 180.0  # Step in radians

        # Store the equations of the line, lengths and orientations
        # for each line segment
        p1 = np.column_stack((self.__lines[:, :2],
                              np.ones(N, dtype=np.float32)))
        p2 = np.column_stack((self.__lines[:, 2:],
                              np.ones(N, dtype=np.float32)))
        cross_p = np.cross(p1, p2)
        dx = p1[:, 0] - p2[:, 0]
        dy = p1[:, 1] - p2[:, 1]
        lengths = np.sqrt(dx * dx + dy * dy)
        orientations = np.arctan2(dy, dx)

        # Perform wraparound - [-pi, pi] --> [0, pi]
        # All negative angles map to their mirrored positive counterpart
        orientations[orientations < 0] = orientations[orientations < 0] + np.pi

        # Keep these around
        self.__cross_p = cross_p
        self.__lengths = lengths
        self.__orientations = orientations

        # Stores the VP hypotheses - 3 per longitude for each RANSAC iteration
        # First dimension - VP triplet proposal for a RANSAC iteration
        # Second dimension - VPs themselves
        # Third dimension - VP component
        vp_hypos = np.zeros((self.__ransac_iter * num_bins_vp2, 3, 3),
                            dtype=np.float32)

        i = 0

        if self.__seed is not None:
            gen = np.random.RandomState(self.__seed)

        # For each iteration...
        while i < self.__ransac_iter:
            # Get two random indices
            if self.__seed is not None:
                (idx1, idx2) = gen.permutation(N)[:2]
            else:
                (idx1, idx2) = np.random.permutation(N)[:2]

            # Get the first VP proposal in the image
            vp1_img = np.cross(cross_p[idx1], cross_p[idx2])

            # Try again if at infinity
            if np.abs(vp1_img[2]) < self.__tol:
                continue

            # Find where it intersects in the sphere
            vp1 = np.zeros(3, dtype=np.float32)
            vp1[:2] = vp1_img[:2] / vp1_img[2] - self._principal_point
            vp1[2] = self._estimated_focal_length

            # Normalize
            vp1 /= np.sqrt(np.sum(np.square(vp1)))

            # Get the other two VPs
            # Search along the circumference of the sphere
            la = np.arange(num_bins_vp2) * vp2_step
            kk = vp1[0] * np.sin(la) + vp1[1] * np.cos(la)
            phi = np.arctan(-vp1[2] / kk)

            # Convert back to Cartesian coordinates
            vp2 = np.column_stack([np.sin(phi) * np.sin(la),
                                   np.sin(phi) * np.cos(la),
                                   np.cos(phi)])

            # Enforce points at infinity to be finite
            vp2[np.abs(vp2[:, 2]) < self.__tol, 2] = self.__zero_value
            # Normalize
            vp2 /= np.sqrt(np.sum(np.square(vp2), axis=1, keepdims=True))
            vp2[vp2[:, 2] < 0, :] *= -1.0  # Ensure direction is +z

            vp3 = np.cross(vp1, vp2)  # Third VP is orthogonal to the two
            vp3[np.abs(vp3[:, 2]) < self.__tol, 2] = self.__zero_value
            vp3 /= np.sqrt(np.sum(np.square(vp3), axis=1, keepdims=True))
            vp3[vp3[:, 2] < 0, :] *= -1.0

            # Place proposals in corresponding locations
            vp_hypos[i * num_bins_vp2: (i + 1) * num_bins_vp2, 0, :] = vp1
            vp_hypos[i * num_bins_vp2: (i + 1) * num_bins_vp2, 1, :] = vp2
            vp_hypos[i * num_bins_vp2: (i + 1) * num_bins_vp2, 2, :] = vp3

            # Move to the next iteration
            i += 1

        return vp_hypos

    def __get_sphere_grids(self):
        """
        Builds spherical voting grid to determine which VP has the most support
        """

        # Determine number of bins for latitude and longitude
        bin_size = np.pi / 180.0
        lat_span = np.pi / 2.0
        long_span = 2.0 * np.pi
        num_bins_lat = int(lat_span / bin_size)
        num_bins_lon = int(long_span / bin_size)

        # Get indices for every unique pair of lines
        combos = list(combinations(range(self.__lines.shape[0]), 2))
        combos = np.asarray(combos, dtype=np.int)

        # For each pair, determine where the lines intersect
        pt_intersect = np.cross(self.__cross_p[combos[:, 0]],
                                self.__cross_p[combos[:, 1]])

        # Ignore if points are at infinity
        mask = np.abs(pt_intersect[:, 2]) >= self.__tol

        # To determine if two points map to the same VP in spherical
        # coordinates, their difference in angle must be less than
        # some threshold
        ang = np.abs(self.__orientations[combos[:, 0]] -
                     self.__orientations[combos[:, 1]])
        ang = np.minimum(np.pi - ang, ang)
        mask = np.logical_and(mask, np.abs(ang) <= self.__angle_tol)

        # Get the points, angles and combinations that are
        # left
        pt_intersect = pt_intersect[mask]
        ang = ang[mask]
        combos = combos[mask]

        # Determine corresponding lat and lon mapped to the sphere
        X = (pt_intersect[:, 0] / pt_intersect[:, 2]) - self._principal_point[0]
        Y = (pt_intersect[:, 1] / pt_intersect[:, 2]) - self._principal_point[1]
        Z = self._estimated_focal_length
        lat = np.arccos(Z / np.sqrt(X * X + Y * Y + Z * Z))
        lon = np.arctan2(X, Y) + np.pi

        # Get corresponding bin locations
        la_bin = (lat / bin_size).astype(np.int)
        lon_bin = (lon / bin_size).astype(np.int)
        la_bin[la_bin >= num_bins_lat] = num_bins_lat - 1
        lon_bin[lon_bin >= num_bins_lon] = num_bins_lon - 1

        # Add their weighted vote to the corresponding bin
        # Get 1D bin coordinate so we can take advantage
        # of bincount method, then reshape back to 2D
        bin_num = la_bin * num_bins_lon + lon_bin
        weights = np.sqrt(self.__lengths[combos[:, 0]] *
                          self.__lengths[combos[:, 1]]) * (np.sin(2.0 * ang) + 0.2)

        sphere_grid = np.bincount(bin_num, weights=weights,
                                  minlength=num_bins_lat * num_bins_lon).reshape(
            (num_bins_lat, num_bins_lon)).astype(np.float32)

        # Add the 3 x 3 smoothed votes on top of the original votes for
        # stability (refer to paper)
        sphere_grid += cv2.filter2D(sphere_grid, -1, (1.0 / 9.0) * np.ones((3, 3)))
        return sphere_grid

    def __get_best_vps_hypo(self, sphere_grid, vp_hypos):
        # Number of hypotheses
        N = vp_hypos.shape[0]

        # Bin size - 1 deg. in radians
        bin_size = np.pi / 180.0

        # Ignore any values whose augmented coordinate are less than
        # the threshold or bigger than magnitude of 1
        # Each row is a VP triplet
        # Each column is the z coordinate
        mask = np.logical_and(np.abs(vp_hypos[:, :, 2]) >= self.__tol,
                              np.abs(vp_hypos[:, :, 2]) <= 1.0)

        # Create ID array for VPs
        ids = np.arange(N).astype(np.int)
        ids = np.column_stack([ids, ids, ids])
        ids = ids[mask]

        # Calculate their respective lat and lon
        lat = np.arccos(vp_hypos[:, :, 2][mask])
        lon = np.arctan2(vp_hypos[:, :, 0][mask],
                         vp_hypos[:, :, 1][mask]) + np.pi

        # Determine which bin they map to
        la_bin = (lat / bin_size).astype(np.int)
        lon_bin = (lon / bin_size).astype(np.int)
        la_bin[la_bin == 90] = 89
        lon_bin[lon_bin == 360] = 359

        # For each hypotheses triplet of VPs, calculate their final
        # votes by summing the contributions of each VP for the
        # hypothesis
        weights = sphere_grid[la_bin, lon_bin]
        votes = np.bincount(ids, weights=weights,
                            minlength=N).astype(np.float32)

        # Find best hypothesis by determining which triplet has the largest
        # votes
        best_idx = np.argmax(votes)
        final_vps = vp_hypos[best_idx]
        vps_2D = self._estimated_focal_length * (final_vps[:, :2] / final_vps[:, 2][:, None])
        vps_2D += self._principal_point

        # Find the coordinate with the largest vertical value
        # This will be the last column of the output
        z_idx = np.argmax(np.abs(vps_2D[:, 1]))
        ind = np.arange(3).astype(np.int)
        mask = np.ones(3, dtype=np.bool)
        mask[z_idx] = False
        ind = ind[mask]

        # Next, figure out which of the other two coordinates has the smallest
        # x coordinate - this would be the left leaning VP
        vps_trim = vps_2D[mask]
        x_idx = np.argmin(vps_trim[:, 0])
        x_idx = ind[x_idx]

        # Finally get the right learning VP
        mask[x_idx] = False
        x2_idx = np.argmax(mask)

        # Re-arrange the order
        # Right VP is first - x-axis would be to the right
        # Left VP is second - y-axis would be to the left
        # Vertical VP is third - z-axis would be vertical
        final_vps = final_vps[[x2_idx, x_idx, z_idx], :]
        vps_2D = vps_2D[[x2_idx, x_idx, z_idx], :]

        # Save for later
        self._vps = final_vps
        self._vps_2D = vps_2D
        return final_vps

    def __cluster_lines(self, vps_hypos):
        """
        Groups the lines based on which VP they contributed to.
        """

        # Extract out line coordinates
        x1 = self.__lines[:, 0]
        y1 = self.__lines[:, 1]
        x2 = self.__lines[:, 2]
        y2 = self.__lines[:, 3]

        # Get midpoint of each line
        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0

        # Get the direction vector of the line detection
        # Also normalize
        dx = x1 - x2
        dy = y1 - y2
        norm_factor = np.sqrt(dx * dx + dy * dy)
        dx /= norm_factor
        dy /= norm_factor

        # Get the direction vector from each detected VP
        # to the midpoint of the line and normalize
        xp = self._vps_2D[:, 0][:, None] - xc[None]
        yp = self._vps_2D[:, 1][:, None] - yc[None]
        norm_factor = np.sqrt(xp * xp + yp * yp)
        xp /= norm_factor
        yp /= norm_factor

        # Calculate the dot product then find the angle between the midpoint
        # of each line and each VPs
        # We calculate the angle that each make with respect to each line and
        # and choose the VP that has the smallest angle with the line
        dotp = dx[None] * xp + dy[None] * yp
        dotp[dotp > 1.0] = 1.0
        dotp[dotp < -1.0] = -1.0
        ang = np.arccos(dotp)
        ang = np.minimum(np.pi - ang, ang)

        # For each line, which VP is the closest?
        # Get both the smallest angle and index of the smallest
        min_ang = np.min(ang, axis=0)
        idx_ang = np.argmin(ang, axis=0)

        # Don't consider any lines where the smallest angle is larger than
        # a similarity threshold
        mask = min_ang <= self._angle_thresh

        # For each VP, figure out the line indices
        # Create a list of 3 elements
        # Each element contains which line index corresponds to which VP
        self.__clusters = [np.where(np.logical_and(mask, idx_ang == i))[0] for i in range(3)]

    def find_image_vps (self, img):
        """
        Find the vanishing points given the input image
        Args:
            img: the image read in with `cv2.imread`

        Returns:
            A np array where each row is a point and each column is a component / coordinate.
            Additionally, the VPs are ordered such that the right most VP is the
            first row, the left most VP is the second row and the vertical VP is
            the last row
        """
        self.__clusters = None
        self.__img = img  # Keep a copy for later

        # Reset principal point if we haven't set it yet
        if self._principal_point is None:
            rows, cols = img.shape[:2]
            self._principal_point = np.array([cols / 2.0, rows / 2.0],
                                             dtype=np.float32)

        # Detect lines
        lines = self.__detect_lines(img)

        # Find VP candidates
        vps_hypos = self.__find_vp_hypotheses_two_lines()
        self.__vps_hypos = vps_hypos  # Save a copy

        # Map VP candidates to sphere
        sphere_grid = self.__get_sphere_grids()

        # Find the final VPs
        best_vps = self.__get_best_vps_hypo(sphere_grid, vps_hypos)
        self.__final_vps = best_vps  # Save a copy
        # Cluster lines based on which VP they contributed to
        if self.__clusters is None:
            self.__cluster_lines(self.__vps_hypos)
#        self.__clusters = None  # Reset because of new image
        return best_vps

    def get_intrinsic_camera_transformation(self):
        """
        Gets the 3x3 intrinsic camera matrix to transform from camera space to image (pixel) space

        :Rtype:
            `np.array`

        cv2.calibrationMatrixValues(cameraMatrix, imageSize, apertureWidth, apertureHeight) → fovx, fovy, focalLength, principalPoint, aspectRatio¶
        """
        # form the appropriate transformation matrix from unit space
        xform = np.zeros((3, 3))
        xform[2, 2] = 1
        xform[0, 0] = xform[1, 1] = self.focal_length
        xform[0, 2] = self.principal_point[0]
        xform[1, 2] = self.principal_point[1]
        return xform

    def calculate_focal_length(self):

        # compute focal length as in "Camera calibration using 2 or 3 vanishing points" (Orghidan et al. 2012)
        vps = self._vps_2D
        vpmin = np.linalg.norm(vps, axis=1).argmin()
        vpsecond = 0 if vpmin == 1 else 1


        self._focal_length = math.sqrt(
            np.linalg.norm((self.principal_point - vps[vpmin]) * (vps[vpsecond] - self.principal_point)))
        error = self._focal_length - self._estimated_focal_length
        print('focal length',self.focal_length, ' computed vs EXIF ', error * error)

    def solve_world_to_cam(self, origin=(0, 0)):
        """
        Solve for the rotation and translation to get from world space to camera space

        :Parameters:
            origin : `tuple`
                Where the origin lies in image space

        :Returns:
            A tuple of (R,t) where R is the rotation matrix and t is the translation vector to get
            into world space
        """
        # based on "Camera calibration using 2 or 3 vanishing points" (Orghidan et al. 2012)

        # first, need to solve for the scaling factor so can get rotation matrix R
        vs = self._vps_2D
        A = np.array([[vs[0][0], vs[1][0], vs[2][0]],
                         [vs[0][1], vs[1][1], vs[2][1]],
                         [vs[0][0] ** 2, vs[1][0] ** 2, vs[2][0] ** 2],
                         [vs[0][1] ** 2, vs[1][1] ** 2, vs[2][1] ** 2],
                         [vs[0][0] * vs[0][1], vs[1][0] * vs[1][1], vs[2][0] * vs[2][1]]])
        b = np.array([self.principal_point[0], self.principal_point[1],
                         self.focal_length ** 2 + self.principal_point[0] ** 2,
                         self.focal_length ** 2 + self.principal_point[1] ** 2,
                         self.principal_point[0] * self.principal_point[1]])
        x = scipy.linalg.pinv2(A).dot(b)
        l1 = math.sqrt(abs(x[0]))
        l2 = math.sqrt(abs(x[1]))
        l3 = math.sqrt(abs(x[2]))

        # now, plug this in to get the rotation matrix
        u1 = vs[0][0]
        v1 = vs[0][1]
        u2 = vs[1][0]
        v2 = vs[1][1]
        u3 = vs[2][0]
        v3 = vs[2][1]
        u0 = self.principal_point[0]
        v0 = self.principal_point[1]

        R = np.array([[l1 * (u1 - u0) / self.focal_length, l2 * (u2 - u0) / self.focal_length,
                          l3 * (u3 - u0) / self.focal_length],
                         [l1 * (v1 - v0) / self.focal_length, l2 * (v2 - v0) / self.focal_length,
                          l3 * (v3 - v0) / self.focal_length],
                         [l1, l2, l3]])

        # finally solve for the translation in image space
        # to do this, we assume the matrix KR maps to a space whose origin is at the camera
        # once we figure out where our given origin maps to in world space, that tells us
        # how much we want to translate in world space to get there
        K = self.get_intrinsic_camera_transformation()
        t = np.linalg.inv(K.dot(R)).dot(np.array([origin[0], origin[1], 1]))
        print('world to cam rotation\n', R)
        print('world to cam translation\n', t)

        def isclose(x, y, rtol=1.e-5, atol=1.e-8):
            return abs(x - y) <= atol + rtol * abs(y)

        def euler_angles_from_rotation_matrix(R):
            '''
            From a paper by Gregory G. Slabaugh (undated),
            "Computing Euler angles from a rotation matrix
            '''
            phi = 0.0
            if isclose(R[2, 0], -1.0):
                theta = math.pi / 2.0
                psi = math.atan2(R[0, 1], R[0, 2])
            elif isclose(R[2, 0], 1.0):
                theta = -math.pi / 2.0
                psi = math.atan2(-R[0, 1], -R[0, 2])
            else:
                theta = -math.asin(R[2, 0])
                cos_theta = math.cos(theta)
                psi = math.atan2(R[2, 1] / cos_theta, R[2, 2] / cos_theta)
                phi = math.atan2(R[1, 0] / cos_theta, R[0, 0] / cos_theta)
            return psi, theta, phi

        # print(euler_angles_from_rotation_matrix(R))
        return (R, t)



    def render_vp_output (self, img):

        colours = 255 * np.eye(3)
        # BGR format
        # First row is red, second green, third blue
        colours = colours[:, ::-1].astype(np.int).tolist()
        colours[2][1] = colours[2][2] = 255

        # Draw the outlier lines as black
        all_clusters = np.hstack(self.__clusters)
        status = np.ones(self.__lines.shape[0], dtype=np.bool)
        status[all_clusters] = False
        ind = np.where(status)[0]
        for (x1, y1, x2, y2) in self.__lines[ind]:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, 'X', (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,0,0), 1)

        for i in range(3):
            vp_x = self._vps_2D[i][0]
            vp_y = self._vps_2D[i][1]
            cv2.circle(img, (vp_x,vp_y), 10, colours[i], 5, cv2.LINE_AA)

        # For each cluster of lines, draw them in their right colour
        # For each lines mark the end that is further from the vp

        for i in range(3):
            for (x1, y1, x2, y2) in self.__lines[self.__clusters[i]]:
                vp_x = self._vps_2D[i][0]
                vp_y = self._vps_2D[i][1]
                dx = vp_x - x1
                dy = vp_y - y1
                d1 = np.sqrt(dx * dx + dy * dy)
                dx = vp_x - x2
                dy = vp_y - y2
                d2 = np.sqrt(dx * dx + dy * dy)
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),colours[i], 2, cv2.LINE_AA)

                if d2 > d1:
                    #cv2.circle(img, (x2, y2), 7, colours[i], 2, cv2.LINE_AA)
                    cv2.putText(img, str(i), (int(x2), int(y2)), cv2.FONT_HERSHEY_PLAIN, 0.8,colours[i], 1)
                else:
                    #cv2.circle(img, (x1, y1), 7, colours[i], 2, cv2.LINE_AA)
                    cv2.putText(img, str(i), (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN, 0.8,colours[i], 1)

        # draw a line between first 2 vps
        cv2.line(img, (int(self._vps_2D[0][0]), int(self._vps_2D[0][1])),
                 (int(self._vps_2D[1][0]), int(self._vps_2D[1][1])),
                 (255, 0, 0), 4, cv2.LINE_AA)
        cv2.line(img, (int(self._vps_2D[0][0]), int(self._vps_2D[0][1])),
                 (int(self._vps_2D[2][0]), int(self._vps_2D[2][1])),
                 (0, 255, 0), 4, cv2.LINE_AA)
        cv2.line(img, (int(self._vps_2D[1][0]), int(self._vps_2D[1][1])),
                 (int(self._vps_2D[2][0]), int(self._vps_2D[2][1])),
                 (0, 0, 255), 4, cv2.LINE_AA)

        return img


    def create_debug_VP_image(self, show_image=False, save_image=None):
        if self.__clusters is None:
            self.__cluster_lines(self.__vps_hypos)

        if save_image is not None and not isinstance(save_image, str):
            raise ValueError('The save_image path should be a string')

        img = self.__img.copy()
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hue, satu, vol = cv2.split(img_hsv)
        img = np.dstack([gray, gray, gray])
        if len(img.shape) == 2:  # If grayscale, artificially make into RGB
            img = np.dstack([img, img, img])

        self.render_vp_output(img)

        # Show image if necessary
        if show_image:
            cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
            cv2.imshow('Display', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Save image if necessary
        if save_image is not None and save_image != '':
            cv2.imwrite(save_image, img)

        return img


def main(input_path, roi):
    # Extract command line arguments
    length_thresh = 30
    principal_point = None
    focal_length = 20
    debug_mode = 1
    debug_show = 1
    debug_path = None
    seed = 1337
    reduce = 1

    print('Input path: {}'.format(input_path))
    print('Seed: {}'.format(seed))
    print('Line length threshold: {}'.format(length_thresh))
    print('Focal length: {}'.format(focal_length))

    # Create object
    vpd = vp_detection(length_thresh, principal_point, focal_length, seed)

    # Run VP detection algorithm
    img = cv2.imread(input_path, -1)
    rows, cols, channels = img.shape
    if not (roi is None):
        if roi[3] > roi[1] and roi[2] > roi[0] and roi[3] < rows and roi[2] < cols:
            img=img[roi[1]:roi[3],roi[0]:roi[2]]
    rows, cols, channels = img.shape
    img = cv2.resize(img, (int(cols / reduce), int(rows / reduce)), cv2.INTER_AREA)

    vps = vpd.find_image_vps(img)
    vpd.calculate_focal_length()
    vpd.solve_world_to_cam()
    K = vpd.get_intrinsic_camera_transformation()
    kp1 = K * np.transpose(K)
    w = np.linalg.inv(kp1)
    print(' Camera Intrinsic Matrix \n', vpd.get_intrinsic_camera_transformation())
    print(' Projective Transformation of the absolute conic\n', w)
    print('Zero Skew', w[0][1] == 0)
    print('Square Pixels', w[0][1] == 0 and w[0][0] == w[1][1])
    print('Principal point: {}'.format(vpd.principal_point))

    # Show VP information
    print("The vanishing points in 3D space are: ")
    for i, vp in enumerate(vps):
        print("Vanishing Point {:d}: {}".format(i + 1, vp))

    vp2D = vpd.vps_2D
    print("\nThe vanishing points in image coordinates are: ")
    x_coords = []
    y_coords = []
    for i, vp in enumerate(vp2D):
        x_coords.append(vp[0])
        y_coords.append(vp[1])
        print("Vanishing Point {:d}: {}".format(i + 1, vp))

    x_max = np.max(x_coords)
    x_min = np.min(x_coords)
    y_max = np.max(y_coords)
    y_min = np.min(y_coords)

    print(x_coords)
    print(y_coords)
    print((x_min, x_max,y_min,y_max))


    # Extra stuff
    if debug_mode or debug_show:
        st = "Creating debug image"
        if debug_show:
            st += " and showing to the screen"
        if debug_path is not None:
            st += "\nAlso writing debug image to: {}".format(debug_path)

        if debug_show or debug_path is not None:
            print(st)
            vpd.create_debug_VP_image(debug_show, debug_path)


if __name__ == "__main__":
    import sys
    import math

    argcnt = len(sys.argv)
    if argcnt < 2 or (not Path(sys.argv[1]).is_file() or not Path(sys.argv[1]).exists()):
        print(' File Does not exist or found ')
    else:
        roi = None
        if argcnt == 6:
            '''
            Indicate roi in the image
            '''
            tlx = int(sys.argv[2])
            tly = int(sys.argv[3])
            brx = int(sys.argv[4])
            bry = int(sys.argv[5])
            roi = [tlx,tly,brx,bry]
        main(sys.argv[1], roi)
