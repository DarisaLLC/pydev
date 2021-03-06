import numpy as np
from simple_logger import Logger
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
#from geometry import angle_x
import geometry as utils


__all__ = ["SegmentClusterer", "SegmentClustererParams", "ClusterCleaningParams"]


class SegmentClustererParams:
    """Parameters used by the :class:`.SegmentClusterer` related to segment clustering."""

    def __init__(self):
        """Initializes the segment clusterer parameters to their default value.

        :ivar num_init: Number of times the cluster detector should reinitialize the cluster search (only works if :attr:`self.cluster_type` is `"knn"`).
        :ivar num_clusters: Number of clusters to be searched.
        :ivar swipe_clusters: Boolean flag which indicates whether the cluster search should sweep over the possible number of clusters or not. (only works if :attr:`self.cluster_type` is `"gmm"`).
        :ivar cluster_type: String specifying which algorithm to used for cluster detection (`"knn"` : K-nearest neighbors, `"gmm"` : Gaussian mixture model).
        :ivar use_angles: Boolean flag indicating if the features to be used for clustering should include the segment angles.
        :ivar use_centers: Boolean flag indicating if the features to be used for clustering should include the segment centers.
        """
        #  Number of initializations to be performed when clustering.
        self.num_init = 10
        # Number of clusters to extract from the parameter space.
        self.num_clusters = 4
        # Boolean flag, if set to 'True' and 'cluster_type' is 'gmm', then the algorithm iterates the clustering
        # procedure over a range of number of clusters from 1 to 'num_clusters' and retains the best result.
        self.swipe_clusters = False
        # Clustering algorithm to be used, must be in ['gmm', 'knn'] which correspond to a full gaussian mixture model,
        # and k-nearest-neighbors respectively.
        self.cluster_type = "gmm"
        # Boolean flag indicating whether to consider angles in the clustering process.
        self.use_angles = True
        # Boolean flag indicating whether to consider segment centroids in the clustering process.
        self.use_centers = False


class ClusterCleaningParams:
    """Parameters used by the :class:`.SegmentClusterer` related to segment filtering."""

    def __init__(self):
        """Initializes the cluster cleaning parameters to their default value.

        :ivar max_angle_variation_mean: Segments whose angle with the mean cluster angle deviates more than this parameter, are rejected.
        :ivar max_merging_angle: Candidate segment pairs for merging whose relative angle deviates more than this threshold are not merged.
        :ivar max_endpoint_distance: Candidate segment pairs for merging whose sum of squared distances between endpoints is larger than the square of this parameter are not merged.
        """
        # Maximal allowed angle between each segment and corresponding cluster mean angle.
        self.max_angle_variation_mean = np.pi / 180 * 5
        # Maximal allowed angle between two segments in order to merge them into a single one.
        self.max_merging_angle = np.pi / 180 * 1
        # Maximal summed distance between segments endpoints and fitted line for merging segments.
        self.max_endpoint_distance = 10.0


class SegmentClusterer:
    """Class responsible for clustering and cleaning the raw segments extracted by the :class:`SegmentDetector` class."""

    def __init__(self, input_segments: np.ndarray, params: SegmentClustererParams = SegmentClustererParams()):
        """Initializes the segment clusterer with the input segments and the semgment clusterer parameters."""
        self.raw_segments = input_segments
        self.params = params

        self.cluster_list = None
        self.cluster_features = None

    def cluster_segments(self) -> None:
        """Clusters the input segments :attr:`self.raw_segments` based on the parameters passed as argument.
        """
        Logger.debug("Clustering segments")
        if self.params.cluster_type not in ["gmm", "knn"]:
            Logger.fatal("Invalid value for cluster type: {}".format(self.params.cluster_type))
            raise ValueError("Invalid value for 'cluster_type': {} "
                             "'cluster_type' should be in ['gmm', 'knn']".format(self.params.cluster_type))

        centers = []
        angles = []
        for segment in self.raw_segments:
            pt1 = segment[0:2]
            pt2 = segment[2:4]
            center = (pt1 + pt2) * 0.5
            centers.append(center)

            # Segment angle lies in [0, pi], multiply by 2 such that complex number associated to similar angles are
            # close on the complex plane (e.g. 180° and 0°)
            angle = utils.angle_x(pt1, pt2) * 2

            # Need to use complex representation as Euclidean distance used in clustering makes sense in complex plane,
            # and does not directly on angles.
            point = np.array([np.cos(angle), np.sin(angle)])
            angles.append(point)

        centers = np.array(centers)
        centers = normalize(centers, axis=0)
        angles = np.array(angles)

        if self.params.use_angles and self.params.use_centers:
            features = np.hstack((angles, centers))
        elif self.params.use_angles:
            features = angles
        elif self.params.use_centers:
            features = centers
        else:
            raise RuntimeError("Can not perform segment clustering without any feature. "
                               "Select 'use_angles=True' and/or 'use_centers=True'.")

        cluster_prediction = None

        if self.params.cluster_type is "knn":
            Logger.debug("Clustering segments using KNN")
            cluster_prediction = KMeans(n_clusters=self.params.num_clusters, n_init=self.params.num_init,
                                        random_state=0).fit_predict(features)
        elif self.params.cluster_type is "gmm":
            Logger.debug("Clustering segments using GMM")
            best_gmm = None
            lowest_bic = np.infty
            bic = []
            n_components_range = range(1, self.params.num_clusters + 1)
            if not self.params.swipe_clusters:
                n_components_range = [self.params.num_clusters]
            for n_components in n_components_range:
                # Fit a Gaussian mixture with EM.
                gmm = GaussianMixture(n_components=n_components, covariance_type='full')
                gmm.fit(features)
                bic.append(gmm.bic(features))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm = gmm

            cluster_prediction = best_gmm.predict(features)

        # Reorder the segments as clusters.
        cluster_segment_list = []
        cluster_feature_list = []
        num_labels = np.max(cluster_prediction) + 1
        for label in range(num_labels):
            cluster_segments = self.raw_segments[cluster_prediction == label]
            if len(cluster_segments) == 0:
                continue
            cluster_features = features[cluster_prediction == label]
            cluster_segment_list.append(cluster_segments)
            cluster_feature_list.append(cluster_features)

        self.cluster_list = cluster_segment_list
        self.cluster_features = cluster_feature_list

    def compute_cluster_mean(self) -> tuple:
        """Computes the mean values (coordinates and angles) for each one of the identified clusters.
        :return The mean angles, and mean coordinates of each cluster.
        """
        mean_centers = []
        mean_angles = []
        for cluster in self.cluster_list:
            centers = 0.5 * (cluster[:, 0:2] + cluster[:, 2:4])

            mean_center = np.mean(centers, axis=0)
            mean_centers.append(mean_center)

            mean_angles.append(utils.mean_segment_angle(cluster))

        return np.array(mean_angles), np.array(mean_centers)

    def clean_clusters(self, mean_angles, params: ClusterCleaningParams = ClusterCleaningParams()) -> None:
        """Cleans the clusters by removing edges outliers (angle deviation from cluster mean is too high), and by merging
        almost collinear segments into a single segment.

        :param mean_angles: List of mean angles computed for each cluster.
        :param params: Parameters used to clean the clusters.
        """

        # Reorder the segments inside the clusters.
        for cluster_index, (cluster, features) in enumerate(zip(self.cluster_list, self.cluster_features)):
            cluster_order = utils.sort_segments(cluster)
            self.cluster_list[cluster_index] = cluster[cluster_order]
            self.cluster_features[cluster_index] = features[cluster_order]

        self.__clean_clusters_angle(mean_angles=mean_angles, max_angle_variation_mean=params.max_angle_variation_mean)
        self.__merge_collinear_segments(max_merging_angle=params.max_merging_angle,
                                        max_endpoint_distance=params.max_endpoint_distance)

    def __clean_clusters_angle(self, mean_angles: np.ndarray, max_angle_variation_mean: float)->None:
        """Removes all segments whose angle deviates more than the passed parameter from the mean cluster angle.

        :param mean_angles: List of cluster means.
        :param max_angle_variation_mean: Maximal angle variation to allow between the cluster segments and the associated mean angle.
        """
        for cluster_index, (cluster, mean_angle) in enumerate(zip(self.cluster_list, mean_angles)):
            invalid_indices = []
            for segment_index, segment in enumerate(cluster):
                # Retrieve angle in [0, pi] of current segment.
                angle = utils.angle_x(segment[0:2], segment[2:4])
                # Compute angle difference between current segment and mean angle of cluster.
                d_angle = utils.angle_diff(angle, mean_angle)
                if d_angle > max_angle_variation_mean:
                    invalid_indices.append(segment_index)
            self.cluster_list[cluster_index] = np.delete(cluster, invalid_indices, axis=0)

    def __merge_collinear_segments(self, max_merging_angle: float, max_endpoint_distance: float):
        """Merges all collinear segments belonging to the same cluster.

        :param max_merging_angle: Maximal angle to allow between segments to be merged.
        :param max_endpoint_distance: Maximal summed distance between segments endpoints and fitted line for merging segments.
        """
        for cluster_index, cluster in enumerate(self.cluster_list):
            merged = []
            merged_segments = []
            for i, segment_i in enumerate(cluster):
                if i in merged:
                    continue
                collinears = [i]
                for j in range(i + 1, len(cluster)):
                    segment_j = cluster[j]
                    if utils.segments_collinear(segment_i, segment_j, max_angle=max_merging_angle,
                                                   max_endpoint_distance=max_endpoint_distance):
                        collinears.append(j)

                merged_segment = utils.merge_segments(cluster[collinears])
                merged_segment = [int(m) for m in merged_segment]
                merged_segments.append(merged_segment)

                for index in collinears:
                    if index not in merged:
                        merged.append(index)

            self.cluster_list[cluster_index] = np.array(merged_segments)


import numpy as np
from simple_logger import Logger


class IntersectionDetectorParams:
    """Parameters used by the :class:`.IntersectionDetector`."""

    def __init__(self):
        """Initializes the intersection detector parameters to their default value.

        :ivar angle_threshold: Only intersections between segments which deviate less than this parameter from the canonical 90° angle are accepted.
        """
        # All intersections between segments whose relative angle is larger than this threshold are ignored.
        self.angle_threshold = np.pi / 180 * 25


class IntersectionDetector:
    """Class responsible for detecting intersections between segments."""

    def __init__(self, input_segments: list, params: IntersectionDetectorParams = IntersectionDetectorParams()):
        """Initializes the intersection detector object.

        Intersections are computed between all segments `s_i` of cluster `i`, against all segments `s_j` of cluster `j`.

        :param input_segments: List of segment clusters. Each element `cluster_i` of this list is a numpy array of shape `[num_segments_i, 4]`
        :param params: Parameters to be used for intersection detection.
        """
        self.segments = input_segments
        self.params = params

        # Collection of intersections divided by clusters:
        # self.cluster_cluster_intersections[i,j] contains the intersections between cluster i and cluster j.
        self.cluster_cluster_intersections = {}
        self.raw_intersections = []

    def detect(self):
        """Detects the intersections between the segments passed to the constructor using the parameters passed to the
        constructor.

        .. note:: The intersections are only computed between segments belonging to different clusters, and never between segments of the same cluster.
        """
        Logger.debug("Detecting intersection")
        self.cluster_cluster_intersections = {}
        self.raw_intersections = []
        num_clusters = len(self.segments)
        for cluster_index_i in range(num_clusters):
            for cluster_index_j in range(cluster_index_i + 1, num_clusters):
                print("Detecting intersections between cluster {} and cluster {}".format(cluster_index_i,
                                                                                                cluster_index_j))
                self.__detect_intersections_between_clusters(cluster_index_i, cluster_index_j)

    def __detect_intersections_between_clusters(self, cluster_index_i: int, cluster_index_j: int) -> None:
        """Detects the intersections between cluster `cluster_index_i` and cluster `cluster_index_j`.

        :param cluster_index_i: Index of first segment cluster to be intersected.
        :param cluster_index_j: Index of second segment cluster to be intersected.
        """

        cluster_i = self.segments[cluster_index_i]
        cluster_j = self.segments[cluster_index_j]
        print((cluster_index_i,cluster_index_j),(len(cluster_i),len(cluster_j)))

        self.cluster_cluster_intersections[cluster_index_i, cluster_index_j] = {}
        cluster_cluster_intersections_raw = []
        for i, segment_i in enumerate(cluster_i):
            intersections_with_i = {}
            angle_i = utils.angle_x(segment_i[0:2], segment_i[2:4])
            for j, segment_j in enumerate(cluster_j):
                print((segment_i,segment_j))

                angle_j = utils.angle_x(segment_j[0:2], segment_j[2:4])
                d_angle = utils.angle_diff(angle_i, angle_j)
                if np.pi / 2 - d_angle > self.params.angle_threshold:
                    continue
                intersection = utils.segment_segment_intersection(seg1=segment_i, seg2=segment_j)
                if intersection is not False:
                    intersections_with_i[j] = intersection
                    cluster_cluster_intersections_raw.append(intersection)
            self.cluster_cluster_intersections[cluster_index_i, cluster_index_j][i] = intersections_with_i
        if cluster_index_j >= cluster_index_i:
            self.raw_intersections.extend(cluster_cluster_intersections_raw)
