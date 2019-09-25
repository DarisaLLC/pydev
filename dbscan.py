"""Functions related to DBSCAN."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from .helper import get_coords_tuple


def find_number_of_clusters_dbscan(eps, coords):
    """Use DBSCAN to form clusters."""
    clustering = DBSCAN(eps=eps, min_samples=2).fit(coords)
    labels = clustering.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return n_clusters


def plot_dbscan_clicks_on_image(path, image_file_path, eps):
    """Plot clicks onto image."""
    coords = get_coords_tuple(path)
    clustering = DBSCAN(eps=eps, min_samples=2).fit(coords)

    centres = find_cluster_centres(clustering, coords)

    img = plt.imread(image_file_path)
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, 1000, -800, 0])

    ax.plot([c[0] for c in coords], [c[1] for c in coords], 'bo')
    ax.plot([r[0] for r in centres], [r[1] for r in centres], 'go')
    fig.savefig('test14.png', dpi=400)


def eps_against_num_of_clusters(path):
    """Output eps values and the number of clusters for a given image."""
    coords = get_coords_tuple(path)

    for eps in range(1, 50, 5):
        number_of_clusters = find_number_of_clusters_dbscan(eps, coords)
        print("EPS: {} CLUSTERS: {}".format(eps, number_of_clusters))


def find_within_interval_eps(
    desired_number_of_clusters,
    interval,
    starting_eps,
    coords
):
    """Find value of b to get within a interval - start eps about 5."""
    eps = starting_eps
    number_of_clusters = find_number_of_clusters_dbscan(eps, coords)

    prev_op_increased = None

    while True:
        if number_of_clusters > desired_number_of_clusters + interval:
            if prev_op_increased is not None and prev_op_increased is False:
                break
            prev_op_increased = True
            eps += interval
        elif number_of_clusters < desired_number_of_clusters - interval:
            if prev_op_increased is not None and prev_op_increased is True:
                break
            prev_op_increased = False
            eps -= interval
            if eps <= 0:
                eps += interval
                break
        else:
            break
        number_of_clusters = find_number_of_clusters_dbscan(eps, coords)
    return eps, number_of_clusters


def find_value_for_eps(cluster_num, path):
    """Find a value of bandwidth for an image given number of penguins."""
    coords = get_coords_tuple(path)

    eps, number_of_clusters = find_within_interval_eps(
        cluster_num, 5, 5, coords)
    eps, number_of_clusters = find_within_interval_eps(
        cluster_num, 1, eps, coords)
    # TODO: Confirm and fix behaviour when interval = 0

    return number_of_clusters, eps


def find_cluster_centres(cluster, coords):
    """Find cluster centres for DBSCAN clustering."""
    # #  Cluster numbering starts at 0 so add 1
    number_of_clusters = np.amax(cluster.labels_) + 1

    # For each cluster create an empty array to store points
    clusters = [[] for x in range(0, number_of_clusters)]
    for x in range(0, len(cluster.labels_)):
        point = cluster.labels_[x]
        clusters[point].append(coords[x])

    centres = []
    for cluster in clusters:
        x = sum([r[0] for r in cluster]) / len(cluster)
        y = sum([r[1] for r in cluster]) / len(cluster)
        centres.append((x, y))
    return centres


def dbscan_and_cluster_centres(eps, coords):
    """Use DBSCAN to form clusters."""
    clustering = DBSCAN(eps=eps, min_samples=2).fit(coords)
    labels = clustering.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    centres = find_cluster_centres(clustering, coords)
    return n_clusters, centres
