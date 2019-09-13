import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans,  SpectralClustering, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture

img = cv2.imread('/Volumes/medvedev/_SP/2019_09_06_BMWi3_natika_camera_images/empty_pad/empty_pad3/image00181.png',0)


def make_skeletonization(img, show_image=False):
    kernel_size=5
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)

    ret,img = cv2.threshold(img,127,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False

    while (not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    skel += img
    if show_image:
        cv2.imshow("skel",skel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return skel


def make_Sobel_filtration(img, show_image=False):
    im = img.astype('int32')
    dx = ndimage.sobel(im, 0)  # horizontal derivative
    dy = ndimage.sobel(im, 1)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    mag *= 255.0 / np.max(mag)  # normalize (Q&D)
    mag = mag.astype(np.uint8)
    if show_image:
        cv2.imshow('Sobel', mag)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return mag


def make_Canny_filtration(img, show_image=False):
    im = img.astype('int32')
    edges = cv2.Canny(img, 100, 200)
    if show_image:
        cv2.imshow('Canny', edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return edges


def convert_mask_to_regression(skel, show_image=False):

    indexes = list(zip(*np.where(skel > 0)))   # вывести все индексы, где яркость выше 0
    xs = [ind_pair[0] for ind_pair in indexes]
    ys = [ind_pair[1] for ind_pair in indexes]
    if show_image:
        plt.scatter(np.array(xs), np.array(ys))
        plt.show()
    return xs, ys


def cluster_mask_dbscan(skel, eps=0.38, show_image=False):

    xs, ys = convert_mask_to_regression(skel)
    X = np.array(list(zip(xs, ys)))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dbscan = DBSCAN(eps=eps)
    clusters = dbscan.fit_predict(X_scaled)

    n_clusters = len(set(clusters))   # количество кластеров
    clusters_points = [X[clusters==i] for i in range(n_clusters)]

    if show_image:
        plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap="plasma")
        plt.title('DBScan')
        plt.xlabel('Feature 0')
        plt.ylabel('Feature 1')
        plt.grid()
        plt.show()
    return clusters_points


def cluster_mask_kmeans(skel, show_image=False):

    xs, ys = convert_mask_to_regression(skel)
    X = np.array(list(zip(xs, ys)))

    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    clusters = kmeans.predict(X)

    n_clusters = len(set(clusters))   # количество кластеров
    clusters_points = [X[clusters==i] for i in range(n_clusters)]

    if show_image:
        plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
        plt.xlabel('Feature 0')
        plt.ylabel('Feature 1')
        plt.title('KMeans')
        plt.grid()
        plt.show()
    return clusters_points


def cluster_mask_OPTICS(skel, show_image=False):

    xs, ys = convert_mask_to_regression(skel)
    X = np.array(list(zip(xs, ys)))

    clust = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05)
    clust.fit(X)


    print(clust.reachability_)
    print(clust.core_distances_)
    print(clust.ordering_)

    labels_050 = cluster_optics_dbscan(reachability=clust.reachability_,
                                       core_distances=clust.core_distances_,
                                       ordering=clust.ordering_, eps=2)

    # Пока не разобрался
    return None


def cluster_mask_GaussianMixture(skel, show_image=False, find_optimal_clusters_num=False):

    xs, ys = convert_mask_to_regression(skel)
    X = np.array(list(zip(xs, ys)))

    n_components = np.arange(1, 21)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X) for n in n_components]

    if find_optimal_clusters_num:
        plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
        plt.plot(n_components, [m.aic(X) for m in models], label='AIC')
        plt.legend(loc='best')
        plt.xlabel('n_components')
        plt.show()

    gmm = GaussianMixture(n_components=4)
    gmm.fit(X)
    clusters_points = gmm.predict(X)
    if show_image:
        plt.grid()
        plt.title('GaussianMixture')
        plt.scatter(X[:, 0], X[:, 1], c=clusters_points, cmap='viridis')
        plt.show()
    return clusters_points


def cluster_mask_SpectralClustering(skel, show_image=False):
    xs, ys = convert_mask_to_regression(skel)
    X = np.array(list(zip(xs, ys)))
    clustering = SpectralClustering(n_clusters=4,
                                    assign_labels="discretize",
                                    random_state=0).fit(X)

    clusters_points = clustering.labels_
    if show_image:
        plt.grid()
        plt.title('Spectral Clustering')
        plt.scatter(X[:, 0], X[:, 1], c=clusters_points, cmap='viridis')
        plt.show()
    return clusters_points


def cluster_mask_AgglomerativeClustering(skel, show_image=False):
    xs, ys = convert_mask_to_regression(skel)
    X = np.array(list(zip(xs, ys)))
    clustering = AgglomerativeClustering(n_clusters=6, linkage='single').fit(X)
    clusters_points = clustering.labels_

    if show_image:
        plt.grid()
        plt.title('Agglomerative Clustering')
        plt.scatter(X[:, 0], X[:, 1], c=clusters_points, cmap='viridis')
        plt.show()
    return clusters_points


def make_skeleton_regression(clusters_points, show_image=False):

    regr = LinearRegression()
    quadratic = PolynomialFeatures(degree=2)
    cubic = PolynomialFeatures(degree=3)

    railsmid = []
    for cluster_points in clusters_points:
        X = np.array(cluster_points[:, 0])
        y = np.array(cluster_points[:, 1])
        
        X = X.reshape(-1, 1)
        X_quad = quadratic.fit_transform(X)
        X_cubic = cubic.fit_transform(X)

        # fit features
        X_fit = np.arange(X.min(), X.max())[:, np.newaxis]

        # Полином 1 степени
        #regr = regr.fit(X, y)
        #y_lin_fit = regr.predict(X_fit)

        # Полином второй степени
        #regr = regr.fit(X_quad, y)
        #y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))

        # Полином третьей степени
        regr = regr.fit(X_cubic, y)
        y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))

        railsmid.append(y_cubic_fit)
        if show_image:
            plt.scatter(X, y, label='Training points', color='lightgray')
            plt.grid()
            plt.plot(X_fit, y_cubic_fit, label='cubic (d=3)', color='green', lw=2, linestyle='--')
            plt.show()
        return railsmid





if __name__ == '__main__':
    skeleton = make_skeletonization(img)
    Sobel = make_Sobel_filtration(img)
    Canny = make_Canny_filtration(img)
    skeleton_Sobel = skeleton + Sobel
    skeleton_Canny = skeleton + Canny 

    cv2.imshow('Skeleton ', skeleton)
    cv2.waitKey(0)
    cv2.imshow('Skeleton + Sobel', skeleton_Sobel)
    cv2.waitKey(0)



    clusters_points = cluster_mask_dbscan(img, show_image=True)
    clusters_points = cluster_mask_kmeans(img, show_image=True)
    cluster_points = cluster_mask_GaussianMixture(skeleton, show_image=True)
    cluster_points = cluster_mask_SpectralClustering(skeleton, show_image=True)
    cluster_points = cluster_mask_AgglomerativeClustering(skeleton, show_image=True)

    #print(clusters_points)
    #regr_cluster_points = make_skeleton_regression(clusters_points, show_image=False)



