import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph


# #############################################################################


def compute_dbscan(X, eps = 0.3, min_samples = 10, labels_true = None):

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
    labels = clustering.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Epsilon: %d' % eps)
    print('Min Samples: %d' % min_samples)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))
    if labels_true is not None:
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        print("Adjusted Rand Index: %0.3f"
              % metrics.adjusted_rand_score(labels_true, labels))
        print("Adjusted Mutual Information: %0.3f"
              % metrics.adjusted_mutual_info_score(labels_true, labels))

    return clustering, labels, core_samples_mask, n_clusters_, n_noise_


# #############################################################################


def plot_result(X, labels, core_samples_mask, n_clusters_):
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


# #############################################################################


def process_datasets(datasets):
    for filename, eps, min_samples in datasets:
        X = np.genfromtxt(repository.format(filename), dtype=float)
        clustering, labels, core_samples_mask, n_clusters_, n_noise_ = compute_dbscan(X, eps, min_samples)

        plot_result(X, labels, core_samples_mask, n_clusters_)


# #############################################################################


repository = "Apprentissage-Non-Supervise/TP-Clustering/cham-data/{}"

datasets = [
    ["t4.8k.dat", 10, 18],
    ["t5.8k.dat", 9.8, 18],
    ["t7.10k.dat", 10, 13],
    ["t8.8k.dat", 10.1, 7],
    ["t4.8k.dat", 12, 22],
    ["t5.8k.dat", 12, 38],
    ["t7.10k.dat", 14, 26],
    ["t8.8k.dat", 12, 6],
]

process_datasets(datasets)


# #############################################################################



