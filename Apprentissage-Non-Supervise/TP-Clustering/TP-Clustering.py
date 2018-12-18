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


# Jeu de données test basique

# centers = [[1, 1], [-1, -1], [1, -1]]
# X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
#                             random_state=0)
#
# X = StandardScaler().fit_transform(X)

repository = "Apprentissage-Non-Supervise/TP-Clustering/cham-data/{}"

datasets = [
    # ["t4.8k.dat", 14, 26],
    # ["t5.8k.dat", 12, 38],
    # ["t7.10k.dat", 14, 26],
    ["t8.8k.dat", 10.1, 6.5],
]

process_datasets(datasets)

# #############################################################################

"""
1. Construct the similarity matrix.
2. Sparsify the similarity matrix using k-nn sparsification.
3. Construct the shared nearest neighbor graph from k-nn sparsified similarity matrix.
4. For every point in the graph, calculate the total strength of links coming out of the
point. (Steps 1-4 are identical to the Jarvis – Patrick scheme.)
5. Identify representative points by choosing the points that have high total link strength.
6. Identify noise points by choosing the points that have low total link strength and
remove them.
7. Remove all links that have weight smaller than a threshold.
8. Take connected components of points to form clusters, where every point
"""

X = np.genfromtxt('Apprentissage-Non-Supervise/TP-Clustering/cham-data/t4.8k.dat', dtype=float)

X_dist = np.array(squareform(pdist(X, metric='euclidean')))

X_sim = 1 / (1 + X_dist)


#Converting adjacency matrix to graph
G = kneighbors_graph(X_sim, 2, mode='distance', include_self= False).toarray()

seuil = 200

X_sim[X_sim < seuil] = 0

# #############################################################################



