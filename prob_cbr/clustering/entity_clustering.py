from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import numpy as np


def dist_mat(query_vecs):
    """
    Calculate cosine sim
    :param query_vecs:
    :return:
    """
    sim = np.matmul(query_vecs, np.transpose(query_vecs))
    return 1 - sim


def cluster_entities(entity_vecs, t=0.7):
    """
    Return cluster assignment of entity vectors.
    :param entity_vecs:
    :param t: desired threshold for hierarchical clustering
    :return:
    """
    dists = dist_mat(entity_vecs.numpy())
    np.fill_diagonal(dists, 0)
    dists = np.clip(dists, 0, None)
    # build tree
    zavg = linkage(squareform(dists), method='average')
    c = fcluster(zavg, criterion='distance', t=t)

    return c
