"""Pressure regions: connectivity-constrained Ward clustering of the surface triangles.

Triangles are grouped by the displacement they need (feature) plus a little position, so each
region is a connected patch that moves coherently. The clustering does not depend on the
pressures, so all coarse-to-fine levels are computed once in fem_setup.
"""

import warnings

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.cluster import AgglomerativeClustering


def ward_labels(feat, adj_pairs, active, K, pos, pos_weight):
    """Per-triangle labels 0..K-1 (-1 where not active), and the K actually used."""
    lab = -np.ones(len(active), dtype=int)
    idx = np.where(active)[0]
    K = int(min(K, len(idx)))
    if K <= 1:
        lab[idx] = 0
        return lab, 1
    remap = -np.ones(len(active), dtype=int)
    remap[idx] = np.arange(len(idx))
    pr = adj_pairs[(remap[adj_pairs[:, 0]] >= 0) & (remap[adj_pairs[:, 1]] >= 0)]
    i, j = remap[pr[:, 0]], remap[pr[:, 1]]
    n = len(idx)
    conn = coo_matrix((np.ones(2 * len(i)), (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    f = feat[idx] / (feat[idx].std(axis=0, keepdims=True) + 1e-12)
    g = pos[idx] - pos[idx].mean(0)
    g = g / (g.std() + 1e-12) * pos_weight
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # connectivity matrix with several components
        lab[idx] = AgglomerativeClustering(n_clusters=K, connectivity=conn,
                                           linkage="ward").fit_predict(np.column_stack([f, g]))
    return lab, K


def all_levels(feat, adj_pairs, active, levels, pos, pos_weight):
    """(K per level, labels (L, T)); levels asking for more regions than triangles are merged."""
    Ks, labels = [], []
    for k in levels:
        lab, K = ward_labels(feat, adj_pairs, active, k, pos, pos_weight)
        if K not in Ks:
            Ks.append(K)
            labels.append(lab)
    return np.array(Ks), np.array(labels)


def region_pairs(tri_labels, adj_pairs):
    """Unique pairs of adjacent regions (for the smoothness penalty)."""
    a, b = tri_labels[adj_pairs[:, 0]], tri_labels[adj_pairs[:, 1]]
    m = (a >= 0) & (b >= 0) & (a != b)
    return np.unique(np.sort(np.column_stack([a[m], b[m]]), axis=1), axis=0).reshape(-1, 2)
