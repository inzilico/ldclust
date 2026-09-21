"""
ldclust - LD-based clustering of SNPs followed by the haplotype-testing
protocol: load an LD (r2) matrix, define distances or a
spectral embedding, cluster SNPs, write LD blocks and PLINK hlist files,
run PLINK --hap-assoc, and select associated blocks.

The pipeline grows one clustering method per wrapper function; benchmark
runners import everything from here.

Author: Gennady Khvorykh, info@inzilico.com
Started: 2026-09-17
"""

import os
import subprocess
import time
import warnings

import h5py
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score

__version__ = "0.2.0"

# Distance transforms: name -> function r2 -> d. d1 = 1 - r2,
# d2 = sqrt(1 - r2) (Euclidean metric for a PSD r2 matrix),
# d3 = arccos(r2) (angular).
DISTANCES = {
    "d1": lambda x: 1.0 - x,
    "d2": lambda x: np.sqrt(1.0 - x),
    "d3": lambda x: np.arccos(x),
}

# LD significance threshold for the haplotype-testing protocol
PROTOCOL_P = 5e-6


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_ids(path):
    """Load SNP ids from a one-column snplist file."""
    return np.loadtxt(path, dtype="str")


def load_ld_h5(path):
    """
    Load an LD matrix saved as HDF5. Two layouts are supported:
    pandas fixed format (group 'r2' with dataset 'block0_values') and
    plain h5py dataset 'r2'. Returns (matrix float64, layout name).
    """
    with h5py.File(path, "r") as hf:
        node = hf["r2"] if "r2" in hf else hf[list(hf.keys())[0]]
        if isinstance(node, h5py.Dataset):
            layout = "dataset:" + node.name
            data = node[()]
        else:
            layout = "pandas:" + node.name + "/block0_values"
            data = node["block0_values"][()]
    a = np.ascontiguousarray(data, dtype=np.float64)
    if a.ndim == 1:  # a flat n*n array
        n = int(round(np.sqrt(a.size)))
        a = a.reshape(n, n)
    return a, layout


def load_ld(prefix):
    """
    Load prefix.ld.h5 + prefix.snplist. Returns (ids, r2, layout);
    exits if files are missing or sizes disagree.
    """
    for ext in [".ld.h5", ".snplist"]:
        if not os.path.isfile(prefix + ext):
            raise FileNotFoundError(prefix + ext)
    ids = load_ids(prefix + ".snplist")
    r2, layout = load_ld_h5(prefix + ".ld.h5")
    if ids.shape[0] != r2.shape[0]:
        raise ValueError(f"Sizes of matrix and snplist differ: "
                         f"{r2.shape[0]} vs {ids.shape[0]}")
    return ids, r2, layout


def drop_nan_snps(r2, ids):
    """Remove SNPs (rows and columns) containing any NaN."""
    keep = ~np.isnan(r2).any(axis=0)
    return r2[keep][:, keep], ids[keep], int((~keep).sum())


def distances(r2, kind="d2"):
    """Precomputed distance matrix from r2 (clipped to [0, 1])."""
    return DISTANCES[kind](np.clip(r2, 0.0, 1.0))


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def spectral_embedding(r2, variance=0.95, k=None):
    """
    Coordinates of the SNPs in the Euclidean space realizing d2:
    eigendecomposition of the (PSD) r2 matrix, top eigenvectors scaled by
    sqrt(eigenvalue). k components are taken as given, or the smallest
    number explaining `variance` of the trace. Note: LD r2 spectra are
    typically flat (tens of thousands of components may be needed for
    95% of trace), so a small fixed
    k is the meaningful denoising choice for clustering.
    Returns (coords n x k float64, k, eigenvalues descending).
    """
    w, v = np.linalg.eigh(r2)
    w = np.clip(w[::-1], 0.0, None)          # descending, numerical zeros >= 0
    v = v[:, ::-1]
    if k is None:
        total = w.sum()
        k = int(np.searchsorted(np.cumsum(w) / total, variance) + 1)
    k = min(k, w.size)
    coords = v[:, :k] * np.sqrt(w[:k])
    return np.ascontiguousarray(coords), k, w


# --------------------------------------------------------------------------
# Clustering wrappers: each returns (labels, probs_or_None, seconds)
# --------------------------------------------------------------------------

def cluster_hdbscan(dist):
    """HDBSCAN on a precomputed distance matrix."""
    import hdbscan
    t = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cl = hdbscan.HDBSCAN(metric="precomputed").fit(dist)
    return cl.labels_, np.round(cl.probabilities_, 4), time.time() - t


def cluster_optics(dist, min_samples=5, xi=0.05):
    """OPTICS on a precomputed distance matrix."""
    from sklearn.cluster import OPTICS
    t = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cl = OPTICS(metric="precomputed", min_samples=min_samples, xi=xi,
                    n_jobs=-1).fit(dist)
    return cl.labels_, None, time.time() - t


def cluster_soptics(coords, min_samples=5, xi=0.05):
    """sOPTICS: OPTICS on the spectral embedding of the r2 matrix."""
    from sklearn.cluster import OPTICS
    t = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cl = OPTICS(min_samples=min_samples, xi=xi, n_jobs=-1).fit(coords)
    return cl.labels_, None, time.time() - t


def cluster_dbscan(dist, eps, min_samples=5):
    """DBSCAN on a precomputed distance matrix."""
    from sklearn.cluster import DBSCAN
    t = time.time()
    cl = DBSCAN(metric="precomputed", eps=eps, min_samples=min_samples
                ).fit(dist)
    return cl.labels_, None, time.time() - t


def cluster_fof(r2, eps):
    """
    Friends-of-friends percolation: link SNP pairs with r2 >= eps,
    connected components are blocks. No noise label (all SNPs
    clustered).
    """
    t = time.time()
    a = np.clip(r2, 0.0, 1.0)
    np.fill_diagonal(a, 0.0)
    graph = csr_matrix(a >= eps)
    n_blocks, labels = connected_components(graph, directed=False)
    return labels.astype(np.int64), None, time.time() - t


def cluster_louvain(r2, gamma, cutoff=0.01, seed=42):
    """
    Louvain community detection on the weighted LD graph (edges r2 >
    cutoff, weight = r2). Resolution
    gamma controls block granularity. All SNPs are assigned: isolated
    nodes become singleton blocks (no noise label).
    """
    import networkx as nx
    t = time.time()
    graph = build_graph(r2, cutoff)
    comms = nx.community.louvain_communities(graph, weight="weight",
                                             resolution=gamma, seed=seed)
    labels = np.full(graph.number_of_nodes(), -1, dtype=np.int64)
    for c, nodes in enumerate(sorted(comms, key=lambda s: min(s))):
        labels[list(nodes)] = c
    return labels, None, time.time() - t


def build_graph(r2, cutoff):
    """Weighted SNP graph for networkx: edges r2 > cutoff, weight = r2."""
    import networkx as nx
    a = np.clip(r2, 0.0, 1.0)
    np.fill_diagonal(a, 0.0)
    rows, cols = np.nonzero(a > cutoff)
    graph = nx.Graph()
    graph.add_nodes_from(range(a.shape[0]))
    graph.add_weighted_edges_from(
        (int(i), int(j), float(a[i, j])) for i, j in zip(rows, cols))
    return graph


def labels_from_communities(n, comms):
    """Community iterables -> labels array, blocks numbered by first member."""
    labels = np.full(n, -1, dtype=np.int64)
    for c, nodes in enumerate(sorted(comms, key=lambda s: min(s))):
        labels[list(nodes)] = c
    return labels


def spectral_eig(r2):
    """
    Eigendecomposition (descending) of the normalized affinity
    D^(-1/2) A D^(-1/2), A = r2 with zero diagonal - the costly step
    of spectral clustering, shared by all theta values.
    """
    a = np.clip(r2, 0.0, 1.0)
    np.fill_diagonal(a, 0.0)
    inv_sqrt_d = 1.0 / np.sqrt(np.maximum(a.sum(axis=1), 1e-12))
    s = a * inv_sqrt_d[:, None] * inv_sqrt_d[None, :]
    w, v = np.linalg.eigh(s)
    return w[::-1].copy(), np.ascontiguousarray(v[:, ::-1])


def load_dosage(prefix):
    """
    Load the additive dosage matrix (individuals x SNPs) from a PLINK
    --recodeA .raw file. Missing genotypes stay NaN. Returns
    (dosage float64 n_ind x n_snp, snp names without the allele suffix,
    in file column order).
    """
    import pandas as pd
    df = pd.read_csv(prefix + ".raw", sep=r"\s+", na_values=["NA"])
    snps = np.array([c.rsplit("_", 1)[0] for c in df.columns[6:]])
    return df.iloc[:, 6:].to_numpy(dtype=np.float64), snps


def cross_product_matrix(prefix):
    """
    Normalised cross-product (relationship) matrix of the SNPs from the
    genotypic matrix, as follows:
      1. mean-centre every SNP column by its expected genotype 2p
         (p = allele frequency from the dosages; missing genotypes are
         imputed to the column mean, i.e. 2p),
      2. K = Z^T Z,
      3. normalise by the total expected heterozygosity sum(2 p (1 - p)).
    Returns (K float64 n_snp x n_snp, snp names). K is a PSD Gram matrix
    (a scaled, margin-rescaled relative of the r2 matrix:
    K_ij / sqrt(K_ii K_jj) = signed r_ij), suitable as the similarity
    substrate for methods that need coordinates or covariances (e.g.
    Gaussian mixtures via spectral_embedding).
    """
    g, snps = load_dosage(prefix)
    p = np.nanmean(g, axis=0) / 2.0
    z = np.where(np.isnan(g), 2.0 * p, g) - 2.0 * p
    del g
    k = z.T @ z
    del z
    k /= float(np.sum(2.0 * p * (1.0 - p)))
    return k, snps


def cluster_gmm(sim=None, k=1000, dim=50, seed=42, max_iter=200,
                reg_covar=1e-5, coords=None):
    """
    Gaussian Mixture Model on the spectral embedding of a similarity
    (relationship) matrix: coordinates = U sqrt(lambda) of the top `dim`
    eigenpairs, then sklearn GaussianMixture with a FULL covariance per
    component in that space. Pass `coords` (n x dim) to skip the
    eigendecomposition when the embedding is cached. All SNPs are
    assigned (no noise label). Randomised initialisation - controlled by
    seed.
    """
    from sklearn.mixture import GaussianMixture
    t = time.time()
    if coords is None:
        if sim is None:
            raise ValueError("provide sim or coords")
        w, v = np.linalg.eigh(sim)
        w = np.clip(w[::-1], 0.0, None)
        v = v[:, ::-1]
        coords = np.ascontiguousarray(v[:, :dim] * np.sqrt(w[:dim]))
        del w, v
    gm = GaussianMixture(n_components=k, covariance_type="full",
                         reg_covar=reg_covar, max_iter=max_iter,
                         random_state=seed).fit(coords)
    labels = gm.predict(coords).astype(np.int64)
    labels = labels_from_communities(
        labels.size,
        [set(np.flatnonzero(labels == c)) for c in np.unique(labels)])
    print(f"  GMM: converged={gm.converged_}, n_iter={gm.n_iter_}",
          flush=True)
    return labels, None, time.time() - t


def cluster_dpgmm(coords=None, sim=None, n_components=3000, dim=50,
                  weight_concentration_prior=None, seed=42, max_iter=100,
                  reg_covar=1e-5):
    """
    Dirichlet-process Gaussian mixture (variational, stick-breaking) on
    the spectral embedding of the relationship matrix - the model-based
    branch without the k grid: sklearn BayesianGaussianMixture with
    weight_concentration_prior_type='dirichlet_process' and a GENEROUS
    component cap (n_components is an upper bound only; the DP empties
    the components the data does not support, and the inferred k is
    read off as the number of occupied components). The concentration
    prior alpha replaces k as the knob: low alpha concentrates mass on
    few components, high alpha lets the mixture spread (None = sklearn
    default 1/n_components). Full per-component covariance as in
    cluster_gmm; pass `coords` (n x dim) to reuse the cached embedding.
    All SNPs are assigned (no noise label). Randomised - seed.
    Returns (labels, None, seconds).
    """
    from sklearn.mixture import BayesianGaussianMixture
    t = time.time()
    if coords is None:
        if sim is None:
            raise ValueError("provide sim or coords")
        w, v = np.linalg.eigh(sim)
        w = np.clip(w[::-1], 0.0, None)
        v = v[:, ::-1]
        coords = np.ascontiguousarray(v[:, :dim] * np.sqrt(w[:dim]))
        del w, v
    bgm = BayesianGaussianMixture(
        n_components=n_components, covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=weight_concentration_prior,
        reg_covar=reg_covar, max_iter=max_iter,
        random_state=seed).fit(coords)
    labels = bgm.predict(coords).astype(np.int64)
    occupied = np.unique(labels).size
    active = int((bgm.weights_ > 1.0 / (2.0 * coords.shape[0])).sum())
    alpha_str = ("default" if weight_concentration_prior is None
                 else str(weight_concentration_prior))
    print(f"  DPGMM: converged={bgm.converged_}, n_iter={bgm.n_iter_}, "
          f"cap={n_components}, active(w>1/2n)={active}, "
          f"occupied={occupied}, alpha={alpha_str}", flush=True)
    labels = labels_from_communities(
        labels.size,
        [set(np.flatnonzero(labels == c)) for c in np.unique(labels)])
    return labels, None, time.time() - t


def cluster_spectral(r2, theta, seed=42, eig=None):
    """
    Spectral clustering (Ng, Jordan, Weiss 2002): k = number of
    eigenvalues of the normalized affinity above theta, row-normalized
    top-k eigenvectors, k-means.
    `eig` allows reusing spectral_eig(r2) across a theta grid.
    """
    from sklearn.cluster import KMeans
    t = time.time()
    w, v = eig if eig is not None else spectral_eig(r2)
    k = max(1, int((w > theta).sum()))
    u = v[:, :k]
    u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    labels = KMeans(n_clusters=k, n_init=10,
                    random_state=seed).fit_predict(u)
    return labels.astype(np.int64), None, time.time() - t


def cluster_ap(r2, preference, damping=0.9, max_iter=500, convergence_iter=30):
    """
    Affinity Propagation (Frey & Dueck 2007) on the raw r2 similarity
    matrix (AP consumes similarities directly - no distance transform).
    `preference` (diagonal self-similarity) controls the number of
    clusters: higher -> more, smaller blocks. All SNPs are assigned
    (no noise label); every cluster has an exemplar SNP.
    Returns (labels, exemplars_as_probs_None, seconds); labels are
    relabelled so that block id = exemplar rank.
    """
    from sklearn.cluster import AffinityPropagation
    t = time.time()
    s = np.clip(r2, 0.0, 1.0).copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cl = AffinityPropagation(affinity="precomputed",
                                 preference=preference, damping=damping,
                                 max_iter=max_iter,
                                 convergence_iter=convergence_iter,
                                 verbose=False).fit(s)
    converged = cl.n_iter_ < max_iter
    labels = cl.labels_.astype(np.int64)
    # stable block ids by exemplar genomic order
    order = np.argsort(cl.cluster_centers_indices_)
    remap = np.full(labels.max() + 1 if labels.size else 1, -1, dtype=np.int64)
    for new, old in enumerate(order):
        remap[old] = new
    labels = remap[labels]
    print(f"  AP: converged={converged}, n_iter={cl.n_iter_}, "
          f"exemplars={len(cl.cluster_centers_indices_)}", flush=True)
    return labels, None, time.time() - t


def cluster_mcl(r2, inflation=2.0, cutoff=0.05, max_iter=100,
                prune=1e-4, tol=1e-3):
    """
    Markov Clustering (van Dongen 2000) on the weighted r2 graph
    (edges r2 > cutoff, weight = r2; self-loops added): alternate
    expansion (M <- M @ M) and inflation (column-wise power gamma,
    renormalised), pruning negligible entries to keep the matrix sparse.
    Convergence: total column change < tol * n_clusters_scale. Blocks are
    connected components of the positive entries of the limit matrix.
    All SNPs are assigned (no noise label). Inflation > 1 controls
    granularity: higher -> more, smaller blocks. Consumes the r2
    similarity directly - no distance transform.
    """
    t = time.time()
    a = np.clip(r2, 0.0, 1.0)
    np.fill_diagonal(a, 0.0)
    n = a.shape[0]
    rows, cols = np.nonzero(a > cutoff)
    idx = np.concatenate([rows, cols, np.arange(n)])
    jdx = np.concatenate([cols, rows, np.arange(n)])
    w = np.concatenate([a[rows, cols], a[cols, rows], np.ones(n)])
    m = coo_matrix((w, (idx, jdx)), shape=(n, n)).tocsr()

    def normalise(mat):
        return mat @ diags(1.0 / np.maximum(np.asarray(mat.sum(axis=0)).ravel(),
                                            1e-12))

    m = normalise(m)
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        prev = m.copy()
        m = normalise(m @ m)          # expansion
        m.data[m.data < prune] = 0.0  # prune negligible flows
        m.eliminate_zeros()
        m.data = m.data ** inflation  # inflation
        m = normalise(m)
        delta = abs(m - prev).sum() / n
        if delta < tol:
            converged = True
            break
    graph = m > 0
    n_blocks, labels = connected_components(graph, directed=False)
    print(f"  MCL: converged={converged}, n_iter={n_iter}, "
          f"delta={delta:.2e}", flush=True)
    return labels.astype(np.int64), None, time.time() - t


def cluster_lpa(r2, cutoff=0.05, seed=42):
    """
    Label Propagation (Raghavan et al. 2007) on the weighted r2 graph
    (networkx asyn_lpa_communities, weight = r2): every node adopts the
    label most frequent among its neighbours until a consensus. All SNPs
    are assigned (no noise label). Randomised - controlled by seed.
    Consumes the r2 similarity directly - no distance transform.
    """
    import networkx as nx
    t = time.time()
    graph = build_graph(r2, cutoff)
    comms = nx.community.asyn_lpa_communities(graph, weight="weight",
                                              seed=seed)
    labels = labels_from_communities(graph.number_of_nodes(), comms)
    return labels, None, time.time() - t


def cluster_kmeans(r2, k, dim=50, seed=42, max_iter=300):
    """
    K-Means (Lloyd 1957 / MacQueen 1967) on the Euclidean embedding of
    the distance-converted LD matrix: d = 1 - r2 is squared Euclidean
    (1 - r2 = 1/2 ||u_i - u_j||^2), so the
    coordinates U sqrt(lambda) of the r2 Gram matrix realise it exactly
    (classical MDS; K-Means is translation-invariant, so no centring is
    needed). All SNPs are assigned (no noise label). Consumes the r2
    similarity via its embedding - no distance transform. Essentially
    spectral clustering without the D^(-1/2) normalisation and row
    rescaling of Ng et al. (2002).
    """
    from sklearn.cluster import KMeans
    t = time.time()
    w, v = np.linalg.eigh(r2)
    w = np.clip(w[::-1], 0.0, None)
    v = v[:, ::-1]
    coords = np.ascontiguousarray(v[:, :dim] * np.sqrt(w[:dim]))
    del w, v
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed,
                    max_iter=max_iter).fit_predict(coords)
    labels = labels.astype(np.int64)
    labels = labels_from_communities(
        labels.size,
        [set(np.flatnonzero(labels == c)) for c in np.unique(labels)])
    return labels, None, time.time() - t


def cluster_minibatch_kmeans(r2, k, dim=50, seed=42, batch_size=1024,
                             max_iter=100):
    """
    Mini-Batch K-Means (Sculley 2010) on the same Euclidean embedding of
    the distance-converted LD matrix as cluster_kmeans: Lloyd updates
    from small random batches instead of full passes - much faster,
    slightly noisier partitions. All SNPs are assigned (no noise label).
    Consumes the r2 similarity via its embedding - no distance transform.
    """
    from sklearn.cluster import MiniBatchKMeans
    t = time.time()
    w, v = np.linalg.eigh(r2)
    w = np.clip(w[::-1], 0.0, None)
    v = v[:, ::-1]
    coords = np.ascontiguousarray(v[:, :dim] * np.sqrt(w[:dim]))
    del w, v
    labels = MiniBatchKMeans(n_clusters=k, batch_size=batch_size,
                             n_init=10, random_state=seed,
                             max_iter=max_iter).fit_predict(coords)
    labels = labels.astype(np.int64)
    labels = labels_from_communities(
        labels.size,
        [set(np.flatnonzero(labels == c)) for c in np.unique(labels)])
    return labels, None, time.time() - t


def cluster_cnm(r2, cutoff=0.05, resolution=1.0):
    """
    Greedy modularity maximisation (Clauset, Newman & Moore 2004) on the
    weighted r2 graph (networkx greedy_modularity_communities). Coarse
    modularity-scale blocks, like Louvain at resolution ~1. All SNPs are
    assigned (no noise label). Consumes the r2 similarity directly - no
    distance transform.
    """
    import networkx as nx
    t = time.time()
    graph = build_graph(r2, cutoff)
    try:
        comms = nx.community.greedy_modularity_communities(
            graph, weight="weight", resolution=resolution)
    except TypeError:  # older networkx without resolution support
        comms = nx.community.greedy_modularity_communities(
            graph, weight="weight")
    labels = labels_from_communities(graph.number_of_nodes(), comms)
    return labels, None, time.time() - t


def cluster_cw(r2, cutoff=0.05, iterations=3, seed=42):
    """
    Chinese Whispers (Biemann 2006): a randomised, near-linear
    agglomeration on the weighted r2 graph - in random order, each node
    adopts the neighbour class with the greatest summed edge weight.
    All SNPs are assigned (no noise label). Randomised - controlled by
    seed. Consumes the r2 similarity directly - no distance transform.
    """
    t = time.time()
    a = np.clip(r2, 0.0, 1.0)
    np.fill_diagonal(a, 0.0)
    n = a.shape[0]
    labels = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    neighbours = [np.flatnonzero(a[i] > cutoff) for i in range(n)]
    for _ in range(iterations):
        changed = 0
        for i in rng.permutation(n):
            nb = neighbours[i]
            if nb.size == 0:
                continue
            classes, inv = np.unique(labels[nb], return_inverse=True)
            scores = np.bincount(inv, weights=a[i, nb])
            best = classes[scores == scores.max()]
            new = best[0] if best.size == 1 else best[rng.integers(best.size)]
            if new != labels[i]:
                labels[i] = new
                changed += 1
        if changed == 0:
            break
    # stable block ids by first member order
    labels = labels_from_communities(
        n, [set(np.flatnonzero(labels == c)) for c in np.unique(labels)])
    return labels, None, time.time() - t


def igraph_graph(r2, cutoff):
    """Weighted SNP graph for python-igraph: edges r2 > cutoff."""
    import igraph as ig
    a = np.clip(r2, 0.0, 1.0)
    np.fill_diagonal(a, 0.0)
    rows, cols = np.nonzero(a > cutoff)
    keep = rows < cols
    g = ig.Graph(n=a.shape[0],
                 edges=list(zip(rows[keep].tolist(), cols[keep].tolist())),
                 directed=False)
    g.es["weight"] = a[rows[keep], cols[keep]].tolist()
    return g


def cluster_leiden(r2, gamma, cutoff=0.05, seed=42):
    """
    Leiden (Traag et al. 2019) with the RBConfiguration (modularity with
    resolution) partition on the weighted r2 graph - the modern repair of
    Louvain's disconnected-community flaw. Resolution gamma controls
    granularity (higher -> more blocks). All SNPs are assigned.
    Consumes the r2 similarity directly - no distance transform.
    """
    import leidenalg
    t = time.time()
    g = igraph_graph(r2, cutoff)
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition, weights="weight",
        resolution_parameter=gamma, seed=seed)
    labels = np.asarray(part.membership, dtype=np.int64)
    return labels, None, time.time() - t


def cluster_walktrap(r2, steps=4, cutoff=0.05):
    """
    Walktrap (Pons & Latapy 2005): short random walks (steps) on the
    weighted r2 graph, agglomerative merging of nodes by walk similarity,
    cut at maximum modularity. All SNPs are assigned. Consumes the r2
    similarity directly - no distance transform.
    """
    t = time.time()
    g = igraph_graph(r2, cutoff)
    clu = g.community_walktrap(weights="weight", steps=steps).as_clustering()
    labels = np.asarray(clu.membership, dtype=np.int64)
    return labels, None, time.time() - t


def cluster_infomap(r2, cutoff=0.05, seed=42):
    """
    Infomap (Rosvall & Bergstrom 2008): minimises the map equation -
    the description length of an infinite random walk - on the weighted
    r2 graph (two-level partition). All SNPs are assigned. Consumes the
    r2 similarity directly - no distance transform.
    """
    from infomap import Infomap
    t = time.time()
    graph = build_graph(r2, cutoff)
    im = Infomap("--two-level --silent", num_trials=1, seed=seed)
    im.add_networkx_graph(graph, weight="weight")
    im.run()
    modules = im.get_modules()
    labels = np.array([modules[i] for i in range(graph.number_of_nodes())],
                      dtype=np.int64)
    labels = labels_from_communities(
        graph.number_of_nodes(),
        [set(np.flatnonzero(labels == c)) for c in np.unique(labels)])
    return labels, None, time.time() - t


def cluster_pam(r2, k, max_iter=100):
    """
    Similarity k-medoids (PAM of Kaufman & Rousseeuw 1990, similarity
    form): choose k medoids so that the total within-cluster similarity
    is maximised - alternating assignment (nearest medoid by r2) and
    medoid update (the member with the greatest within-cluster
    similarity). Medoids are seeded evenly along genomic order (LD
    blocks are genomic intervals; max-similarity seeding degenerates
    because isolated SNPs attract no members). All SNPs are assigned
    (no noise label); every block is centred on an exemplar SNP.
    Consumes the r2 similarity directly - no distance transform.
    """
    t = time.time()
    s = np.clip(r2, 0.0, 1.0)
    np.fill_diagonal(s, 0.0)
    n = s.shape[0]
    medoids = np.linspace(0, n - 1, k).astype(np.int64)
    labels = None
    for _ in range(max_iter):
        new_labels = np.argmax(s[:, medoids], axis=1).astype(np.int64)
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            idx = np.flatnonzero(labels == c)
            if idx.size == 0:
                labels[medoids[c]] = c  # keep the medoid as a singleton
            else:
                inner = s[np.ix_(idx, idx)]
                medoids[c] = idx[int(np.argmax(inner.sum(axis=1)))]
    # stable block ids by medoid genomic order
    order = np.argsort(medoids)
    remap = np.empty(order.size, dtype=np.int64)
    remap[order] = np.arange(order.size)
    return remap[labels], None, time.time() - t


def _core_size(branch_size, min_cluster_size):
    """dynamicTreeCut .CoreSize (R): size of a branch's tight core."""
    base = min_cluster_size / 2.0 + 1.0
    if base < branch_size:
        return int(base + np.sqrt(branch_size - base))
    return int(branch_size)


def dynamic_tree_cut(z, dist, min_cluster_size=20, deep_split=2,
                     cut_height=None):
    """
    Dynamic Tree Cut, tree stage - a port of the reference algorithm
    (Langfelder, Zhang & Horvath 2008, R package dynamicTreeCut,
    cutreeHybrid with pamStage = FALSE, no external split criteria).

    Walk the agglomerative merges in height order (z rows sorted by
    height, as scipy linkage outputs for the monotone methods). At every
    meeting of two basic branches, the weaker one is absorbed into the
    stronger (fails: size < min_cluster_size, core scatter too diffuse,
    attachment gap too shallow, merge below the reference height);
    otherwise both survive as separate clusters and the merge continues
    as composite bookkeeping. Top basic branches that still satisfy the
    cluster criteria at their last attachment height become clusters;
    everything else is noise (-1), the WGCNA 'grey' objects.

    deep_split 0..4 selects the R presets exactly:
    maxCoreScatter = (0.64, 0.73, 0.82, 0.91, 0.95),
    minGap = (1 - maxCoreScatter) * 3/4 (higher -> more, smaller blocks).
    cutHeight defaults to the R rule: 99% of the height range above the
    5%-of-merges reference height.

    z: scipy linkage matrix; dist: the full n x n distance matrix (the
    core scatter needs core-to-core distances). Deterministic.
    Returns int64 labels (noise = -1), blocks numbered by first member.
    """
    n = z.shape[0] + 1
    heights = z[:, 2]
    n_merge = int(heights.size)
    ref_merge = max(1, int(round(0.05 * n_merge)))   # R is 1-indexed
    ref_height = float(heights[ref_merge - 1])
    if cut_height is None:
        cut_height = 0.99 * (heights[-1] - ref_height) + ref_height
    cut_height = min(cut_height, float(heights[-1]))

    def_mcs = (0.64, 0.73, 0.82, 0.91, 0.95)
    mcs = def_mcs[deep_split]
    mg = (1.0 - mcs) * 3.0 / 4.0
    max_abs_core_scatter = ref_height + mcs * (cut_height - ref_height)
    min_abs_gap = mg * (cut_height - ref_height)
    # minSplitHeight = 0 -> minAbsSplitHeight = refHeight (R default)
    min_abs_split_height = ref_height

    is_basic, is_top, size, n_sing = [], [], [], []
    merged_into, attach_h, singletons = [], [], []
    ind = np.full(n_merge, -1, dtype=np.int64)       # merge row -> branch

    def new_branch(basic, sz, sings):
        is_basic.append(basic)
        is_top.append(basic)
        size.append(sz)
        n_sing.append(len(sings))
        merged_into.append(0)
        attach_h.append(None)
        singletons.append(list(sings))
        return len(is_basic) - 1                      # 0-based branch ids

    def core_scatter(br):
        k = _core_size(n_sing[br], min_cluster_size)
        idx = np.asarray(singletons[br][:k])
        if idx.size < 2:
            return 0.0
        sub = dist[np.ix_(idx, idx)]
        return float(sub.sum() / (idx.size * (idx.size - 1)))

    for j in range(n_merge):
        h = float(heights[j])
        if h > cut_height:
            break
        a, b = int(z[j, 0]), int(z[j, 1])
        leaf_a, leaf_b = a < n, b < n
        if leaf_a and leaf_b:                        # two singletons start
            ind[j] = new_branch(True, 2, [a, b])     # a basic branch
        elif leaf_a != leaf_b:                       # singleton joins a
            gene = a if leaf_a else b                # branch
            br = ind[(b if leaf_a else a) - n]
            size[br] += 1
            if is_basic[br]:
                singletons[br].append(gene)
                n_sing[br] += 1
            ind[j] = br
        else:                                        # two branches meet
            c1, c2 = ind[a - n], ind[b - n]
            if c1 < 0 or c2 < 0:
                continue                             # above-cut component
            if size[c1] <= size[c2]:
                small, large = c1, c2
            else:
                small, large = c2, c1
            victim = -1
            for cand in (small, large):              # R checks the smaller
                if not is_basic[cand]:               # first, then swaps
                    continue
                cs = core_scatter(cand)
                if (size[cand] < min_cluster_size
                        or cs > max_abs_core_scatter
                        or h - cs < min_abs_gap
                        or h < min_abs_split_height):
                    victim = cand
                    break
            if victim >= 0:                          # absorb the failing
                other = large if victim == small else small
                merged_into[victim] = other          # branch into the other
                attach_h[victim] = h
                is_top[victim] = False
                if is_basic[other]:
                    singletons[other].extend(singletons[victim])
                    n_sing[other] += n_sing[victim]
                size[other] += size[victim]
                ind[j] = other
            elif is_basic[large]:                    # both survive: start
                nb = new_branch(False, size[large] + size[small], [])
                merged_into[large] = nb              # a composite branch
                merged_into[small] = nb
                attach_h[large] = h
                attach_h[small] = h
                ind[j] = nb
            else:                                    # grow the composite
                merged_into[small] = large
                attach_h[small] = h
                size[large] += size[small]
                ind[j] = large

    clusters = []
    for br in range(len(is_basic)):
        if not is_top[br]:
            continue
        ah = attach_h[br] if attach_h[br] is not None else cut_height
        cs = core_scatter(br)
        if (size[br] >= min_cluster_size
                and cs < max_abs_core_scatter
                and ah - cs > min_abs_gap):
            clusters.append(br)

    labels = np.full(n, -1, dtype=np.int64)
    order = sorted((singletons[br] for br in clusters),
                   key=lambda ix: ix[0])
    for c_id, members in enumerate(order):
        labels[members] = c_id
    n_noise = int((labels == -1).sum())
    print(f"  DTC: cut_height={cut_height:.3f}, {len(is_basic)} branches, "
          f"{len(clusters)} clusters, noise {n_noise}/{n}", flush=True)
    return labels


def cluster_hc(dist, linkage_method="average", min_cluster_size=3,
               deep_split=2, cut_height=None):
    """
    Agglomerative hierarchical clustering + dynamic tree cut
    (Langfelder et al. 2008), the WGCNA recipe transplanted to LD:
    scipy linkage (average = the WGCNA default) on the condensed
    d = 1 - r2 - the analogue of WGCNA's 1 - |cor| dissimilarity - then
    the dynamic tree cut (tree stage, see dynamic_tree_cut) instead of a
    single global cut height: branches split by shape (core scatter and
    attachment gaps). Blocks smaller than min_cluster_size end up as
    noise (-1). Deterministic; consumes a precomputed distance matrix.
    """
    from scipy.cluster.hierarchy import linkage
    t = time.time()
    n = dist.shape[0]
    y = np.empty(n * (n - 1) // 2, dtype=np.float64)  # condensed distance
    start = 0
    for i in range(n - 1):
        seg = dist[i, i + 1:]
        y[start:start + seg.size] = seg
        start += seg.size
    z = linkage(y, method=linkage_method)
    del y
    labels = dynamic_tree_cut(z, dist, min_cluster_size, deep_split,
                              cut_height)
    return labels, None, time.time() - t


def _greedy_tag_count(sub):
    """
    Greedy set-cover tag count for one candidate block: sub[i, t] is
    True iff r2(i, t) >= tag_thr within the block. The minimum tag set
    is NP-hard (set cover); greedy max-new-coverage is the standard
    approximation (Zhang et al. 2002 also select htSNPs greedily).
    Two exact fast paths: a fully-connected block needs one tag, a
    block without any sharing (all row sums 1) needs one tag per SNP.
    Returns the number of tags.
    """
    n = sub.shape[0]
    if sub.all():
        return 1
    if (sub.sum(axis=1) == 1).all():
        return n
    covered = np.zeros(n, dtype=bool)
    n_tags = 0
    while not covered.all():
        counts = sub[~covered].sum(axis=0)
        best = int(np.argmax(counts))
        covered |= sub[:, best]
        n_tags += 1
    return n_tags


def cluster_dpblocks(r2, tag_thr=0.5, max_len=30):
    """
    Dynamic-programming haplotype-block partitioning: the Zhang, Deng,
    Chen, Waterman & Sun (2002, PNAS) DP framework with an r2-tagging
    cost. The genomically ordered SNPs are partitioned into contiguous
    blocks minimising the TOTAL number of tag SNPs, where a block's cost
    is the (greedy) minimum number of tags such that every SNP in the
    block has r2 >= tag_thr with some tag - Carlson-style LD bins inside
    Zhang's recurrence best[e] = min_len best[e - len] + cost(e - len,
    e - 1), len = 1..max_len. Ties keep the finest partition (smallest
    len). Contiguity caps every block at max_len SNPs, so haplotype-EM
    feasibility is enforced by construction (the protocol's DNF failure
    mode of the modularity family). Zhang's original per-block cost
    (minimum htSNPs distinguishing common haplotypes) needs an EM run
    per candidate block - intractable at 23.6k SNPs x max_len
    candidates; the r2 criterion is its LD-bin surrogate (the r2 matrix
    already condenses the genotypes). All SNPs are assigned, in blocks
    of size 1..max_len; no noise label. Deterministic; consumes the
    ordered r2 matrix directly - no distance transform.
    Returns (labels, None, seconds); total tags = the optimum printed.
    """
    t = time.time()
    b = np.clip(r2, 0.0, 1.0) >= tag_thr            # cover matrix
    n = b.shape[0]
    inf = np.int64(1) << 40
    best = np.full(n + 1, inf, dtype=np.int64)
    best[0] = 0
    back = np.ones(n + 1, dtype=np.int32)
    for e in range(1, n + 1):
        for ln in range(1, min(e, max_len) + 1):
            s = e - ln
            cand = best[s] + _greedy_tag_count(
                np.ascontiguousarray(b[s:e, s:e]))
            if cand < best[e]:                      # strict <: finest on
                best[e] = cand                      # ties (ln ascending)
                back[e] = ln
    bounds = []
    i = n
    while i > 0:
        ln = int(back[i])
        bounds.append((i - ln, i))
        i -= ln
    bounds.reverse()
    labels = np.empty(n, dtype=np.int64)
    for c, (s, e) in enumerate(bounds):
        labels[s:e] = c
    sizes = np.diff(np.asarray(bounds).reshape(-1, 2), axis=1)
    print(f"  DP: optimum {int(best[n])} tags in {len(bounds)} blocks "
          f"(median {int(np.median(sizes))}, max {int(sizes.max())}), "
          f"thr {tag_thr}, len <= {max_len}", flush=True)
    return labels, None, time.time() - t


def cluster_sbm(r2, cutoff=0.05, deg_corr=True, weighted=True, seed=42,
                nested=False):
    """
    Degree-corrected stochastic block model (Karrer & Newman 2011) on the
    weighted r2 graph (edges r2 > cutoff), fitted by minimising the
    microcanonical description length (Peixoto 2017) with graph-tool's
    minimise_blockmodel_dl - the principled replacement for the modularity
    family: the number of blocks B is selected by the description length
    itself, and degree correction lets blocks keep heterogeneous SNP
    degrees (the plain SBM, deg_corr=False, is the ablation). With
    weighted=True the r2 values enter as a continuous 'real-exponential'
    edge covariate. All SNPs are assigned (isolated nodes form singleton
    blocks via relabelling). Deterministic given seed. Needs graph-tool
    (not pip-installable - use its conda environment, see README).
    Returns (labels, None, seconds); prints B and the DL in nats.
    """
    import graph_tool as gt
    from graph_tool.inference import minimize_blockmodel_dl
    from graph_tool.inference.blockmodel import BlockState, WeightedBlockState
    t = time.time()
    a = np.clip(r2, 0.0, 1.0)
    np.fill_diagonal(a, 0.0)
    rows, cols = np.nonzero(a > cutoff)
    keep = rows < cols
    n = a.shape[0]
    g = gt.Graph(directed=False)
    g.add_vertex(n)
    g.add_edge_list(np.stack([rows[keep], cols[keep]], axis=1))
    if weighted:
        ew = g.new_ep("double")
        ew.a = a[rows[keep], cols[keep]]
        # graph-tool 3.x: weighted SBM = WeightedBlockState with the r2
        # values as a continuous 'real-exponential' edge covariate
        state_cls = WeightedBlockState
        state_args = {"rec": [ew], "rec_types": ["real-exponential"],
                      "deg_corr": deg_corr}
    else:
        state_cls = BlockState
        state_args = {"deg_corr": deg_corr}
    gt.seed_rng(seed)
    if nested:
        # nested SBM: level 0 is the finest partition; upper levels model
        # the block graph, letting level 0 go finer than the flat optimum
        from graph_tool.inference import minimize_nested_blockmodel_dl
        state = minimize_nested_blockmodel_dl(g, base_state=state_cls,
                                              base_state_args=state_args)
        levels_b = [lv.get_B() for lv in state.levels]
        blocks = state.levels[0].get_blocks()
        b_str = "/".join(str(x) for x in levels_b)
    else:
        state = minimize_blockmodel_dl(g, state=state_cls,
                                       state_args=state_args)
        blocks = state.get_blocks()
        b_str = str(state.get_B())
    labels = np.fromiter((int(blocks[v]) for v in g.vertices()),
                         dtype=np.int64, count=n)
    labels = labels_from_communities(
        n, [set(np.flatnonzero(labels == c)) for c in np.unique(labels)])
    print(f"  SBM: B={b_str} (level0..top), DL={state.entropy():.0f} nats, "
          f"deg_corr={deg_corr}, weighted={weighted}, "
          f"m={int(keep.sum())} edges", flush=True)
    return labels, None, time.time() - t


def cluster_cdhit(r2, thr=0.5, priority="degree", recruit="seed",
                  degree_cutoff=0.05):
    """
    Greedy incremental seed-and-recruit clustering in the manner of
    CD-HIT (Li & Godzik 2006) / UCLUST / MMseqs2, transplanted to LD:
    SNPs are processed by PRIORITY - the weighted degree (sum of r2
    above degree_cutoff, the LD-graph hub-iness; 'order' = genomic
    order as the alternative) - and the top still-unclustered SNP
    becomes the seed of a new block, recruiting every remaining
    unclustered SNP with r2 >= thr to the block. With recruit='seed'
    (CD-HIT representative semantics) the recruit must match the SEED
    itself - no transitive chaining, the anti-FoF: blocks are stars
    around high-LD hubs, so block size is bounded by the seed's r2
    neighbourhood and single-linkage chains cannot over-merge.
    recruit='member' lets a SNP join if it matches ANY current member,
    looping to a fixpoint over still-unclustered SNPs. Because the
    seed's closure always reaches its entire connected component (no
    earlier seed can sit inside it), member-recruit is provably the FoF
    partition - verified empirically (identical co-membership to
    cluster_fof on real matrices, via a completely separate code
    path). Kept as an internal consistency check. All SNPs are assigned (seed-only blocks are
    singletons); every block has a seed. Deterministic (stable priority
    ties). Consumes the r2 similarity directly - no distance transform.
    """
    t = time.time()
    a = np.clip(r2, 0.0, 1.0)
    np.fill_diagonal(a, 0.0)
    n = a.shape[0]
    if priority == "degree":
        score = np.where(a > degree_cutoff, a, 0.0).sum(axis=1)
        # round before sorting: float-sum order decides exact ties
        # arbitrarily; rounding restores genomic-index tie-breaking
        order = np.lexsort((np.arange(n), -np.round(score, 9)))
    elif priority == "order":
        order = np.arange(n)
    else:
        raise ValueError(f"unknown priority {priority!r}")
    labels = np.full(n, -1, dtype=np.int64)
    seeds = []
    for i in order:
        if labels[i] != -1:
            continue
        c = len(seeds)
        seeds.append(i)
        labels[i] = c
        members = [i]
        while True:
            open_idx = np.flatnonzero(labels == -1)
            if open_idx.size == 0:
                break
            if recruit == "seed":
                hit = open_idx[a[i, open_idx] >= thr]
            else:               # recruit == 'member': fixpoint closure
                hit = open_idx[
                    a[np.ix_(members, open_idx)].max(axis=0) >= thr]
            if hit.size == 0:
                break
            labels[hit] = c
            members.extend(hit.tolist())
            if recruit == "seed":
                break           # representative semantics: one pass only
    # stable block ids by seed genomic order
    rank = np.empty(len(seeds), dtype=np.int64)
    rank[np.argsort(seeds)] = np.arange(len(seeds))
    labels = rank[labels]
    sizes = np.bincount(labels)
    print(f"  CDHIT: {len(seeds)} blocks from {len(seeds)} seeds "
          f"(median {int(np.median(sizes))}, max {int(sizes.max())}), "
          f"thr {thr}, priority {priority}, recruit {recruit}",
          flush=True)
    return labels, None, time.time() - t


def coassociation(labels_list):
    """
    Co-association (evidence-accumulation) matrix of base partitions:
    C[i, j] = fraction of partitions in which SNPs i and j share a
    block (Fred & Jain 2005; Strehl & Ghosh 2002). Noise labels (-1)
    are treated as singleton blocks. C is PSD - each partition's
    co-membership matrix is a sum of PSD block indicators 1_b 1_b^T -
    so 1 - C plays exactly the role of 1 - r2 on the LD substrate.
    """
    labels_list = [np.asarray(l) for l in labels_list]
    n = labels_list[0].size
    c = np.zeros((n, n), dtype=np.float64)
    for lab in labels_list:
        if lab.size != n:
            raise ValueError("base partitions differ in size")
        if (lab < 0).any():
            lab = lab.copy()
            nxt = int(lab.max()) + 1
            for i in np.flatnonzero(lab < 0):
                lab[i] = nxt
                nxt += 1
        for cl in np.unique(lab):
            idx = np.flatnonzero(lab == cl)
            c[np.ix_(idx, idx)] += 1.0
    return c / float(len(labels_list))


def cluster_consensus(labels_list, method="dtc", thr=None, deep_split=2,
                      min_cluster_size=3, linkage="average"):
    """
    Consensus (ensemble) clustering of base partitions via the
    co-association route: C = coassociation(labels_list) (PSD, see
    there), then a consensus function on C:
      dtc - average linkage on d = 1 - C + the dynamic tree cut
            (cluster_hc): C is a correlation-like similarity exactly as
            r2, so the LD machinery transfers verbatim;
      fof - percolation on C >= thr (thr = 1.0: the intersection core,
            pairs co-clustered by every base and transitively chained;
            2/3: the majority core of three partitions).
    All SNPs are assigned under fof; dtc leaves singletons/small
    branches as noise (-1) as usual. Deterministic. Returns (labels,
    None, seconds); the base partitions enter as a list of label
    arrays (same n, same SNP order).
    """
    t = time.time()
    c = coassociation(labels_list)
    if method == "dtc":
        labels, _, _ = cluster_hc(1.0 - c, linkage_method=linkage,
                                  min_cluster_size=min_cluster_size,
                                  deep_split=deep_split)
    elif method == "fof":
        if thr is None:
            raise ValueError("fof consensus needs thr")
        labels, _, _ = cluster_fof(c, thr)
    else:
        raise ValueError(f"unknown consensus method {method!r}")
    vals, counts = np.unique(c[np.triu_indices(c.shape[0], 1)],
                             return_counts=True)
    print(f"  CONSENSUS: {len(labels_list)} bases, {method}"
          + (f" thr={thr}" if thr else "")
          + f", off-diagonal C levels "
          + "/".join(f"{v:.2f}x{cnt}" for v, cnt in zip(vals, counts)
                     if cnt > 0), flush=True)
    return labels, None, time.time() - t



# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------

def write_blocks(path, ids, labels):
    """LD blocks: '<snp> <block>' per line, no header (noise = -1)."""
    with open(path, "w") as f:
        for s, c in zip(ids, labels):
            f.write(f"{s} {c}\n")


def write_hdbscan_file(path, ids, labels, probs=None):
    """.hdbscan text format: '<snp> <cluster> <probability>'."""
    probs = np.zeros(labels.size) if probs is None else probs
    with open(path, "w") as f:
        for s, c, p in zip(ids, labels, probs):
            f.write(f"{s} {c} {p}\n")


def write_hlist(path, ids, labels):
    """
    hlist for PLINK --hap: one '** <locus> <snps...>' line per
    cluster (noise excluded), clusters in ascending id order. The locus
    name is a sequential number - this way single-SNP blocks keep a
    non-empty SNP list (PLINK 1.07 rejects '** <snp>' lines with
    "must have at least one SNP").
    """
    with open(path, "w") as f:
        for i, c in enumerate(sorted(set(labels) - {-1})):
            f.write(f"** {i} " + " ".join(ids[labels == c]) + "\n")


# --------------------------------------------------------------------------
# Haplotype-testing protocol (PLINK 1.07)
# --------------------------------------------------------------------------

def run_hap_assoc(prefix, hlist, out_prefix,
                  plink=os.path.expanduser("~/tools/plink-1.07-x86_64/plink"),
                  timeout_sec=None):
    """
    Run PLINK 1.07 --hap-assoc on prefix.ped/map with the given hlist.
    Returns the assoc.hap path.
    Raises TimeoutExpired if timeout_sec (seconds) is exceeded - blocks
    with hundreds of SNPs make the haplotype EM intractable, so callers
    should bound the runtime.
    """
    cmd = [plink, "--file", prefix, "--hap-assoc", "--hap", hlist,
           "--allow-no-sex", "--noweb", "--nonfounders", "--out", out_prefix]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        raise
    out = out_prefix + ".assoc.hap"
    if res.returncode != 0 or not os.path.isfile(out):
        raise RuntimeError(f"PLINK failed ({res.returncode}):\n"
                           f"{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
    return out


def parse_assoc_hap(path, pos=None):
    """
    OMNIBUS (block-level) rows of a PLINK assoc.hap file. If pos (dict
    snp -> bp) is given, block start/end coordinates are added.
    """
    rows = []
    with open(path) as f:
        for line in f:
            v = line.split()
            if len(v) >= 8 and v[1] == "OMNIBUS":
                snps = v[7].split("|")
                row = {"locus": v[0],
                       "p": float(v[6]) if v[6] != "NA" else np.nan,
                       "chisq": float(v[4]) if v[4] != "NA" else np.nan,
                       "df": v[5], "n_snps": len(snps), "snps": snps}
                if pos is not None:
                    p_pos = [pos[s] for s in snps if s in pos]
                    row["start"] = min(p_pos) if p_pos else np.nan
                    row["end"] = max(p_pos) if p_pos else np.nan
                rows.append(row)
    return rows


def select_blocks(rows, thr=PROTOCOL_P):
    """Blocks with OMNIBUS p <= thr."""
    return [r for r in rows if np.isfinite(r["p"]) and r["p"] <= thr]


def causal_snps(disease_csv):
    """Causal (disease) SNP ids: rows with a non-None cl column."""
    causal = []
    with open(disease_csv) as f:
        header = f.readline().strip().split(",")
        i_rs, i_cl = header.index("rs"), header.index("cl")
        for line in f:
            v = line.rstrip("\n").split(",")
            if v[i_cl] != "None":
                causal.append(v[i_rs])
    return sorted(set(causal))


def snp_positions(bim):
    """dict snp -> bp from a PLINK .bim/.map file."""
    pos = {}
    with open(bim) as f:
        for line in f:
            v = line.split()
            pos[v[1]] = int(v[3])
    return pos


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def clustering_stats(labels):
    """Block count, noise fraction and block-size statistics."""
    n = labels.size
    noise = float((labels == -1).mean())
    sizes = np.bincount(labels[labels != -1]) if (labels != -1).any() \
        else np.array([])
    return {
        "n_snps": int(n),
        "n_blocks": int(sizes.size),
        "noise": noise,
        "size_median": float(np.median(sizes)) if sizes.size else np.nan,
        "size_max": int(sizes.max()) if sizes.size else 0,
        "size_var": float(np.var(sizes, ddof=1)) if sizes.size > 1 else 0.0,
    }


def pairwise_ari(labels_dict):
    """Adjusted Rand index between every pair of labelings."""
    names = list(labels_dict)
    ari = {f"{a}|{b}": adjusted_rand_score(labels_dict[a], labels_dict[b])
           for i, a in enumerate(names) for b in names[i + 1:]}
    return ari
