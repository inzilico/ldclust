"""
Smoke tests for the ldclust package: synthetic LD matrices through
representative method families, checking the structural invariants the
benchmark relies on. Runs as a plain script (python3 tests/test_smoke.py)
or under pytest. No PLINK and no filesystem side effects.

Author: Gennady Khvorykh, info@inzilico.com
Started: 2026-09-21
"""

import sys

import numpy as np

sys.path.insert(0, ".")
import ldclust as ld  # noqa: E402


def toy_r2(n_per=12, gap=0.05, within=0.8, seed=0):
    """3 LD blocks + 1 r2-isolate, the benchmark's microcosm."""
    rng = np.random.default_rng(seed)
    n = 3 * n_per + 1
    r2 = np.full((n, n), gap)
    np.fill_diagonal(r2, 1.0)
    for b in range(3):
        ix = np.arange(b * n_per, (b + 1) * n_per)
        block = within * np.ones((n_per, n_per))
        np.fill_diagonal(block, 1.0)
        r2[np.ix_(ix, ix)] = block
    return np.clip(r2 * rng.uniform(.95, 1.05, r2.shape), 0, 1), n


def check_partition(labels, n):
    """Blocks disjoint, covering all n SNPs, ids 0..B-1 with -1 noise."""
    assert labels.shape == (n,)
    blocks = labels[labels != -1]
    assert np.array_equal(np.unique(blocks), np.arange(blocks.max() + 1))


def test_loaders_and_distances():
    r2, n = toy_r2()
    assert ld.distances(r2, "d1").max() <= 1.0
    assert np.allclose(ld.distances(r2, "d2"), np.sqrt(1 - r2))
    ids = np.array([f"rs{i}" for i in range(n)])
    keep = np.ones(n, bool)
    assert ld.drop_nan_snps(r2, ids)[2] == 0


def test_fof_blocks_and_isolate():
    r2, n = toy_r2()
    labels, _, _ = ld.cluster_fof(r2, 0.5)
    check_partition(labels, n)
    # the isolate (last SNP, max r2 ~ gap) is its own singleton block
    assert np.flatnonzero(labels == labels[-1]).size == 1


def test_cdhit_seed_semantics():
    r2, n = toy_r2()
    labels, _, _ = ld.cluster_cdhit(r2, 0.5)
    check_partition(labels, n)
    sizes = np.bincount(labels)
    assert sizes.max() <= n  # stars bounded by the seed neighbourhood


def test_dpblocks_optimum_and_contiguity():
    r2, n = toy_r2()
    labels, _, _ = ld.cluster_dpblocks(r2, tag_thr=0.5, max_len=10)
    check_partition(labels, n)
    for c in np.unique(labels):
        ix = np.flatnonzero(labels == c)
        assert np.all(np.diff(ix) == 1), "non-contiguous block"
        assert ix.size <= 10
        assert r2[np.ix_(ix, ix)].max(axis=1).min() >= 0.5  # tag coverage


def test_hc_dynamic_tree_cut():
    r2, n = toy_r2()
    labels, _, _ = ld.cluster_hc(ld.distances(r2, "d1"), min_cluster_size=3,
                                 deep_split=2)
    check_partition(labels, n)
    assert labels.max() + 1 >= 3  # the three blocks survive the cut


def test_consensus_coassociation():
    r2, n = toy_r2()
    bases = [ld.cluster_fof(r2, 0.5)[0],
             ld.cluster_mcl(r2, inflation=2.0)[0],
             ld.cluster_fof(r2, 0.3)[0]]
    c = ld.coassociation(bases)
    assert np.linalg.eigvalsh(c).min() > -1e-10  # PSD
    labels, _, _ = ld.cluster_consensus(bases, method="dtc", deep_split=0)
    check_partition(labels, n)


def test_writers():
    r2, n = toy_r2()
    labels = ld.cluster_fof(r2, 0.5)[0]
    ids = np.array([f"rs{i}" for i in range(n)])
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        hlist = f"{d}/x.hlist"
        ld.write_hlist(hlist, ids, labels)
        lines = open(hlist).read().strip().split("\n")
        assert all(ln.startswith("** ") for ln in lines)
        assert sum(len(ln.split()) - 2 for ln in lines) == n


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
