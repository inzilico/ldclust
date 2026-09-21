"""
ldclust command line: run one clustering method on an LD matrix and
write the benchmark artifacts (blocks csv, labels npy, hlist for PLINK
--hap-assoc, and a row in <out-dir>/summary.tsv).

Methods:
  hdbscan  HDBSCAN on precomputed distances (--dist d1|d2, default d2)
  optics   OPTICS on precomputed distances d2 (sklearn)
  soptics  OPTICS on the spectral embedding of the r2 matrix
  fof      friends-of-friends percolation on r2 >= --eps (no noise label)
  dbscan   DBSCAN on precomputed distances d2 with --eps
  louvain  modularity communities on the weighted r2 graph
  spectral normalized-affinity spectral clustering (Ng et al. 2002)
  ap       affinity propagation on the raw r2 similarity
  mcl      Markov clustering on the weighted r2 graph
  lpa      asynchronous label propagation on the r2 graph
  leiden   Leiden RBConfiguration partition on the r2 graph
  walktrap Walktrap communities on the r2 graph
  infomap  Infomap two-level partition on the r2 graph
  pam      similarity k-medoids on the raw r2
  cnm      greedy modularity (CNM) on the r2 graph
  cw       Chinese Whispers on the r2 graph
  gmm      Gaussian mixture on the cross-product-matrix embedding
  kmeans   K-Means on the Euclidean embedding of 1 - r2
  minibatch Mini-Batch K-Means on the same embedding
  hc       agglomerative linkage on 1-r2 + dynamic tree cut (noise = -1)
  dpblocks dynamic-programming contiguous blocks minimizing tag SNPs
  sbm      degree-corrected stochastic block model (needs graph-tool;
           run inside its conda environment - see README)
  cdhit    CD-HIT-style greedy seed-and-recruit on r2 >= --eps
  consensus evidence-accumulation consensus of existing runs' labels
           (--bases tag1,tag2,... --consensus dtc|fof)
  dpgmm    Dirichlet-process GMM on the gmm embedding (--k = cap only)

Usage:
  ldclust -p <prefix> --method <name> [--eps 0.5] [--dist d2]
      [--out-dir data/bench] [--check <labels.npy>]
  (prefix names prefix.ld.h5 + prefix.snplist; --method hdbscan|gmm
  also read prefix.gmm.raw via PLINK --recodeA output)

Author: Gennady Khvorykh, info@inzilico.com
Started: 2026-09-21
"""

import argparse
import os
import time

import numpy as np

from . import __version__
from . import library as ld


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ldclust",
        description="LD-based clustering runner (ldclust package)")
    parser.add_argument("-p", "--prefix", required=True,
                        help="/path/to/prefix of prefix.ld.h5 file")
    parser.add_argument("--method", required=True,
                        choices=["hdbscan", "optics", "soptics", "fof",
                                 "dbscan", "louvain", "spectral", "ap",
                                 "mcl", "lpa", "leiden", "walktrap",
                                 "infomap", "pam", "cnm", "cw", "gmm",
                                 "kmeans", "minibatch", "hc", "dpblocks",
                                 "sbm", "cdhit", "consensus", "dpgmm"])
    parser.add_argument("--dist", choices=["d1", "d2"], default="d2")
    parser.add_argument("--eps", type=float, default=None,
                        help="threshold for fof and cdhit (r2), dbscan "
                             "(distance), consensus-fof (co-association)")
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--xi", type=float, default=0.05)
    parser.add_argument("--variance", type=float, default=0.95)
    parser.add_argument("--k", type=int, default=None,
                        help="components: soptics embedding dim / pam "
                             "blocks / gmm components / dpgmm cap")
    parser.add_argument("--dim", type=int, default=50,
                        help="embedding dimension for gmm/dpgmm")
    parser.add_argument("--gamma", type=float, default=None,
                        help="resolution for louvain")
    parser.add_argument("--theta", type=float, default=None,
                        help="eigenvalue cutoff for spectral "
                             "(k = #(w > theta))")
    parser.add_argument("--preference", type=float, default=None,
                        help="self-similarity (diagonal) for ap")
    parser.add_argument("--inflation", type=float, default=None,
                        help="inflation for mcl")
    parser.add_argument("--cutoff", type=float, default=0.05,
                        help="edge cutoff on r2 for graph methods; "
                             "consensus bases count; sbm")
    parser.add_argument("--steps", type=int, default=4,
                        help="random-walk length for walktrap")
    parser.add_argument("--linkage", choices=["average", "complete"],
                        default="average", help="agglomeration for hc")
    parser.add_argument("--min-cluster-size", type=int, default=3,
                        help="dynamic tree cut minimum block size for hc")
    parser.add_argument("--deep-split", type=int, default=None,
                        choices=[0, 1, 2, 3, 4],
                        help="dynamic tree cut deepSplit preset for hc")
    parser.add_argument("--tag-thr", type=float, default=None,
                        help="r2 tag-coverage threshold for dpblocks")
    parser.add_argument("--max-len", type=int, default=None,
                        help="maximum block length in SNPs for dpblocks")
    parser.add_argument("--no-deg-corr", action="store_true",
                        help="plain SBM (no degree correction) for sbm")
    parser.add_argument("--binary", action="store_true",
                        help="ignore r2 weights (binary graph) for sbm")
    parser.add_argument("--nested", action="store_true",
                        help="nested SBM (level-0 partition) for sbm")
    parser.add_argument("--priority", choices=["degree", "order"],
                        default="degree", help="seed priority for cdhit")
    parser.add_argument("--recruit", choices=["seed", "member"],
                        default="seed", help="recruit semantics for cdhit")
    parser.add_argument("--bases", default=None,
                        help="comma-separated base run tags for consensus "
                             "(labels.npy in --out-dir)")
    parser.add_argument("--consensus", choices=["dtc", "fof"], default="dtc",
                        help="consensus function on the co-association "
                             "matrix")
    parser.add_argument("--conc", type=float, default=None,
                        help="DP weight concentration prior for dpgmm "
                             "(default: sklearn 1/cap)")
    parser.add_argument("--out-dir", default="data/bench")
    parser.add_argument("--check", default=None,
                        help="optional labels.npy to verify determinism")
    parser.add_argument("--version", action="version",
                        version="%(prog)s " + __version__)
    return parser


def main(argv=None):
    t1 = time.time()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.method == "fof" and args.eps is None:
        parser.error("--eps is required for fof")
    if args.method == "dbscan" and args.eps is None:
        parser.error("--eps is required for dbscan")
    if args.method == "louvain" and args.gamma is None:
        parser.error("--gamma is required for louvain")
    if args.method == "spectral" and args.theta is None:
        parser.error("--theta is required for spectral")
    if args.method == "ap" and args.preference is None:
        parser.error("--preference is required for ap")
    if args.method == "mcl" and args.inflation is None:
        parser.error("--inflation is required for mcl")
    if args.method == "leiden" and args.gamma is None:
        parser.error("--gamma is required for leiden")
    if args.method == "pam" and args.k is None:
        parser.error("--k is required for pam")
    if args.method == "gmm" and args.k is None:
        parser.error("--k is required for gmm")
    if args.method == "kmeans" and args.k is None:
        parser.error("--k is required for kmeans")
    if args.method == "minibatch" and args.k is None:
        parser.error("--k is required for minibatch")
    if args.method == "dpgmm" and args.k is None:
        parser.error("--k (component cap) is required for dpgmm")
    if args.method == "hc" and args.deep_split is None:
        parser.error("--deep-split is required for hc")
    if args.method == "cdhit" and args.eps is None:
        parser.error("--eps is required for cdhit")
    if args.method == "consensus" and not args.bases:
        parser.error("--bases is required for consensus")
    if (args.method == "consensus" and args.consensus == "fof"
            and args.eps is None):
        parser.error("--eps (co-association threshold) is required for "
                     "fof consensus")
    if args.method == "dpblocks" and (args.tag_thr is None
                                      or args.max_len is None):
        parser.error("--tag-thr and --max-len are required for dpblocks")

    os.makedirs(args.out_dir, exist_ok=True)
    name = os.path.basename(args.prefix)

    ids, r2, layout = ld.load_ld(args.prefix)
    r2, ids, n_dropped = ld.drop_nan_snps(r2, ids)
    if n_dropped:
        print(f"Dropped {n_dropped} SNPs with NaN")
    print(f"Loaded {ids.size} SNPs ({layout})")

    substrate = "precomputed-" + args.dist
    if args.method == "hdbscan":
        labels, probs, sec = ld.cluster_hdbscan(ld.distances(r2, args.dist))
        tag = f"hdbscan.{args.dist}"
    elif args.method == "optics":
        labels, probs, sec = ld.cluster_optics(ld.distances(r2, "d2"),
                                               args.min_samples, args.xi)
        tag = "optics"
    elif args.method == "soptics":
        ksfx = f".emb{args.k}" if args.k else ""
        emb_file = os.path.join(args.out_dir, f"{name}{ksfx}.embedding.npy")
        if os.path.isfile(emb_file):
            coords = np.load(emb_file)
            print(f"Loaded embedding from {emb_file} (k = {coords.shape[1]})")
        else:
            coords, k, _ = ld.spectral_embedding(r2, args.variance, args.k)
            np.save(emb_file, coords)
            print(f"Built spectral embedding (k = {k}), saved to {emb_file}")
        labels, probs, sec = ld.cluster_soptics(coords, args.min_samples,
                                                args.xi)
        tag = "soptics" + (f"k{args.k}" if args.k else "")
        substrate = "spectral-embedding"
    elif args.method == "fof":
        labels, probs, sec = ld.cluster_fof(r2, args.eps)
        tag = f"fof{args.eps}"
        substrate = "r2-graph"
    elif args.method == "louvain":
        labels, probs, sec = ld.cluster_louvain(r2, args.gamma)
        tag = f"louvain{args.gamma}"
        substrate = "r2-graph"
    elif args.method == "spectral":
        labels, probs, sec = ld.cluster_spectral(r2, args.theta)
        tag = f"spectral{args.theta}"
        substrate = "normalized-affinity"
    elif args.method == "ap":
        labels, probs, sec = ld.cluster_ap(r2, args.preference)
        tag = f"ap{args.preference}"
        substrate = "r2-similarity"
    elif args.method == "mcl":
        labels, probs, sec = ld.cluster_mcl(r2, args.inflation, args.cutoff)
        tag = f"mcl{args.inflation}"
        substrate = "r2-graph"
    elif args.method == "lpa":
        labels, probs, sec = ld.cluster_lpa(r2, args.cutoff)
        tag = "lpa"
        substrate = "r2-graph"
    elif args.method == "leiden":
        labels, probs, sec = ld.cluster_leiden(r2, args.gamma, args.cutoff)
        tag = f"leiden{args.gamma}"
        substrate = "r2-graph"
    elif args.method == "walktrap":
        labels, probs, sec = ld.cluster_walktrap(r2, args.steps, args.cutoff)
        tag = f"walktrap{args.steps}"
        substrate = "r2-graph"
    elif args.method == "infomap":
        labels, probs, sec = ld.cluster_infomap(r2, args.cutoff)
        tag = "infomap"
        substrate = "r2-graph"
    elif args.method == "pam":
        labels, probs, sec = ld.cluster_pam(r2, args.k)
        tag = f"pam{args.k}"
        substrate = "r2-similarity"
    elif args.method == "cnm":
        labels, probs, sec = ld.cluster_cnm(r2, args.cutoff)
        tag = "cnm"
        substrate = "r2-graph"
    elif args.method == "cw":
        labels, probs, sec = ld.cluster_cw(r2, args.cutoff)
        tag = "cw"
        substrate = "r2-graph"
    elif args.method == "gmm":
        # relationship matrix from the genotype dosages, then GMM on
        # its spectral embedding with a full per-component covariance
        gmm_prefix = args.prefix + ".gmm"  # genotype prefix must expose .raw
        emb_file = os.path.join(args.out_dir,
                                f"{name}.gmm{args.dim}.embedding.npy")
        if os.path.isfile(emb_file):
            coords = np.load(emb_file)
            print(f"Loaded embedding from {emb_file} (d = {coords.shape[1]})")
        else:
            k_grm, raw_snps = ld.cross_product_matrix(gmm_prefix)
            if not np.array_equal(raw_snps, ids):
                print("Cross-product SNP order differs from snplist - "
                      "reordering")
                pos_of = {s: i for i, s in enumerate(raw_snps)}
                order = np.array([pos_of[s] for s in ids])
                k_grm = k_grm[np.ix_(order, order)]
            coords, dim_used, _ = ld.spectral_embedding(k_grm, k=args.dim)
            del k_grm
            np.save(emb_file, coords)
            print(f"Built cross-product embedding (d = {dim_used}), "
                  f"saved to {emb_file}")
        labels, probs, sec = ld.cluster_gmm(k=args.k, dim=args.dim,
                                            coords=coords)
        tag = f"gmm{args.k}d{args.dim}"
        substrate = "cross-product-embedding"
    elif args.method == "kmeans":
        # K-Means on the Euclidean embedding of the distance-converted LD
        # matrix (d = 1 - r2 is squared Euclidean; classical MDS coords)
        emb_file = os.path.join(
            args.out_dir, f"{name}.kmeans{args.dim}.embedding.npy")
        if os.path.isfile(emb_file):
            coords = np.load(emb_file)
            print(f"Loaded embedding from {emb_file} (d = {coords.shape[1]})")
        else:
            coords, dim_used, _ = ld.spectral_embedding(r2, k=args.dim)
            np.save(emb_file, coords)
            print(f"Built r2 embedding (d = {dim_used}), saved to {emb_file}")
        labels, probs, sec = ld.cluster_kmeans(r2, args.k, args.dim)
        tag = f"kmeans{args.k}d{args.dim}"
        substrate = "r2-embedding"
    elif args.method == "minibatch":
        # Mini-Batch K-Means on the same embedding as kmeans (shared cache)
        emb_file = os.path.join(
            args.out_dir, f"{name}.kmeans{args.dim}.embedding.npy")
        if os.path.isfile(emb_file):
            coords = np.load(emb_file)
            print(f"Loaded embedding from {emb_file} (d = {coords.shape[1]})")
        else:
            coords, dim_used, _ = ld.spectral_embedding(r2, k=args.dim)
            np.save(emb_file, coords)
            print(f"Built r2 embedding (d = {dim_used}), saved to {emb_file}")
        labels, probs, sec = ld.cluster_minibatch_kmeans(r2, args.k, args.dim)
        tag = f"mbk{args.k}d{args.dim}"
        substrate = "r2-embedding"
    elif args.method == "hc":
        # WGCNA-style: scipy linkage on the condensed d1 = 1 - r2 (the
        # analogue of WGCNA's 1 - |cor|), then the dynamic tree cut
        # (Langfelder et al. 2008, tree stage) on the dendrogram
        labels, probs, sec = ld.cluster_hc(ld.distances(r2, "d1"),
                                           args.linkage,
                                           args.min_cluster_size,
                                           args.deep_split)
        tag = f"dtc{args.deep_split}" + ("" if args.linkage == "average"
                                         else args.linkage[0])
        substrate = "precomputed-d1"
    elif args.method == "dpblocks":
        # Zhang et al. 2002 DP framework, r2-tagging cost: optimal
        # contiguous partition of the ordered SNPs minimizing total tag
        # SNPs (every SNP covered at r2 >= tag_thr; greedy set cover per
        # candidate block)
        labels, probs, sec = ld.cluster_dpblocks(r2, args.tag_thr,
                                                 args.max_len)
        tag = f"dp{args.tag_thr}L{args.max_len}"
        substrate = "r2-ordered"
    elif args.method == "sbm":
        # degree-corrected SBM on the weighted r2 graph (Peixoto 2017 MDL,
        # graph-tool); run this CLI inside the graph-tool environment
        labels, probs, sec = ld.cluster_sbm(r2, args.cutoff,
                                            deg_corr=not args.no_deg_corr,
                                            weighted=not args.binary,
                                            nested=args.nested)
        tag = ("sbm" + ("nd" if args.no_deg_corr else "")
               + ("b" if args.binary else "w") + str(args.cutoff)
               + ("n" if args.nested else ""))
        substrate = "r2-graph-mdl"
    elif args.method == "cdhit":
        # CD-HIT-style greedy seed-and-recruit: hub SNPs (weighted degree)
        # seed blocks first, recruits need r2 >= eps to the seed (no
        # transitive chaining - the anti-FoF)
        labels, probs, sec = ld.cluster_cdhit(r2, args.eps, args.priority,
                                              args.recruit)
        tag = (f"cd{args.eps}" + ("o" if args.priority == "order" else "")
               + ("m" if args.recruit == "member" else ""))
        substrate = "r2-similarity"
    elif args.method == "consensus":
        # evidence-accumulation consensus: co-association matrix of the
        # base partitions, then DTC on 1 - C (C is PSD like r2) or
        # percolation
        bases = args.bases.split(",")
        base_labels = [np.load(os.path.join(
            args.out_dir, f"{name}.{b}.labels.npy")) for b in bases]
        ds = args.deep_split if args.deep_split is not None else 2
        labels, probs, sec = ld.cluster_consensus(
            base_labels, method=args.consensus, thr=args.eps, deep_split=ds,
            min_cluster_size=args.min_cluster_size)
        if args.consensus == "dtc":
            tag = f"consdtc{ds}"
        else:
            tag = f"consfof{args.eps:g}"
        substrate = "coassociation-of:" + "+".join(bases)
    elif args.method == "dpgmm":
        # Dirichlet-process GMM on the same cached embedding as gmm; --k is
        # only an upper bound - the DP empties unsupported components
        emb_file = os.path.join(
            args.out_dir, f"{name}.gmm{args.dim}.embedding.npy")
        if os.path.isfile(emb_file):
            coords = np.load(emb_file)
            print(f"Loaded embedding from {emb_file} (d = {coords.shape[1]})")
        else:
            gmm_prefix = args.prefix + ".gmm"
            k_grm, raw_snps = ld.cross_product_matrix(gmm_prefix)
            if not np.array_equal(raw_snps, ids):
                print("Cross-product SNP order differs from snplist - "
                      "reordering")
                pos_of = {s: i for i, s in enumerate(raw_snps)}
                order = np.array([pos_of[s] for s in ids])
                k_grm = k_grm[np.ix_(order, order)]
            coords, dim_used, _ = ld.spectral_embedding(k_grm, k=args.dim)
            del k_grm
            np.save(emb_file, coords)
            print(f"Built cross-product embedding (d = {dim_used}), "
                  f"saved to {emb_file}")
        labels, probs, sec = ld.cluster_dpgmm(
            coords=coords, n_components=args.k, dim=args.dim,
            weight_concentration_prior=args.conc)
        tag = f"dpgmm{args.k}" + (f"a{args.conc:g}" if args.conc else "adef")
        substrate = "cross-product-embedding"
    elif args.method == "dbscan":
        labels, probs, sec = ld.cluster_dbscan(ld.distances(r2, "d2"),
                                               args.eps, args.min_samples)
        tag = f"dbscan{args.eps}"

    stats = ld.clustering_stats(labels)
    print(f"{tag}: {stats['n_blocks']} blocks, noise {stats['noise']:.2%}, "
          f"median size {stats['size_median']}, max {stats['size_max']}, "
          f"{sec:.0f} s")

    if args.check:
        ref = np.load(args.check)
        if ref.size == labels.size:
            print(f"Determinism check vs {args.check}: "
                  f"{int((ref != labels).sum())} label differences")

    ld.write_blocks(os.path.join(args.out_dir, f"{name}.{tag}.blocks.csv"),
                    ids, labels)
    np.save(os.path.join(args.out_dir, f"{name}.{tag}.labels.npy"), labels)
    ld.write_hlist(os.path.join(args.out_dir, f"{name}.{tag}.hlist"),
                   ids, labels)

    row = {"method": args.method, "tag": tag, "substrate": substrate,
           "params": f"eps={args.eps};min_samples={args.min_samples};"
                     f"xi={args.xi};dist={args.dist}", **stats,
           "seconds": sec}
    summary = os.path.join(args.out_dir, "summary.tsv")
    header = not os.path.isfile(summary)
    with open(summary, "a") as f:
        if header:
            f.write("\t".join(row.keys()) + "\n")
        f.write("\t".join(str(v) for v in row.values()) + "\n")
    print("Appended summary row to", summary)
    dur = time.strftime("%H:%M:%S", time.gmtime(time.time() - t1))
    print(f"Done {tag}, time elapsed: {dur}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
