import gzip
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np


# --------------------------------------------------------------------------
def load_observation_dict(d: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (H, pos) from a dict {'word_embeddings': ..., 'pos_tags': ...}.
    H : (n_layers, N, D), pos : (N,). Fixes the axis order if needed.
    """
    H = np.asarray(d["word_embeddings"])
    pos = np.asarray(d["pos_tags"])

    if H.ndim != 3:
        raise ValueError(f"word_embeddings must be 3D, got {H.shape}")

    N = pos.shape[0]
    if H.shape[0] == N and H.shape[1] != N:
        H = H.transpose(1, 0, 2)              # (N, L, D) -> (L, N, D)
    elif H.shape[1] != N and H.shape[0] != N:
        raise ValueError(
            f"no axis of word_embeddings {H.shape} matches N={N}"
        )
    return H, pos


def load_observation(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rb") as f:
        d = pickle.load(f)
    return load_observation_dict(d)



def _prep(X: np.ndarray, metric: str, dtype=np.float32) -> np.ndarray:
    """Contiguous copy in the chosen dtype; L2-normalized if metric='cosine'."""
    X = np.ascontiguousarray(X, dtype=dtype)
    if metric == "cosine":
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X


def _sqnorm(X: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", X, X)


def _dist_block(X, XT, sq, rows_blk):
    """Squared distances from rows_blk to every point, self = inf."""
    d2 = X[rows_blk] @ XT
    d2 *= -2.0
    d2 += sq[rows_blk, None]
    d2 += sq[None, :]
    d2[np.arange(len(rows_blk)), rows_blk] = np.inf
    return d2


def _nn_rows(X, sq, rows, k, word_ids=None, chunk=1024) -> np.ndarray:
    """
    Indices of the k nearest global neighbours of the points in rows (self
    excluded; lexical clones excluded too if word_ids is given).
    """
    XT = np.ascontiguousarray(X.T)
    out = np.empty((len(rows), k), dtype=np.int64)
    for s in range(0, len(rows), chunk):
        r = rows[s:s + chunk]
        d2 = _dist_block(X, XT, sq, r)
        if word_ids is not None:
            d2[word_ids[r, None] == word_ids[None, :]] = np.inf
        if k == 1:
            out[s:s + chunk, 0] = np.argmin(d2, axis=1)
        else:
            part = np.argpartition(d2, k - 1, axis=1)[:, :k]
            order = np.argsort(np.take_along_axis(d2, part, axis=1), axis=1)
            out[s:s + chunk] = np.take_along_axis(part, order, axis=1)
    return out


def _ranks_rows(Y, sq, rows, targets, chunk=1024) -> np.ndarray:
    """
    rank^Y(i, targets[i, c]) for i in rows, c in [0, k). Ranks are counted
    over ALL N points (self excluded), as in the original: #{j : d(i,j) <
    d(i,target)} + 1.
    """
    YT = np.ascontiguousarray(Y.T)
    out = np.empty(targets.shape, dtype=np.float64)
    for s in range(0, len(rows), chunk):
        r = rows[s:s + chunk]
        d2 = _dist_block(Y, YT, sq, r)
        dt = np.take_along_axis(d2, targets[s:s + chunk], axis=1)   # (m, k)
        for c in range(targets.shape[1]):
            out[s:s + chunk, c] = (d2 < dt[:, c][:, None]).sum(axis=1) + 1
    return out


# --------------------------------------------------------------------------
def conditional_ii(
    H: np.ndarray,
    pos: np.ndarray,
    layer_a: int,
    layer_b: int,
    pos_tag: Optional[str] = None,
    k: int = 1,
    metric: str = "euclidean",
    word_ids: Optional[np.ndarray] = None,
    dtype=np.float32,
    chunk: int = 1024,
) -> Tuple[float, float, int]:
    """
    Delta_p(l_a -> l_b) and Delta_p(l_b -> l_a), same semantics as the
    original. For full profiles use `profile`, which shares work across PoS
    tags and layers; this one is kept for single calls.
    """
    N = H.shape[1]
    rows = np.arange(N) if pos_tag is None else np.flatnonzero(pos == pos_tag)
    if rows.size == 0:
        return np.nan, np.nan, 0

    A = _prep(H[layer_a], metric, dtype)
    B = _prep(H[layer_b], metric, dtype)
    sqA, sqB = _sqnorm(A), _sqnorm(B)
    norm = N / 2.0

    nnA = _nn_rows(A, sqA, rows, k, word_ids, chunk)
    nnB = _nn_rows(B, sqB, rows, k, word_ids, chunk)
    d_ab = float(_ranks_rows(B, sqB, rows, nnA, chunk).mean(axis=1).mean() / norm)
    d_ba = float(_ranks_rows(A, sqA, rows, nnB, chunk).mean(axis=1).mean() / norm)
    return d_ab, d_ba, rows.size


# --------------------------------------------------------------------------
def _select_nn(d2, k, rows_blk, word_ids, out):
    """argmin/argpartition over a block of distances, with the clone mask."""
    if word_ids is not None:
        d2[word_ids[rows_blk, None] == word_ids[None, :]] = np.inf
    s = rows_blk[0]
    if k == 1:
        out[s:s + len(rows_blk), 0] = np.argmin(d2, axis=1)
    else:
        part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(d2, part, axis=1), axis=1)
        out[s:s + len(rows_blk)] = np.take_along_axis(part, order, axis=1)


def _rank_matrix_and_nn(X, sq, k, word_ids, chunk):
    """
    A single pass over the layer, returning
      R  : (N, N) int32, R[i, j] = #{m : d(i,m) < d(i,j)} + 1 (self excluded),
           ties handled exactly (lowest rank, matching the strict count of
           the original);
      nn : (N, k) nearest neighbours (clones excluded if word_ids is given).
    """
    N = X.shape[0]
    XT = np.ascontiguousarray(X.T)
    R = np.empty((N, N), dtype=np.int32)
    nn = np.empty((N, k), dtype=np.int64)
    col = np.arange(N)[None, :]
    for s in range(0, N, chunk):
        rows_blk = np.arange(s, min(s + chunk, N))
        d2 = _dist_block(X, XT, sq, rows_blk)
        order = np.argsort(d2, axis=1)
        sd = np.take_along_axis(d2, order, axis=1)
        # lowest rank of a tie group = position of its first occurrence
        start = np.empty(sd.shape, dtype=bool)
        start[:, 0] = True
        np.greater(sd[:, 1:], sd[:, :-1], out=start[:, 1:])
        minrank = np.where(start, col, 0)
        np.maximum.accumulate(minrank, axis=1, out=minrank)
        blk = np.empty(sd.shape, dtype=np.int32)
        np.put_along_axis(blk, order, (minrank + 1).astype(np.int32), axis=1)
        R[rows_blk] = blk
        _select_nn(d2, k, rows_blk, word_ids, nn)          #after the ranks!
    return R, nn


def _layer_pass(X, sq, k, targets_list, word_ids, chunk):
    """
    A single GEMM pass over layer X: for every target matrix T in
    targets_list compute rank^X(i, T[i, :]) and, from the same distance
    block, also take the k nearest neighbours nn_X(i).
    """
    N = X.shape[0]
    XT = np.ascontiguousarray(X.T)
    nn = np.empty((N, k), dtype=np.int64)
    outs = [np.empty(T.shape, dtype=np.float64) for T in targets_list]
    for s in range(0, N, chunk):
        rows_blk = np.arange(s, min(s + chunk, N))
        d2 = _dist_block(X, XT, sq, rows_blk)
        for T, out in zip(targets_list, outs):
            dt = np.take_along_axis(d2, T[rows_blk], axis=1)
            for c in range(T.shape[1]):
                out[rows_blk, c] = (d2 < dt[:, c][:, None]).sum(axis=1) + 1
        _select_nn(d2, k, rows_blk, word_ids, nn)          # after the ranks!
    return nn, outs


def profile(
    H: np.ndarray,
    pos: np.ndarray,
    pos_tags: Optional[Sequence[str]] = None,
    k: int = 1,
    metric: str = "cosine",
    word_ids: Optional[np.ndarray] = None,
    dtype=np.float32,
    chunk: int = 1024,
    max_rank_matrix_gb: float = 2.0,
):
    """
    Same output as the original. The Deltas are computed as per-token vectors
    and then averaged per PoS group (conditioning touches neither neighbours
    nor ranks, only the final average). Two paths:

      * fast path (default when 2*N^2*4 bytes <= max_rank_matrix_gb): rank
        matrices for the reference layers are built once, then ONE GEMM pass
        per intermediate layer;
      * constant-memory path: same, but the ranks towards first/last are
        recounted at every layer (4 passes per layer).
    """
    import pandas as pd

    L, N, _ = H.shape
    allr = np.arange(N)
    norm = N / 2.0
    if pos_tags is None:
        pos_tags = sorted(set(pos.tolist()))
    groups = {p: np.flatnonzero(pos == p) for p in pos_tags}

    Hn = [_prep(H[i], metric, dtype) for i in range(L)]
    sq = [_sqnorm(X) for X in Hn]

    use_matrix = 2 * N * N * 4 <= max_rank_matrix_gb * 1e9

    if use_matrix:
        R_first, nn_first = _rank_matrix_and_nn(Hn[0], sq[0], k, word_ids, chunk)
        if L == 1:
            R_last, nn_last = R_first, nn_first
        else:
            R_last, nn_last = _rank_matrix_and_nn(Hn[-1], sq[-1], k, word_ids, chunk)
        rowsel = allr[:, None]

        def deltas(i):
            if i == 0:
                nn_i, (r_f_i, r_l_i) = nn_first, (
                    R_first[rowsel, nn_first].astype(np.float64),
                    R_first[rowsel, nn_last].astype(np.float64))
            elif i == L - 1:
                nn_i, (r_f_i, r_l_i) = nn_last, (
                    R_last[rowsel, nn_first].astype(np.float64),
                    R_last[rowsel, nn_last].astype(np.float64))
            else:
                nn_i, (r_f_i, r_l_i) = _layer_pass(
                    Hn[i], sq[i], k, [nn_first, nn_last], word_ids, chunk)
            d_i_f = R_first[rowsel, nn_i].mean(axis=1) / norm
            d_i_l = R_last[rowsel, nn_i].mean(axis=1) / norm
            return d_i_f, r_f_i.mean(axis=1) / norm, d_i_l, r_l_i.mean(axis=1) / norm
    else:
        nn_first = _nn_rows(Hn[0], sq[0], allr, k, word_ids, chunk)
        nn_last = nn_first if L == 1 else _nn_rows(Hn[-1], sq[-1], allr, k, word_ids, chunk)

        def deltas(i):
            if i == 0:
                nn_i = nn_first
            elif i == L - 1:
                nn_i = nn_last
            else:
                nn_i = _nn_rows(Hn[i], sq[i], allr, k, word_ids, chunk)
            d_i_f = _ranks_rows(Hn[0], sq[0], allr, nn_i, chunk).mean(axis=1) / norm
            d_i_l = _ranks_rows(Hn[-1], sq[-1], allr, nn_i, chunk).mean(axis=1) / norm
            d_f_i = _ranks_rows(Hn[i], sq[i], allr, nn_first, chunk).mean(axis=1) / norm
            d_l_i = _ranks_rows(Hn[i], sq[i], allr, nn_last, chunk).mean(axis=1) / norm
            return d_i_f, d_f_i, d_i_l, d_l_i

    def gmean(v, g):
        return float(v[g].mean()) if g.size else np.nan

    recs = []
    for i in range(L):
        d_i_f, d_f_i, d_i_l, d_l_i = deltas(i)
        for p in pos_tags:
            g = groups[p]
            recs.append({
                "pos": p, "layer": i, "n": int(g.size),
                "d_i_to_first": gmean(d_i_f, g), "d_first_to_i": gmean(d_f_i, g),
                "d_i_to_last": gmean(d_i_l, g), "d_last_to_i": gmean(d_l_i, g),
            })
    return pd.DataFrame(recs)


def nn_pos_distribution(H: np.ndarray, pos: np.ndarray, layer: int,
                        pos_tag: str, metric: str = "euclidean",
                        dtype=np.float32, chunk: int = 1024) -> Dict[str, float]:
    """Which PoS the nearest global neighbour of the pos_tag tokens has, at one layer."""
    rows = np.flatnonzero(pos == pos_tag)
    if rows.size == 0:
        return {}
    X = _prep(H[layer], metric, dtype)
    idx = _nn_rows(X, _sqnorm(X), rows, 1, None, chunk)[:, 0]
    c = Counter(pos[idx])
    tot = sum(c.values())
    return {p: v / tot for p, v in c.most_common()}


# --------------------------------------------------------------------------
def analyze_dict(
    data: Dict[str, np.ndarray],
    pos_tags: Optional[Sequence[str]] = None,
    k: int = 1,
    metric: str = "euclidean",
    exclude_lexical_clones: bool = False,
    nn_dist_at: Sequence[int] = (),
    name: str = "observation",
    dtype=np.float32,
    chunk: int = 1024,
):

    
    H, pos = load_observation_dict(data)
    L, N, D = H.shape
    print(f"{name}: {L} layers, {N} tokens, D={D}")
    print("PoS:", {p: int((pos == p).sum()) for p in sorted(set(pos.tolist()))})

    word_ids = None
    if exclude_lexical_clones:
        _, word_ids = np.unique(
            np.asarray(H[0], dtype=np.float64).round(4), axis=0, return_inverse=True
        )

    df = profile(H, pos, pos_tags, k=k, metric=metric,
                 word_ids=word_ids, dtype=dtype, chunk=chunk)

    for i in nn_dist_at:
        print(f"\n--- nearest neighbour by PoS, layer {i} ---")
        for p in (pos_tags or sorted(set(pos.tolist()))):
            dist = nn_pos_distribution(H, pos, i, p, metric, dtype, chunk)
            print(f"  {p:6s}", {q: round(v, 2) for q, v in list(dist.items())[:4]})

    return df