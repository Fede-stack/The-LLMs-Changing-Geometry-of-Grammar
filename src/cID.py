"""
Per-PoS conditional intrinsic dimension, sentence by sentence, with a
conditional binomial likelihood (ABIDE restricted to one group).

Shared-scale strategy:
  1. For each (sentence, layer), ABIDE iterates over ALL tokens of the
     sentence: k*_i, tau and the global ID converge using the whole cloud.
  2. At convergence, the estimate restricted to a group G (one PoS) is the
     MLE of the conditional likelihood

         L_G(d) = prod_{i in G} Binomial(k_A,i | k_B,i, tau^d)

     which has a closed form:

         d_G = log( mean_G(k_A) / mean_G(k_B) ) / log(tau)

     A single binomial factor does not depend on the local density rho_i
     (it cancels in the ratio tau^d = mu(A)/mu(B)), so the product
     restricted to G is a legitimate likelihood for d. The neighbours of a
     point in G can have any PoS: k_B counts every point in the ball, and
     binomiality only depends on local uniformity.

  The geometry (k*_i, tau) is shared across groups: all the d_G of the same
  (sentence, layer) are measured at the same scale and are therefore
  comparable. The per-group error uses the restricted Fisher information
  (Cramer-Rao with |G| in place of n).
"""

import os
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np

POS_LIST_DEFAULT = ["ADJ", "ADP", "ADV", "PRON", "NOUN", "DET", "VERB"]


# --------------------------------------------------------------------------
def abide_conditional(
    X: np.ndarray,
    pos: np.ndarray,
    pos_list: Sequence[str] = POS_LIST_DEFAULT,
    n_iter: int = 5,
    alpha: float = 0.01,
    r: str = "opt",
    initial_id: Optional[float] = None,
    min_group_size: int = 10,
) -> Dict:
    """
    Global ABIDE over all tokens of a sentence + conditional per-PoS estimate.

    Args:
        X: (N, D) representations of all tokens of the sentence at ONE layer.
        pos: (N,) PoS tag per token.
        pos_list: the groups to compute the restricted estimate for.
        n_iter: ABIDE iterations (5 are enough, see the paper).
        alpha: quantile of the constant-density test (0.01 = Dthr 6.635).
        r: 'opt' for tau = 0.2032^(1/d), or a float in (0,1).
        initial_id: if None, start from the 2NN.
        min_group_size: below this size the group reports NaN (the restricted
            estimate is too noisy to mean anything).

    Returns:
        dict with:
          'id_global', 'id_global_err'
          for each p in pos_list: 'id_<p>', 'err_<p>', 'n_<p>'
    """
    from dadapy import Data
    from dadapy._utils import utils as ut

    X = np.ascontiguousarray(X, dtype=np.float64)
    N = X.shape[0]

    # --- smearing against duplicates ---------------------------------------
    #Repeated tokens (e.g. 'the', punctuation) can have identical
    #remove_identical_points() because dropping rows would break the
    #breaks the ties without moving the points noticeably.
    scale = X.std()
    if scale > 0:
        jit_rng = np.random.default_rng(0)          # fixed: reproducible
        X = X + (1e-8 * scale) * jit_rng.standard_normal(X.shape)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = Data(X, verbose=False)
        if initial_id is None:
            data.compute_id_2NN(algorithm="base")
        else:
            data.compute_distances()
            data.set_id(initial_id)

    id_est, id_err, r_eff, kstar, n_in_A = np.nan, np.nan, np.nan, None, None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_iter):
            data.compute_kstar(alpha)
            r_eff = (min(0.95, 0.2032 ** (1.0 / data.intrinsic_dim))
                     if r == "opt" else float(r))
            # radius of ball B = distance to the k*-th neighbour
            rk = np.array([dd[data.kstar[j]] for j, dd in enumerate(data.distances)])
            rn = rk * r_eff
            # count in ball A (includes the point itself, as in the original)
            n_in_A = np.sum(
                [dd < rn[j] for j, dd in enumerate(data.distances)], axis=1
            )
            # GLOBAL estimate: identical to the original ABIDE code
            id_est = (np.log((n_in_A.mean() - 1) / (data.kstar.mean() - 1))
                      / np.log(r_eff))
            id_err = ut._compute_binomial_cramerrao(
                id_est, data.kstar - 1, r_eff, N
            )
            kstar = data.kstar.copy()
            data.set_id(id_est)

    out = {"id_global": float(id_est), "id_global_err": float(id_err),
           "n_tokens": int(N)}

    # ---- conditional estimate: same geometry (kstar, r_eff), mean over G ----
    kA = n_in_A - 1          # points in ball A, centre excluded
    kB = kstar - 1           # points in ball B, centre excluded
    for p in pos_list:
        G = np.flatnonzero(pos == p)
        out[f"n_{p}"] = int(G.size)
        if G.size < min_group_size:
            out[f"id_{p}"] = np.nan
            out[f"err_{p}"] = np.nan
            continue
        mA, mB = kA[G].mean(), kB[G].mean()
        if mA <= 0 or mB <= 0 or mA >= mB:
            out[f"id_{p}"] = np.nan
            out[f"err_{p}"] = np.nan
            continue
        d_G = np.log(mA / mB) / np.log(r_eff)
        # restricted Cramer-Rao: Fisher information with the group's k_B and |G|
        err_G = ut._compute_binomial_cramerrao(d_G, kB[G], r_eff, G.size)
        out[f"id_{p}"] = float(d_G)
        out[f"err_{p}"] = float(err_G)

    return out


# --------------------------------------------------------------------------
def run_conditional_id_h5(
    h5_path: str,
    out_dir: str,
    pos_list: Sequence[str] = POS_LIST_DEFAULT,
    layers: Sequence[int] = tuple(range(1, 23)),
    sentence_ids: Optional[Sequence[str]] = None,
    n_iter: int = 5,
    alpha: float = 0.01,
    min_group_size: int = 10,
    verbose: bool = True,
):
    """
    For every sentence in the HDF5 file and every layer, run global ABIDE on
    the sentence and the conditional estimate for each PoS. Writes one CSV
    per PoS in out_dir: rows = sentences, columns = group token count plus an
    ID (and an error) per layer.

    Expected h5 layout (one group per sentence):
        f[sent_id]['word_embeddings'] : (n_layers, N, D)
        f[sent_id]['pos_tags']        : (N,) strings
    """
    import h5py
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    rows_per_pos: Dict[str, List[dict]] = {p: [] for p in pos_list}

    with h5py.File(h5_path, "r") as f:
        ids_all = list(f.keys()) if sentence_ids is None else list(sentence_ids)

        for si, sent_id in enumerate(ids_all):
            grp = f[sent_id]
            pos = grp["pos_tags"].asstr()[:]
            pos = np.asarray(pos)

            per_layer: Dict[str, dict] = {p: {"sent_id": sent_id} for p in pos_list}

            for layer in layers:
                X = grp["word_embeddings"][layer, :, :]
                try:
                    res = abide_conditional(
                        X, pos, pos_list=pos_list, n_iter=n_iter,
                        alpha=alpha, min_group_size=min_group_size,
                    )
                except Exception as e:  # pathological sentence: keep the run going
                    warnings.warn(f"{sent_id} layer {layer}: {e}")
                    res = {}
                for p in pos_list:
                    per_layer[p][f"id_layer_{layer}"] = res.get(f"id_{p}", np.nan)
                    per_layer[p][f"err_layer_{layer}"] = res.get(f"err_{p}", np.nan)
                    per_layer[p]["n"] = res.get(f"n_{p}", 0)

            for p in pos_list:
                rows_per_pos[p].append(per_layer[p])

            if verbose:
                print(f"[{si+1}/{len(ids_all)}] {sent_id} fatto")

    paths = {}
    for p in pos_list:
        df = pd.DataFrame(rows_per_pos[p])
        cols = (["sent_id", "n"]
                + [f"id_layer_{l}" for l in layers]
                + [f"err_layer_{l}" for l in layers])
        df = df[[c for c in cols if c in df.columns]]
        path = os.path.join(out_dir, f"conditional_id_{p}.csv")
        df.to_csv(path, index=False)
        paths[p] = path
        if verbose:
            print(f"salvato {path}  ({len(df)} frasi)")
    return paths