# The-LLMs-Changing-Geometry-of-Grammar

# Conditional geometry of text embeddings

Two independent modules:

- `conditional_ii.py` — neighbourhood overlap between layers (information
  imbalance), conditioned on PoS.
- `conditional_id.py` — intrinsic dimension per sentence and per PoS (ABIDE
  with a conditional binomial likelihood).


## Data format

Both expect, for one sentence or one observation:

```
word_embeddings : (n_layers, N, D)   float
pos_tags        : (N,)               strings, e.g. "NOUN", "VERB"
```

`conditional_ii.py` reads it from a pickle (`.pkl` or `.pkl.gz`) or from a
dict already in memory; `conditional_id.py` reads it from an HDF5 file with
one group per sentence.

## conditional_ii.py

Full profile over all layers, one row per (PoS, layer):

```python
from conditional_ii import load_observation, profile

H, pos = load_observation("observation.pkl.gz")
df = profile(H, pos, metric="cosine", k=1)
df.head()
```

Columns: `d_i_to_first`, `d_first_to_i`, `d_i_to_last`, `d_last_to_i` —
the imbalance of each layer against the first and the last one, in both
directions. Values near 0 mean the neighbourhoods are preserved.

Same thing plus the diagnostics, starting from a dict:

```python
from conditional_ii import analyze_dict

df = analyze_dict(data, pos_tags=["NOUN", "VERB", "DET"],
                  metric="cosine")
```

## conditional_id.py


Whole file, all layers, one CSV per PoS:

```python
from conditional_id import run_conditional_id_h5

run_conditional_id_h5("sentences.h5", "results/", layers=range(1, 23))
```


Note: the geometry (k*, tau) is estimated on the whole sentence, so the IDs
of different PoS in the same sentence and layer are measured at the same
scale and can be compared directly.
