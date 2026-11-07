"""Dirichlet partition of WSN-DS over K clients (label skew over the five traffic types + quantity skew).

For each traffic type c in {Normal, Blackhole, Grayhole, Flooding, TDMA} the rows are split across the K
clients with proportions p_c ~ Dirichlet(alpha * 1_K) (Hsu et al. 2019). Small alpha = extreme skew,
large alpha = nearly IID. The draw is repeated (seed, seed+1000, ...) until every client has at least
MIN_ROWS rows and, for each of the two classes, either none or at least MIN_CLASS rows, so every client
can form a stratified split.

usage: python make_partition_dirichlet.py ALPHA K SEED OUTDIR
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "data")
MIN_ROWS, MIN_CLASS = 300, 20


def partition(df, alpha, K, seed):
    types = df["_atk"].values
    for attempt in range(500):
        rng = np.random.default_rng(seed + 1000 * attempt)
        owner = np.empty(len(df), dtype=int)
        for t in np.unique(types):
            idx = np.where(types == t)[0]
            rng.shuffle(idx)
            p = rng.dirichlet(alpha * np.ones(K))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                owner[chunk] = k
        ok = True
        for k in range(K):
            m = owner == k
            n_norm = int((types[m] == "Normal").sum()); n_att = int(m.sum()) - n_norm
            if m.sum() < MIN_ROWS or (0 < n_norm < MIN_CLASS) or (0 < n_att < MIN_CLASS):
                ok = False; break
        if ok:
            return owner, attempt
    raise RuntimeError("no valid partition found")


def main(alpha, K, seed, out):
    df = pd.concat([pd.read_csv(os.path.join(SRC, f"WSN-DS_part{i}.csv")) for i in (1, 2, 3)], ignore_index=True)
    df.columns = df.columns.str.strip()
    df["_atk"] = df["Attack type"].astype(str).str.strip()
    owner, attempt = partition(df, alpha, K, seed)
    os.makedirs(out, exist_ok=True)
    for k in range(K):
        sub = df[owner == k].sample(frac=1.0, random_state=seed + k).drop(columns=["_atk"])
        sub.to_csv(os.path.join(out, f"WSN-DS_part{k + 1}.csv"), index=False)
        y = sub["Attack type"].astype(str).str.strip() != "Normal"
        print(f"client {k + 1:>2}: {len(sub):>7} rows | attack rate {y.mean():.3f}")
    np.save(os.path.join(out, "input_shape.npy"), np.array(9))
    print(f"alpha={alpha} K={K} seed={seed} attempts={attempt + 1} -> {out}")


if __name__ == "__main__":
    main(float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
