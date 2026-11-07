"""Build a HIGHLY non-IID 5-client partition of WSN-DS for the robustness study.

Same philosophy as make_partition.py (3 clients) but with K = 5 so that a
trimmed-mean / Krum aggregator can actually discard a full client:
    beta = 0.2, K = 5  ->  floor(0.2 * 5) = 1 trimmed each end
    Krum with f = 1    ->  needs K > 2f + 2 = 4, satisfied

Attack-type skew: each of the four attack families lives on exactly one client;
the fifth client is benign-only (an extreme heterogeneity case).
    client 1 : Blackhole
    client 2 : Grayhole
    client 3 : Flooding
    client 4 : TDMA (Scheduling)
    client 5 : benign-heavy (Normal + a small mixed sample of every attack, so
               both classes are present for metric computation)
Label / quantity skew: the Normal rows are split 40 / 20 / 12 / 8 / 20 (%).

Output: noniid/data_k5/WSN-DS_part{1..5}.csv (+ input_shape.npy), same schema
as the other splits.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "data_k5")
SEED = 42

ATTACK_TO_CLIENT = {"Blackhole": 1, "Grayhole": 2, "Flooding": 3, "TDMA": 4}
NORMAL_FRACTIONS = {1: 0.40, 2: 0.20, 3: 0.12, 4: 0.08, 5: 0.20}
# fraction of each attack type held out for the benign-heavy client 5
C5_ATTACK_FRAC = 0.04


def main():
    df = pd.concat([pd.read_csv(os.path.join(SRC, f"WSN-DS_part{i}.csv")) for i in (1, 2, 3)],
                   ignore_index=True)
    df.columns = df.columns.str.strip()
    df["_atk"] = df["Attack type"].astype(str).str.strip()

    normal = df[df["_atk"] == "Normal"].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n = len(normal)
    bounds, acc = {}, 0.0
    cuts = []
    for c in (1, 2, 3, 4):
        acc += NORMAL_FRACTIONS[c]
        cuts.append(int(acc * n))
    idx = [0] + cuts + [n]
    normal_split = {c: normal.iloc[idx[c - 1]:idx[c]] for c in (1, 2, 3, 4, 5)}

    # hold out a small mixed attack sample for client 5
    c5_attacks = []
    remaining = {}
    for atk in ATTACK_TO_CLIENT:
        pool = df[df["_atk"] == atk].sample(frac=1.0, random_state=SEED + 99)
        k = int(C5_ATTACK_FRAC * len(pool))
        c5_attacks.append(pool.iloc[:k])
        remaining[atk] = pool.iloc[k:]

    os.makedirs(OUT, exist_ok=True)
    for c in (1, 2, 3, 4, 5):
        parts = [normal_split[c]]
        if c == 5:
            parts.extend(c5_attacks)
        else:
            for atk, dst in ATTACK_TO_CLIENT.items():
                if dst == c:
                    parts.append(remaining[atk])
        client = (pd.concat(parts, ignore_index=True)
                  .sample(frac=1.0, random_state=SEED + c)
                  .drop(columns=["_atk"]))
        client.to_csv(os.path.join(OUT, f"WSN-DS_part{c}.csv"), index=False)
        y = (client["Attack type"].astype(str).str.strip() != "Normal")
        comp = client["Attack type"].astype(str).str.strip().value_counts().to_dict()
        print(f"client {c}: {len(client):>7} rows | pos rate {y.mean():.3f} | {comp}")

    np.save(os.path.join(OUT, "input_shape.npy"), np.array(9))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
