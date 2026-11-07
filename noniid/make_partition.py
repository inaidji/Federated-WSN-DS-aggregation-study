"""Build a HIGHLY non-IID 3-client partition of WSN-DS.

Two skews are combined so the client distributions differ as much as possible:

1. Attack-type skew  - every attack type lives on exactly one client, so the
   feature distribution of the positive ("Attack") class is disjoint per client.
       client 1 : Blackhole only
       client 2 : Grayhole + Flooding
       client 3 : TDMA (Scheduling) only
2. Label / quantity skew - the 340k Normal rows are split 80 / 18 / 2, so the
   positive rate and the client size are wildly different.
       client 1 : ~272k Normal + 10k Blackhole      -> pos rate ~3.5 %, huge
       client 2 : ~61k  Normal + 18k Grayhole/Flood -> pos rate ~23 %,  medium
       client 3 : ~7k   Normal + 6.6k TDMA          -> pos rate ~49 %,  tiny

Output: noniid/data/WSN-DS_part{1,2,3}.csv  (+ input_shape.npy), same schema as
the IID split, so common/fl_model.py loads them unchanged when WSN_DATA_DIR
points here.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "data")
SEED = 42

ATTACK_TO_CLIENT = {
    "Blackhole": 1,
    "Grayhole": 2,
    "Flooding": 2,
    "TDMA": 3,
}
NORMAL_FRACTIONS = {1: 0.80, 2: 0.18, 3: 0.02}


def main():
    rng = np.random.default_rng(SEED)
    df = pd.concat([pd.read_csv(os.path.join(SRC, f"WSN-DS_part{i}.csv")) for i in (1, 2, 3)],
                   ignore_index=True)
    df.columns = df.columns.str.strip()
    df["_atk"] = df["Attack type"].astype(str).str.strip()

    normal = df[df["_atk"] == "Normal"].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n = len(normal)
    n1 = int(NORMAL_FRACTIONS[1] * n)
    n2 = int(NORMAL_FRACTIONS[2] * n)
    normal_split = {1: normal.iloc[:n1],
                    2: normal.iloc[n1:n1 + n2],
                    3: normal.iloc[n1 + n2:]}

    os.makedirs(OUT, exist_ok=True)
    for c in (1, 2, 3):
        parts = [normal_split[c]]
        for atk, dst in ATTACK_TO_CLIENT.items():
            if dst == c:
                parts.append(df[df["_atk"] == atk])
        client = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=SEED + c)
        client = client.drop(columns=["_atk"])
        path = os.path.join(OUT, f"WSN-DS_part{c}.csv")
        client.to_csv(path, index=False)
        y = (client["Attack type"].astype(str).str.strip() != "Normal")
        comp = client["Attack type"].astype(str).str.strip().value_counts().to_dict()
        print(f"client {c}: {len(client):>7} rows | pos rate {y.mean():.3f} | {comp}")

    np.save(os.path.join(OUT, "input_shape.npy"), np.array(9))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
