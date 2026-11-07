"""Aggregate the v4 poisoning study (K=5): per (aggregator, condition) mean/std over seeds of the
metric suite, plus how often Krum selects a malicious client. Writes analysis/out/poison_v4.json."""
import glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runio import run_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "noniid", "v4", "poison")
OUT = os.path.join(HERE, "out", "poison_v4.json")
ATTACKERS = {"clean": [], "lf1": [3], "lf2": [3, 4], "sf1": [3], "sf2": [3, 4]}
KEYS = ["test_atk_f1", "test_mcc", "test_pr_auc", "test_log_loss", "test_accuracy", "test_macro_f1",
        "platt_log_loss", "test_auc"]
STRATS = ["fedavg", "fedprox", "fedtrimmedavg", "krum", "fedmedian"]
CONDS = ["clean", "lf1", "lf2", "sf1", "sf2"]

runs, krum = {}, {}
for d in sorted(glob.glob(os.path.join(ROOT, "s*_*"))):
    m = re.match(r"s(\d+)_(clean|lf1|lf2|sf1|sf2)_(\w+)$", os.path.basename(d))
    if not m:
        continue
    seed, cond, strat = int(m.group(1)), m.group(2), m.group(3)
    met = run_metrics(d)
    if met is None:
        continue
    runs.setdefault((strat, cond), {})[seed] = met
    if strat == "krum":
        f = glob.glob(os.path.join(d, "krum_selection_*.json"))
        if f:
            log = json.load(open(f[0]))
            bad = ATTACKERS[cond]
            krum.setdefault(cond, {})[seed] = {
                "frac_rounds_malicious_selected": float(np.mean([r["selected"] in bad for r in log])) if bad else 0.0,
                "selected_counts": {str(c): int(sum(r["selected"] == c for r in log)) for c in range(1, 6)}}
S = {}
for (strat, cond), per in runs.items():
    S.setdefault(strat, {})[cond] = {"n": len(per), **{
        k: {"mean": float(np.mean([per[s][k] for s in per])),
            "std": float(np.std([per[s][k] for s in per], ddof=1)) if len(per) > 1 else 0.0,
            "values": [per[s][k] for s in sorted(per)]} for k in KEYS}}
json.dump({"summary": S, "krum_selection": krum}, open(OUT, "w"), indent=1)
for key in ("test_atk_f1", "test_log_loss"):
    print(f"\n{key} (mean±std over seeds)")
    print(f"{'':14}" + "".join(f"{c:>16}" for c in CONDS))
    for s in STRATS:
        row = f"{s:14}"
        for c in CONDS:
            v = S.get(s, {}).get(c)
            row += f"{(str(round(v[key]['mean'],3))+'±'+str(round(v[key]['std'],3))+' n'+str(v['n'])) if v else '--':>16}"
        print(row)
print("\nKrum: fraction of rounds a malicious client was selected")
for c, per in krum.items():
    print(c, {s: round(v["frac_rounds_malicious_selected"], 2) for s, v in per.items()})
