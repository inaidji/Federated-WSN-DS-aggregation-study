"""Aggregate the v4 main grid (K=3): per-config mean/std over seeds, validation-based selection,
paired comparisons, post-hoc calibration controls. Writes analysis/out/main_v4.json."""
import glob, json, os, re, sys
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runio import run_metrics

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "noniid", "v4", "main")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "main_v4.json")
KEYS = ["test_accuracy", "test_atk_precision", "test_atk_recall", "test_atk_f1", "test_macro_f1", "test_weighted_f1",
        "test_mcc", "test_auc", "test_pr_auc", "test_log_loss", "test_brier", "test_ece", "test_tpr_at_fpr1",
        "test_tpr_at_fpr5", "val_macro_f1", "val_log_loss", "temp_log_loss", "temp_ece", "platt_log_loss",
        "platt_ece", "platt_atk_f1", "temp_T"]


def collect():
    runs = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "s*_*"))):
        m = re.match(r"s(\d+)_(.+)$", os.path.basename(d))
        if not m:
            continue
        met = run_metrics(d)
        if met is None:
            continue
        runs.setdefault(m.group(2), {})[int(m.group(1))] = met
    return runs


def summarize(runs):
    S = {}
    for tag, per in runs.items():
        S[tag] = {"n": len(per), "seeds": sorted(per)}
        for k in KEYS:
            v = np.array([per[s][k] for s in sorted(per)])
            sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            ci = float(stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v))) if len(v) > 1 else 0.0
            S[tag][k] = {"mean": float(v.mean()), "std": sd, "ci95": ci, "values": v.tolist()}
        S[tag]["_reliability"] = np.nanmean([per[s]["_reliability"] for s in sorted(per)], axis=0).tolist()
    return S


def paired(runs, a, b, key):
    sa, sb = runs[a], runs[b]
    seeds = sorted(set(sa) & set(sb))
    d = np.array([sa[s][key] - sb[s][key] for s in seeds])
    if len(d) < 2:
        return None
    sd = d.std(ddof=1)
    ci = stats.t.ppf(0.975, len(d) - 1) * sd / np.sqrt(len(d))
    p = stats.ttest_rel([sa[s][key] for s in seeds], [sb[s][key] for s in seeds]).pvalue if sd > 0 else 1.0
    return {"mean_diff": float(d.mean()), "ci95": float(ci), "p": float(p), "n": len(d), "wins": int((d < 0).sum())}


if __name__ == "__main__":
    runs = collect()
    S = summarize(runs)
    # validation-based selection inside each swept family
    fams = {"fedprox": [t for t in S if t.startswith("fedprox")], "fedadam": [t for t in S if t.startswith("fedadam")],
            "fedyogi": [t for t in S if t.startswith("fedyogi")]}
    sel = {f: max(ts, key=lambda t: S[t]["val_macro_f1"]["mean"]) for f, ts in fams.items() if ts}
    cmp = {}
    if "fedavg" in runs:
        for other in [t for t in runs if t != "fedavg"]:
            for key in ("test_atk_f1", "test_log_loss", "test_mcc"):
                cmp[f"{other}_vs_fedavg|{key}"] = paired(runs, other, "fedavg", key)
    # Holm step-down correction within each metric family (10 comparisons vs FedAvg)
    for key in ("test_atk_f1", "test_log_loss", "test_mcc"):
        names = [n for n in cmp if n.endswith("|" + key) and cmp[n]]
        order = sorted(names, key=lambda n: cmp[n]["p"])
        run_max = 0.0
        for i, n in enumerate(order):
            run_max = max(run_max, min(1.0, (len(order) - i) * cmp[n]["p"]))
            cmp[n]["p_holm"] = run_max
    json.dump({"summary": S, "selected_on_validation": sel, "paired_vs_fedavg": cmp}, open(OUT, "w"), indent=1)
    print(f"{'config':18} n | atkF1          macroF1        MCC            PR-AUC         logloss        ECE            Platt-ll")
    for t in sorted(S):
        f = lambda k: f"{S[t][k]['mean']:.3f}±{S[t][k]['std']:.3f}"
        print(f"{t:18} {S[t]['n']} | {f('test_atk_f1')} {f('test_macro_f1')} {f('test_mcc')} {f('test_pr_auc')} {f('test_log_loss')} {f('test_ece')} {f('platt_log_loss')}")
    print("selected on validation macro-F1:", sel)
