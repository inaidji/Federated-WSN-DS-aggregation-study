"""Aggregate the heterogeneity sweep and the fairer FedAdam/FedYogi tuning. Writes analysis/out/sweep_v5.json.
 (1) K=3 main partition: batch-norm-averaged FedAdam/FedYogi (selected on seed-0 validation) and FedAvgM-BN, 5 seeds,
     paired against FedAvg from main_v4 (Holm over the comparisons made here);
 (2) K=10 Dirichlet partitions, alpha x aggregator, 3 seeds (a fresh partition per seed)."""
import glob, json, os, re, sys
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from runio import run_metrics
SW = os.path.join(HERE, "..", "noniid", "sweep")
KEYS = ["test_atk_f1", "test_mcc", "test_pr_auc", "test_tpr_at_fpr1", "test_log_loss", "platt_log_loss", "test_weighted_f1", "test_auc"]

def summ(per):
    out = {"n": len(per), "seeds": sorted(per)}
    for k in KEYS:
        v = np.array([per[s][k] for s in sorted(per)])
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "values": v.tolist()}
    return out

main = json.load(open(os.path.join(HERE, "out", "main_v4.json")))["summary"]
sel = json.load(open(os.path.join(SW, "selected_adaptive.json")))
k3, per3 = {}, {}
for opt, h in sel.items():
    per = {0: run_metrics(os.path.join(SW, "tune_bn", f"s0_{opt}_eta{h['eta']}_tau{h['tau']}"))}
    for s in (1, 2, 3, 4):
        per[s] = run_metrics(os.path.join(SW, "tuned", f"s{s}_{opt}"))
    per3[f"{opt}_bn"] = per; k3[f"{opt}_bn"] = {"eta": h["eta"], "tau": h["tau"], **summ(per)}
per = {s: run_metrics(os.path.join(SW, "bn", f"s{s}_fedavgm_bn")) for s in range(5)}
per3["fedavgm_bn"] = per; k3["fedavgm_bn"] = summ(per)
# paired vs FedAvg (v4), Holm over the 3 comparisons per metric
fa = dict(zip(main["fedavg"]["seeds"], range(5)))
cmp = {}
for key in ("test_atk_f1", "test_log_loss"):
    names = list(per3)
    rows = {}
    for n in names:
        a = np.array([per3[n][s][key] for s in range(5)]); b = np.array(main["fedavg"][key]["values"])
        d = a - b; sd = d.std(ddof=1)
        rows[n] = {"mean_diff": float(d.mean()), "ci95": float(stats.t.ppf(.975, 4) * sd / np.sqrt(5)),
                   "p": float(stats.ttest_rel(a, b).pvalue)}
    order = sorted(rows, key=lambda n: rows[n]["p"]); run = 0.0
    for i, n in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * rows[n]["p"])); rows[n]["p_holm"] = run
    for n in names: cmp[f"{n}_vs_fedavg|{key}"] = rows[n]
# K=10 Dirichlet
TAGS = ["fedavg", "fedprox_mu1.0", "fedmedian", "fedtrimmedavg", "fedadam_tuned", "fedyogi_tuned"]
dr = {}
for d in glob.glob(os.path.join(SW, "dir", "a*_s*_*")):
    m = re.match(r"a([\d.]+)_s(\d+)_(.+)$", os.path.basename(d))
    met = run_metrics(d)
    if m and met:
        dr.setdefault((m.group(1), m.group(3)), {})[int(m.group(2))] = met
K10 = {}
for (al, tag), per in dr.items():
    K10.setdefault(al, {})[tag] = summ(per)
json.dump({"k3": k3, "k3_paired_vs_fedavg": cmp, "k10": K10, "selected": sel}, open(os.path.join(HERE, "out", "sweep_v5.json"), "w"), indent=1)

f = lambda d, k: f"{d[k]['mean']:.3f}±{d[k]['std']:.3f}"
print("K=3 (5 seeds)         atkF1          MCC            PR-AUC         logloss        Platt-ll       wF1")
for n in ("fedavg", "fedmedian", "fedtrimmedavg", "fedprox_mu1.0"):
    d = main[n]; print(f"{n:20}  " + " ".join(f"{f(d, k):14}" for k in ("test_atk_f1", "test_mcc", "test_pr_auc", "test_log_loss", "platt_log_loss", "test_weighted_f1")))
for n, d in k3.items():
    print(f"{n:20}  " + " ".join(f"{f(d, k):14}" for k in ("test_atk_f1", "test_mcc", "test_pr_auc", "test_log_loss", "platt_log_loss", "test_weighted_f1")), d.get("eta", ""), d.get("tau", ""))
for k, v in cmp.items():
    if "atk_f1" in k: print(k, {a: round(b, 3) for a, b in v.items()})
print("\nK=10 Dirichlet (attack-class F1 mean±std, n seeds)")
for al in sorted(K10, key=float):
    print(f"alpha={al}: " + "  ".join(f"{t}={f(K10[al][t], 'test_atk_f1')}(n{K10[al][t]['n']})" for t in TAGS if t in K10[al]))
