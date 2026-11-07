"""Pick FedAdam / FedYogi (eta, tau) by validation macro-F1 on the seed-0 K=3 grid, separately for
the Flower default (server optimizer applied to every array, incl. batch-norm statistics: sweep/tune
plus the two v4 settings) and the BN-averaged variant (sweep/tune_bn). The BN-averaged selection is
written to selected_adaptive.json (used by gen_jobs.py B); both grids go to analysis/out/adaptive_grid.json."""
import glob, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "analysis"))
from runio import run_metrics

KEYS = ("val_macro_f1", "val_atk_f1", "test_atk_f1", "test_pr_auc", "test_log_loss", "test_mcc", "test_weighted_f1")
def grid(patterns):
    rows = {}
    for pat, tau_default in patterns:
        for d in glob.glob(pat):
            m = re.match(r"s0_(fedadam|fedyogi)_eta([^_]+)(?:_tau(\S+))?$", os.path.basename(d))
            if not m: continue
            met = run_metrics(d)
            if met: rows[(m.group(1), m.group(2), m.group(3) or tau_default)] = met
    return rows
variants = {"flower_default": grid([(os.path.join(HERE, "tune", "s0_*"), None), (os.path.join(HERE, "..", "v4", "main", "s0_fed*_eta*"), "1e-2")]),
            "bn_averaged": grid([(os.path.join(HERE, "tune_bn", "s0_*"), None)])}
out, sel = [], {}
for var, rows in variants.items():
    for opt in ("fedadam", "fedyogi"):
        cand = {k: v for k, v in rows.items() if k[0] == opt}
        if not cand: continue
        best = max(cand, key=lambda k: cand[k]["val_macro_f1"])
        print(f"\n[{var}] {opt}: eta tau | val macroF1 val atkF1 | test atkF1 PR-AUC logloss")
        for k in sorted(cand, key=lambda k: (float(k[1]), float(k[2]))):
            v = cand[k]
            print(f"  {k[1]:>5} {k[2]:>5} | {v['val_macro_f1']:.3f} {v['val_atk_f1']:.3f} | {v['test_atk_f1']:.3f} {v['test_pr_auc']:.3f} {v['test_log_loss']:.3f}{'  <== selected' if k == best else ''}")
            out.append({"variant": var, "opt": opt, "eta": k[1], "tau": k[2], "selected": k == best, **{c: v[c] for c in KEYS}})
        if var == "bn_averaged": sel[opt] = {"eta": best[1], "tau": best[2]}
json.dump(sel, open(os.path.join(HERE, "selected_adaptive.json"), "w"), indent=1)
json.dump(out, open(os.path.join(HERE, "..", "..", "analysis", "out", "adaptive_grid.json"), "w"), indent=1)
print("\nselected (BN-averaged):", sel)
