"""Figures for the revised paper, from analysis/out/*.json and the per-round histories."""
import glob, json, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "..", "figures")
os.makedirs(IMG, exist_ok=True)
M = json.load(open(os.path.join(HERE, "out", "main_v4.json")))["summary"]
P = json.load(open(os.path.join(HERE, "out", "poison_v4.json")))["summary"]
B = json.load(open(os.path.join(HERE, "..", "centralized", "baselines_v4.json")))["summary"]

LAB = {"fedavg": "FedAvg", "fedmedian": "FedMedian", "fedtrimmedavg": "FedTrimmedAvg", "fedavgm": "FedAvgM",
       "fedprox_mu0.1": "FedProx\n$\\mu{=}0.1$", "fedprox_mu0.5": "FedProx\n$\\mu{=}0.5$", "fedprox_mu1.0": "FedProx\n$\\mu{=}1.0$",
       "fedadam_eta1e-3": "FedAdam\n$\\eta{=}10^{-3}$", "fedadam_eta3e-3": "FedAdam\n$\\eta{=}3{\\times}10^{-3}$",
       "fedyogi_eta1e-3": "FedYogi\n$\\eta{=}10^{-3}$", "fedyogi_eta3e-3": "FedYogi\n$\\eta{=}3{\\times}10^{-3}$"}
ORDER = ["fedavg", "fedmedian", "fedtrimmedavg", "fedprox_mu0.1", "fedprox_mu0.5", "fedprox_mu1.0", "fedavgm",
         "fedadam_eta1e-3", "fedadam_eta3e-3", "fedyogi_eta1e-3", "fedyogi_eta3e-3"]
COL = {"fedavg": "#555555", "fedmedian": "#2ca02c", "fedtrimmedavg": "#8c564b", "fedavgm": "#ff7f0e"}
for t in ORDER:
    COL.setdefault(t, "#1f77b4" if "prox" in t else "#d62728" if "adam" in t else "#9467bd")

# 1. attack-class F1, per seed
fig, ax = plt.subplots(figsize=(11, 4.2))
for i, t in enumerate(ORDER):
    v = np.array(M[t]["test_atk_f1"]["values"])
    ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color=COL[t], s=26, zorder=3)
    ax.errorbar(i, v.mean(), yerr=v.std(ddof=1), color="k", capsize=4, lw=1.4, zorder=4)
    ax.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color="k", lw=2, zorder=4)
for name, lab, ls in [("cnn", "centralized CNN", "--"), ("mlp", "centralized MLP", "-."), ("hgb", "centralized boosting", ":")]:
    ax.axhline(B[name]["atk_f1"][0], color="gray", ls=ls, lw=1.2)
    ax.text(len(ORDER) - 0.45, B[name]["atk_f1"][0] + 0.012, lab, ha="right", fontsize=8, color="gray")
ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([LAB[t] for t in ORDER], fontsize=8)
ax.set_ylabel("Attack-class F1 (pooled test set)"); ax.set_ylim(-0.03, 1.0); ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(IMG, "fig_atkf1.png"), dpi=130); plt.close(fig)

# 2. reliability diagrams (raw), mean over seeds of last-round bins
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.plot([0, 1], [0, 1], "k--", lw=1)
for t in ["fedavg", "fedprox_mu1.0", "fedtrimmedavg", "fedmedian", "fedadam_eta3e-3"]:
    r = np.array(M[t]["_reliability"], dtype=float)
    ok = ~np.isnan(r[:, 0]) & (r[:, 2] > 0)
    ax.plot(r[ok, 0], r[ok, 1], "o-", color=COL[t], label=LAB[t].replace("\n", " "), ms=4, lw=1.2)
ax.set_xlabel("Mean predicted attack probability"); ax.set_ylabel("Observed attack frequency")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=7.5); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(IMG, "fig_reliability.png"), dpi=130); plt.close(fig)

# 3. convergence: per-round (client-weighted) AUC and log loss, mean +- std over seeds
ROOT = os.path.join(HERE, "..", "noniid", "v4", "main")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for t in ["fedavg", "fedmedian", "fedtrimmedavg", "fedprox_mu1.0", "fedadam_eta3e-3", "fedyogi_eta3e-3"]:
    H = [json.load(open(glob.glob(f"{ROOT}/s{s}_{t}/metrics_history_*.json")[0])) for s in range(5)]
    for ax, key in zip(axes, ["auc_roc", "log_loss"]):
        a = np.array([h[key] for h in H]); x = np.arange(1, a.shape[1] + 1)
        ax.plot(x, a.mean(0), color=COL[t], label=LAB[t].replace("\n", " "), lw=1.6)
        ax.fill_between(x, a.mean(0) - a.std(0), a.mean(0) + a.std(0), color=COL[t], alpha=0.12)
axes[0].set_ylabel("AUC-ROC (client-weighted)"); axes[1].set_ylabel("Log loss (client-weighted)")
for ax in axes: ax.set_xlabel("Communication round"); ax.grid(True, alpha=0.3)
axes[0].legend(fontsize=7.5, loc="lower right"); axes[1].set_ylim(0, 1.2)
fig.tight_layout(); fig.savefig(os.path.join(IMG, "fig_convergence.png"), dpi=130); plt.close(fig)

# 4. poisoning: attack-class F1 per aggregator and condition
CONDS = ["clean", "lf1", "lf2", "sf1", "sf2"]
CL = ["Clean", "Label-flip\n1/5", "Label-flip\n2/5", "Boosted sign-flip\n1/5", "Boosted sign-flip\n2/5"]
ST = [("fedavg", "FedAvg", "#555555"), ("fedprox", "FedProx ($\\mu{=}1$)", "#1f77b4"),
      ("fedtrimmedavg", "FedTrimmedAvg", "#8c564b"), ("krum", "Krum", "#d62728"), ("fedmedian", "FedMedian", "#2ca02c")]
fig, ax = plt.subplots(figsize=(11, 4))
w = 0.16
for i, (s, lab, c) in enumerate(ST):
    mu = [P[s][k]["test_atk_f1"]["mean"] for k in CONDS]; sd = [P[s][k]["test_atk_f1"]["std"] for k in CONDS]
    ax.bar(np.arange(5) + (i - 2) * w, mu, w, yerr=sd, capsize=2, color=c, label=lab, edgecolor="white", lw=0.5)
ax.set_xticks(range(5)); ax.set_xticklabels(CL, fontsize=9); ax.set_ylabel("Attack-class F1 (pooled test set)")
ax.set_ylim(0, 1.0); ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13))
fig.tight_layout(); fig.savefig(os.path.join(IMG, "fig_poison.png"), dpi=130); plt.close(fig)
print("figures written")

# 5. Dirichlet heterogeneity sweep (K=10): attack-class F1 vs alpha
SW = json.load(open(os.path.join(HERE, "out", "sweep_v5.json")))["k10"]
TAGS = [("fedavg", "FedAvg", "#555555", "o"), ("fedprox_mu1.0", "FedProx ($\\mu{=}1$)", "#1f77b4", "s"),
        ("fedmedian", "FedMedian", "#2ca02c", "^"), ("fedtrimmedavg", "FedTrimmedAvg", "#8c564b", "v"),
        ("fedadam_tuned", "FedAdam (tuned)", "#d62728", "D"), ("fedyogi_tuned", "FedYogi (tuned)", "#9467bd", "P")]
alphas = sorted(SW, key=float)
fig, ax = plt.subplots(figsize=(6.2, 4.0))
for j, (t, lab, c, mk) in enumerate(TAGS):
    xs = np.arange(len(alphas)) + (j - 2.5) * 0.045
    m = [SW[a][t]["test_atk_f1"]["mean"] for a in alphas]; s = [SW[a][t]["test_atk_f1"]["std"] for a in alphas]
    ax.errorbar(xs, m, yerr=s, color=c, marker=mk, ms=5, capsize=2.5, lw=1.2, label=lab)
ax.set_xticks(range(len(alphas))); ax.set_xticklabels([f"$\\alpha={a}$" for a in alphas])
ax.set_ylabel("Attack-class F1 (pooled test set)"); ax.set_ylim(0, 1.0); ax.grid(True, axis="y", alpha=0.3)
ax.legend(fontsize=8, ncol=2, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(IMG, "fig_dirichlet.png"), dpi=130); plt.close(fig)
