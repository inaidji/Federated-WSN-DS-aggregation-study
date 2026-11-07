"""Write job lists for the heterogeneity sweep: python gen_jobs.py A|C|B  (order A, C, select_adaptive.py, B)
Each line: RUNDIR|N_CLIENTS|ENV assignments (space separated). Phase B reads the validation-selected
FedAdam/FedYogi settings from selected_adaptive.json (written by select_adaptive.py)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
NONIID = os.path.dirname(HERE)
K3 = os.path.join(NONIID, "data")
phase = sys.argv[1]
jobs = []
def job(rundir, nc, data, strat, seed, extra=""):
    tag = os.path.basename(rundir)
    jobs.append(f"{rundir}|{nc}|WSN_DATA_DIR={data} WSN_MIN_CLIENTS={nc} WSN_STRATEGY={strat} WSN_STRATEGY_TAG={tag} WSN_SEED={seed} {extra}".strip())
if phase == "C":                                        # BN-averaged FedAdam/FedYogi grid (seed 0) + FedAvgM-BN (5 seeds)
    BN = "WSN_ADAPTIVE_BN_AVG=1"
    for opt in ("fedadam", "fedyogi"):
        for eta in ("3e-3", "1e-2", "3e-2", "1e-1"):
            for tau in ("1e-3", "1e-2", "1e-1"):
                job(f"{HERE}/tune_bn/s0_{opt}_eta{eta}_tau{tau}", 3, K3, opt, 0, f"WSN_ETA={eta} WSN_TAU={tau} {BN}")
        job(f"{HERE}/tune_bn/s0_{opt}_eta3e-1_tau1e-2", 3, K3, opt, 0, f"WSN_ETA=3e-1 WSN_TAU=1e-2 {BN}")
    for s in range(5):
        job(f"{HERE}/bn/s{s}_fedavgm_bn", 3, K3, "fedavgm", s, BN)
elif phase == "A":
    for opt in ("fedadam", "fedyogi"):                      # stage 1: coarse grid, K=3, seed 0
        for eta in ("1e-2", "3e-2", "1e-1", "3e-1"):
            for tau in ("1e-3", "1e-2", "1e-1"):
                job(f"{HERE}/tune/s0_{opt}_eta{eta}_tau{tau}", 3, K3, opt, 0, f"WSN_ETA={eta} WSN_TAU={tau}")
    for alpha in ("0.1", "0.5", "5.0"):                       # K=10 Dirichlet, fixed-hyperparameter rules
        for seed in (0, 1, 2):
            data = f"{HERE}/data/a{alpha}_s{seed}"
            for tag, strat, extra in (("fedavg", "fedavg", ""), ("fedprox_mu1.0", "fedprox", "WSN_MU=1.0"),
                                      ("fedmedian", "fedmedian", ""), ("fedtrimmedavg", "fedtrimmedavg", "WSN_BETA=0.2")):
                job(f"{HERE}/dir/a{alpha}_s{seed}_{tag}", 10, data, strat, seed, extra)
else:
    sel = json.load(open(os.path.join(HERE, "selected_adaptive.json")))   # {"fedadam": {"eta":..,"tau":..}, ...}
    for opt, h in sel.items():
        for seed in (1, 2, 3, 4):                              # stage 2: selected config on remaining K=3 seeds
            job(f"{HERE}/tuned/s{seed}_{opt}", 3, K3, opt, seed, f"WSN_ETA={h['eta']} WSN_TAU={h['tau']} WSN_ADAPTIVE_BN_AVG=1")
    for alpha in ("0.1", "0.5", "5.0"):
        for seed in (0, 1, 2):
            data = f"{HERE}/data/a{alpha}_s{seed}"
            for opt, h in sel.items():
                job(f"{HERE}/dir/a{alpha}_s{seed}_{opt}_tuned", 10, data, opt, seed, f"WSN_ETA={h['eta']} WSN_TAU={h['tau']} WSN_ADAPTIVE_BN_AVG=1")
open(os.path.join(HERE, f"jobs_{phase}.txt"), "w").write("\n".join(jobs) + "\n")
print(len(jobs), "jobs")
