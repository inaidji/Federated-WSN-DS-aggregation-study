"""Load the per-client probability dumps of one federated run and compute the metric suite."""
import glob, os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import metrics_suite as ms


def pooled_rounds(rundir):
    """{round: dict(y_test,p_test,y_val,p_val)} pooled over clients, for the dumped rounds."""
    files = glob.glob(os.path.join(rundir, "preds", "client_*_r*.npz"))
    rounds = sorted({int(re.search(r"_r(\d+)\.npz", f).group(1)) for f in files})
    out = {}
    for r in rounds:
        parts = [np.load(f) for f in sorted(glob.glob(os.path.join(rundir, "preds", f"client_*_r{r}.npz")))]
        out[r] = {k: np.concatenate([d[k] for d in parts]) for k in ("y_test", "p_test", "y_val", "p_val")}
    return out


def run_metrics(rundir):
    """Mean over the dumped (last-5) rounds of: raw test metrics, validation metrics,
    and post-hoc-calibrated (temperature / Platt, fitted on validation) test metrics."""
    R = pooled_rounds(rundir)
    if len(R) < 5:
        return None
    acc = {}
    for r, d in R.items():
        row = {f"test_{k}": v for k, v in ms.all_metrics(d["y_test"], d["p_test"]).items()}
        val = ms.all_metrics(d["y_val"], d["p_val"])
        row.update({"val_macro_f1": val["macro_f1"], "val_atk_f1": val["atk_f1"], "val_log_loss": val["log_loss"]})
        ft, T = ms.fit_temperature(d["y_val"], d["p_val"])
        fp, ab = ms.fit_platt(d["y_val"], d["p_val"])
        pt, pp = ft(d["p_test"]), fp(d["p_test"])
        row.update(temp_T=T, temp_log_loss=ms.log_loss_np(d["y_test"], pt), temp_ece=ms.ece_binary(d["y_test"], pt),
                   temp_brier=float(np.mean((pt - d["y_test"]) ** 2)),
                   platt_log_loss=ms.log_loss_np(d["y_test"], pp), platt_ece=ms.ece_binary(d["y_test"], pp),
                   platt_brier=float(np.mean((pp - d["y_test"]) ** 2)),
                   platt_atk_f1=ms.all_metrics(d["y_test"], pp)["atk_f1"])
        for k, v in row.items():
            acc.setdefault(k, []).append(v)
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    last = R[max(R)]
    out["_reliability"] = ms.reliability_bins(last["y_test"], last["p_test"], 10)
    return out
