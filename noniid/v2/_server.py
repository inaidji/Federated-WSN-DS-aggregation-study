"""Generic server for the non-IID v2 sweep.

Everything is driven by environment variables so one script covers every
(strategy, hyper-parameter) combination:

    WSN_STRATEGY   fedavg | fedprox | fedadam | fedyogi | fedtrimmedavg
    WSN_MU         FedProx proximal_mu           (default 0.1)
    WSN_ETA        FedAdam/FedYogi server lr     (default 1e-2)
    WSN_TAU        FedAdam/FedYogi tau           (default 1e-3)
    WSN_BETA       FedTrimmedAvg trim fraction   (default 0.2)
    FL_NUM_ROUNDS  number of rounds              (default 8)
    WSN_STRATEGY_TAG  suffix for output filenames

Data dir is pinned to noniid/data.  Run with the run directory as CWD so the
metrics_history / plots land there.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_NONIID = os.path.dirname(_HERE)
_REPO = os.path.dirname(_NONIID)
os.environ["WSN_DATA_DIR"] = os.path.join(_NONIID, "data")
sys.path.insert(0, _REPO)

from common.fl_server import run_server  # noqa: E402

if __name__ == "__main__":
    strat = os.environ["WSN_STRATEGY"].lower()
    overrides = {}
    if strat == "fedprox":
        overrides["proximal_mu"] = float(os.environ.get("WSN_MU", "0.1"))
    elif strat in ("fedadam", "fedyogi"):
        overrides["eta"] = float(os.environ.get("WSN_ETA", "1e-2"))
        overrides["tau"] = float(os.environ.get("WSN_TAU", "1e-3"))
    elif strat == "fedtrimmedavg":
        overrides["beta"] = float(os.environ.get("WSN_BETA", "0.2"))
    run_server(strat, **overrides)
