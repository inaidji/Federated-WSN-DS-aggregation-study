"""Generic server for the K=5 robustness study (poisoning attacks).

Env vars:
    WSN_STRATEGY   fedavg | fedprox | fedtrimmedavg | krum | fedmedian
    WSN_MU         FedProx proximal_mu            (default 1.0)
    WSN_BETA       FedTrimmedAvg trim fraction    (default 0.2)
    WSN_KRUM_F     Krum assumed #malicious        (default 1)
    WSN_MIN_CLIENTS  min fit/eval clients         (default 5 here)
    FL_NUM_ROUNDS, WSN_STRATEGY_TAG, WSN_LIGHT, WSN_LOCAL_EPOCHS
Data dir pinned to noniid/data_k5.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_NONIID = os.path.dirname(_HERE)
_REPO = os.path.dirname(_NONIID)
os.environ["WSN_DATA_DIR"] = os.path.join(_NONIID, "data_k5")
os.environ.setdefault("WSN_MIN_CLIENTS", "5")
sys.path.insert(0, _REPO)

from common.fl_server import run_server  # noqa: E402

if __name__ == "__main__":
    strat = os.environ["WSN_STRATEGY"].lower()
    overrides = {}
    if strat == "fedprox":
        overrides["proximal_mu"] = float(os.environ.get("WSN_MU", "1.0"))
    elif strat == "fedtrimmedavg":
        overrides["beta"] = float(os.environ.get("WSN_BETA", "0.2"))
    elif strat == "krum":
        overrides["num_malicious_clients"] = int(os.environ.get("WSN_KRUM_F", "1"))
    run_server(strat, **overrides)
