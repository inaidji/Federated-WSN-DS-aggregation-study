"""Server for the heterogeneity sweep; the data dir comes from WSN_DATA_DIR (set by the runner)."""
import os, sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
