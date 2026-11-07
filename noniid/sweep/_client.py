"""Client for the heterogeneity sweep; partition index from WSN_PART, data dir from WSN_DATA_DIR."""
import os, sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

from common.fl_client import run_client  # noqa: E402

if __name__ == "__main__":
    run_client(int(os.environ["WSN_PART"]))
