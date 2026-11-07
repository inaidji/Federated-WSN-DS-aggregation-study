"""Client for the K=5 robustness study. Partition index from WSN_PART.
Poisoning is configured via WSN_POISON / WSN_POISON_PARTS (read inside common.fl_client)."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_NONIID = os.path.dirname(_HERE)
_REPO = os.path.dirname(_NONIID)
os.environ["WSN_DATA_DIR"] = os.path.join(_NONIID, "data_k5")
sys.path.insert(0, _REPO)

from common.fl_client import run_client  # noqa: E402

if __name__ == "__main__":
    run_client(int(os.environ["WSN_PART"]))
