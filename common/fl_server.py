"""Shared Flower server utilities for the WSN-DS experiments.

``MetricsMixin`` reproduces the behaviour of the original ``CustomFedAvg``
(round-by-round logging, weighted aggregation of client evaluation metrics and
global metric plots).  It is mixed into whichever base strategy is requested so
that FedProx / FedAdam / FedYogi / FedTrimmedAvg all share identical bookkeeping.
"""
import json
import logging
import os

import flwr as fl
import numpy as np
from flwr.server.strategy import (FedAdam, FedAvg, FedAvgM, FedMedian, FedProx, FedTrimmedAvg,
                                  FedYogi, Krum)

from common.fl_model import get_initial_parameters, load_input_shape, seed_everything

logger = logging.getLogger(__name__)

NUM_ROUNDS = 8
SERVER_ADDRESS = f"localhost:{os.environ.get('WSN_PORT', '8081')}"

# "Light" mode (WSN_LIGHT=1): only render the global metrics figure every 5th
# round and on the final round (the JSON history is still written every round).
_LIGHT = os.environ.get("WSN_LIGHT", "") not in ("", "0", "false")

_ADAPTIVE_BN_AVG = os.environ.get("WSN_ADAPTIVE_BN_AVG", "0") not in ("", "0", "false")

_METRIC_KEYS = ['accuracy', 'loss', 'precision', 'recall', 'f1_score', 'auc_roc', 'log_loss']


class MetricsMixin:
    """Adds config injection, weighted evaluate aggregation and plotting."""

    strategy_name = "FL"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_history = {k: [] for k in _METRIC_KEYS}
        self.krum_log = []  # which client Krum selects each round (only used for Krum)

    def configure_fit(self, server_round, parameters, client_manager):
        logger.info("[Round %d] Sending weights to clients for training...", server_round)
        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        extra = {"server_round": server_round}
        return [
            (client, fl.common.FitIns(ins.parameters, {**ins.config, **extra}))
            for client, ins in fit_ins
        ]

    def aggregate_fit(self, server_round, results, failures):
        if isinstance(self, Krum) and results:
            from flwr.common import parameters_to_ndarrays
            from flwr.server.strategy.aggregate import _compute_distances
            W = [parameters_to_ndarrays(r.parameters) for _, r in results]
            D = _compute_distances(W)
            nc = max(1, len(W) - self.num_malicious_clients - 2)
            scores = [float(np.sum(D[i, np.argsort(D[i])[1:nc + 1]])) for i in range(len(W))]
            cids = [int(r.metrics.get("client_id", -1)) for _, r in results]
            self.krum_log.append({"round": server_round, "selected": cids[int(np.argmin(scores))],
                                  "scores": {str(c): s for c, s in zip(cids, scores)}})
            with open(f"krum_selection_{self.strategy_name}.json", "w") as fh:
                json.dump(self.krum_log, fh, indent=1)
        agg = super().aggregate_fit(server_round, results, failures)
        if _ADAPTIVE_BN_AVG and isinstance(self, (FedAdam, FedYogi, FedAvgM)) and agg[0] is not None and results:
            agg = self._average_bn_stats(agg, results)
        return agg

    def _average_bn_stats(self, agg, results):
        """Server-optimizer variants only (FedAdam, FedYogi, FedAvgM): keep the server-optimizer update for trainable weights but replace the
        non-trainable batch-norm running statistics by the plain example-weighted client average
        (treating running variances as parameters of Adam/Yogi is not meaningful)."""
        from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
        from common.fl_model import build_model
        mask = [not w.trainable for w in build_model().weights]
        new = parameters_to_ndarrays(agg[0])
        n = float(sum(r.num_examples for _, r in results))
        cl = [(parameters_to_ndarrays(r.parameters), r.num_examples / n) for _, r in results]
        for i, is_stat in enumerate(mask):
            if is_stat:
                new[i] = sum(w[i] * f for w, f in cl)
        params = ndarrays_to_parameters(new)
        if isinstance(self, FedAvgM):
            self.initial_parameters = params      # FedAvgM keeps its running global model here
        else:
            self.current_weights = [np.array(x) for x in new]
        return params, agg[1]

    def configure_evaluate(self, server_round, parameters, client_manager):
        logger.info("[Round %d] Sending weights to clients for evaluation...", server_round)
        eval_ins = super().configure_evaluate(server_round, parameters, client_manager)
        extra = {"server_round": server_round}
        return [
            (client, fl.common.EvaluateIns(ins.parameters, {**ins.config, **extra}))
            for client, ins in eval_ins
        ]

    def aggregate_evaluate(self, server_round, results, failures):
        logger.info("[Round %d] Aggregating client evaluation metrics...", server_round)
        if not results:
            logger.warning("[Round %d] No evaluation results received.", server_round)
            return None, {}
        if failures:
            logger.warning("[Round %d] Failures during evaluation: %s", server_round, failures)

        totals = {k: 0.0 for k in _METRIC_KEYS}
        n = 0
        for _, res in results:
            for k in _METRIC_KEYS:
                totals[k] += float(res.metrics.get(k, 0.0)) * res.num_examples
            n += res.num_examples

        agg = {k: (totals[k] / n if n else 0.0) for k in _METRIC_KEYS}
        for k in _METRIC_KEYS:
            self.metrics_history[k].append(agg[k])

        logger.info("[Round %d] Global metrics:", server_round)
        for k, v in agg.items():
            logger.info("  %s: %.4f", k, v)

        self._plot(server_round)
        self._dump_history()
        return agg['loss'], agg

    def _plot(self, server_round):
        final = int(os.environ.get("FL_NUM_ROUNDS", NUM_ROUNDS))
        if _LIGHT and server_round % 5 != 0 and server_round != final:
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.metrics_history['accuracy']:
            return
        rounds = list(range(1, len(self.metrics_history['accuracy']) + 1))
        plt.figure(figsize=(15, 8))
        for i, k in enumerate(_METRIC_KEYS, 1):
            plt.subplot(2, 4, i)
            plt.plot(rounds, self.metrics_history[k], marker='o')
            plt.title(f"Global {k} - {self.strategy_name}")
            plt.xlabel("Round"); plt.ylabel(k); plt.grid(True)
        plt.tight_layout()
        fname = f"metrics_plot_global_{self.strategy_name}_round_{server_round}.png"
        plt.savefig(fname)
        plt.close()
        logger.info("Saved %s", fname)

    def _dump_history(self):
        with open(f"metrics_history_{self.strategy_name}.json", "w") as fh:
            json.dump(self.metrics_history, fh, indent=2)


_BASE = {
    "fedprox": FedProx,
    "fedadam": FedAdam,
    "fedyogi": FedYogi,
    "fedtrimmedavg": FedTrimmedAvg,
    "fedmedian": FedMedian,
    "fedavgm": FedAvgM,
    "krum": Krum,
    "fedavg": FedAvg,
}


def build_strategy(name: str, **overrides):
    name = name.lower()
    base = _BASE[name]
    tag = os.environ.get("WSN_STRATEGY_TAG", "")
    sname = base.__name__ + (f"_{tag}" if tag else "")
    cls = type(f"Custom{base.__name__}", (MetricsMixin, base), {"strategy_name": sname})

    mc = int(os.environ.get("WSN_MIN_CLIENTS", "3"))
    kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=mc,
        min_evaluate_clients=mc,
        min_available_clients=mc,
    )

    if name == "fedprox":
        kwargs["proximal_mu"] = overrides.pop("proximal_mu", 0.1)
        kwargs["initial_parameters"] = get_initial_parameters()
    elif name in ("fedadam", "fedyogi"):
        # Adaptive server optimisers require initial parameters.
        kwargs["initial_parameters"] = get_initial_parameters()
        kwargs["eta"] = overrides.pop("eta", 1e-2)
        kwargs["eta_l"] = overrides.pop("eta_l", 1e-2 if name == "fedadam" else 0.0316)
        kwargs["tau"] = overrides.pop("tau", 1e-3)
    elif name == "fedtrimmedavg":
        kwargs["beta"] = overrides.pop("beta", 0.2)
        kwargs["initial_parameters"] = get_initial_parameters()
    elif name == "fedavgm":
        kwargs["server_learning_rate"] = float(overrides.pop("server_learning_rate", 1.0))
        kwargs["server_momentum"] = float(overrides.pop("server_momentum", 0.5))
        kwargs["initial_parameters"] = get_initial_parameters()
    elif name == "krum":
        # plain Krum selects the single most central update (num_clients_to_keep=0);
        # >0 turns it into multi-Krum averaging that many.
        kwargs["num_malicious_clients"] = int(overrides.pop("num_malicious_clients", 1))
        kwargs["num_clients_to_keep"] = int(overrides.pop("num_clients_to_keep", 0))
        kwargs["initial_parameters"] = get_initial_parameters()
    else:
        kwargs["initial_parameters"] = get_initial_parameters()

    kwargs.update(overrides)
    return cls(**kwargs)


def run_server(name: str, num_rounds: int = NUM_ROUNDS, **overrides):
    num_rounds = int(os.environ.get("FL_NUM_ROUNDS", num_rounds))
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    logging.getLogger("flwr").setLevel(logging.WARNING)

    seed_everything()
    load_input_shape()  # parity check with the original server
    strategy = build_strategy(name, **overrides)
    logger.info("Starting Flower server (%s) on %s for %d rounds...",
                strategy.strategy_name, SERVER_ADDRESS, num_rounds)
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        grpc_max_message_length=536870912,
    )
