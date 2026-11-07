"""Shared Flower client for the WSN-DS experiments.

The same client is used for every strategy.  When the server sends a
``proximal_mu`` entry in the fit config (FedProx), local training switches to a
manual loop that adds the FedProx proximal term  ``(mu/2) * ||w - w_global||^2``
to the loss.  Otherwise it behaves exactly like the original FedAvg client
(``model.fit`` with class weights).
"""
import logging
import os

import numpy as np
import tensorflow as tf
import flwr as fl
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score,
                             roc_auc_score)

logger = logging.getLogger(__name__)

LE_CLASSES = ['Normal', 'Attack']
LOCAL_EPOCHS = int(os.environ.get("WSN_LOCAL_EPOCHS", "5"))
BATCH_SIZE = 32

# "Light" mode (WSN_LIGHT=1): skip the per-round model checkpoint + PNG plots and
# only write them on the final round.  Used by the long multi-round sweeps so 40
# rounds don't produce 40 checkpoints / 80 figures per client.
_LIGHT = os.environ.get("WSN_LIGHT", "") not in ("", "0", "false")
_FINAL_ROUND = int(os.environ.get("FL_NUM_ROUNDS", "0"))

# ---- Poisoning / Byzantine clients (robustness study) -----------------------
# WSN_POISON        none | labelflip | signflip
# WSN_POISON_PARTS  comma-separated 1-based partition indices that are malicious
# WSN_POISON_LAMBDA sign-flip strength (weights sent = g - lambda*(local - g))
_POISON = os.environ.get("WSN_POISON", "none").strip().lower()
_POISON_PARTS = {int(x) for x in os.environ.get("WSN_POISON_PARTS", "").replace(" ", "").split(",") if x}
_POISON_LAMBDA = float(os.environ.get("WSN_POISON_LAMBDA", "1.0"))
# WSN_ATTACKER_IGNORES_MU  malicious clients train WITHOUT the FedProx proximal term
#                          (a real attacker does not obey the protocol)
# WSN_ATTACK_TRAINABLE_ONLY  sign-flip only the trainable weights; batch-norm running
#                          statistics are sent as honest local values (negating a running
#                          variance can make it negative and NaN the model)
_ATTACKER_IGNORES_MU = os.environ.get("WSN_ATTACKER_IGNORES_MU", "0") not in ("", "0", "false")
_ATTACK_TRAINABLE_ONLY = os.environ.get("WSN_ATTACK_TRAINABLE_ONLY", "0") not in ("", "0", "false")
# WSN_DUMP_LAST=n: in the last n rounds, save each client's val/test probabilities of the
# global model to preds/<client>_r<round>.npz for offline analysis (ECE, PR-AUC, MCC, ...)
_DUMP_LAST = int(os.environ.get("WSN_DUMP_LAST", "0"))


def _proximal_train(model, X_train, y_train, class_weight_dict, mu):
    """Local training with the FedProx proximal term (mu/2)||w - w_global||^2.

    The global weights are snapshotted from the model *after* the server's
    parameters have been loaded, so they line up with ``trainable_variables``.
    """
    global_vars = [tf.constant(v.numpy()) for v in model.trainable_variables]
    optimizer = tf.keras.optimizers.Adam(0.001)
    bce = tf.keras.losses.BinaryCrossentropy()
    mu_t = tf.constant(mu, dtype=tf.float32)

    w0 = tf.constant(float(class_weight_dict.get(0, 1.0)), dtype=tf.float32)
    w1 = tf.constant(float(class_weight_dict.get(1, 1.0)), dtype=tf.float32)

    ds = (tf.data.Dataset.from_tensor_slices((X_train.astype("float32"),
                                              y_train.astype("float32")))
          .shuffle(8192, seed=int(os.environ.get("WSN_SEED", "42")))
          .batch(BATCH_SIZE)
          .prefetch(tf.data.AUTOTUNE))

    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            preds = tf.squeeze(model(xb, training=True), axis=-1)
            sample_w = yb * w1 + (1.0 - yb) * w0
            loss = bce(yb, preds, sample_weight=sample_w)
            if model.losses:
                loss += tf.add_n(model.losses)
            prox = tf.add_n([
                tf.reduce_sum(tf.square(v - g))
                for v, g in zip(model.trainable_variables, global_vars)
            ])
            loss += (mu_t / 2.0) * prox
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def run_epoch(dataset):
        """Whole-epoch loop compiled into one graph -> no per-step Python
        dispatch (the batch-32 microbatch dispatch was ~13x slower than
        model.fit); numerically identical to calling train_step per batch."""
        last = tf.constant(0.0)
        for xb, yb in dataset:
            last = train_step(xb, yb)
        return last

    for epoch in range(LOCAL_EPOCHS):
        last = run_epoch(ds)
        logger.info("  [FedProx] epoch %d/%d done (mu=%.3f, loss=%.4f)",
                    epoch + 1, LOCAL_EPOCHS, mu, float(last))


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test, class_weight_dict, client_id,
                 poison="none", X_val=None, y_val=None):
        self.model = model
        self.X_val = X_val
        self.y_val = y_val
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.class_weight_dict = class_weight_dict
        self.client_id = client_id
        self.poison = poison  # "none" | "labelflip" | "signflip"
        self.fit_round = 0
        self.fit_metrics_history = []
        os.makedirs(f"models_{client_id}", exist_ok=True)
        if poison != "none":
            logger.warning("Client %s is MALICIOUS (attack=%s)", client_id, poison)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.fit_round += 1
        mu = float(config.get("proximal_mu", 0.0))
        if self.poison != "none" and _ATTACKER_IGNORES_MU:
            mu = 0.0  # the attacker does not honour the server-requested proximal term
        logger.info("Client %s round %d (config: %s, mu used: %s)", self.client_id,
                    self.fit_round, config, mu)
        if parameters:
            self.model.set_weights(parameters)

        # data-poisoning: train on inverted labels; honest labels kept for eval
        y_fit = (1 - self.y_train) if self.poison == "labelflip" else self.y_train

        if mu > 0.0:
            _proximal_train(self.model, self.X_train, y_fit,
                            self.class_weight_dict, mu)
        else:
            self.model.fit(
                self.X_train, y_fit,
                class_weight=self.class_weight_dict,
                epochs=LOCAL_EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0,
                validation_data=((self.X_val, self.y_val) if self.X_val is not None
                                 else (self.X_test, self.y_test)),
            )

        m = self._compute_metrics()
        m['round'] = self.fit_round
        self.fit_metrics_history.append(m)
        logger.info("Local metrics for %s (round %d):", self.client_id, self.fit_round)
        logger.info("\n%s", classification_report(
            self.y_test, m.pop('_y_pred'), labels=[0, 1], target_names=LE_CLASSES, zero_division=0))
        for k in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'log_loss']:
            logger.info("  %s: %.4f", k, m[k])

        if not _LIGHT or self.fit_round >= _FINAL_ROUND:
            self.model.save(f"models_{self.client_id}/model_round_{self.fit_round}.h5")
            self._plot()

        weights = self.model.get_weights()
        # model-poisoning: replace the honest update with one pointing away from it
        if self.poison == "signflip" and parameters:
            trainable = [bool(v.trainable) for v in self.model.weights]
            flipped = []
            for g, l, tr in zip(parameters, weights, trainable):
                g, l = np.asarray(g), np.asarray(l)
                if _ATTACK_TRAINABLE_ONLY and not tr:
                    flipped.append(l)  # keep honest batch-norm statistics
                else:
                    flipped.append(g - _POISON_LAMBDA * (l - g))
            weights = flipped
        return weights, len(self.X_train), {"client_id": int(self.client_id.split("_")[1])}

    def evaluate(self, parameters, config):
        server_round = config.get('server_round', 0)
        logger.info("Evaluating global model for %s (round %d)", self.client_id, server_round)
        if parameters:
            self.model.set_weights(parameters)
        m = self._compute_metrics()
        m.pop('_y_pred')
        m.pop('round', None)
        if _DUMP_LAST and _FINAL_ROUND and server_round > _FINAL_ROUND - _DUMP_LAST:
            self._dump_preds(server_round)
        return m['loss'], len(self.X_test), m

    def _dump_preds(self, server_round):
        def prob(X):
            p = self.model.predict(X, verbose=0).flatten()
            return np.clip(np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0).astype("float32")
        os.makedirs("preds", exist_ok=True)
        arrs = dict(y_test=np.asarray(self.y_test).astype("int8"), p_test=prob(self.X_test))
        if self.X_val is not None:
            arrs.update(y_val=np.asarray(self.y_val).astype("int8"), p_val=prob(self.X_val))
        np.savez_compressed(f"preds/{self.client_id}_r{server_round}.npz", **arrs)

    # ------------------------------------------------------------------
    def _compute_metrics(self):
        proba = self.model.predict(self.X_test, verbose=0).flatten()
        # A diverged global model (e.g. under a strong model-poisoning attack) can
        # produce non-finite scores; degrade gracefully instead of crashing so the
        # round still completes and the failure is recorded.
        if not np.all(np.isfinite(proba)):
            logger.warning("Non-finite predictions for %s -> reporting degenerate metrics", self.client_id)
            proba = np.nan_to_num(proba, nan=0.5, posinf=1.0, neginf=0.0)
        proba = np.clip(proba, 0.0, 1.0)
        pred = (proba > 0.5).astype(int)
        multi = len(np.unique(self.y_test)) > 1
        ll = float(log_loss(self.y_test, proba, labels=[0, 1]))
        return {
            'accuracy': float(accuracy_score(self.y_test, pred)),
            'loss': ll,
            'log_loss': ll,
            'precision': float(precision_score(self.y_test, pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(self.y_test, pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(self.y_test, pred, average='weighted', zero_division=0)),
            'auc_roc': float(roc_auc_score(self.y_test, proba)) if multi else 0.0,
            '_y_pred': pred,
        }

    def _plot(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        h = self.fit_metrics_history
        rounds = [x['round'] for x in h]
        keys = [('accuracy', 'blue'), ('precision', 'orange'), ('recall', 'green'),
                ('f1_score', 'purple'), ('auc_roc', 'red'), ('log_loss', 'brown')]
        plt.figure(figsize=(15, 8))
        for i, (k, c) in enumerate(keys, 1):
            plt.subplot(2, 3, i)
            plt.plot(rounds, [x[k] for x in h], marker='o', color=c)
            plt.title(f"{k} (Fit) - {self.client_id}")
            plt.xlabel("Round"); plt.ylabel(k); plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"metrics_{self.client_id}_round_{rounds[-1]}.png")
        plt.close()

        proba = self.model.predict(self.X_test, verbose=0).flatten()
        cm = confusion_matrix(self.y_test, (proba > 0.5).astype(int))
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=LE_CLASSES, yticklabels=LE_CLASSES)
        plt.title(f"Confusion Matrix - {self.client_id} (Round {rounds[-1]})")
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_{self.client_id}_round_{rounds[-1]}.png")
        plt.close()


def run_client(part: int, server_address: str = None):
    server_address = server_address or f"localhost:{os.environ.get('WSN_PORT', '8081')}"
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    _logging.getLogger("flwr").setLevel(_logging.WARNING)

    from common.fl_model import build_model, load_client_data, load_client_data_val, seed_everything

    seed_everything()
    client_id = f"client_{part}"
    val_frac = float(os.environ.get("WSN_VAL_FRAC", "0"))
    if val_frac > 0:
        X_train, X_val, X_test, y_train, y_val, y_test, cw = load_client_data_val(part, val_frac)
    else:
        X_train, X_test, y_train, y_test, cw = load_client_data(part)
        X_val = y_val = None
    model = build_model()
    poison = _POISON if (_POISON != "none" and part in _POISON_PARTS) else "none"
    client = FlowerClient(model, X_train, y_train, X_test, y_test, cw, client_id, poison=poison,
                          X_val=X_val, y_val=y_val)
    fl.client.start_numpy_client(server_address=server_address, client=client)
