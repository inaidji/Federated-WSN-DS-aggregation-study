"""Centralized baselines on the pooled training data (same per-client splits as the FL runs):
the CNN used in the federation, a plain MLP, and a histogram gradient-boosting model.
Evaluated on the pooled test set with the full metric suite, 5 seeds."""
import json, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["WSN_DATA_DIR"] = os.path.join(_HERE, "..", "noniid", "data")
sys.path.insert(0, os.path.dirname(_HERE))
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight
from common.fl_model import build_model, load_client_data_val, seed_everything
from common import metrics_suite as ms
import tensorflow as tf

EPOCHS, BATCH, SEEDS = 40, 32, [0, 1, 2, 3, 4]


def build_mlp():
    m = tf.keras.Sequential([tf.keras.layers.Input(shape=(9,)),
                             tf.keras.layers.Dense(32, activation="relu"),
                             tf.keras.layers.Dropout(0.4),
                             tf.keras.layers.Dense(16, activation="relu"),
                             tf.keras.layers.Dense(1, activation="sigmoid")])
    m.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(1e-3))
    return m


def run(seed):
    seed_everything(seed)
    tr, va, te = [], [], []
    for part in (1, 2, 3):
        Xtr, Xva, Xte, ytr, yva, yte, _ = load_client_data_val(part, 0.1)
        tr.append((Xtr, ytr)); va.append((Xva, yva)); te.append((Xte, yte))
    cat = lambda L, i: np.concatenate([a[i] for a in L])
    Xtr, ytr, Xva, yva, Xte, yte = cat(tr, 0), cat(tr, 1), cat(va, 0), cat(va, 1), cat(te, 0), cat(te, 1)
    cw = dict(enumerate(compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)))
    flat = lambda X: X.reshape(len(X), -1)
    out = {}
    cnn = build_model(); cnn.fit(Xtr, ytr, class_weight=cw, epochs=EPOCHS, batch_size=BATCH, verbose=0)
    out["cnn"] = (cnn.predict(Xva, verbose=0).ravel(), cnn.predict(Xte, verbose=0).ravel())
    mlp = build_mlp(); mlp.fit(flat(Xtr), ytr, class_weight=cw, epochs=EPOCHS, batch_size=BATCH, verbose=0)
    out["mlp"] = (mlp.predict(flat(Xva), verbose=0).ravel(), mlp.predict(flat(Xte), verbose=0).ravel())
    gb = HistGradientBoostingClassifier(class_weight="balanced", early_stopping=True, random_state=seed).fit(flat(Xtr), ytr)
    out["hgb"] = (gb.predict_proba(flat(Xva))[:, 1], gb.predict_proba(flat(Xte))[:, 1])
    res = {}
    for name, (pva, pte) in out.items():
        r = ms.all_metrics(yte, pte)
        f_t, T = ms.fit_temperature(yva, pva); f_p, ab = ms.fit_platt(yva, pva)
        r["log_loss_temp"] = ms.log_loss_np(yte, f_t(pte)); r["ece_temp"] = ms.ece_binary(yte, f_t(pte))
        r["log_loss_platt"] = ms.log_loss_np(yte, f_p(pte)); r["ece_platt"] = ms.ece_binary(yte, f_p(pte))
        res[name] = r
    return res


if __name__ == "__main__":
    allr = [run(s) for s in SEEDS]
    summary = {}
    for name in ("cnn", "mlp", "hgb"):
        summary[name] = {k: [float(np.mean([r[name][k] for r in allr])), float(np.std([r[name][k] for r in allr]))]
                         for k in allr[0][name]}
    json.dump({"per_seed": allr, "summary": summary}, open(os.path.join(_HERE, "baselines_v4.json"), "w"), indent=1)
    for name, s in summary.items():
        print(name, {k: f"{v[0]:.3f}+-{v[1]:.3f}" for k, v in s.items() if k in
              ("accuracy", "atk_f1", "macro_f1", "mcc", "auc", "pr_auc", "log_loss", "ece", "log_loss_temp", "log_loss_platt")})
