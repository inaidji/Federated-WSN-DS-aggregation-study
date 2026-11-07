"""Shared model definition and data loading for the WSN-DS federated experiments.

Identical CNN architecture to the original manual FedAvg implementation so that
FedProx / FedAdam / FedYogi / FedTrimmedAvg results are directly comparable.
"""
import logging
import os
import sys

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

# Repo layout: <repo>/common/fl_model.py  ->  data lives in <repo>/data by default.
# Set WSN_DATA_DIR to point the clients at a different partition (e.g. the
# highly non-IID split in <repo>/noniid/data).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("WSN_DATA_DIR", os.path.join(REPO_ROOT, "data"))

# Global seed (WSN_SEED). Controls model init, minibatch order, the train/test
# split and the FedProx shuffle so that repeated runs differ only by this value.
SEED = int(os.environ.get("WSN_SEED", "42"))


def seed_everything(seed: int = SEED):
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:  # very old TF
        tf.random.set_seed(seed)
    logger.info("Seeded everything with %d", seed)

EXPECTED_COLUMNS = [
    'id', 'Time', 'Is_CH', 'who CH', 'Dist_To_CH', 'ADV_S', 'ADV_R', 'JOIN_S',
    'JOIN_R', 'SCH_S', 'SCH_R', 'Rank', 'DATA_S', 'DATA_R', 'Data_Sent_To_BS',
    'dist_CH_To_BS', 'send_code', 'Expaned Energy', 'Attack type'
]

FEATURES = [
    'Is_CH', 'Dist_To_CH', 'Expaned Energy', 'DATA_S', 'DATA_R',
    'ADV_S', 'ADV_R', 'JOIN_S', 'JOIN_R'
]


def load_input_shape():
    """Return the number of input features (kept for parity with the FedAvg server)."""
    path = os.path.join(DATA_DIR, "input_shape.npy")
    try:
        data = np.load(path, allow_pickle=True)
        n_features = int(data.item())
        if n_features <= 0:
            raise ValueError("Input shape must be a positive integer")
        logger.info("Loaded input shape: %d features", n_features)
        return n_features
    except FileNotFoundError:
        logger.error("input_shape.npy not found at %s", path)
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        logger.error("Error loading input_shape.npy: %s", e)
        sys.exit(1)


def build_model(n_features: int = 9) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(3, 3, 1)),
        tf.keras.layers.Conv2D(8, (2, 2), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.02)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (2, 2), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.02)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.02)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy']
    )
    return model


def get_initial_parameters():
    """Serialized initial weights, used by the adaptive server strategies."""
    import flwr as fl
    weights = build_model().get_weights()
    logger.info("Shapes of initial weights: %s", [w.shape for w in weights])
    return fl.common.ndarrays_to_parameters([np.array(w) for w in weights])


def load_client_data(part: int):
    """Load and preprocess one WSN-DS partition. Mirrors the original client code."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_class_weight

    csv_path = os.path.join(DATA_DIR, f"WSN-DS_part{part}.csv")
    df = pd.read_csv(csv_path)
    logger.info("Initial rows in dataset: %d", len(df))
    df.columns = df.columns.str.strip()
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Expected columns {EXPECTED_COLUMNS}, but got {df.columns.tolist()}")

    df['attack_encoded'] = df['Attack type'].apply(lambda x: 0 if str(x).strip() == 'Normal' else 1).to_numpy()
    df = df.dropna().reset_index(drop=True)
    logger.info("Rows after dropping NaN: %d", len(df))
    df = df.drop_duplicates().reset_index(drop=True)

    X = df[FEATURES].values
    y = df['attack_encoded'].values

    X = StandardScaler().fit_transform(X)
    X = X.reshape(-1, 3, 3, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    logger.info("X_train: %s  X_test: %s", X_train.shape, X_test.shape)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    logger.info("Class weights: %s", class_weight_dict)

    return X_train, X_test, y_train, y_test, class_weight_dict


def load_client_data_val(part: int, val_frac: float = 0.1):
    """Like load_client_data but carves a validation split out of the training data.

    The test set is identical to load_client_data's (same first split); the
    validation split is stratified and seeded with SEED+1, and the class weights
    are recomputed on the reduced training set.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    X_train, X_test, y_train, y_test, _ = load_client_data(part)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_frac, random_state=SEED + 1, stratify=y_train
    )
    cw = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)))
    return X_tr, X_val, X_test, y_tr, y_val, y_test, cw
