"""Part 1 - TIME OVERRUN prediction. Target: execution_days > 365
(MPLADS Guidelines Para 3.13's own norm). 21.0% of 13,119 works exceed it.
Routing: work_type segments with >= MIN_SEGMENT_SIZE works use an ANN;
thinner segments fall back to a pooled SVM.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import tensorflow as tf
from tensorflow import keras

from Time_Cost_Overrun.leakage_safe.build_features import build_features, FEATURES_NUM, FEATURES_CAT

MIN_SEGMENT_SIZE = 150
LABEL_COL = "time_overrun_label"


def _make_xy(df, label_col, cat_encoder=None, scaler=None, fit=False):
    X_num = df[FEATURES_NUM].to_numpy(dtype=float)
    if fit:
        scaler = StandardScaler().fit(X_num)
    X_num = scaler.transform(X_num)
    if fit:
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[FEATURES_CAT])
    X_cat = cat_encoder.transform(df[FEATURES_CAT])
    X = np.hstack([X_num, X_cat])
    return X, df[label_col].to_numpy(), cat_encoder, scaler


def build_ann(input_dim):
    inp = keras.Input(shape=(input_dim,))
    h = keras.layers.Dense(16, activation="relu")(inp)
    h = keras.layers.Dense(8, activation="relu")(h)
    out = keras.layers.Dense(1, activation="sigmoid")(h)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    return model


def train_and_evaluate(seed: int = 42, label_col: str = LABEL_COL):
    df = build_features()
    counts = df["work_type"].value_counts()
    large = counts[counts >= MIN_SEGMENT_SIZE].index
    df["segment_route"] = np.where(df["work_type"].isin(large), "ANN", "SVM")

    train_df, test_df = train_test_split(df, test_size=0.25, random_state=seed, stratify=df[label_col])
    results = {}

    ann_train = train_df[train_df["segment_route"] == "ANN"]
    ann_test = test_df[test_df["segment_route"] == "ANN"]
    X_tr, y_tr, enc, scaler = _make_xy(ann_train, label_col, fit=True)
    X_te, y_te, _, _ = _make_xy(ann_test, label_col, enc, scaler)

    tf.keras.utils.set_random_seed(seed)
    ann = build_ann(X_tr.shape[1])
    ann.fit(X_tr, y_tr, epochs=30, batch_size=64, verbose=0, validation_split=0.1,
            class_weight={0: 1.0, 1: (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)})
    ann_proba = ann.predict(X_te, verbose=0).ravel()
    results["ANN segments"] = dict(n_train=len(ann_train), n_test=len(ann_test),
                                   AUC=roc_auc_score(y_te, ann_proba),
                                   accuracy=accuracy_score(y_te, ann_proba > 0.5))

    svm_train = train_df[train_df["segment_route"] == "SVM"]
    svm_test = test_df[test_df["segment_route"] == "SVM"]
    Xs_tr, ys_tr, enc_s, scaler_s = _make_xy(svm_train, label_col, fit=True)
    Xs_te, ys_te, _, _ = _make_xy(svm_test, label_col, enc_s, scaler_s)

    svm = SVC(kernel="rbf", probability=True, random_state=seed, class_weight="balanced")
    svm.fit(Xs_tr, ys_tr)
    svm_proba = svm.predict_proba(Xs_te)[:, 1]
    results["SVM segments"] = dict(n_train=len(svm_train), n_test=len(svm_test),
                                   AUC=roc_auc_score(ys_te, svm_proba),
                                   accuracy=accuracy_score(ys_te, svm_proba > 0.5))
    return results, df


if __name__ == "__main__":
    print("*** Part 1: TIME OVERRUN ***\n")
    results, df = train_and_evaluate()
    print(f"ANN: {(df['segment_route']=='ANN').sum()} works | SVM: {(df['segment_route']=='SVM').sum()} works\n")
    for name, r in results.items():
        print(f"{name}: train={r['n_train']} test={r['n_test']} AUC={r['AUC']:.3f} accuracy={r['accuracy']:.3f}")