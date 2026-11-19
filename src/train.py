"""Training utilities: split loading, LR / LightGBM training, evaluation."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score, roc_curve)

from src.config_loader import Cfg


def load_splits(cfg: Cfg) -> dict[str, pd.DataFrame]:
    """Returns {"train", "early_stop", "cal", "test"}: transformed features + target column, original row index."""
    p = cfg.paths
    return {
        "train": pd.read_parquet(p.train_parquet),
        "early_stop": pd.read_parquet(p.early_stop_parquet),
        "cal": pd.read_parquet(p.calibration_parquet),
        "test": pd.read_parquet(p.test_parquet),
    }


def train_lr(X, y, cfg: Cfg) -> LogisticRegression:
    model = LogisticRegression(random_state=cfg.modeling.random_seed,
                               **cfg.modeling.logistic_regression)
    return model.fit(X, y)


def train_lgbm(X_tr, y_tr, X_es, y_es, cfg: Cfg) -> lgb.LGBMClassifier:
    """Early stopping on the early_stop set only (AUC). Curve in model.evals_result_['early_stop']."""
    model = lgb.LGBMClassifier(random_state=cfg.modeling.random_seed,
                               **cfg.modeling.lightgbm)
    model.fit(
        X_tr, y_tr,
        eval_X=(X_es,),
        eval_y=(y_es,),
        eval_names=["early_stop"],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    return model


def evaluate_model(model, X, y, label: str) -> dict:
    p = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, p)
    return {
        "label": label,
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "ks": float(np.max(tpr - fpr)),
    }