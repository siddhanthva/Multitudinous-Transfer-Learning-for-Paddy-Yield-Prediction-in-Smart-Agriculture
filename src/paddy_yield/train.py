from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from .config import PipelineConfig
from .data import (
    FoldScalers,
    inverse_transform_yield,
    fit_fold_scalers,
    transform_aux_targets_scaled,
    transform_features,
    transform_target_yield_scaled,
)
from .features import as_feature_sequence
from .metrics import RegressionMetrics, regression_metrics
from .models import build_bilstm_model, build_mtl_model


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except Exception:
        # TensorFlow may not be installed yet; this will be surfaced at runtime when training.
        return


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_val: int
    metrics_scaled: RegressionMetrics
    metrics_original: RegressionMetrics

    def to_dict(self) -> dict:
        d = asdict(self)
        d["metrics_scaled"] = self.metrics_scaled.to_dict()
        d["metrics_original"] = self.metrics_original.to_dict()
        return d


@dataclass
class CVResult:
    model_name: str
    feature_cols: list[str]
    fold_results: list[FoldResult]

    def summary(self) -> dict:
        scaled = [fr.metrics_scaled.to_dict() for fr in self.fold_results]
        orig = [fr.metrics_original.to_dict() for fr in self.fold_results]

        def _mean_std(items: list[dict], keys: list[str]) -> dict:
            out: dict[str, dict[str, float]] = {}
            for k in keys:
                vals = np.array([it[k] for it in items], dtype=float)
                out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0)}
            return out

        keys = ["r2", "mse", "mae", "rmse", "accuracy"]
        return {
            "model": self.model_name,
            "n_folds": len(self.fold_results),
            "scaled": _mean_std(scaled, keys),
            "original": _mean_std(orig, keys),
        }

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "feature_cols": self.feature_cols,
            "folds": [fr.to_dict() for fr in self.fold_results],
            "summary": self.summary(),
        }


def _kfold(cfg: PipelineConfig) -> KFold:
    return KFold(n_splits=cfg.n_splits, shuffle=cfg.shuffle_kfold, random_state=cfg.random_state)


def run_cv_bilstm(
    df,
    feature_cols: list[str],
    cfg: PipelineConfig,
    out_dir: str | Path,
) -> CVResult:
    set_global_seed(cfg.random_state)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    from tensorflow import keras

    fold_results: list[FoldResult] = []
    preds_rows: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(_kfold(cfg).split(df), start=1):
        scalers: FoldScalers = fit_fold_scalers(df, feature_cols, cfg, train_idx=train_idx)

        x_train = transform_features(df, feature_cols, scalers, train_idx)
        x_val = transform_features(df, feature_cols, scalers, val_idx)
        y_train_s = transform_target_yield_scaled(df, cfg, scalers, train_idx)
        y_val_s = transform_target_yield_scaled(df, cfg, scalers, val_idx)

        x_train_seq = as_feature_sequence(x_train)
        x_val_seq = as_feature_sequence(x_val)

        model = build_bilstm_model(
            timesteps=x_train_seq.shape[1],
            lr=cfg.bilstm_lr,
            units=cfg.bilstm_units,
        )

        cb = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.bilstm_patience,
                restore_best_weights=True,
            )
        ]

        model.fit(
            x_train_seq,
            y_train_s,
            validation_data=(x_val_seq, y_val_s),
            epochs=cfg.bilstm_epochs,
            batch_size=cfg.bilstm_batch_size,
            verbose=0,
            callbacks=cb,
        )

        y_pred_s = model.predict(x_val_seq, verbose=0).reshape(-1, 1)
        y_true_s = y_val_s.reshape(-1, 1)

        # Metrics in scaled space ([0,1])
        m_scaled = regression_metrics(
            y_true=y_true_s.reshape(-1),
            y_pred=y_pred_s.reshape(-1),
            tol=cfg.scaled_accuracy_tol,
        )

        # Metrics in original units (kg/ha)
        y_pred = inverse_transform_yield(y_pred_s, scalers)
        y_true = inverse_transform_yield(y_true_s, scalers)
        m_orig = regression_metrics(y_true=y_true, y_pred=y_pred, tol=cfg.original_accuracy_tol)

        fold_results.append(
            FoldResult(
                fold=fold,
                n_train=int(len(train_idx)),
                n_val=int(len(val_idx)),
                metrics_scaled=m_scaled,
                metrics_original=m_orig,
            )
        )

        for i, row_id in enumerate(val_idx):
            preds_rows.append(
                {
                    "fold": fold,
                    "row_index": int(row_id),
                    "y_true_scaled": float(y_true_s[i, 0]),
                    "y_pred_scaled": float(y_pred_s[i, 0]),
                    "y_true": float(y_true[i]),
                    "y_pred": float(y_pred[i]),
                }
            )

    pd.DataFrame(preds_rows).to_csv(out_dir / "predictions.csv", index=False)
    cv = CVResult(model_name="bilstm", feature_cols=feature_cols, fold_results=fold_results)
    (out_dir / "metrics.json").write_text(json.dumps(cv.to_dict(), indent=2))
    return cv


def run_cv_mtl(
    df,
    feature_cols: list[str],
    cfg: PipelineConfig,
    out_dir: str | Path,
) -> CVResult:
    set_global_seed(cfg.random_state)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    from tensorflow import keras

    if len(cfg.aux_target_cols) != 2:
        raise ValueError("This pipeline currently supports exactly two auxiliary targets for MTL.")

    aux1, aux2 = cfg.aux_target_cols
    lw = cfg.loss_weights

    fold_results: list[FoldResult] = []
    preds_rows: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(_kfold(cfg).split(df), start=1):
        scalers: FoldScalers = fit_fold_scalers(df, feature_cols, cfg, train_idx=train_idx)

        x_train = transform_features(df, feature_cols, scalers, train_idx)
        x_val = transform_features(df, feature_cols, scalers, val_idx)

        y_train_s = transform_target_yield_scaled(df, cfg, scalers, train_idx)
        y_val_s = transform_target_yield_scaled(df, cfg, scalers, val_idx)

        aux_train = transform_aux_targets_scaled(df, cfg, scalers, train_idx)
        aux_val = transform_aux_targets_scaled(df, cfg, scalers, val_idx)

        model = build_mtl_model(
            n_features=x_train.shape[1],
            lr=cfg.mtl_lr,
            shared_units=cfg.mtl_shared_units,
            shared_layers=cfg.mtl_shared_layers,
            dropout=cfg.mtl_dropout,
            aux_heads=(aux1, aux2),
            loss_weights=lw,
        )

        cb = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.mtl_patience,
                restore_best_weights=True,
            )
        ]

        model.fit(
            x_train,
            {"yield": y_train_s, aux1: aux_train[aux1], aux2: aux_train[aux2]},
            validation_data=(x_val, {"yield": y_val_s, aux1: aux_val[aux1], aux2: aux_val[aux2]}),
            epochs=cfg.mtl_epochs,
            batch_size=cfg.mtl_batch_size,
            verbose=0,
            callbacks=cb,
        )

        y_pred_s, _, _ = model.predict(x_val, verbose=0)
        y_pred_s = np.asarray(y_pred_s).reshape(-1, 1)
        y_true_s = y_val_s.reshape(-1, 1)

        m_scaled = regression_metrics(
            y_true=y_true_s.reshape(-1),
            y_pred=y_pred_s.reshape(-1),
            tol=cfg.scaled_accuracy_tol,
        )

        y_pred = inverse_transform_yield(y_pred_s, scalers)
        y_true = inverse_transform_yield(y_true_s, scalers)
        m_orig = regression_metrics(y_true=y_true, y_pred=y_pred, tol=cfg.original_accuracy_tol)

        fold_results.append(
            FoldResult(
                fold=fold,
                n_train=int(len(train_idx)),
                n_val=int(len(val_idx)),
                metrics_scaled=m_scaled,
                metrics_original=m_orig,
            )
        )

        for i, row_id in enumerate(val_idx):
            preds_rows.append(
                {
                    "fold": fold,
                    "row_index": int(row_id),
                    "y_true_scaled": float(y_true_s[i, 0]),
                    "y_pred_scaled": float(y_pred_s[i, 0]),
                    "y_true": float(y_true[i]),
                    "y_pred": float(y_pred[i]),
                }
            )

    pd.DataFrame(preds_rows).to_csv(out_dir / "predictions.csv", index=False)
    cv = CVResult(model_name="mtl", feature_cols=feature_cols, fold_results=fold_results)
    (out_dir / "metrics.json").write_text(json.dumps(cv.to_dict(), indent=2))
    return cv

