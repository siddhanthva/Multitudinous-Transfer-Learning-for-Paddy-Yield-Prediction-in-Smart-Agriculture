from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import PipelineConfig


@dataclass
class PreparedData:
    df: pd.DataFrame
    feature_cols: list[str]


@dataclass
class FoldScalers:
    x_scaler: MinMaxScaler
    y_scaler: MinMaxScaler
    aux_scalers: dict[str, MinMaxScaler]


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def prepare_dataframe(df_raw: pd.DataFrame, cfg: PipelineConfig) -> PreparedData:
    df = df_raw.copy()

    # Basic column checks
    if cfg.target_col not in df.columns:
        raise ValueError(f"Missing target column `{cfg.target_col}` in CSV.")
    if cfg.date_col not in df.columns:
        raise ValueError(f"Missing date column `{cfg.date_col}` in CSV.")

    # Drop id-like columns if present
    for c in cfg.drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Parse date and add engineered date features.
    dt = pd.to_datetime(df[cfg.date_col], errors="coerce")
    df["year"] = dt.dt.year.astype("Int64")
    df["month"] = dt.dt.month.astype("Int64")
    df["dayofyear"] = dt.dt.dayofyear.astype("Int64")
    df = df.drop(columns=[cfg.date_col])

    # Coerce everything numeric (after dropping date string)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop missing rows (paper: drop missing rows + min-max scaling)
    df = df.dropna(axis=0).reset_index(drop=True)

    # Feature columns are everything except target
    feature_cols = [c for c in df.columns if c != cfg.target_col]
    if not feature_cols:
        raise ValueError("No feature columns found after preprocessing.")

    return PreparedData(df=df, feature_cols=feature_cols)


def fit_fold_scalers(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    cfg: PipelineConfig,
    train_idx: np.ndarray,
) -> FoldScalers:
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_scaler.fit(df.loc[train_idx, list(feature_cols)].to_numpy())
    y_scaler.fit(df.loc[train_idx, [cfg.target_col]].to_numpy())

    aux_scalers: dict[str, MinMaxScaler] = {}
    for col in cfg.aux_target_cols:
        if col not in df.columns:
            raise ValueError(f"Aux target `{col}` not present in dataframe.")
        sc = MinMaxScaler()
        sc.fit(df.loc[train_idx, [col]].to_numpy())
        aux_scalers[col] = sc

    return FoldScalers(x_scaler=x_scaler, y_scaler=y_scaler, aux_scalers=aux_scalers)


def transform_features(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    scalers: FoldScalers,
    idx: np.ndarray,
) -> np.ndarray:
    return scalers.x_scaler.transform(df.loc[idx, list(feature_cols)].to_numpy()).astype(
        np.float32
    )


def transform_target_yield_scaled(
    df: pd.DataFrame, cfg: PipelineConfig, scalers: FoldScalers, idx: np.ndarray
) -> np.ndarray:
    y = df.loc[idx, [cfg.target_col]].to_numpy()
    return scalers.y_scaler.transform(y).astype(np.float32)


def transform_aux_targets_scaled(
    df: pd.DataFrame, cfg: PipelineConfig, scalers: FoldScalers, idx: np.ndarray
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for col in cfg.aux_target_cols:
        out[col] = scalers.aux_scalers[col].transform(df.loc[idx, [col]].to_numpy()).astype(
            np.float32
        )
    return out


def inverse_transform_yield(
    y_scaled: np.ndarray, scalers: FoldScalers
) -> np.ndarray:
    y_scaled = np.asarray(y_scaled).reshape(-1, 1)
    return scalers.y_scaler.inverse_transform(y_scaled).reshape(-1)

