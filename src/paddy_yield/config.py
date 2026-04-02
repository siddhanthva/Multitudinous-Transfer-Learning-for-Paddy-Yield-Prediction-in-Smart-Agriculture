from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    random_state: int = 42
    n_splits: int = 5
    shuffle_kfold: bool = True

    # Columns
    target_col: str = "yield_kg_per_ha"
    drop_cols: tuple[str, ...] = ("sl_no",)
    date_col: str = "date"

    # Bi-LSTM: treat each sample's feature vector as a "sequence"
    # (timesteps = n_features, feature_dim = 1)
    bilstm_epochs: int = 100
    bilstm_batch_size: int = 32
    bilstm_patience: int = 10
    bilstm_lr: float = 1e-3
    bilstm_units: int = 128

    # MTL: shared dense trunk + heads for yield and auxiliary targets
    mtl_epochs: int = 100
    mtl_batch_size: int = 32
    mtl_patience: int = 10
    mtl_lr: float = 1e-3
    mtl_shared_units: int = 64
    mtl_shared_layers: int = 2
    mtl_dropout: float = 0.1

    # Auxiliary tasks for MTL
    aux_target_cols: tuple[str, ...] = ("n", "pH")
    loss_weights: tuple[float, float, float] = (0.7, 0.15, 0.15)  # yield, n, pH

    # "Accuracy" as used in the paper is not precisely defined.
    # We report tolerance-based accuracy in both scaled and original space.
    # For scaled yield, tol=0.10 means within 0.10 of [0,1] range.
    scaled_accuracy_tol: float = 0.10
    # For original yield units (kg/ha), this is used as an absolute tolerance.
    original_accuracy_tol: float = 100.0

