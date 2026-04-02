from __future__ import annotations

import numpy as np


def as_feature_sequence(x_2d: np.ndarray) -> np.ndarray:
    """
    Reshape tabular data (n_samples, n_features) into a sequence-like tensor
    (n_samples, timesteps=n_features, feature_dim=1) so sequence models like
    Bi-LSTM can be applied without requiring time windows.
    """
    x_2d = np.asarray(x_2d, dtype=np.float32)
    if x_2d.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x_2d.shape}")
    return x_2d.reshape((x_2d.shape[0], x_2d.shape[1], 1))

