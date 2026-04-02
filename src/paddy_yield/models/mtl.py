from __future__ import annotations

from typing import Iterable

from tensorflow import keras


def build_mtl_model(
    n_features: int,
    lr: float,
    shared_units: int = 64,
    shared_layers: int = 2,
    dropout: float = 0.1,
    aux_heads: Iterable[str] = ("n", "pH"),
    loss_weights: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> keras.Model:
    aux_heads = list(aux_heads)
    if len(aux_heads) != 2:
        raise ValueError("This implementation expects exactly two auxiliary heads.")
    if len(loss_weights) != 3:
        raise ValueError("loss_weights must be (yield, aux1, aux2).")

    inputs = keras.Input(shape=(n_features,), name="x")
    x = inputs
    for i in range(shared_layers):
        x = keras.layers.Dense(shared_units, activation="relu", name=f"shared_dense_{i+1}")(x)
        if dropout and dropout > 0:
            x = keras.layers.Dropout(dropout, name=f"shared_dropout_{i+1}")(x)

    y_out = keras.layers.Dense(1, activation="linear", name="yield")(x)
    aux1 = keras.layers.Dense(1, activation="linear", name=aux_heads[0])(x)
    aux2 = keras.layers.Dense(1, activation="linear", name=aux_heads[1])(x)

    model = keras.Model(inputs=inputs, outputs=[y_out, aux1, aux2], name="mtl_yield")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss={"yield": "mse", aux_heads[0]: "mse", aux_heads[1]: "mse"},
        loss_weights={"yield": loss_weights[0], aux_heads[0]: loss_weights[1], aux_heads[1]: loss_weights[2]},
    )
    return model

