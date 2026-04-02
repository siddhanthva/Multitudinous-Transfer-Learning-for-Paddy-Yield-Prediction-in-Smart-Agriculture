from __future__ import annotations

from tensorflow import keras


def build_bilstm_model(timesteps: int, lr: float, units: int = 128) -> keras.Model:
    inputs = keras.Input(shape=(timesteps, 1), name="x")
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(units, return_sequences=False),
        name="bilstm",
    )(inputs)
    x = keras.layers.Dense(64, activation="relu", name="dense_1")(x)
    x = keras.layers.Dense(32, activation="relu", name="dense_2")(x)
    outputs = keras.layers.Dense(1, activation="linear", name="y")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="bilstm_yield")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
    )
    return model

