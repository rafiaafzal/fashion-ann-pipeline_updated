import os
import numpy as np
import yaml
import tensorflow as tf

params = yaml.safe_load(open("params.yaml"))["train"]
tf.keras.utils.set_random_seed(params["seed"])

x_train = np.load("data/processed/x_train.npy")
y_train = np.load("data/processed/y_train.npy")
x_val = np.load("data/processed/x_val.npy")
y_val = np.load("data/processed/y_val.npy")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(params["units"], activation="relu"),
    tf.keras.layers.Dropout(params["dropout"]),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

os.makedirs("models", exist_ok=True)
model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=params["epochs"],
    batch_size=params["batch_size"],
    callbacks=[tf.keras.callbacks.CSVLogger("models/history.csv")],
)

model.save("models/model.h5")
print("Saved models/model.h5 and models/history.csv")