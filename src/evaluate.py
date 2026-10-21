import json
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/model.h5")

x_test = np.load("data/processed/x_test.npy")
y_test = np.load("data/processed/y_test.npy")

loss, acc = model.evaluate(x_test, y_test, verbose=0)

with open("metrics.json", "w") as f:
    json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, f, indent=2)

print(f"Test accuracy: {acc:.4f}")