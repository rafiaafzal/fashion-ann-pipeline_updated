import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["preprocess"]
x = np.load("data/raw/x_train.npy").astype("float32") / 255.0 - 0.286
y = np.load("data/raw/y_train.npy")
x_test = np.load("data/raw/x_test.npy").astype("float32") / 255.0 - 0.286
y_test = np.load("data/raw/y_test.npy")

x_train, x_val, y_train, y_val = train_test_split(
    x, y,
    test_size=params["test_size"],
    random_state=params["seed"],
    stratify=y,
)

os.makedirs("data/processed", exist_ok=True)
np.save("data/processed/x_train.npy", x_train)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/x_val.npy", x_val)
np.save("data/processed/y_val.npy", y_val)
np.save("data/processed/x_test.npy", x_test)
np.save("data/processed/y_test.npy", y_test)

print("Processed data saved to data/processed/")

# TODO: add data augmentation