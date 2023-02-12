import numpy as np
import pandas as pd
import os

CSV = "CSV/"

# Load Dataset as numpy arrays
X = np.load("X.npy")
X_reduced = np.load("X_reduced.npy")
y = np.load("y.npy")
y_hard = np.load("y_hard.npy")
# Real classes used to compare performances
y_real = np.load("y_real.npy")
X_real = np.load("X_real.npy")
classes = np.load("classes.npy")


if not os.path.exists(CSV):
    os.mkdir(CSV)

np.savetxt(CSV + "classes.csv", np.array([range(classes.shape[0]), classes]).T, delimiter=';', fmt="%s")
np.savetxt(CSV + "X_512.csv", X, delimiter=';')
np.savetxt(CSV + "X.csv", X_reduced, delimiter=';')
np.savetxt(CSV + "X_pictures.csv", X_real, delimiter=';', fmt="%s")
np.savetxt(CSV + "y_true.csv", y_real, delimiter=';', fmt="%d")
np.savetxt(CSV + "y.csv", y, delimiter=';')
np.savetxt(CSV + "y_hard.csv", y_hard, delimiter=';', fmt="%d")