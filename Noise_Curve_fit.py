
# Import modules
import numpy as np
import scipy.optimize as opt
import scipy.stats as st
import math
import matplotlib.pyplot as plt
import warnings
import random
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import librosa
import librosa.display

# Import wav file and save as a CSV file
audio_data = "tone.wav"
x, sr = librosa.load(audio_data)
np.savetxt("tone.csv", x, delimiter=",")

# import CSV and define DataFrame of a sine wave
df = pd.read_csv("tone.csv", delimiter=",")
df2 = df.iloc[0:100, :].to_numpy()

# Import sine wave y avlues from DataFrame
# Introduce noise using gaussinan distrubution
yv = [list(row) for row in df.iloc[0:100, :].values]
yv2 = [y + random.gauss(0, 0.1) for y in df2]

# Define x input values
xv = np.linspace(0, 100, 100, endpoint=False)
xv2 = np.linspace(0, 100, 100, endpoint=False)


# Generate model using Ridge()
model = make_pipeline(PolynomialFeatures(12), Ridge(2))
model.fit(xv.reshape(-1, 1), yv2)

# Define Y plot of the geerated Prediction
y_plot = model.predict(xv2.reshape(-1, 1))

# Plot original curve and predicted curve
plt.plot(xv, yv, "-", color="red", label="Original")
plt.plot(y_plot, "--", color="Blue", label="Prediction")
plt.scatter(xv, yv2, alpha=0.3, s=3, color="Green", label="Noise")
plt.legend(loc="lower left")
plt.show()