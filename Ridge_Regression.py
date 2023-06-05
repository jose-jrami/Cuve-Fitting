

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


# Define 5th order polynomail
def f(x):
    return 3 * x ** 5 - x ** 4 - 9 * x ** 3 + x ** 2 + 3 * x + 2


# x input values for testing
x_plot = np.linspace(-3, 3, 10000)

# x input values for testing
x = np.linspace(-3, 3, 10000)

# Generate y value from the defined polynomial function
y = f(x)

# change x input array dimensions
X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]

# Generate model to predict the curve using Ridge()
model = make_pipeline(PolynomialFeatures(5), Ridge(1))
model.fit(X, y)
y_plot = model.predict(X_plot)

# Plot the original polynomial curve and the generated predicted curve
plt.plot(x_plot, f(x_plot), "-", color="red", label="Original")
plt.plot(x_plot, y_plot, "--", color="blue", label="Prediction")
plt.legend(loc="lower left")
plt.ylim(-6, 15)
plt.xlim(-6, 6)
plt.show()
