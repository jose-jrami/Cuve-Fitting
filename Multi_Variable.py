

# Import Modules
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


# Declare indpendant variables x0 and x1
# x0 is a sine wave and returns sin value outputs
# x1 is a list on intigers
x0 = [math.sin(x / 10) for x in range(100)]
x1 = [x / 10 + x / 100 for x in range(100)]
xa = (x0, x1)


# Define function with independant varibles x0 and x0
# function is a sine wave minus a cosine wave
def f(X, fx0, fx1):
    x0, x1 = X
    return np.sin(np.multiply(x0, fx0)) - np.cos(np.multiply(x1, fx1))


# Generate y values using independant input variables x0 ad x1
y = f((x0, x1), 2, 3)


# Generate the predcited curve
popt, pcov = opt.curve_fit(f, xa, y, (2.8, 3.1))

# Plot the original curve and the predicted curve
plt.plot(f((x0, x1), 2, 3), "-", color="red", label="Original")
plt.plot(f((x0, x1), *popt), "--", color="Blue", label="Prediction")
plt.legend(loc="lower left")
plt.show()
