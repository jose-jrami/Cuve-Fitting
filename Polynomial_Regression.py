

# Imported modules
import numpy as np
import scipy.optimize as opt
import scipy.stats as st
import math
import matplotlib.pyplot as plt
import warnings
import pandas as pd

# Defined Polynomiala using poly1d
p_original = np.poly1d([3, -1, -9, 1, 3, 2])


# Datapoints for original polynomial curve
X = np.linspace(-3, 3, 100, endpoint=True)
Y = p_original(X)

# Fit curve using polyfit while catching catch_warnings
# Reference: EE104 Lectrue Notes
with warnings.catch_warnings():
    warnings.simplefilter("ignore", np.RankWarning)
    p5 = np.poly1d(np.polyfit(X, Y, 6))

# 100 points betwwen -3 and +3
xp = np.linspace(-3, 3, 100)


# Plot using matplotlib
plt.plot(X, Y, ".", color="red", label="Data Points")
plt.plot(X, Y, "-", color="red", label="Original")
plt.plot(xp, p5(xp), "--", color="Blue", label="Prediction")
plt.legend(loc="lower left")
plt.ylim(-6, 15)
plt.show()
