
# Imported Modules
import numpy as np
import scipy.optimize as opt
import scipy.stats as st
import math
import matplotlib.pyplot as plt
import warnings


# Defining Damped sin wave function
# xin = iput Data
# sm = sine multiplier
# sp = sin phase
# ew = exponent weight
# gain = gain factor
# Reference: EE 104 Lecture notes
def fds(xin, sm, sp, ew, gain):
    return [gain * math.sin(sm * x + sp) * math.exp(-ew * x) for x in xin]


# x input values to train Model
# input values for sin function definition
# xin = xv
# sm = 5
# sp = 9
# ew = 0.3
# gain = 3
# yv = generated y values
xv = [x / 30 for x in range(400)]
yv = fds(xv, 5, 9, 0.3, 3)

# fit curve to the defined sin fucntion suing x values
# and generated y values
popt, pcov = opt.curve_fit(fds, xv, yv)


# Plot using matplotlib
plt.plot(xv, yv, "-", color="red", label="Original")
plt.plot(xv, fds(xv, *popt), "--", color="Blue", label="Prediction")
plt.legend(loc="lower left")
plt.show()
