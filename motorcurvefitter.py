import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

df = pd.read_csv('GoBilda-MotorCurve-Combined.csv')

def model(X, a, b, c):
    v, w, sign_w = X
    return a * v - b * w - sign_w * c

v = df['voltage'] / 12
w = df['speed'] * 2 * np.pi / 60
t = df['torque']

x = np.vstack((v, w, np.sign(w)))

params, covariance = curve_fit(model, x, t)

print(params)

T_pred = model(x, *params)

# Calculate R^2
ss_res = np.sum((t - T_pred) ** 2)
ss_tot = np.sum((t - np.mean(t)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(r_squared)
