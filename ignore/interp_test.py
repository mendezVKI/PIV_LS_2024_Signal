# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 18:55:34 2024

@author: admin
"""

# new tools for interpolation in scipy
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return 2 * x**3 + 3 * y**2 

x = np.linspace(1, 4, 11)
y = np.linspace(4, 7, 22)
xg, yg = np.meshgrid(x, y, indexing='ij', sparse=True)
data = f(xg, yg)

interp = RegularGridInterpolator((x, y), data)

xf = np.linspace(1, 4, 110)
yf = np.linspace(4, 7, 220)
xfg,yfg=np.meshgrid(xf, yf, indexing='ij', sparse=True)

data_f=interp((xfg,yfg))



plt.figure(1)
plt.contourf(data)


plt.figure(2)
plt.contourf(data_f)


# let's say you use a scattered list
n_p=1000
x_s=np.random.uniform(np.min(xf),np.max(xf),n_p)
y_s=np.random.uniform(np.min(yf),np.max(yf),n_p)

random_points = np.column_stack((x_s, y_s))
Z_random = interp(random_points)


plt.figure(3)
plt.scatter(x_s,y_s, c=Z_random, edgecolor='k', cmap="viridis")







