import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import loadmat
import cv2
import pandas as pd
from matplotlib import gridspec
from time import time
from concurrent.futures import ThreadPoolExecutor


plt.rcParams['image.cmap'] = 'viridis'

fontsize = 17.8
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize-3
plt.rcParams['font.size'] = fontsize
plt.rcParams["text.latex.preamble"] = r"\usepackage{bm}"  # Include the bm package


#%% Load the data

Fol_In = 'data_piv'
data = np.load(Fol_In + os.sep + 'piv_data.npz')

X = data['X']
Y = data['Y']
U = data['U']
V = data['V']

x = np.unique(X)
y = np.unique(Y)
x_pcm = np.concatenate([
    [x[0] - (x[1] - x[0]) / 2],
    (x[:-1] + x[1:]) / 2,
    [x[-1] + (x[-1] - x[-2]) / 2]
    ])
y_pcm = np.concatenate([
    [y[0] - (y[1] - y[0]) / 2],
    (y[:-1] + y[1:]) / 2,
    [y[-1] + (y[-1] - y[-2]) / 2]
    ])
X_pcm, Y_pcm = np.meshgrid(x_pcm, y_pcm)

#%%

valid = np.logical_and(
    np.isfinite(U),
    np.isfinite(V)
    )

valid_sum = valid.sum(axis=0)

U_mean = np.nansum(U * valid, axis=0) / valid_sum
V_mean = np.nansum(V * valid, axis=0) / valid_sum

fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
ax.pcolormesh(X_pcm, Y_pcm, U_mean, vmin=0, vmax=0.004, cmap=plt.get_cmap('viridis', lut=15))
ax.set_xlabel('$x$\,[mm]')
ax.set_xlabel('$y$\,[mm]')
ax.set_aspect(1)
fig.tight_layout()


fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
ax.pcolormesh(X_pcm, Y_pcm, V_mean, vmin=0, vmax=12, cmap=plt.get_cmap('viridis', lut=15))
ax.set_xlabel('$x$\,[mm]')
ax.set_xlabel('$y$\,[mm]')
ax.set_aspect(1)
fig.tight_layout()


#%%

U_prime = U - U_mean[np.newaxis, :]
V_prime = V - V_mean[np.newaxis, :]

uu_mean = np.nansum(U_prime * U_prime * valid, axis=0) / (valid_sum - 1)
vv_mean = np.nansum(V_prime * V_prime * valid, axis=0) / (valid_sum - 1)
uv_mean = np.nansum(U_prime * V_prime * valid, axis=0) / (valid_sum - 1)


