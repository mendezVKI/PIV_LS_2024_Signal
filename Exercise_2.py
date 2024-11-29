# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:12:40 2024

@author: mendez, rigutto

Script for Exercise 2: Coherency analysis between two probes.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Setting for the plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)

# Generate the output folder
Fol_Plots = 'plots_exercise_2'
if not os.path.exists(Fol_Plots):
    os.makedirs(Fol_Plots)

# Load the two probes
data = np.load('Sampled_PROBES.npz')
n_T = 3000
Fs = 3000
dt = 1/Fs
P2_U = data['P2_U'][0:n_T]
P3_U = data['P3_U'][0:n_T]

# Compute the correlation coefficient
rho_1_2 = np.corrcoef(P2_U, P3_U)

# Scatter plot
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(P2_U, P3_U, 'ko')
ax.set_xlabel('$u(P2) [m/s]$', fontsize=18)
ax.set_ylabel('$u(P3) [m/s]$', fontsize=18)
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
fig.savefig(Fol_Plots + os.sep + 'Scatter_P2_P3.pdf', dpi=200)


# Prepare the time axis
t = np.linspace(0, dt*(n_T-1), n_T) # prepare the time axis#
N_P = 512
f, Cxy = signal.coherence(P2_U, P3_U, Fs, nperseg=N_P)
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(f, Cxy,label='$C_{x,y}$')
ax.set_xlabel('$f [Hz]$', fontsize=18)
ax.set_ylabel('$|C_{x,y}| [db]$', fontsize=18)
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
fig.savefig(Fol_Plots + os.sep + 'Cross_Coherency.pdf', dpi=200)


# Look for phase lag:
from scipy.signal import csd

f, Pxy = csd(P2_U, P3_U, fs=Fs, nperseg=N_P)

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(f, np.abs(Pxy), label='$C_{x,y}$')
ax.set_xlabel('$f [Hz]$', fontsize=18)
ax.set_ylabel('$|S_{x,y}| [db]$', fontsize=18)
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
fig.savefig(Fol_Plots + os.sep + 'Cross_Spectral_Density.pdf', dpi=200)
