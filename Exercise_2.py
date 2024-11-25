# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:12:40 2024

@author: mendez
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Setting for the plots
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=13)
plt.rc('ytick',labelsize=13)


# Load the two probes
data = np.load('Sampled_PROBES.npz')
n_T=3000; Fs=3000; dt=1/Fs
P2_U=data['P2_U'][0:n_T];  
P3_U=data['P3_U'][0:n_T]; 


# Compute the correlation coefficient
rho_1_2=np.corrcoef(P2_U, P3_U)


# Scatter plot
fig, ax = plt.subplots(figsize=(6, 3)) 
plt.plot(P2_U,P3_U,'ko')
plt.xlabel('$u(P2) [m/s]$',fontsize=18)
plt.ylabel('$u(P3) [m/s]$',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='Scatter_P2_P3.pdf'
plt.savefig(Name, dpi=200) 


# Prepare the time axis
t=np.linspace(0,dt*(n_T-1),n_T) # prepare the time axis# 
N_P=512
f, Cxy = signal.coherence(P2_U, P3_U, Fs, nperseg=N_P)
fig, ax = plt.subplots(figsize=(6, 3)) 
plt.plot(f, Cxy,label='$C_{x,y}$')
plt.xlabel('$f [Hz]$',fontsize=18)
plt.ylabel('$|C_{x,y}| [db]$',fontsize=18)
#plt.legend(fontsize=16)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='Cross_Coherency.pdf'
plt.savefig(Name, dpi=200) 


# Look for phase lag:
from scipy.signal import csd

f, Pxy = csd(P2_U, P3_U, fs=Fs, nperseg=N_P)

fig, ax = plt.subplots(figsize=(6, 3)) 
plt.plot(f, np.abs(Pxy),label='$C_{x,y}$')
plt.xlabel('$f [Hz]$',fontsize=18)
plt.ylabel('$|S_{x,y}| [db]$',fontsize=18)
#plt.legend(fontsize=16)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='Cross_Spectral_Density.pdf'
plt.savefig(Name, dpi=200) 



# Looking at the phase:
#phase_diff = np.angle(Pxy,deg=True)  # Phase difference in radians   
#plt.plot(f,phase_diff)    
    
    










