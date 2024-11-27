# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:52:19 2024

@author: mendez
"""

# We study the statistics of an Ornstein-Uhlenbeck Process
# and study its autocorrelation. In particular, compute the
# Integral Time Scale of the samples and the ensemble average

import numpy as np
import matplotlib.pyplot as plt
import os

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=13)
plt.rc('ytick',labelsize=13)

# Generate the output folder
Fol_Plots = 'plots_exercise_1'
if not os.path.exists(Fol_Plots):
    os.makedirs(Fol_Plots)

# Initial and final time
t_0 = 0
t_end = 10
# Number of Samples
n_t = 1001
# Process Parameters
kappa = 1.2
theta = 3
sigma = 0.5
# Create the time scale
t = np.linspace(t_0,t_end,n_t); dt=t[2]-t[1]

def U_O_Process(kappa, theta, sigma, t):
    n_T = len(t)
    # Initialize the output
    y = np.zeros(n_T)
    # Define Drift and Diffusion functions in the process
    drift = lambda y,t: kappa*(theta-y)
    diff = lambda y,t: sigma
    noise=np.random.normal(loc=0,scale=1,size=n_T)*np.sqrt(dt)
    # Solve Stochastic Difference Equation
    for i in range(1,n_T):
        y[i] = y[i-1] + drift(y[i-1], i*dt)*dt + diff(y[i-1], i*dt)*noise[i]
    return y

#%% Step two: Produce the ensemble of n_lambda=50 realizations
n_r = 5000
U_N = np.zeros((n_t,n_r))

for l in range(n_r):
    U_N[:,l] = U_O_Process(kappa,theta,sigma,t)


fig, ax = plt.subplots(figsize=(5, 3)) # This creates the figure
ax.plot(t, U_N[:, 1])
ax.plot(t, U_N[:, 10])
ax.plot(t, U_N[:, 22])
ax.plot(t, U_N[:, 55])
ax.set_xlabel('$t[s]$',fontsize=18)
ax.set_ylabel('$u_{n}$',fontsize=18)
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
fig.savefig(Fol_Plots + os.sep + 'Ex1_samples.pdf', dpi=200)

# Compute the sample mean and plot it
# ensemble Mean
U_Mean = np.mean(U_N,axis=1)
# ensemble STD
U_STD = np.std(U_N,axis=1)


fig, ax = plt.subplots(figsize=(5, 3)) # This creates the figure
ax.plot(t, U_Mean)
ax.plot(t, U_Mean+U_STD,'r--')
ax.plot(t, U_Mean-U_STD,'r--')
ax.set_xlabel('$t[s]$', fontsize=18)
ax.set_ylabel('$\tilde{\mu}_u\pm\tilde{\sigma}_u$', fontsize=18)
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
fig.savefig(Fol_Plots + os.sep + 'Ex2_Means.pdf', dpi=200)



#%% Compute the ensemble autocorrelations

# Definition not normalized
def ensemble_Autocorr(U_N, W_N, k, j):
    n_t, n_r = np.shape(U_N)
    # Select all realizations at time t_k for U
    U_N_k = U_N[k,:]
    # Select all realizations at time t_kj for W
    W_N_k=W_N[j,:]
    # Note (these are row vectors)
    # Compute the average products
    PR=(U_N_k-np.mean(U_N_k)).dot((W_N_k-np.mean(W_N_k)))/n_r
    R_UW=PR/(np.std(U_N_k)*np.std(W_N_k))
    return R_UW


# Define lag (in number of samples)
lag = 50
# Study the autocorrelation between two points at equal lags
N_S = 100
R_UW = np.zeros(N_S)
# Select a 100 random points i (larger than 500)
J = np.random.randint(500, 800, N_S)
K = J + 50
for n in range(N_S):
  R_UW[n] = ensemble_Autocorr(U_N, U_N, J[n], K[n])

#%% Finally, analyze the convergence as a function of n_r
# Create the vector of n_r's that will be tested
n_R = np.round(np.logspace(0.1, 3, num=41))
# Prepare the outputs at k=100
mu_10 = np.zeros(len(n_R))
sigma_10 = np.zeros(len(n_R))
# Prepare the outputs at k=700
mu_700 = np.zeros(len(n_R))
sigma_700 = np.zeros(len(n_R))

# Loop over all n_R's.
for n in range(len(n_R)):
    # show progress
    print('Computing n=' + str(n) + ' of ' + str(len(n_R)))
    n_r = int(n_R[n]) # Define the number of ensembles
    U_N = np.zeros((n_t, n_r)) # Initialize the ensemble set
    for l in range(n_r): # Fill the ensemble Matrix
        U_N[:, l] = U_O_Process(kappa, theta, sigma, t)
    # Compute the mean and the std's
    mu_10[n] = np.mean(U_N[10, :]) # ensemble Mean
    sigma_10[n] = np.std(U_N[10, :]) # ensemble STD
    mu_700[n] = np.mean(U_N[700, :]) # ensemble Mean
    sigma_700[n] = np.std(U_N[700, :]) # ensemble STD


fig, ax = plt.subplots(figsize=(5, 3)) # This creates the figure
ax.plot(n_R, mu_10, 'ko:', label='k=10')
ax.plot(n_R, mu_700, 'rs:', label='k=700')
ax.set_xscale('log')
ax.set_xlabel('$n_r$', fontsize=18)
ax.set_ylabel('$\mu_U$', fontsize=18)
plt.legend()
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
fig.savefig(Fol_Plots + os.sep + 'Ex1_Mean_Conv.pdf', dpi=200)


fig, ax = plt.subplots(figsize=(5, 3)) # This creates the figure
ax.plot(n_R, sigma_10, 'ko:', label='k=10')
ax.plot(n_R, sigma_700, 'rs:', label='k=700')
ax.set_xscale('log')
ax.set_xlabel('$n_r$', fontsize=18)
ax.set_ylabel('$\sigma_U$', fontsize=18)
plt.legend()
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
fig.savefig(Fol_Plots + os.sep + 'Ex1_Sigma_Conv.pdf', dpi=200)


# Auto-Correlation Matrix
R_UU = 1/n_r * U_N.dot(U_N.T)
U_prime = U_N - np.outer(U_Mean, np.ones(n_r))
C_UU = 1/n_r * U_prime.dot(U_prime.T)



# #%% The scripts that follows were used in the section 2

# # Check the autocorrelation (linear or circular )

# u = np.array([1,2,3,4])

# from scipy import signal
# R_UUL = signal.correlate(u, u, method = 'direct')
# print(R_UUL)


# from numpy.fft import fft, ifft

# R_UUC = ifft(fft(u) * fft(u).conj()).real
# print(R_UUC)

# def CC_Norm_zeroPAD(x,y):
#     result = np.correlate(x, y, mode='same')/len(x)/(np.std(x)*np.std(y))
#     return result[result.size//2:]

# plt.figure()
# plt.plot(CC_Norm_zeroPAD(U_N[:, 1], U_N[:, 2]))



# def CC_Norm_fft(x,y):
#     r2 = np.fft.ifft(np.abs(np.fft.fft(x)*np.fft.fft(y))).real
#     c = (r2/len(x)-(np.mean(x)*np.mean(y)))/(np.std(x)*np.std(y))
#     return c[:len(x)//2]
