"""
Created on Mon Nov 25 17:31:27 2024

@author: ratz, mendez

Script for Exercise 6: meshless POD via RBF integration.
Not fully parallelized yet.

"""


import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from Functions_Chapter_10 import Plot_Field_TEXT_Cylinder
from scipy.fft import fft, fftfreq

import sys
sys.path.insert(0, 'spicy_newrelease')
# from _basis_RBF import Phi_RBF_2D
from spicy_vki.utils._basis_RBF import Phi_RBF_2D

# Setting for the plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)

Fol_Plots = 'plots_exercise_6'
if not os.path.exists(Fol_Plots):
    os.makedirs(Fol_Plots)

#%% Load the (gridded PIV data)

n_snapshots = 1000
# Define the input folder and only take the first 1000 frames
Fol_Piv = 'PIV_DATA_CYLINDER'
file_names = sorted([file for file in os.listdir(Fol_Piv) if 'MESH' not in file])[:n_snapshots]

n_t = len(file_names)

# Load the mesh information
Name = Fol_Piv + os.sep + file_names[0]
Name_Mesh = Fol_Piv + os.sep + 'MESH.dat'
n_s, Xg, Yg, Vxg, Vyg, X_S, Y_S = Plot_Field_TEXT_Cylinder(Name, Name_Mesh, PLOT=False)
nxny = int(n_s/2)


def load_piv_data(file_name):
    data = np.genfromtxt(file_name)[1:, :]
    return data

# Parallel processing to load the files
num_workers = 4
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    results = np.array(list(tqdm(executor.map(lambda file_name: load_piv_data(Fol_Piv + os.sep + file_name),
        file_names),
        total=len(file_names), desc='Loading PIV Data')))


#%% Compute the POD using the modulo package

D = np.transpose(results, axes=(0, 2, 1)).reshape(n_t, n_s).T

from modulo_vki import ModuloVKI
modu = ModuloVKI(data=np.nan_to_num(D), n_Modes=1000)
Phi_grid, Psi_grid, Sigma_grid = modu.compute_POD_K()

#%% Visualization of the gridded modes

# Plot the amplitudes
fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
ax.plot(Sigma_grid/Sigma_grid[0], 'ko')
ax.set_yscale('log')
ax.set_title('Gridded amplitude $\sigma$')
ax.set_xlim(-0.5, 50.5)
ax.set_ylim(1e-3, 1)
fig.tight_layout()
fig.savefig(Fol_Plots + os.sep + 'Sigmas_grid.pdf')


# And the temporal modes

# Sampling frequency to use for the fft
Fs = 3000
dt = 1/Fs
t = np.linspace(0, n_snapshots*dt, endpoint=False)
freqs = fftfreq(n_snapshots, dt)[:n_snapshots//2]

fig, axes = plt.subplots(figsize=(5, 5), dpi=100, ncols=1, nrows=3, sharex=True)
for i in range(axes.shape[0]):
    # Perform the fft and plot it
    axes[i].plot(freqs, 2.0/n_snapshots * np.abs(fft(Psi_grid[:, i])[0:n_snapshots//2]))
    axes[i].set_ylabel('Mode ' + str(i))
axes[2].set_xlabel('$f$ [Hz]')
fig.suptitle('Gridded FFTs of $\psi$')
fig.tight_layout()
fig.savefig(Fol_Plots + os.sep + 'Psi_fft_grid.pdf')


# And the spatial modes
n_plot = 3  # Only plot the first 3 spatial modes
fig, axes = plt.subplots(figsize=(8, 6), ncols=2, nrows=n_plot, sharex=True, sharey=True)
for i in range(axes.shape[0]):
    axes[i, 0].imshow(Phi_grid[:nxny, i].reshape(Xg.T.shape))
    axes[i, 1].imshow(Phi_grid[nxny:, i].reshape(Xg.T.shape))
    axes[i, 0].set_ylabel('Mode ' + str(i))

axes[0, 0].set_title('$U$')
axes[0, 1].set_title('$V$')

fig.suptitle('Gridded spatial basis $\phi$')
fig.tight_layout()
fig.savefig(Fol_Plots + os.sep + 'Phi_grid.pdf')


#%% Meshless POD, computation of I matrix

# Input folder of the RBF weights
Fol_Rbf = 'RBF_DATA_CYLINDER'

weight_list = sorted([file for file in os.listdir(Fol_Rbf) if 'RBF' not in file])

# Function for the integrand in equation 77
def func(x, y, x_c_n, x_c_m, y_c_n, y_c_m, c_n, c_m):
    return np.exp(-c_n**2 * ((x-x_c_n)**2 + (y-y_c_n)**2)) * \
        np.exp(-c_m**2 * ((x-x_c_m)**2 + (y-y_c_m)**2))

# load the RBF data
X_C, Y_C, c_k = np.genfromtxt(Fol_Rbf + os.sep + 'RBFs.dat').T
n_b = c_k.shape[0]

# Integration domain (for classic integration)
x_integrate = np.linspace(Xg.min(), Xg.max(), 151)
y_integrate = np.linspace(Yg.min(), Yg.max(), 61)
X_integrate, Y_integrate = np.meshgrid(x_integrate, y_integrate)
X_integrate = X_integrate.ravel()
Y_integrate = Y_integrate.ravel()


# We compute a single matrix I since the RBFs do not change in between time steps, only their weights.
# This allows to save a lot of computational cost since instead of computing 1000 matrices, we only need 1
I_meshless = np.zeros((n_b, n_b))
for m in tqdm(range(n_b), mininterval=1, desc='Filling I matrix'):
    for n in range(0, n_b):
        I_meshless[m, n] = func(X_integrate, Y_integrate, X_C[n], X_C[m], Y_C[n], Y_C[m], c_k[n], c_k[m]).sum()
# Divide by the area to normalize the integral
area = (Xg.max() - Xg.min()) / (Yg.max() - Yg.min())
I_meshless = I_meshless / area

#%% Meshless POD, computation of K matrix

from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm

# Function to compute a single element of the covariance matrix
def compute_K_element(i, j, w_U_all, w_V_all, I_meshless):
    w_U_i = w_U_all[i]; w_V_i = w_V_all[i]
    w_U_j = w_U_all[j]; w_V_j = w_V_all[j]
    # Calculate the element of the covariance matrix K[i, j]
    K_value = w_U_i.T @ I_meshless @ w_U_j + w_V_i.T @ I_meshless @ w_V_j
    return (i, j, K_value)

# Load all weights ahead of time to avoid repeated I/O
w_U_all = []; w_V_all = []

# Assuming `weight_list` and `Fol_Rbf` are already defined
for i in tqdm(range(len(weight_list)), desc='Loading weights'):
    w_U_i, w_V_i = np.genfromtxt(Fol_Rbf + os.sep + weight_list[i]).T
    w_U_all.append(w_U_i)
    w_V_all.append(w_V_i)

# Stack weights for easy indexing
w_U_all = np.stack(w_U_all)
w_V_all = np.stack(w_V_all)

# Number of snapshots
n_t = len(weight_list)

# Use joblib to parallelize the double loop calculation
results = Parallel(n_jobs=-1)(
    delayed(compute_K_element)(i, j, w_U_all, w_V_all, I_meshless)
    for i in tqdm(range(n_t), desc="Computing covariance matrix")
    for j in range(i + 1)  # Only compute for j <= i since K is symmetric
)

# Create an empty covariance matrix
K_meshless = np.zeros((n_t, n_t))

# Fill in the covariance matrix with the results from the parallel computation
for i, j, value in results:
    K_meshless[i, j] = value
    K_meshless[j, i] = value  # Since K is symmetric


# Plot the two different Ks
fig, axes = plt.subplots(figsize=(10, 5), dpi=100, ncols=2)
# Use the same vmin and vmax for both plots to ensure the color scale is the same
im1 = axes[0].imshow(modu.K/np.max(modu.K), vmin=0, vmax=1)
im2 = axes[1].imshow(K_meshless/np.max(K_meshless), vmin=0, vmax=1)
axes[0].set_title('Gridded')
axes[1].set_title('Meshless')

# Add a colorbar to the figure
fig.colorbar(im1, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
fig.tight_layout()
plt.show()
fig.savefig(Fol_Plots + os.sep + 'K_matrix.pdf')

#%% Compute the temporal basis Psi and Sigma

# Import the eigensolver for the decomposition
from scipy.linalg import eigh

n_modes = 1000  # Full POD

# The computation of Sigma and Psi is the same for the meshless POD (since time is still a discrete variable)
n = np.shape(K_meshless)[0]
Lambda_meshless, Psi_meshless = eigh(K_meshless, subset_by_index=[n - n_modes, n - 1])
idx = np.flip(np.argsort(Lambda_meshless))
Lambda_meshless = Lambda_meshless[idx]
Psi_meshless = Psi_meshless[:, idx]
Sigma_meshless = np.sqrt(Lambda_meshless)

#%% Compute the spatial basis

# load all the weights
w_U = np.zeros((n_t, n_b))
w_V = np.zeros((n_t, n_b))
for i in tqdm(range(n_t)):
    w_U[i, :], w_V[i, :] = np.genfromtxt(Fol_Rbf + os.sep + weight_list[i]).T

# The meshless Phi can be computed on ANY set of points. For reference, we take the same ones as the gridded POD
Phi_meshless = np.zeros(Phi_grid.shape)

# The Gamma matrix is the same at every step
Gamma = Phi_RBF_2D(Xg.ravel(), Yg.ravel(), X_C, Y_C, c_k, basis='gauss')
for i in tqdm(range(n_modes), mininterval=1, desc='Computing Meshless Phi'):
    weights_U_projected = np.squeeze(Psi_meshless[:, i][np.newaxis, :].dot(w_U))
    weights_V_projected = np.squeeze(Psi_meshless[:, i][np.newaxis, :].dot(w_V))

    Phi_meshless[:nxny, i] = Gamma.dot(weights_U_projected) / Sigma_meshless[i]
    Phi_meshless[nxny:, i] = Gamma.dot(weights_V_projected) / Sigma_meshless[i]


#%% Visualization of the meshless modes

# Plot the amplitudes
fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
ax.plot(Sigma_meshless/Sigma_meshless[0], 'ko')
ax.set_yscale('log')
ax.set_title('Meshless amplitude $\sigma$')
ax.set_xlim(-0.5, 50.5)
ax.set_ylim(1e-3, 1)
fig.tight_layout()
fig.savefig(Fol_Plots + os.sep + 'Sigmas_meshless.pdf')


# And the temporal modes

# Sampling frequency to use for the fft
Fs = 3000
dt = 1/Fs
t = np.linspace(0, n_snapshots*dt, endpoint=False)
freqs = fftfreq(n_snapshots, dt)[:n_snapshots//2]

fig, axes = plt.subplots(figsize=(5, 5), dpi=100, ncols=1, nrows=3, sharex=True)
for i in range(axes.shape[0]):
    # Perform the fft and plot it
    axes[i].plot(freqs, 2.0/n_snapshots * np.abs(fft(Psi_meshless[:, i])[0:n_snapshots//2]))
    axes[i].set_ylabel('Mode ' + str(i))
axes[2].set_xlabel('$f$ [Hz]')
fig.suptitle('Meshless FFTs of $\psi$')
fig.tight_layout()
fig.savefig(Fol_Plots + os.sep + 'Psi_fft_meshless.pdf')


# And the spatial modes
n_plot = 3  # Only plot the first 3 spatial modes
fig, axes = plt.subplots(figsize=(8, 6), ncols=2, nrows=n_plot, sharex=True, sharey=True)
for i in range(axes.shape[0]):
    axes[i, 0].imshow(Phi_meshless[:nxny, i].reshape(Xg.shape).T)
    axes[i, 1].imshow(Phi_meshless[nxny:, i].reshape(Xg.shape).T)
    axes[i, 0].set_ylabel('Mode ' + str(i))

axes[0, 0].set_title('$U$')
axes[0, 1].set_title('$V$')

fig.suptitle('Meshless spatial basis $\phi$')
fig.tight_layout()
fig.savefig(Fol_Plots + os.sep + 'Phi_meshless.pdf')

