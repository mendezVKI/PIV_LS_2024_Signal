import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from Functions_Chapter_10 import Plot_Field_TEXT_Cylinder, Gamma_RBF

# Setting for the plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)

Fol_Plots = 'plots_exercise_5'
if not os.path.exists(Fol_Plots):
    os.makedirs(Fol_Plots)

#%% Load the (gridded PIV data)

def load_piv_data(file_name):
    data = np.genfromtxt(file_name, usecols=[0, 1], max_rows=nxny+1)[1:, :]
    return data

# Define the input folder and only take the first 1000 frames
Fol_Piv = 'PIV_DATA_CYLINDER'
file_names = sorted([file for file in os.listdir(Fol_Piv) if 'MESH' not in file])[:1000]

n_t = len(file_names)

# Load the mesh information
Name = Fol_Piv + os.sep + file_names[0]
Name_Mesh = Fol_Piv + os.sep + 'MESH.dat'
n_s, Xg, Yg, Vxg, Vyg, X_S, Y_S = Plot_Field_TEXT_Cylinder(Name, Name_Mesh, PLOT=False)
nxny = int(n_s/2)

# Parallel processing to load the files
num_workers = 6
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    results = np.array(list(tqdm(executor.map(lambda file_name: load_piv_data(Fol_Piv + os.sep + file_name),
        file_names),
        total=len(file_names), disable=False, desc='Loading PIV Data')))

#%% Compute the POD using the modulo package (this is just to check the decompositions later on)
# TODO Delete this cell later on
from modulo_vki import ModuloVKI

D = np.transpose(results, axes=(0, 2, 1))
# D = D - np.mean(D, axis=0)[np.newaxis, :, :]
D = D.reshape(n_t, n_s).T

modu = ModuloVKI(data=np.nan_to_num(D), n_Modes=1000)
Phi_POD, Psi_POD, Sigma_POD = modu.compute_POD_K()

K_grid = modu.K

#%% Meshless POD, computation of I matrix

# Input folder of the RBF weights
Fol_Rbf = 'RBF_DATA_CYLINDER'

weight_list = sorted([file for file in os.listdir(Fol_Rbf) if 'RBF' not in file])


def func(x, y, x_c_n, x_c_m, y_c_n, y_c_m, c_n, c_m):
    return np.exp(-c_n**2 * ((x-x_c_n)**2 + (y-y_c_n)**2)) * \
        np.exp(-c_m**2 * ((x-x_c_m)**2 + (y-y_c_m)**2))

# load the RBF data
X_C, Y_C, c_k = np.genfromtxt(Fol_Rbf + os.sep + 'RBFs.dat').T
n_b = c_k.shape[0]


# # This was to test dblquad, but for some reason it did not work well, so we just use a summation for now
# from scipy.integrate import dblquad
# n = 1
# m = 33
# integral, error = dblquad(
#     func,
#     Xg.min(), Xg.max(),
#     lambda x: Yg.min(),
#     lambda x: Yg.max(),
#     args=(X_C[n], X_C[m], Y_C[n], Y_C[m], c_k[n], c_k[m])
#     )

# Integration domain (for classic integration)
n_integrate = 151
x_integrate = np.linspace(Xg.min(), Xg.max(), n_integrate)
y_integrate = np.linspace(Yg.min(), Yg.max(), int(n_integrate / 0.41))
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

# Initialize the meshless POD K matrix
K_meshless = np.zeros((n_t, n_t))

# Double loop seems inefficient, but the matrix multiplication is the most expensive part and it is already parallelized
for i in tqdm(range(n_t), desc='Computing meshless K'):
    w_U_i, w_V_i = np.genfromtxt(Fol_Rbf + os.sep + weight_list[i]).T
    for j in range(i+1):
        w_U_j, w_V_j = np.genfromtxt(Fol_Rbf + os.sep + weight_list[j]).T

        # K is symmetric, so we only compute the lower diagonal matrix and mirror it while filling
        K_meshless[i, j] = w_U_i.T@I_meshless@w_U_j + w_V_i.T@I_meshless@w_V_j
        K_meshless[j, i] = K_meshless[i, j]


# Plot the two different Ks
fig, axes = plt.subplots(figsize=(10, 5), dpi=100, ncols=2)
axes[0].imshow(K_grid)
axes[1].imshow(K_meshless)
axes[0].set_title('Gridded')
axes[1].set_title('Meshless')
fig.tight_layout()
fig.savefig(Fol_Plots + os.sep + 'K_matrix.pdf')

#%% Compute the temporal basis Psi and Sigma

# Import the eigensolver for the decomposition
from scipy.linalg import eigh

n_modes = 1000  # Full POD

# Perform the eigendecomposition
n = np.shape(K_grid)[0]
Lambda_grid, Psi_grid = eigh(K_grid, subset_by_index=[n - n_modes, n - 1])
# Sort eigenvalues by their magnitude
idx = np.flip(np.argsort(Lambda_grid))
Lambda_grid = Lambda_grid[idx]
Psi_grid = Psi_grid[:, idx]
Sigma_grid = np.sqrt(Lambda_grid)

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

# The spatial basis for gridded data just requires matrix multiplications
Sigma_grid_Inv_V = 1 / Sigma_grid
Sigma_grid_Inv = np.diag(Sigma_grid_Inv_V)
Phi_grid = np.linalg.multi_dot([D, Psi_grid, Sigma_grid_Inv])

# The meshless Phi can be computed on ANY set of points. For reference, we take the same ones as the gridded POD
Phi_meshless = np.zeros(Phi_grid.shape)

# The Gamma matrix is the same at every step
Gamma = Gamma_RBF(Xg.ravel(), Yg.ravel(), X_C, Y_C, c_k)
for i in tqdm(range(n_modes), mininterval=1, desc='Computing Meshless Phi'):
    weights_U_projected = np.squeeze(Psi_meshless[:, i][np.newaxis, :].dot(w_U))
    weights_V_projected = np.squeeze(Psi_meshless[:, i][np.newaxis, :].dot(w_V))

    Phi_meshless[:nxny, i] = Gamma.dot(weights_U_projected) / Sigma_meshless[i]
    Phi_meshless[nxny:, i] = Gamma.dot(weights_V_projected) / Sigma_meshless[i]


#%% Visualization of the modes

fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
ax.plot(Sigma_grid/np.nansum(Sigma_grid), 'ko', label='Gridded')
ax.plot(Sigma_meshless/np.nansum(Sigma_meshless), 'rs', label='Meshless', alpha=0.5)
ax.set_yscale('log')
ax.legend()
ax.set_title('Sigmas')
ax.set_xlim(-0.5, 50.5)
ax.set_ylim(1e-3, 1)
fig.tight_layout()
fig.savefig(Fol_Plots + os.sep + 'Sigmas.pdf')

fig, axes = plt.subplots(figsize=(10, 15), dpi=100, ncols=1, nrows=6, sharex=True)
for i in range(axes.shape[0]):
    axes[i].plot(Psi_grid[:, i], label='Gridded')
    axes[i].plot(Psi_meshless[:, i], label='Meshless')
axes[0].legend()
fig.suptitle('Temporal basis Psi')
fig.tight_layout()
fig.savefig(Fol_Plots + os.sep + 'Psi.pdf')


n_plot = 5  # Only plot the first 5 spatial modes
fig, axes = plt.subplots(figsize=(14, 12), ncols=4, nrows=n_plot, sharex=True, sharey=True)
for i in range(axes.shape[0]):
    axes[i, 0].imshow(Phi_grid[:nxny, i].reshape(Xg.T.shape))
    axes[i, 1].imshow(Phi_grid[nxny:, i].reshape(Xg.T.shape))
    axes[i, 2].imshow(Phi_meshless[:nxny, i].reshape(Xg.shape).T)
    axes[i, 3].imshow(Phi_meshless[nxny:, i].reshape(Xg.shape).T)

axes[0, 0].set_title('Gridded $U$')
axes[0, 1].set_title('Gridded $V$')
axes[0, 2].set_title('Meshless $U$')
axes[0, 3].set_title('Meshless $V$')

fig.suptitle('Temporal basis Phi')
fig.tight_layout()
fig.savefig(Fol_Plots + os.sep + 'Psi.pdf')


# fig, axes = plt.subplots(figsize=(10, 12), ncols=2, nrows=n_modes, sharex=True, sharey=True)
# for i in range(axes.shape[0]):
#     axes[i, 0].imshow(Phi_grid[nxny:, i].reshape(Xg.T.shape))

#     weights_projected = np.squeeze(Psi_meshless[:, i][np.newaxis, :].dot(weights))

#     Gamma = Gamma_RBF(Xg.ravel(), Yg.ravel(), X_C, Y_C, c_k)

#     Phi_V = Gamma.dot(weights_projected[1*n_b:2*n_b]) / Sigma_meshless[i]

#     axes[i, 1].imshow(Phi_V.reshape(Xg.shape).T)

# fig.tight_layout()
