# -*- coding: utf-8 -*-
"""
Created on Thu Nov 7 18:02:43 2024

@author: mendez, ratz

This script downloads PIV data, performs PTV analysis using multiprocessing,
and generates visual comparisons between the PIV and PTV fields, including animations.
"""

from tqdm import tqdm
import os, numpy as np, matplotlib.pyplot as plt
from Functions_Chapter_10 import Plot_Field_TEXT_Cylinder
from scipy.interpolate import RegularGridInterpolator
from concurrent.futures import ProcessPoolExecutor, as_completed
import urllib.request, imageio  # Library to create GIF animations
from zipfile import ZipFile

plt.rc('text', usetex=True); plt.rc('font', family='serif'); plt.rc('xtick', labelsize=12); plt.rc('ytick', labelsize=12)

# %% Step 1: Download the Data
print('Downloading Cylinder data for Chapter 10...')
url = 'https://osf.io/47ftd/download'; urllib.request.urlretrieve(url, 'Ex_5_TR_PIV_Cylinder.zip')
print('Download Completed! Preparing the data folder...')

# Unzip the file
with ZipFile('Ex_5_TR_PIV_Cylinder.zip', 'r') as zf: zf.extractall('./PIV_DATA_CYLINDER')

# %% Step 2: Load Field and Prepare the Grid
n_t_full, Fs, dt = 13200, 3000, 1 / 3000
n_t = 3000  # Number of snapshots for PIV and PTV
n_t_probes = n_t  # Number of snapshots for probes

t = np.linspace(0, dt * (n_t_probes - 1), n_t_probes)
Fol_In = 'PIV_DATA_CYLINDER'
file_names = sorted([file for file in os.listdir(Fol_In) if 'MESH' not in file])[:n_t]
Name, Name_Mesh = os.path.join(Fol_In, file_names[0]), os.path.join(Fol_In, 'MESH.dat')
n_s, X_g, Y_g, _, _, _, _ = Plot_Field_TEXT_Cylinder(Name, Name_Mesh, PLOT=False)
nxny, (n_x, n_y) = int(n_s / 2), X_g.shape


# %% Step 3: Generate PTV Data using Multi-Processing
n_p_range, n_p_sigma = 2000, 0.02  # Approximate number of particles and fluctuation
Fol_Ptv = 'PTV_DATA_CYLINDER'; os.makedirs(Fol_Ptv, exist_ok=True)
Num_WORKERS = 8
num_workers, block_size = Num_WORKERS, n_t // Num_WORKERS

# Function to process data in parallel for PTV interpolation
def process_block(worker_idx):
    block_start, block_end, local_results = worker_idx * block_size, min((worker_idx + 1) * block_size, n_t), []
    for idx in tqdm(range(block_start, block_end), desc=f'Worker {worker_idx}', position=worker_idx + 1):
        file_name = file_names[idx]  # Only process files for interpolation (PTV)
        U, V = np.genfromtxt(os.path.join(Fol_In, file_name), skip_header=1).T  # Read data from file
        U_grid, V_grid = np.reshape(U, (n_y, n_x)).T, np.reshape(V, (n_y, n_x)).T  # Reshape as grid
        
        # Interpolation for random particle positions (only for PTV snapshots)
        interp_u, interp_v = RegularGridInterpolator((X_g[:, 0], Y_g[1, :]), U_grid, method='cubic'), RegularGridInterpolator((X_g[:, 0], Y_g[1, :]), V_grid, method='cubic')
        n_particles = np.random.randint(int(n_p_range * (1 - n_p_sigma)), int(n_p_range * (1 + n_p_sigma)))
        x_ran, y_ran = np.random.uniform(np.min(X_g[:, 0]), np.max(X_g[:, 0]), n_particles), np.random.uniform(np.min(Y_g[1, :]), np.max(Y_g[1, :]), n_particles)
        random_points = np.column_stack((x_ran, y_ran))
        U_ran, V_ran = interp_u(random_points), interp_v(random_points)

        # Store the generated PTV data for later export
        local_results.append((file_name, np.column_stack((x_ran, y_ran, U_ran, V_ran))))
    return local_results

# Use ProcessPoolExecutor to process each block in parallel with progress tracking
with tqdm(total=n_t, desc='Processing PTV blocks in parallel', position=0) as progress_bar:
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_block, worker_idx) for worker_idx in range(num_workers)]
        for future in as_completed(futures):
            results = future.result()
            progress_bar.update(block_size)

            # Write results to files after processing each block
            for file_name, data in results:
                np.savetxt(os.path.join(Fol_Ptv, file_name), data, delimiter='\t', fmt='%.4g')

# %% Step 4: Sample the Probes Sequentially (Not in Parallel)
# Define probe locations in the grid
i1, j1, i2, j2, i3, j3 = 2, 4, 14, 18, 23, 11
xp1, yp1, xp2, yp2, xp3, yp3 = X_g[i1, j1], Y_g[i1, j1], X_g[i2, j2], Y_g[i2, j2], X_g[i3, j3], Y_g[i3, j3]


# Initialize Probes P1,P2,P3
P1_U=np.zeros(n_t_full,dtype="float32") 
P1_V=np.zeros(n_t_full,dtype="float32") 
P2_U=np.zeros(n_t_full,dtype="float32") 
P2_V=np.zeros(n_t_full,dtype="float32") 
P3_U=np.zeros(n_t_full,dtype="float32") 
P3_V=np.zeros(n_t_full,dtype="float32") 


print('Sampling probes sequentially...')
for idx in tqdm(range(n_t_full), desc='Sampling probes'):
    file_name = file_names[idx % n_t]  # Loop back for probe data
    U, V = np.genfromtxt(os.path.join(Fol_In, file_name), skip_header=1).T  # Read data from file
    U_grid, V_grid = np.reshape(U, (n_y, n_x)).T, np.reshape(V, (n_y, n_x)).T  # Reshape as grid

    # Sample the probes for all snapshots
    P1_U[idx], P2_U[idx], P3_U[idx] = U_grid[i1, j1], U_grid[i2, j2], U_grid[i3, j3]
    P1_V[idx], P2_V[idx], P3_V[idx] = V_grid[i1, j1], V_grid[i2, j2], V_grid[i3, j3]

np.savez('Sampled_PROBES', P1_U=P1_U, P2_U=P2_U, P3_U=P3_U, P1_V=P1_V, P2_V=P2_V, P3_V=P3_V)


# %% Step 5: Generate Snapshots (Only generate a few snapshots for the animations)
n_plots, Fol_Out_A = 20, 'plots_exercise_0/PIV_vs_PTV_Animation_CYL'
os.makedirs(Fol_Out_A, exist_ok=True)

for idx in tqdm(range(n_plots), desc='Generating plots'):
    U_g, V_g = np.genfromtxt(os.path.join(Fol_In, file_names[idx]), skip_header=1).T
    Magn_g = np.sqrt(U_g ** 2 + V_g ** 2).reshape(n_y, n_x).T
    x_ran, y_ran, U_ran, V_ran = np.genfromtxt(os.path.join(Fol_Ptv, file_names[idx])).T
    Magn_ran = np.sqrt(U_ran ** 2 + V_ran ** 2)

    fig, axes = plt.subplots(nrows=2, figsize=(6, 6), sharex=True, sharey=True)
    axes[0].set_title('PIV Field'); 
    contour = axes[0].contourf(X_g, Y_g, Magn_g, cmap='viridis',vmin=0, vmax=16); 
    plt.colorbar(contour, ax=axes[0]);  
    axes[1].set_title('PTV Field'); 
    scatter = axes[1].scatter(x_ran, y_ran, c=Magn_ran, cmap='viridis',vmin=0, vmax=16); 
    plt.colorbar(scatter, ax=axes[1])
    axes[1].quiver(x_ran, y_ran, U_ran, V_ran, color='k', scale=400)

    for ax in axes:
        ax.set_aspect('equal'); ax.set_xlim([0, 50]); ax.set_ylim([-10, 10])
    fig.tight_layout(); fig.savefig(os.path.join(Fol_Out_A, f'Snapshot_U_{idx:03d}.png'), dpi=100); plt.close(fig)

# %% Step 6: Create the GIF from Snapshots
gifname = 'plots_exercise_0/PIV_vs_PTV.gif'
images = [imageio.imread(os.path.join(Fol_Out_A, f'Snapshot_U_{idx:03d}.png')) for idx in range(n_plots)]
imageio.mimsave(gifname, images, duration=0.1, loop=0)


