# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 18:02:43 2024

@author: admin
"""

from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from Functions_Chapter_10 import Plot_Field_TEXT_Cylinder
from scipy.interpolate import RegularGridInterpolator

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)

from concurrent.futures import ThreadPoolExecutor

#%% Step 1: Download the data

import urllib.request
print('Downloading Data PIV for Chapter 10...')
url = 'https://osf.io/47ftd/download'
urllib.request.urlretrieve(url, 'Ex_5_TR_PIV_Cylinder.zip')
print('Download Completed! I prepare the data Folder')
# Unzip the file
from zipfile import ZipFile
String = 'Ex_5_TR_PIV_Cylinder.zip'
zf = ZipFile(String,'r');
zf.extractall('./PIV_DATA_CYLINDER')
zf.close()


#%% Step 2: Print a Field and prepare the grid


# Prepare the time axis
n_t = 13000
Fs = 3000
dt = 1/Fs
t = np.linspace(0, dt*(n_t-1), n_t) # prepare the time axis

Fol_In = 'PIV_DATA_CYLINDER'
file_names = sorted([file for file in os.listdir(Fol_In) if 'MESH' not in file])[:n_t]
Name = Fol_In + os.sep + file_names[0]
Name_Mesh = Fol_In + os.sep + 'MESH.dat'
n_s, X_g, Y_g, _, _, _, _ = Plot_Field_TEXT_Cylinder(Name, Name_Mesh, PLOT=False)
# Name='Snapshot_Cylinder.png'
nxny = int(n_s/2)
# Get number of columns and rows
n_x, n_y = np.shape(X_g)

# for exercise 2, prepare to export
# three time series at different locations
# Define the location of the probes in the grid
i1, j1 = 2, 4
i2, j2 = 14, 18
i3, j3 = 23, 11
xp1, yp1 = X_g[i1, j1], Y_g[i1, j1]
xp2, yp2 = X_g[i2, j2], Y_g[i2, j2]
xp3, yp3 = X_g[i3, j3], Y_g[i3, j3]

# Initialize Probes P1,P2,P3
P1_U = np.zeros(n_t, dtype=np.float32)
P1_V = np.zeros(n_t, dtype=np.float32)
P2_U = np.zeros(n_t, dtype=np.float32)
P2_V = np.zeros(n_t, dtype=np.float32)
P3_U = np.zeros(n_t, dtype=np.float32)
P3_V = np.zeros(n_t, dtype=np.float32)


#%% Step 3: Generate PTV data using multi processing


# This is the interpolation step, which we run with multi-processing
def process_data(idx, file_name):
    # Read data from a file
    # Here we have the two colums
    U, V = np.genfromtxt(Fol_In + os.sep + file_name, skip_header=1).T
    # Get U component and reshape as grid
    U_grid = np.reshape(U, (n_y, n_x)).T
    # Get V component and reshape as grid
    V_grid = np.reshape(V, (n_y, n_x)).T

    # Sample the three probes for both components
    # Sample the U components
    P1_U[idx] = U_grid[i1, j1]
    P2_U[idx] = U_grid[i2, j2]
    P3_U[idx] = U_grid[i3, j3]
    # Sample the V Components
    P1_V[idx] = V_grid[i1, j1]
    P2_V[idx] = V_grid[i2, j2]
    P3_V[idx] = V_grid[i3, j3]

    # Compute an interpolator
    # This is for the U component
    interp_u = RegularGridInterpolator((X_g[:,0], Y_g[1,:]), U_grid, method='cubic')
    # This is for the U component
    interp_v = RegularGridInterpolator((X_g[:,0], Y_g[1,:]), V_grid, method='cubic')

    # Define the number of particles:
    n_particles = np.random.randint(int(n_p_range*(1-n_p_sigma)),
                                    int(n_p_range*(1+n_p_sigma)))

    # Random locations of these particles
    x_ran = np.random.uniform(np.min(X_g[:,0]), np.max(X_g[:,0]), n_particles)
    y_ran = np.random.uniform(np.min(Y_g[1,:]), np.max(Y_g[1,:]), n_particles)

    random_points = np.column_stack((x_ran, y_ran))
    U_ran = interp_u(random_points)
    V_ran = interp_v(random_points)

    # Export the data in the PTV folder
    output_file = Fol_Ptv + os.sep + file_name
    data = np.column_stack((x_ran, y_ran, U_ran, V_ran))
    np.savetxt(output_file, data, delimiter='\t', fmt='%.4g')


# We will not use a grid refinement for the sampling but just a random
# sampling of the interpolant.

# Approx number of particles
n_p_range = 2000
# Fluctuation around that number
n_p_sigma = 0.02 # this is 2%

# General output folder where everything is placed
Fol_Plots = 'plots_exercise_0'

# This is a folder for the temporary images for the GIF
Fol_Out_A = Fol_Plots + os.sep + 'PIV_vs_PTV_Animation_CYL'
if not os.path.exists(Fol_Out_A):
    os.makedirs(Fol_Out_A)

# This is a folder with all the PTV data
Fol_Ptv = 'PTV_DATA_CYLINDER'
if not os.path.exists(Fol_Ptv):
    os.makedirs(Fol_Ptv)

total_datasets = len(file_names)

# Number of processors to use for the parallel processing
num_workers = 4

with ThreadPoolExecutor(max_workers=num_workers) as executor:
    results = np.array(list(tqdm(executor.map(lambda enum:
                                              process_data(enum[0], enum[1]),
        enumerate(file_names)),
        total=len(file_names), mininterval=1, desc='Generating PTV data in parallel')))

#%% Step 4: Generate the figures of the snapshots

n_plots = 100
# 3. Pile up the images into a video
for idx in tqdm(range(n_plots), desc='Generating plots'):

    # Load the PIV
    U_g, V_g = np.genfromtxt(Fol_In + os.sep + file_names[idx], skip_header=1).T
    Magn_g = np.sqrt(U_g**2 + V_g**2).reshape(n_y, n_x).T
    # And the generated PTV
    x_ran, y_ran, U_ran, V_ran = np.genfromtxt(Fol_Ptv + os.sep + file_names[idx]).T
    Magn_ran = np.sqrt(U_ran**2 + V_ran**2)

    # Show the PIV and the PTV
    fig, axes = plt.subplots(nrows=2, figsize=(6, 6), sharex=True, sharey=True)
    axes[0].set_title('PIV Field')
    axes[0].contourf(X_g, Y_g, Magn_g)
    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlabel('$x[mm]$', fontsize=13)
        ax.set_ylabel('$y[mm]$', fontsize=13)
        ax.set_xticks(np.arange(0, 70, 10))
        ax.set_yticks(np.arange(-10, 11, 10))
        ax.set_xlim([0, 50])
        ax.set_ylim(-10,10)
        circle = plt.Circle((0, 0), 2.5, fill=True, color='r', edgecolor='k', alpha=0.5)
        ax.add_patch(circle)


    # Plot the location of the probes
    axes[0].plot(xp1, yp1, 'ro', markersize=10)
    axes[0].annotate(r'P1', xy=(9, -9), color='black',\
        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.9))
    axes[0].plot(xp2, yp2, 'ro',markersize=10)
    axes[0].annotate(r'P2', xy=(19, 4), color='black',\
        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.9))
    axes[0].plot(xp3,yp3,'ro',markersize=10)
    axes[0].annotate(r'P3', xy=(27, -3), color='black',\
        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.9))

    axes[1].set_title('PTV Field')
    axes[1].scatter(x_ran, y_ran, c=Magn_ran)
    axes[1].quiver(x_ran, y_ran, U_ran, V_ran)

    fig.tight_layout()
    Name = Fol_Out_A + os.sep + 'Snapshot_U_{0:03d}.png'.format(idx)
    fig.savefig(Name, dpi=100)
    plt.close('all')


#%% Step 5: Create the gif from the video tensor
# Create the gif with the 100 saved snapshots
# prepare the GIF
import imageio # powerful library to build animations
gifname = Fol_Plots + os.sep + 'PIV_vs_PTV.gif'

images = []
for idx in range(n_plots):
    print('Mounting Im '+ str(idx) + ' of ' + str(100))
    fig_name = Fol_Out_A + os.sep + 'Snapshot_U_{0:03d}.png'.format(idx)
    images.append(imageio.imread(fig_name))

imageio.mimsave(gifname, images, duration=0.1, loop=0)

# import shutil  # nice and powerfull tool to delete a folder and its content
# shutil.rmtree(Fol_Out_A)

#%% Step 6: Sample the probes

# We collect the sample probes in a separate loop:
for idx in tqdm(range(n_t), mininterval=1, desc='Sampling the probes'):
    # Read data from a file
    U, V = np.genfromtxt(Fol_In + os.sep + file_names[idx], skip_header=1).T
    # Get U component and reshape as grid
    U_grid = np.reshape(U, (n_y, n_x)).T
    # Get V component and reshape as grid
    V_grid = np.reshape(V, (n_y, n_x)).T

    # Sample the three probes for both components
    # Sample the U components
    P1_U[idx] = U_grid[i1, j1]
    P2_U[idx] = U_grid[i2, j2]
    P3_U[idx] = U_grid[i3, j3]
    # Sample the V Components
    P1_V[idx] = V_grid[i1, j1]
    P2_V[idx] = V_grid[i2, j2]
    P3_V[idx] = V_grid[i3, j3]



np.savez('Sampled_PROBES', P1_U=P1_U,P2_U=P2_U,P3_U=P3_U,
                           P1_V=P1_V,P2_V=P2_V,P3_V=P3_V)


from scipy import signal
# Compute the power spectral density for chapter Figure
P2 = np.sqrt(P2_U**2+P2_V**2)
fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
f, Pxx_den = signal.welch(P2-np.mean(P2), Fs, nperseg=2048*2)
ax.plot(f, 20*np.log10(Pxx_den),'k')
ax.set_xlim([100,1500])
ax.set_xlabel('$f [Hz]$',fontsize=18)
ax.set_ylabel('$PSD_{U2} [dB]$',fontsize=18)
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
plt.savefig(Fol_Plots + os.sep + 'PSD_U_2.pdf', dpi=200)
