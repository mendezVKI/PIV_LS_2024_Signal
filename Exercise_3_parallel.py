# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:31:27 2024

@author: mendez
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist
from scipy.stats import qmc # sample method
from scipy import linalg
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Computation of all the w_u, w_b.
import multiprocessing

# # import functions from the previous script
# from Exercise_3 import RBF_Collocation, Gamma_RBF, RBF_U_Reg, Evaluate_RBF

import sys
sys.path.insert(0, 'spicy_newrelease')
from spicy_vki.spicy import Spicy

# Generate the output folder
Fol_Plots = 'plots_exercise_3_spicy'
if not os.path.exists(Fol_Plots):
    os.makedirs(Fol_Plots)

#%% Parallel processing of the regressions

# Step 1 Function that takes in input a range of snapshots
# and produce the associated w_u, w_v.

def process_Spicy_parallel(file_name):
    Name = Fol_In + os.sep + file_name
    X_p, Y_p, U_p, V_p = np.genfromtxt(Name).T

    SP = Spicy(points=[X_p, Y_p], data=[U_p, V_p], model='scalar', verbose=0)

    # Instead of calling this collocation function
    # SP.collocation(n_K=[5], method='semirandom', r_mM=[0.001, 50], eps_l=0.8)
    # We assign it to be the same for all snapshots. This helps us a lot in exercise 5
    SP.X_C, SP.Y_C, SP.c_k = np.genfromtxt(Fol_Rbf + os.sep + 'RBFs.dat').T

    SP.Assembly_Regression()
    SP.Solve(K_cond=1e11)

    data = np.column_stack((SP.w_list[0], SP.w_list[1]))
    np.savetxt(Fol_Rbf + os.sep + file_name, data, delimiter='\t',fmt='%.6g')


Fol_In = 'PTV_DATA_CYLINDER'


Fol_Out_A = 'PTV_vs_RBF_Animation_CYL_spicy'
if not os.path.exists(Fol_Out_A):
    os.mkdir(Fol_Out_A)

# This is a folder with all the PTV data
Fol_Rbf = 'RBF_DATA_CYLINDER_spicy'
if not os.path.exists(Fol_Rbf):
    os.mkdir(Fol_Rbf)

total_datasets = 1000

file_names = sorted([file for file in os.listdir(Fol_In) if 'MESH' not in file])[:total_datasets]

num_workers = 4

# We set up the collocation info. It is the same in each snapshot
Name = Fol_In + os.sep + file_names[0]
X_p, Y_p, U_p, V_p = np.genfromtxt(Name).T

SP = Spicy(points=[X_p, Y_p], data=[U_p, V_p], model='scalar', verbose=0)
SP.collocation(n_K=[5], method='semirandom', r_mM=[0.001, 50], eps_l=0.8)
collocation_path = Fol_Rbf + os.sep + 'RBFs.dat'  # name for the RBF info
np.savetxt(collocation_path, np.column_stack((SP.X_C, SP.Y_C, SP.c_k)), delimiter='\t',fmt='%.6g')

with ThreadPoolExecutor(max_workers=num_workers) as executor:
    results = np.array(list(tqdm(executor.map(lambda file_name:
                                              process_Spicy_parallel(file_name),
        file_names),
        total=len(file_names), disable=False, desc='Processing with SPICY in parallel')))

#%% Generating animation

print('Generating animation of the velocity regressions')
# Evaluate this on a new grid
n_x = 100
n_y = 50
x_g = np.linspace(np.min(X_p), np.max(X_p), n_x)
y_g = np.linspace(np.min(Y_p), np.max(Y_p), n_y)

X_g, Y_g = np.meshgrid(x_g, y_g)
# Get the solution on this new grid

for idx in tqdm(range(100)):
    # Load the weights and assign them to the Spicy object
    weights = np.genfromtxt((Fol_Rbf + os.sep + file_names[idx]))
    SP.w_list = [weights[:, 0], weights[:, 1]]
    # Get the solution on the new grid
    U_reg, V_reg = SP.get_sol(points=[X_g.ravel(), Y_g.ravel()], order=0, shape=(X_g.shape))

    # Show PTV and RBF regression of the velocity data
    fig, axes = plt.subplots(2, figsize=(6, 6), dpi=100, sharex=True, sharey=True)

    axes[0].set_title('PTV Field')
    axes[0].scatter(X_p, Y_p, c=np.sqrt(U_p**2 + V_p**2), vmin=0, vmax=18)
    axes[0].quiver(X_p, Y_p, U_p, V_p)

    axes[1].set_title('RBF Regression')
    axes[1].contourf(X_g, Y_g, np.sqrt(U_reg**2 + V_reg**2), levels=30, vmin=0, vmax=18)
    axes[1].quiver(X_g, Y_g, U_reg, V_reg)

    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlabel('$x[mm]$', fontsize=13)
        ax.set_ylabel('$y[mm]$', fontsize=13)
        ax.set_xticks(np.arange(0, 70, 10))
        ax.set_yticks(np.arange(-10, 11, 10))
        ax.set_xlim([0, 50])
        ax.set_ylim(-10, 10)
        circle = plt.Circle((0, 0), 2.5, fill=True, color='r', edgecolor='k', alpha=0.5)
        ax.add_patch(circle)

    Name = Fol_Out_A + os.sep + 'Snapshot_U_{0:03d}.png'.format(idx)
    plt.savefig(Name,dpi=100)
    plt.close('all')

#%% Animation of the RBF regression
# Create the gif with the 100 saved snapshots
# prepare the GIF
import imageio # powerful library to build animations
gifname = 'PTV_vs_RBF.gif'

images=[]
# 3. Pile up the images into a video
for k in range(100):
    print('Mounting Im '+ str(k)+' of ' + str(100))
    Name = Fol_Out_A + os.sep + 'Snapshot_U_{0:03d}.png'.format(idx)
    images.append(imageio.imread(Name))

# 4. Create the gif from the video tensor
imageio.mimsave(gifname, images, duration=0.1, loop=-1)
