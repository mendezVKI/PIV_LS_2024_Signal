# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 12:20:23 2024

@author: rigutto, ratz, mendez

This script downloads the data from PIV and PTV measurements of a jet flow.
Note 1: a lot of NaNs are present for the PIV due to poor illumination.
Note 2: the PTV fields are overly sparse near the border due to limits in the 
        illumination and the processing.
Note 3: the PTV data is provided as a single file (one very large ensamble!)        
"""

import urllib.request
from zipfile import ZipFile
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

print('Downloading JET data for Chapter 10...')
url = 'https://osf.io/s5zgr/download'
urllib.request.urlretrieve(url, 'Chapter_X_PIV_LS.zip')
print('Download Completed! Preparing the data folder...')

# Define the paths for the folders
ptv_folder = "PTV_DATA_JET"
piv_folder = "PIV_DATA_JET"

# Create the folders if they do not exist
if not os.path.exists(ptv_folder):
    os.mkdir(ptv_folder)

if not os.path.exists(piv_folder):
    os.mkdir(piv_folder)

# Path to the zip file
zip_file_path = "Chapter_X_PIV_LS.zip"

# Extract the specific files from the zip archive
with ZipFile(zip_file_path, 'r') as zip_ref:
    # Get the list of files in the zip
    file_names = zip_ref.namelist()
    
    # Extract and save the files in their respective folders
    for file_name in file_names:
        if "ptv_JET.npz" in file_name:
            # Extract to a temporary location
            extracted_path = zip_ref.extract(file_name, ptv_folder)
            # Move the file to the target folder root
            final_path = os.path.join(ptv_folder, os.path.basename(file_name))
            shutil.move(extracted_path, final_path)
            print(f"Moved {file_name} to {final_path}")
        elif "piv_JET.npz" in file_name:
            # Extract to a temporary location
            extracted_path = zip_ref.extract(file_name, piv_folder)
            # Move the file to the target folder root
            final_path = os.path.join(piv_folder, os.path.basename(file_name))
            shutil.move(extracted_path, final_path)
            print(f"Moved {file_name} to {final_path}")

# Clean up any leftover empty subdirectories
for folder in [ptv_folder, piv_folder]:
    for subdir in os.listdir(folder):
        subdir_path = os.path.join(folder, subdir)
        if os.path.isdir(subdir_path):
            shutil.rmtree(subdir_path)
            print(f"Removed leftover folder: {subdir_path}")

# Plot one velocity field from PIV

# Input folder
Fol_In = 'PIV_DATA_JET'

data = np.load(Fol_In + os.sep + 'piv_JET.npz')

X = data['X']
Y = data['Y']
U = data['U']
V = data['V']

# this is for plotting purpose
x = np.unique(X)
y = np.unique(Y)
n_x = x.shape[0]
n_y = y.shape[0]

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

# array of the X and Y coordinates
X_pcm, Y_pcm = np.meshgrid(x_pcm, y_pcm)

index=10
fig, axes = plt.subplots(figsize=(6, 6), sharex=True, sharey=True)
contour = plt.pcolormesh(X_pcm, Y_pcm, 
                     np.sqrt(U[index,:,:]**2+V[index,:,:]**2), cmap='viridis',vmin=0, vmax=16); 






# Plot an ensamble of vectors from PTV


data = np.load('PTV_DATA_JET/ptv_JET.npz')
X, Y, U, V = data['X'], data['Y'], data['U'], data['V']

fig, ax = plt.subplots(figsize=(4, 7), dpi=300)
plt.quiver(X,Y,U,V)





