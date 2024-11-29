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

print('Downloading JET data for Chapter 10...')
url = 'https://osf.io/s5zgr/download'; urllib.request.urlretrieve(url, 'Chapter_X_PIV_LS.zip')
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
zip_file_path = "Chapter_X_PIV_LS.zip"  # Replace with the path to your zip file

# Extract the specific files from the zip archive
with ZipFile(zip_file_path, 'r') as zip_ref:
    # Get the list of files in the zip
    file_names = zip_ref.namelist()
    
    # Extract and save the files in their respective folders
    for file_name in file_names:
        if "ptv_JET.npz" in file_name:
            zip_ref.extract(file_name, ptv_folder)
            print(f"Extracted {file_name} to {ptv_folder}")
        elif "piv_JET.npz" in file_name:
            zip_ref.extract(file_name, piv_folder)
            print(f"Extracted {file_name} to {piv_folder}")



