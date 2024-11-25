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



# import functions from the previous script
from Exercise_3 import RBF_Collocation, Gamma_RBF, RBF_U_Reg, Evaluate_RBF


# This is a folder with all the PTV data 
FOL_RBF='RBF_DATA_CYLINDER'
if not os.path.exists(FOL_RBF):
    os.mkdir(FOL_RBF)
    

# This is a folder for the temporary images for the GIF
FOL_OUT_A='PTV_vs_RBF_Animation_CYL'
if not os.path.exists(FOL_OUT_A):
    os.mkdir(FOL_OUT_A)
    


Plots=True


# Step 1 Function that takes in input a range of snapshots
# and produce the associated w_u, w_v.

def process_data(start,end):
 #loop over the start/end snapshots   
 for k in tqdm(range(start,end)):
  Name='PTV_DATA_CYLINDER'+os.sep+'Res%05d'%(k+1)+'.dat' 
  DATA = np.genfromtxt(Name,usecols=np.arange(0,4)) 
  X_p=DATA[:,0]; Y_p=DATA[:,1]
  U_p=DATA[:,2] ; V_p=DATA[:,3]
  RBF_data=FOL_RBF+os.sep+'RBFs.dat'
  DATA = np.genfromtxt(RBF_data,usecols=np.arange(0,3)) 
  X_c, Y_c, c= DATA[:,0],DATA[:,1],DATA[:,2]
  w_u, w_v=RBF_U_Reg(X_p,Y_p,U_p,V_p,X_c,Y_c,c,K_cond=1e11)
  # Export the RBFs data for all the other workers
  output_file='RBF_DATA_CYLINDER'+os.sep+'Res%05d'%(k+1)+'.dat'
  data=np.column_stack((w_u,w_v))
  np.savetxt(output_file, data, delimiter='\t',fmt='%.4g') 
   
  if Plots and k<100:
   # Evaluate this on a new grid
   n_x=100; n_y=50
   x_v=np.linspace(np.min(X_p),np.max(X_p),n_x)
   y_v=np.linspace(np.min(Y_p),np.max(Y_p),n_y)

   Xg,Yg=np.meshgrid(x_v,y_v)
   # Get the solution on this new grid
   U_reg, V_reg=Evaluate_RBF(Xg.reshape(-1),
                             Yg.reshape(-1),
                             X_c,Y_c,c,w_u,w_v)
   Ug=U_reg.reshape(n_y,n_x)
   Vg=V_reg.reshape(n_y,n_x)
   
   # Show low resolution vs high resolution modes
   fig, (ax1, ax2) = plt.subplots(2,figsize=(6, 6))
   ax2.set_title('RBF Regression')
   #ax1.set_yticks([]); ax1.set_xticks([])
   ax2.contourf(Xg,Yg,np.sqrt(Ug**2+Vg**2),levels=30,
                vmin=0, vmax=18)
   ax2.quiver(Xg.reshape(-1),Yg.reshape(-1),
           U_reg,V_reg)
   ax2.set_aspect('equal')
   ax2.set_xlabel('$x[mm]$',fontsize=13)
   ax2.set_ylabel('$y[mm]$',fontsize=13)
   ax2.set_xticks(np.arange(0,70,10))
   ax2.set_yticks(np.arange(-10,11,10))
   ax2.set_xlim([0,50])
   ax2.set_ylim(-10,10)
   circle = plt.Circle((0,0),2.5,fill=True,color='r',edgecolor='k',alpha=0.5)
   ax2.add_patch(circle)
   plt.tight_layout()    
   
   #ax1.quiver(Xg.T,Yg.T,Phi_U,Phi_V)
   ax1.set_title('PTV Field')
   ax1.scatter(X_p,Y_p,c=np.sqrt(U_p**2+V_p**2))
   ax1.quiver(X_p,Y_p,U_p,V_p)
   #ax2.set_yticks([]); ax2.set_xticks([])
   ax1.set_aspect('equal')
   ax1.set_xlabel('$x[mm]$',fontsize=13)
   ax1.set_ylabel('$y[mm]$',fontsize=13)
   ax1.set_xticks(np.arange(0,70,10))
   ax1.set_yticks(np.arange(-10,11,10))
   ax1.set_xlim([0,50])
   ax1.set_ylim(-10,10)
   circle = plt.Circle((0,0),2.5,fill=True,color='r',edgecolor='k',alpha=0.5)
   ax1.add_patch(circle)
   plt.tight_layout()
   Name=FOL_OUT_A+os.sep+'Snapshot_U_'+str(k)+'.png'
   plt.savefig(Name,dpi=100)
   plt.close('all')
   

# Computation of all the w_u, w_b.
import multiprocessing


if __name__ == "__main__":
    
    # This is a folder with all the PTV data 
    FOL_RBF='RBF_DATA_CYLINDER'
    if not os.path.exists(FOL_RBF):
     os.mkdir(FOL_RBF)
    
    total_datasets = 1000
    num_workers = 4
    chunk_size = total_datasets // num_workers

    ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_workers)]
    ranges[-1] = (ranges[-1][0], total_datasets)
    
    # Read the first data
    Name='PTV_DATA_CYLINDER'+os.sep+'Res%05d'%(1)+'.dat' 
    DATA = np.genfromtxt(Name,usecols=np.arange(0,4)) 
    X_p=DATA[:,0]; Y_p=DATA[:,1]
    U_p=DATA[:,2] ; V_p=DATA[:,3]
    # Prepare the rbfs
    R_b=5; gamma_max=0.8; R_max=0.5
    X_c, Y_c, c=RBF_Collocation(X_p,Y_p,R_max,R_b,gamma_max)
    RBF_data=FOL_RBF+os.sep+'RBFs.dat' # name for the RBF info
    data=np.column_stack((X_c, Y_c, c))
    np.savetxt(RBF_data, data, delimiter='\t',fmt='%.4g') 


    
    with multiprocessing.Pool(num_workers) as pool:
        pool.starmap(process_data, ranges)


# Animation of the RBF regression


# Create the gif with the 100 saved snapshots
# prepare the GIF
import imageio # powerful library to build animations
GIFNAME='PTV_vs_RBF.gif'

images=[]
# 3. Pile up the images into a video
for k in range(100):
 MEX= 'Mounting Im '+ str(k)+' of ' + str(100)
 print(MEX)
 FIG_NAME=FOL_OUT_A+os.sep+'Snapshot_U_'+str(k)+'.png' 
 images.append(imageio.imread(FIG_NAME))

# 4. Create the gif from the video tensor
imageio.mimsave(GIFNAME, images,duration=0.1)
