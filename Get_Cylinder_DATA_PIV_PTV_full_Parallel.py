# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 18:02:43 2024

@author: admin
"""



from tqdm import tqdm 



#%% Step 2: Print a Field and prepare the grid

import urllib.request
print('Downloading Data PIV for Chapter 10...')
url = 'https://osf.io/47ftd/download'
urllib.request.urlretrieve(url, 'Ex_5_TR_PIV_Cylinder.zip')
print('Download Completed! I prepare data Folder')
# Unzip the file 
from zipfile import ZipFile
String='Ex_5_TR_PIV_Cylinder.zip'
zf = ZipFile(String,'r'); 
zf.extractall('./DATA_CYLINDER'); zf.close()


#%% Step 2: Print a Field and prepare the grid
import os
import numpy as np

import matplotlib.pyplot as plt

plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)


from Functions_Chapter_10 import Plot_Field_TEXT_Cylinder

FOLDER='DATA_CYLINDER'
Name=FOLDER+os.sep+'Res%05d'%10+'.dat' # Check it out: print(Name)
Name_Mesh=FOLDER+os.sep+'MESH.dat'
n_s, Xg, Yg, Vxg, Vyg, X_S,Y_S=Plot_Field_TEXT_Cylinder(Name,Name_Mesh) 
Name='Snapshot_Cylinder.png'
nxny=int(n_s/2) 
# Get number of columns and rows
n_x,n_y=np.shape(Xg)


# Prepare the time axis (take only the first second!)
n_t=13000; Fs=3000; dt=1/Fs 
t=np.linspace(0,dt*(n_t-1),n_t) # prepare the time axis# 


# for exercise 2, prepare to export
# three time series at different locations
# Define the location of the probes in the grid
i1,j1=2,4; xp1=Xg[i1,j1]; yp1=Yg[i1,j1]
i2,j2=14,18; xp2=Xg[i2,j2]; yp2=Yg[i2,j2]
i3,j3=23,11; xp3=Xg[i3,j3]; yp3=Yg[i3,j3]

# Initialize Probes P1,P2,P3
P1_U=np.zeros(n_t,dtype="float32") 
P1_V=np.zeros(n_t,dtype="float32") 
P2_U=np.zeros(n_t,dtype="float32") 
P2_V=np.zeros(n_t,dtype="float32") 
P3_U=np.zeros(n_t,dtype="float32") 
P3_V=np.zeros(n_t,dtype="float32") 




#%% Step 3: Generate PTV data using multi processing

Plots=True
# We will not use a grid refinement for the sampling but just a random
# sampling of the interpolant.

# Approx number of particles
n_p_range=1000
# Fluctuation around that number 
n_p_sigma=0.02 # this is 2%


from scipy.interpolate import RegularGridInterpolator


# This is a folder for the temporary images for the GIF
FOL_OUT_A='PIV_vs_PTV_Animation'
if not os.path.exists(FOL_OUT_A):
    os.mkdir(FOL_OUT_A)
    
# This is a folder with all the PTV data 
FOL_PTV='PTV_data'
if not os.path.exists(FOL_PTV):
    os.mkdir(FOL_PTV)
    

# This is the interpolation step, which we run with multi-processing
# Define a function that process data from r1 to r2

def process_data(start,end): 
 for k in tqdm(range(start,end)):
   # load the velocity field
   # Name of the file to read
   Name=FOLDER+os.sep+'Res%05d'%(k+1)+'.dat' 
   # Read data from a file
   # Here we have the two colums
   DATA = np.genfromtxt(Name,usecols=np.arange(0,2),max_rows=nxny+1) 
   Dat=DATA[1:,:] # Remove the first raw with the header
   # Get U component and reshape as grid
   V_X=Dat[:,0]; U_F=np.reshape(V_X,(n_y,n_x)).T
   # Get V component and reshape as grid
   V_Y=Dat[:,1]; V_F=np.reshape(V_Y,(n_y,n_x)).T   
   Magn=np.sqrt(U_F**2+V_F**2)
   
   # # Sample the three probes for both components
   # # Sample the U components
   # P1_U[k]=U_F[i1,j1];P2_U[k]=U_F[i2,j2];P3_U[k]=U_F[i3,j3];
   # # Sample the V Components
   # P1_V[k]=V_F[i1,j1];P2_V[k]=V_F[i2,j2];P3_V[k]=V_F[i3,j3];
   
   #print('Interpolating snapshot '+str(k))
   # Compute an interpolator
   
   # This is for the U component
   interp_u=RegularGridInterpolator((Xg[:,0], Yg[1,:]), U_F,method='cubic')
   # This is for the U component
   interp_v=RegularGridInterpolator((Xg[:,0], Yg[1,:]), V_F,method='cubic')
         
   # Define the number of particles:
   n_particles=np.random.randint(int(n_p_range*(1-n_p_sigma)),
                                 int(n_p_range*(1+n_p_sigma)))    
   
   # Random locations of these particles
   x_ran=np.random.uniform(np.min(Xg[:,0]),np.max(Xg[:,0]),n_particles)
   y_ran=np.random.uniform(np.min(Yg[1,:]),np.max(Yg[1,:]),n_particles)
   
       
   random_points = np.column_stack((x_ran, y_ran))
   U_ran = interp_u(random_points)
   V_ran = interp_v(random_points)
   Magn_2=np.sqrt(U_ran**2+V_ran**2)      
   
   # Export the data in the PTV folder
   output_file=FOL_PTV+os.sep+'Res%05d'%(k+1)+'.dat'
   data=np.column_stack((x_ran,y_ran,U_ran,V_ran))
   np.savetxt(output_file, data, delimiter='\t',fmt='%.4g')

   
   # Plot some snapshots to make a gif
   if Plots and k<100:
    # Show low resolution vs high resolution modes
    fig, (ax1, ax2) = plt.subplots(2,figsize=(6, 6))
    ax1.set_title('PIV Field')
   # ax1.set_yticks([]); ax1.set_xticks([])
    ax1.contourf(Xg,Yg,Magn)
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
    
    # Plot the location of the probes
    ax1.plot(xp1,yp1,'ro',markersize=10)
    ax1.annotate(r'P1', xy=(9, -9),color='black',\
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.9))
    ax1.plot(xp2,yp2,'ro',markersize=10)
    ax1.annotate(r'P2', xy=(19, 4),color='black',\
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.9))
    ax1.plot(xp3,yp3,'ro',markersize=10)
    ax1.annotate(r'P3', xy=(27, -3),color='black',\
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.9))

    #ax1.quiver(Xg.T,Yg.T,Phi_U,Phi_V)
    ax2.set_title('PTV Field')
    ax2.scatter(x_ran,y_ran,c=Magn_2)
    ax2.quiver(x_ran,y_ran,U_ran,V_ran)
    #ax2.set_yticks([]); ax2.set_xticks([])
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
    Name=FOL_OUT_A+os.sep+'Snapshot_U_'+str(k)+'.png'
    plt.savefig(Name,dpi=100)
    plt.close('all')
       
    
     
    
    
# Test: process data 
# process_data(1,10)

import multiprocessing

if __name__ == "__main__":
    total_datasets = 13000
    num_workers = 4
    chunk_size = total_datasets // num_workers

    ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_workers)]
    ranges[-1] = (ranges[-1][0], total_datasets)

    with multiprocessing.Pool(num_workers) as pool:
        pool.starmap(process_data, ranges)


# Create the gif with the 100 saved snapshots
# prepare the GIF
import imageio # powerful library to build animations
GIFNAME='PIV_vs_PTV.gif'

images=[]
# 3. Pile up the images into a video
for k in range(100):
 MEX= 'Mounting Im '+ str(k)+' of ' + str(100)
 print(MEX)
 FIG_NAME=FOL_OUT_A+os.sep+'Snapshot_U_'+str(k)+'.png' 
 images.append(imageio.imread(FIG_NAME))

# 4. Create the gif from the video tensor
imageio.mimsave(GIFNAME, images,duration=0.1)

#import shutil  # nice and powerfull tool to delete a folder and its content
#shutil.rmtree(FOL_OUT_A)


# We collect the sample probes in a separate loop:
print('Sampling the probes')
for k in tqdm(range(n_t)):
  Name=FOLDER+os.sep+'Res%05d'%(k+1)+'.dat'
  # Read data from a file
  # Here we have the two colums
  DATA = np.genfromtxt(Name,usecols=np.arange(0,2),max_rows=nxny+1) 
  Dat=DATA[1:,:] # Remove the first raw with the header
  # Get U component and reshape as grid
  V_X=Dat[:,0]; U_F=np.reshape(V_X,(n_y,n_x)).T
  # Get V component and reshape as grid
  V_Y=Dat[:,1]; V_F=np.reshape(V_Y,(n_y,n_x)).T   
  Magn=np.sqrt(U_F**2+V_F**2)
  
  # Sample the three probes for both components
  # Sample the U components
  P1_U[k]=U_F[i1,j1];P2_U[k]=U_F[i2,j2];P3_U[k]=U_F[i3,j3];
  # Sample the V Components
  P1_V[k]=V_F[i1,j1];P2_V[k]=V_F[i2,j2];P3_V[k]=V_F[i3,j3];

    
    
np.savez('Sampled_PROBES',P1_U=P1_U,P2_U=P2_U,P3_U=P3_U,
                          P1_V=P1_V,P2_V=P2_V,P3_V=P3_V)


from scipy import signal
# Compute the power spectral density for chapter Figure
P2=np.sqrt(P2_U**2+P2_V**2)
fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
f, Pxx_den = signal.welch(P2-np.mean(P2), Fs, nperseg=2048*2)
plt.plot(f, 20*np.log10(Pxx_den),'k')
plt.xlim([100,1500])
# ax.set_yscale('log')
# ax.set_xscale('log')
# xticks=np.array([100, 200, 300, 1000,1500])
# ax.xaxis.set_ticklabels( xticks )
# ax.set_xticks([100,200, 300, 500, 1000])
# ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xlabel('$f [Hz]$',fontsize=18)
plt.ylabel('$PSD_{U2} [dB]$',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='PSD_U_2.pdf'
plt.savefig(Name, dpi=200) 




