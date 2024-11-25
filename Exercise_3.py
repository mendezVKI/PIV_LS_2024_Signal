# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:24:02 2024

@author: mendez
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist
from scipy.stats import qmc # sample method

from scipy import linalg

# Setting for the plots
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=13)
plt.rc('ytick',labelsize=13)

Name='PIV_DATA_CYLINDER/Res00017.dat'
DATA = np.genfromtxt(Name,usecols=np.arange(0,4)) 
X_p=DATA[:,0]; Y_p=DATA[:,1]
U_p=DATA[:,2] ; V_p=DATA[:,3]


#%% Functions 1: RBF Collocation

def RBF_Collocation(X_p,Y_p,R_max,R_b,gamma_max):
 # Step 1: Define number of bases and collocate them
 n_p=len(X_p); n_b=round(n_p/R_b)
 # Define sampler and collocation points in [0,1]x [0,1]
 sampler = qmc.Halton(d=2, scramble=True)
 collocation_points_01 = sampler.random(n=n_b)
 domain_bounds=np.array([[np.min(X_p),np.max(X_p)],
                         [np.min(Y_p),np.max(Y_p)]])
 # Define the collocation points
 collocation_points=qmc.scale(collocation_points_01,
                              domain_bounds[:,0],domain_bounds[:,1])
 X_c, Y_c=collocation_points[:,0],collocation_points[:,1]
 # Step 2 Define the shape factors
 pairwise_distances = cdist(collocation_points, 
                            collocation_points)
 c  = np.zeros(n_b)

 for i in range(n_b):
     distances = pairwise_distances[i]
     distances[i] = np.inf  # Ignore self-distance
     nearest_distance = np.min(distances)
     if nearest_distance > 0:
         c[i] = np.sqrt(-np.log(gamma_max)) / nearest_distance
     else:
         c[i] = np.sqrt(-np.log(gamma_max)) / R_max # cap limit
 
 return X_c, Y_c, c


#%% Functions 2: Basis Matrix generation

def Gamma_RBF(X_p,Y_p,X_C,Y_C,c_k):
    n_b=len(X_C) # number of bases
    n_p=len(X_p) # number of points for the evaluation
    Gamma_matrix=np.zeros((n_p,n_b))
    for r in range(n_b):
      gaussian=np.exp(-c_k[r]**2 * ((X_p-X_C[r])**2 + (Y_p-Y_C[r])**2))
      Gamma_matrix[:,r]=gaussian  
    return Gamma_matrix # matrix of basis functions

#%% Functions 3: Regularize and solve system

def RBF_U_Reg(X_p,Y_p,U_p,V_p,X_c,Y_c,c,K_cond=1e11):
    Gamma_matrix=Gamma_RBF(X_p,Y_p,X_c,Y_c,c)
    A=Gamma_matrix.T@Gamma_matrix
    # Regularize it
    lambda_A = eigsh(A, 1, return_eigenvectors=False) # Largest eigenvalue
    alpha = (lambda_A-K_cond*2.2e-16) / K_cond
    A=A+alpha*np.eye(np.shape(A)[0])
    # Compute the RHSides
    b_U=Gamma_matrix.T.dot(U_p);  b_V=Gamma_matrix.T.dot(V_p)
    # Cholesky Factorization
    L_A, low = linalg.cho_factor(A, overwrite_a = True, check_finite = False, lower = True)
    # solve for w_u
    w_u = linalg.cho_solve((L_A, low), b_U, check_finite = False) 
    # solve for w_v
    w_v = linalg.cho_solve((L_A, low), b_V, check_finite = False)     
    return w_u, w_v

#%% Function 4: evaluate a given RBF regression

def Evaluate_RBF(X_g,Y_g,X_c,Y_c,c,w_u,w_v):
    Gamma=Gamma_RBF(X_g,Y_g,X_c,Y_c,c) # basis on new points X_g, Y_g
    U_reg=Gamma.dot(w_u)
    V_reg=Gamma.dot(w_v)
    return U_reg, V_reg


R_b=5 # number of particles per basis.
gamma_max=0.8
R_max=0.5
X_c, Y_c, c=RBF_Collocation(X_p,Y_p,R_max,R_b,gamma_max)


fig, ax=plt.subplots(1,figsize=(6,3))
plt.quiver(X_p,Y_p,U_p,V_p)
plt.plot(X_c,Y_c,'ro')
for i in range(len(c)):
  circle = plt.Circle([X_c[i],Y_c[i]], np.sqrt(np.log(2)/c[i]**2), edgecolor='blue', facecolor='none') 
  ax.add_patch(circle)    
ax.set_aspect('equal')
ax.set_xlabel('$x[mm]$',fontsize=13)
ax.set_ylabel('$y[mm]$',fontsize=13)
ax.set_xticks(np.arange(0,70,10))
ax.set_yticks(np.arange(-10,11,10))
ax.set_xlim([0,50])
ax.set_ylim(-10,10)
circle = plt.Circle((0,0),2.5,fill=True,color='r',edgecolor='k',alpha=0.5)
ax.add_patch(circle)
plt.tight_layout()
Name='RBF_Collocation.png'
plt.savefig(Name,dpi=100)
plt.close('all')


# Solve for the weights
w_u, w_v=RBF_U_Reg(X_p,Y_p,U_p,V_p,X_c,Y_c,c,K_cond=1e11)
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


# reshape it back
fig, axes=plt.subplots(nrows=1, ncols=2,figsize=(12,3))
axes[0].quiver(X_p,Y_p,U_p,V_p)
axes[0].set_aspect('equal')
axes[0].set_xlabel('$x[mm]$',fontsize=13)
axes[0].set_ylabel('$y[mm]$',fontsize=13)
axes[0].set_xticks(np.arange(0,70,10))
axes[0].set_yticks(np.arange(-10,11,10))
axes[0].set_xlim([0,50])
axes[0].set_ylim(-10,10)
circle = plt.Circle((0,0),2.5,fill=True,
                    color='r',edgecolor='k',alpha=0.5)
axes[0].add_patch(circle)

axes[1].contourf(Xg,Yg,np.sqrt(Ug**2+Vg**2),levels=30)
axes[1].quiver(Xg.reshape(-1),Yg.reshape(-1),
           U_reg,V_reg)
axes[1].set_aspect('equal')
axes[1].set_xlabel('$x[mm]$',fontsize=13)
axes[1].set_ylabel('$y[mm]$',fontsize=13)
axes[1].set_xticks(np.arange(0,70,10))
axes[1].set_yticks(np.arange(-10,11,10))
axes[1].set_xlim([0,50])
axes[1].set_ylim(-10,10)
circle = plt.Circle((0,0),2.5,fill=True,
                    color='r',edgecolor='k',alpha=0.5)
axes[1].add_patch(circle)
plt.tight_layout()


Name='RBF_Regression_VS_PTV.png'
plt.savefig(Name,dpi=100)   
plt.close('all')



