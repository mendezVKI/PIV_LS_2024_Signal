import numpy as np
import os
import matplotlib.pyplot as plt

fontsize = 13
# Setting for the plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['font.size'] = fontsize

# Generate the output folder
Fol_Plots = 'plots_exercise_3'
if not os.path.exists(Fol_Plots):
    os.makedirs(Fol_Plots)

#%% Load the data

# Input folfer
Fol_In = 'Exercise_3_PIV_Data'

data = np.load(Fol_In + os.sep + 'piv_data.npz')

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

#%% (1) Mean flow

# indices of valid vectors
valid = np.logical_and(np.isfinite(U),
                       np.isfinite(V))

# number of valid vectors
valid_sum = valid.sum(axis=0)

# compute mean on valide vectors
U_mean = np.nansum(U * valid, axis=0) / valid_sum
V_mean = np.nansum(V * valid, axis=0) / valid_sum

plt.close('all')


u_min, u_max = 0, 225
v_min, v_max = -13, 13

# plot u and v fields
fig, axes = plt.subplots(1, 2, figsize=(8, 5), dpi=300, sharex=True, sharey=True, layout='constrained')

# u_mean field
clb = axes[0].pcolormesh(X_pcm, Y_pcm, U_mean, cmap=plt.get_cmap('viridis', lut=15),
                         vmin=u_min, vmax=u_max)
cbar = fig.colorbar(clb, ax=axes[0], pad=0.02, label=r'$u$\,[mm/s]')
axes[0].set_xlabel(r'$x$\,[mm]')
axes[0].set_ylabel(r'$y$\,[mm]')
axes[0].set_aspect(1)

# v_mean field
clb = axes[1].pcolormesh(X_pcm, Y_pcm, V_mean, cmap=plt.get_cmap('viridis', lut=15),
                     vmin=v_min, vmax=v_max)
cbar = fig.colorbar(clb, ax=axes[1], pad=0.02, label=r'$v$\,[mm/s]')
axes[1].set_xlabel(r'$x$\,[mm]')
axes[1].set_aspect(1)

fig.set_constrained_layout_pads(wspace=0.05, w_pad=0.05)

# save figure
fig.savefig(Fol_Plots + os.sep + 'Ex3_u_mean_v_mean_PIV.png')


#%% (2) TKE

# compute fluctuation fields
U_prime = U - U_mean[np.newaxis, :]
V_prime = V - V_mean[np.newaxis, :]

# compute average fluctuations
uu_mean = np.nansum(U_prime * U_prime * valid, axis=0) / (valid_sum - 1)
vv_mean = np.nansum(V_prime * V_prime * valid, axis=0) / (valid_sum - 1)
uv_mean = np.nansum(U_prime * V_prime * valid, axis=0) / (valid_sum - 1)

#%%

# compute mean TKE
fill_value = np.zeros_like(uu_mean)

# assemble the reynold stress tensor
R_ij = np.array([
    [uu_mean,       uv_mean,        fill_value  ],
    [uv_mean,       vv_mean,        fill_value  ],
    [fill_value,    fill_value,     vv_mean     ]
    ])

# comptute the turbulence kinetic energy
k = np.sum(np.diagonal(R_ij), axis=2) / 2

TI = 1/ np.max(U_mean) * np.sqrt(2/3 * k)

# plot ans save
fig, axes = plt.subplots(figsize=(8, 4), dpi=300, layout='constrained', ncols=2)
clb = axes[0].pcolormesh(X_pcm, Y_pcm, k, cmap=plt.get_cmap('viridis', lut=15), vmin=0, vmax=1600)
cbar = fig.colorbar(clb, ax=axes[0], pad=0.02, shrink=1, label=r'$k$\,[mm²/s²]')
axes[0].set_xlabel(r'$x$\,[mm]', fontsize=fontsize)
axes[0].set_ylabel(r'$y$\,[mm]')
axes[0].set_aspect(1)

clb = axes[1].pcolormesh(X_pcm, Y_pcm, TI*100, cmap=plt.get_cmap('viridis', lut=15), vmin=0, vmax=16)
cbar = fig.colorbar(clb, ax=axes[1], pad=0.02, shrink=1, label=r'TI\,[\%]')
axes[1].set_xlabel(r'$x$\,[mm]')
axes[1].set_aspect(1)
fig.set_constrained_layout_pads()
fig.savefig(Fol_Plots + os.sep + 'Ex3_TKE.png')

#%% (3) Anisotrpy

# compute the anisotropic tensor
A_ij = R_ij / (2*k[np.newaxis, np.newaxis, :]) - np.diag(np.full(3, 1/3))[:, :, np.newaxis, np.newaxis]

# norm of the tensor
A_norm = np.linalg.norm(A_ij, axis=(0, 1))

# plot and save
fig, ax = plt.subplots(figsize=(3.5, 4), dpi=300, layout='constrained')
clb = ax.pcolormesh(X_pcm, Y_pcm, A_norm, cmap=plt.get_cmap('viridis', lut=15), vmin=0, vmax=0.4)
cbar = fig.colorbar(clb, ax=ax, pad=0.02, shrink=1, label=r'$||A||$\,[-]')
ax.set_xlabel(r'$x$\,[mm]')
ax.set_ylabel(r'$y$\,[mm]')
ax.set_aspect(1)
fig.set_constrained_layout_pads()
fig.savefig(Fol_Plots + os.sep + 'Ex3_A_tensor.png')

#%% (4) Lumley triangle

# We cut some parts at the boundary since these regions are particularly delicate for the
# analysis of the lumley triangle. This is consistent between PIV and PTV
border_cut = 3

# Boundaries of the invariant map
x_1C = np.array([2/3, -1/3, -1/3])
x_2C = np.array([1/6, 1/6, -1/3])
x_3C = np.array([0, 0, 0])

# II cordinate of the triangle
II_1C = x_1C[0]**2 + x_1C[0]*x_1C[1] + x_1C[1]**2
II_2C = x_2C[0]**2 + x_2C[0]*x_2C[1] + x_2C[1]**2
II_3C = x_3C[0]**2 + x_3C[0]*x_3C[1] + x_3C[1]**2

# III coordinates of the triangle
III_1C = -x_1C[0]*x_1C[1] * (x_1C[0] + x_1C[1])
III_2C = -x_2C[0]*x_2C[1] * (x_2C[0] + x_2C[1])
III_3C = -x_3C[0]*x_3C[1] * (x_3C[0] + x_3C[1])

# Number of points to draw the limiting curves
n_p = 101

# Curve from 1 to 3
x_13 = np.array([
    np.linspace(0, 2/3, n_p),
    np.linspace(0, -1/3, n_p),
    np.linspace(0, -1/3, n_p),
    ])

# Convert into II and III coordinates
II_13 = x_13[0, :]**2 + x_13[0, :]*x_13[1, :] + x_13[1, :]**2
III_13 = -x_13[0, :]*x_13[1, :] * (x_13[0, :] + x_13[1, :])

# Curve from 2 to 3
x_23 = np.array([
    np.linspace(0, -1/3, n_p),
    np.linspace(0, 1/6, n_p),
    np.linspace(0, 1/6, n_p)
    ])

# Convert into II and III coordinates
II_23 = x_23[0, :]**2 + x_23[0, :]*x_23[1, :] + x_23[1, :]**2
III_23 = -x_23[0, :]*x_23[1, :] * (x_23[0, :] + x_23[1, :])

# Curve from 1 to 2
x_12 = np.array([
    np.linspace(2/3, 1/6, n_p),
    np.linspace(-1/3, -1/3, n_p),
    1/3 - np.linspace(2/3, 1/6, n_p)
    ])

# Convert into II and III coordinates
II_12 = x_12[0, :]**2 + x_12[0, :]*x_12[1, :] + x_12[1, :]**2
III_12 = -x_12[0, :]*x_12[1, :] * (x_12[0, :] + x_12[1, :])

# No need for the 2 last dimension of the array (x and y dimension)
# A is flatenned into a 3x3xn_vectors array
a_ij = np.reshape(A_ij, [3, 3, np.shape(A_ij)[2]*np.shape(A_ij)[3]]).T

# Compute eigen values of the anisotropic tensor
eig_vals = np.linalg.eigvals(a_ij).T
eig_vals = eig_vals - eig_vals.mean(axis=0)[np.newaxis, :]
eig_vals = np.sort(eig_vals, axis=0)[::-1, :]

# Convert to II and III coordinates
II = eig_vals[0, :]**2 + eig_vals[0, :]*eig_vals[1, :] + eig_vals[1, :]**2
III = -eig_vals[0, :]*eig_vals[1, :] * (eig_vals[0, :] + eig_vals[1, :])


# Plot and save
lw = 1
color = 'black'

fig, ax = plt.subplots(figsize=(10, 5), dpi=300, layout='constrained')
ax.plot(III_13, II_13, lw=lw, c=color)
ax.plot(III_23, II_23, lw=lw, c=color)
ax.plot(III_12, II_12, lw=lw, c=color)
ax.set_xlim(-0.02, 0.08)
ax.set_ylim(-0.01, 0.35)
ax.set_xlabel(r'$III$')
ax.set_ylabel(r'$II$')
ax.scatter(III.reshape(n_y, n_x)[border_cut:-border_cut, :],
           II.reshape(n_y, n_x)[border_cut:-border_cut, :],
           s=10, facecolor='None', edgecolor='r')
fig.set_constrained_layout_pads()
fig.savefig(Fol_Plots + os.sep + 'Ex3_Lumley_triangle.png')
