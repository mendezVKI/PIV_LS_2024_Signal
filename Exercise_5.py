import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
# from spicy import spicy
from shapely import geometry
from time import time



fontsize = 17.8
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize-3
plt.rcParams['font.size'] = fontsize
plt.rcParams["text.latex.preamble"] = r"\usepackage{bm}"  # Include the bm package
plt.rcParams['image.cmap'] = 'viridis'

import sys
sys.path.insert(0, 'spicy_newrelease')
from spicy_vki.spicy import Spicy


# Generate the output folder
Fol_Plots = 'plots_exercise_5'
if not os.path.exists(Fol_Plots):
    os.makedirs(Fol_Plots)

#%% Data loading

data = np.load('JET_PTV.npz')
X, Y, U, V = data['X'], data['Y'], data['U'], data['V']

np.random.seed(42)

n_p = 750000

idcs = np.arange(X.shape[0])
np.random.shuffle(idcs)
idcs = idcs[:n_p]

X = X[idcs]
Y = Y[idcs]
U = U[idcs]
V = V[idcs]

bounds = [90, 140, 20, 120]
x_min, x_max, y_min, y_max = bounds
scaling = max(x_max - x_min, y_max - y_min)


# We also load the PIV grid to compute the RBF solution on it. Note though that this could be
# any set of points, uniform or not
data = np.load('Exercise_3_PIV_Data' + os.sep + 'piv_data.npz')
X_Piv = data['X']
Y_Piv = data['Y']

# this is for plotting purpose
x = np.unique(X_Piv)
y = np.unique(Y_Piv)
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


#%% Pruning particles in training data

# Here we define different refinement regions to use more particles where there is stronger turbulence.
# Outside of the jet, many particles will not actually help the regression, since turbulence is quite low.
# Thus, we prune particles from the core of the jet and the outside.
# These regions are unknown beforehand, but we can estimate them from a preliminary binning of PIV analysis

# These regions are a bit abstract, so it is much better to plot them
refinement_1 = np.array([
        [x_min, x_max, x_max, x_min],
        [105, 115, 23, 35],
    ])
polygon_points_1 = geometry.Polygon(refinement_1.T)


refinement_2 = np.array([
        [x_min, x_max, x_max, x_min],
        [85, 75, 63, 51],
    ])
polygon_points_2 = geometry.Polygon(refinement_2.T)


data_stack = np.stack((X, Y))

# This is a trick to not compute for every point whether it is inside the polygon. Very useful for large datasets
in_box_1 = np.logical_and(
    np.logical_and(data_stack[0, :] >= refinement_1[0, :].min(),
                   data_stack[0, :] <= refinement_1[0, :].max()),
    np.logical_and(data_stack[1, :] >= refinement_1[1, :].min(),
                   data_stack[1, :] <= refinement_1[1, :].max())
    )
in_polygon_1 = np.zeros(data_stack.shape[1], dtype=bool)
in_polygon_1[in_box_1] = np.array([polygon_points_1.contains(geometry.Point(p)) for p in data_stack[:, in_box_1].T])


in_box_2 = np.logical_and(
    np.logical_and(data_stack[0, in_polygon_1] >= refinement_2[0, :].min(),
                   data_stack[0, in_polygon_1] <= refinement_2[0, :].max()),
    np.logical_and(data_stack[1, in_polygon_1] >= refinement_2[1, :].min(),
                   data_stack[1, in_polygon_1] <= refinement_2[1, :].max())
    )
in_polygon_2 = np.zeros(data_stack[:, in_polygon_1].shape[1], dtype=bool)
in_polygon_2[in_box_2] = np.array([polygon_points_2.contains(geometry.Point(p)) for p in data_stack[:, in_polygon_1][:, in_box_2].T])


# Extract the sets of points
X_out_glo = X[~in_polygon_1]
Y_out_glo = Y[~in_polygon_1]
U_out_glo = U[~in_polygon_1]
V_out_glo = V[~in_polygon_1]

X_shear_glo = X[in_polygon_1][~in_polygon_2]
Y_shear_glo = Y[in_polygon_1][~in_polygon_2]
U_shear_glo = U[in_polygon_1][~in_polygon_2]
V_shear_glo = V[in_polygon_1][~in_polygon_2]

X_core_glo = X[in_polygon_1][in_polygon_2]
Y_core_glo = Y[in_polygon_1][in_polygon_2]
U_core_glo = U[in_polygon_1][in_polygon_2]
V_core_glo = V[in_polygon_1][in_polygon_2]


fig, ax = plt.subplots(figsize=(4, 7), dpi=300)
ax.scatter(X_out_glo[::10], Y_out_glo[::10])
ax.scatter(X_shear_glo[::10], Y_shear_glo[::10])
ax.scatter(X_core_glo[::10], Y_core_glo[::10])
ax.plot(*polygon_points_1.exterior.xy, c='k')
ax.plot(*polygon_points_2.exterior.xy, c='r')
ax.set_aspect(1)
ax.set_xlabel(r'$x$\,[mm]')
ax.set_ylabel(r'$y$\,[mm]')
fig.tight_layout()
fig.savefig(Fol_Plots + os.sep + 'Ex5_regions_points.png')


np.random.seed(42)

# Here, we do the pruning in each region. These are the fractions of particles which are kept in each area
fraction_out = 0.6
fraction_shear = 1.0
fraction_core = 0.4

idcs_out = np.arange(X_out_glo.shape[0])
np.random.shuffle(idcs_out)
idcs_out = idcs_out[:int(X_out_glo.shape[0] * fraction_out)]

X_out = X_out_glo[idcs_out]
Y_out = Y_out_glo[idcs_out]
U_out = U_out_glo[idcs_out]
V_out = V_out_glo[idcs_out]


idcs_shear = np.arange(X_shear_glo.shape[0])
np.random.shuffle(idcs_shear)
idcs_shear = idcs_shear[:int(X_shear_glo.shape[0] * fraction_shear)]

X_shear = X_shear_glo[idcs_shear]
Y_shear = Y_shear_glo[idcs_shear]
U_shear = U_shear_glo[idcs_shear]
V_shear = V_shear_glo[idcs_shear]


idcs_core = np.arange(X_core_glo.shape[0])
np.random.shuffle(idcs_core)
idcs_core = idcs_core[:int(X_core_glo.shape[0] * fraction_core)]

X_core = X_core_glo[idcs_core]
Y_core = Y_core_glo[idcs_core]
U_core = U_core_glo[idcs_core]
V_core = V_core_glo[idcs_core]


X_train = np.concatenate((X_out, X_shear, X_core))
Y_train = np.concatenate((Y_out, Y_shear, Y_core))
U_train = np.concatenate((U_out, U_shear, U_core))
V_train = np.concatenate((V_out, V_shear, V_core))

print(str(X_core.size) + ' particles in jet core')
print(str(X_shear.size) + ' particles in shear layer')
print(str(X_out.size) + ' particles outside of the jet')


#%% Refinement areas clustering

# Here, we define refinement regions for the clustering. Like for the training data
# we also want to refine the basis in regions of strong gradients. Again, this knowledge
# can come from a coarse binning or PIV analysis

# first shear layer
refinement_rbf_1 = np.array([
        (np.array([x_min, x_max, x_max, x_min]) - x_min) / scaling,
        (np.array([105, 115, 75, 85]) - y_min) / scaling,
    ])
poly_refinement_1 = geometry.Polygon(refinement_rbf_1.T)

# second shear layer
refinement_rbf_2 = np.array([
        (np.array([x_min, x_max, x_max, x_min]) - x_min) / scaling,
        (np.array([51, 63, 23, 35]) - y_min) / scaling,
    ])
poly_refinement_2 = geometry.Polygon(refinement_rbf_2.T)

# jet core
refinement_rbf_3 = np.array([
        (np.array([x_min, x_max, x_max, x_min]) - x_min) / scaling,
        (np.array([85, 75, 63, 51]) - y_min) / scaling,
    ])
poly_refinement_3 = geometry.Polygon(
    refinement_rbf_3.T
    )

# plot the result and every 10th training point
fig, ax = plt.subplots(figsize=(4, 7), dpi=300)
ax.scatter(
    (X_train[::10] - x_min) / scaling,
    (Y_train[::10] - y_min) / scaling,
    c=U_train[::10]
    )
ax.plot(*poly_refinement_1.exterior.xy, c='k')
ax.plot(*poly_refinement_2.exterior.xy, c='k')
ax.plot(*poly_refinement_3.exterior.xy, c='r')
ax.set_aspect(1)
ax.set_xlabel(r'$x_\textrm{norm}\,[-]$')
ax.set_ylabel(r'$y_\textrm{norm}\,[-]$')
fig.tight_layout()
fig.savefig(Fol_Plots + os.sep + 'Ex5_regions_basis.png')

#%% Regression of mean flow

# Average number of particles per basis in each refinement level
refines = [150, 150, 450, 1500]
eps_l = 0.9

SP = Spicy(
    points=[
        (X_train - x_min) / scaling,
        (Y_train - y_min) / scaling
        ],
    data=[U_train, V_train],
    basis='gauss',
    model='scalar'
    )

SP.collocation(
    n_K=refines,
    Areas=[poly_refinement_1, poly_refinement_2, poly_refinement_3, None],
    r_mM=[0.05, 0.8],
    eps_l=eps_l
    )

# Visualize the reinfed bases
SP.plot_RBFs(level=0)  # acts on first shear layer
SP.plot_RBFs(level=1)  # acts on second shear layer
SP.plot_RBFs(level=2)  # acts on core
SP.plot_RBFs(level=3)  # acts everywhere

SP.Assembly_Regression()

SP.Solve(K_cond=1e11)

U_sol, V_sol = SP.get_sol(
    points=[
        (X_Piv.ravel() - x_min) / scaling,
        (Y_Piv.ravel() - y_min) / scaling
        ]
    )

U_sol = U_sol.reshape(n_y, n_x)
V_sol = V_sol.reshape(n_y, n_x)

#%% (1) Plot the mean

u_min, u_max = 0, 225
v_min, v_max = -13, 13

# plot u and v fields
fig, axes = plt.subplots(1, 2, figsize=(8, 5), dpi=300, sharex=True, sharey=True, layout='constrained')

# u_mean field
clb = axes[0].pcolormesh(X_pcm, Y_pcm, U_sol, cmap=plt.get_cmap('viridis', lut=15),
                         vmin=u_min, vmax=u_max)
cbar = fig.colorbar(clb, ax=axes[0], pad=0.02, label=r'$u$\,[mm/s]')
axes[0].set_xlabel(r'$x$\,[mm]')
axes[0].set_ylabel(r'$y$\,[mm]')
axes[0].set_aspect(1)

# v_mean field
clb = axes[1].pcolormesh(X_pcm, Y_pcm, V_sol, cmap=plt.get_cmap('viridis', lut=15),
                      vmin=v_min, vmax=v_max
                     )
cbar = fig.colorbar(clb, ax=axes[1], pad=0.02, label=r'$v$\,[mm/s]')
axes[1].set_xlabel(r'$x$\,[mm]')
axes[1].set_aspect(1)

fig.set_constrained_layout_pads(wspace=0.05, w_pad=0.05)

# save figure
fig.savefig(Fol_Plots + os.sep + 'Ex5_u_mean_v_mean_PIV.png')



#%%
# Get the mean velocity in the training data to subtract
U_mean_train, V_mean_train = SP.get_sol(
        points=[
            (X_train - x_min) / scaling,
            (Y_train - y_min) / scaling
            ]
        )

u_train_prime = U_train - U_mean_train
v_train_prime = V_train - V_mean_train

# Compute the field of correlations
uu_train = u_train_prime * u_train_prime
vv_train = v_train_prime * v_train_prime
uv_train = u_train_prime * v_train_prime

SP_stat = Spicy(
    points=[
        (X_train - x_min) / scaling,
        (Y_train - y_min) / scaling
        ],
    data=[uu_train, vv_train, uv_train],
    basis='gauss',
    model='scalar'
    )
SP_stat.collocation(
    n_K=refines,
    Areas=[poly_refinement_1, poly_refinement_2, poly_refinement_3, None],
    r_mM=[0.05, 0.8],
    eps_l=eps_l
    )
SP_stat.Assembly_Regression()
# We borrow the cholesky factorization from the mean flow since the training data is the same
# Careful! Spicy rescales your data range internally, so we need to adapt this
SP_stat.L_A = SP.L_A
SP_stat.b_1 = SP_stat.b_1 * SP_stat.scale_U / SP.scale_U

SP_stat.Solve(K_cond=1e11)
uu_sol, vv_sol, uv_sol = SP_stat.get_sol(
        points=[
            (X_Piv.ravel() - x_min) / scaling,
            (Y_Piv.ravel() - y_min) / scaling
            ]
        )

uu_sol = uu_sol.reshape(n_y, n_x)
vv_sol = vv_sol.reshape(n_y, n_x)
uv_sol = uv_sol.reshape(n_y, n_x)


#%%

lims = [
        [0, 12],
        [-0.5, 0.5],
        [0, 12],
        [0, 5],
        [0, 5],
        [-2, 2]
        ]

method_idx = 1

plots = [
    U_sol,
    V_sol,
    np.sqrt(U_sol**2 + V_sol**2),
    uu_sol,
    vv_sol,
    uv_sol,
    ]


cmap = plt.get_cmap('viridis', lut=15)
cmap_sym = plt.get_cmap('coolwarm', lut=15)

cmaps = [
    cmap,
    cmap_sym,
    cmap,
    cmap,
    cmap,
    cmap_sym
    ]

titles = [
    'U',
    'V',
    'magnitude',
    'uu',
    'vv',
    'uv',
    ]

fig, axes = plt.subplots(figsize=(15, 10), dpi=300, ncols=3, nrows=2, sharex=True, sharey=True)
for ax, plot, cmap, title, lim in zip(axes.ravel(), plots, cmaps, titles, lims):
    ax.imshow(plot.T,
              # vmin=lim[0], vmax=lim[1],
              cmap=cmap)
    ax.set_title(title)

#%%

fill_value = np.zeros(uu_sol.size)

R_ij = np.array([
    [uu_sol.ravel(), uv_sol.ravel(), fill_value     ],
    [uv_sol.ravel(), vv_sol.ravel(), fill_value     ],
    [fill_value,      fill_value,      vv_sol.ravel()]
    ])

k = (np.sum(np.diagonal(R_ij), axis=1) / 2).reshape(n_y, n_x)

TI = 1/ np.max(U_sol) * np.sqrt(2/3 * k)

# plot and save
fig, axes = plt.subplots(figsize=(8, 4), dpi=300, layout='constrained', ncols=2)
clb = axes[0].pcolormesh(X_pcm, Y_pcm, k, cmap=plt.get_cmap('viridis', lut=15),
                          vmin=0, vmax=1600
                         )
cbar = fig.colorbar(clb, ax=axes[0], pad=0.02, shrink=1, label=r'$k$\,[mm²/s²]')
axes[0].set_xlabel(r'$x$\,[mm]', fontsize=fontsize)
axes[0].set_ylabel(r'$y$\,[mm]')
axes[0].set_aspect(1)

clb = axes[1].pcolormesh(X_pcm, Y_pcm, TI*100, cmap=plt.get_cmap('viridis', lut=15),
                          vmin=0, vmax=16
                         )
cbar = fig.colorbar(clb, ax=axes[1], pad=0.02, shrink=1, label=r'TI\,[\%]')
axes[1].set_xlabel(r'$x$\,[mm]')
axes[1].set_aspect(1)
fig.set_constrained_layout_pads()
fig.savefig(Fol_Plots + os.sep + 'Ex5_TKE.png')

#%% (3) Anisotrpy

# compute the anisotropic tensor
A_ij = (R_ij / (2*k.ravel()[np.newaxis, :]) - np.diag(np.full(3, 1/3))[:, :, np.newaxis]).transpose(2, 0, 1)

# norm of the tensor
A_norm = np.linalg.norm(A_ij, axis=(1, 2)).reshape(n_y, n_x)

# plot and save
fig, ax = plt.subplots(figsize=(3.5, 4), dpi=300, layout='constrained')
clb = ax.pcolormesh(X_pcm, Y_pcm, A_norm, cmap=plt.get_cmap('viridis', lut=15), vmin=0, vmax=0.4)
cbar = fig.colorbar(clb, ax=ax, pad=0.02, shrink=1, label=r'$||A||$\,[-]')
ax.set_xlabel(r'$x$\,[mm]')
ax.set_ylabel(r'$y$\,[mm]')
ax.set_aspect(1)
fig.set_constrained_layout_pads()
fig.savefig(Fol_Plots + os.sep + 'Ex5_A_tensor.png')





#%% (4) Lumley triangle

# We cut some parts at the boundary since these regions are particularly delicate for the
# analysis of the lumley triangle. This is consistent between PIV and PTV
border_cut = 3

# Eigenvalues of A tensor
eig_vals = np.linalg.eigvals(A_ij).T
eig_vals = eig_vals - eig_vals.mean(axis=0)[np.newaxis, :]
eig_vals = np.sort(eig_vals, axis=0)[::-1, :]


# Convert to II and III coordinates
II = eig_vals[0, :]**2 + eig_vals[0, :]*eig_vals[1, :] + eig_vals[1, :]**2
III = -eig_vals[0, :]*eig_vals[1, :] * (eig_vals[0, :] + eig_vals[1, :])

# Do the mentioned cropping
II = II.reshape(n_y, n_x)[border_cut:-border_cut]
III = III.reshape(n_y, n_x)[border_cut:-border_cut]

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

# Plot and save

lw = 3
color = 'black'

fig, ax = plt.subplots(figsize=(10, 5), dpi=300, layout='constrained')
ax.plot(III_13, II_13, lw=lw, c=color)
ax.plot(III_23, II_23, lw=lw, c=color)
ax.plot(III_12, II_12, lw=lw, c=color)

ax.scatter(III, II, s=10, facecolor='None', edgecolor='r')
ax.set_xlim(-0.02, 0.08)
ax.set_ylim(-0.01, 0.35)
ax.set_xlabel(r'$III$')
ax.set_ylabel(r'$II$')
fig.set_constrained_layout_pads()

fig.tight_layout()
fig.savefig(Fol_Plots + os.sep + 'Ex5_Lumley_triangle.png')

