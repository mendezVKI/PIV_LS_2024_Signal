import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from spicy import spicy
from shapely import geometry




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


x_min = 352
x_max = 1472
y_min = 16
y_max = 1184

sigma = 0.66
n_bins_x = 141

# Compute the number of collocation points along y and create the grid
n_bins_y = int(np.rint(n_bins_x * (y_min - y_max) / (x_min - x_max)))
x_bin = np.linspace(x_min, x_max, n_bins_x)
y_bin = np.linspace(y_min, y_max, n_bins_y)

spacing_x = x_bin[1] - x_bin[0]
spacing_y = y_bin[1] - y_bin[0]

X_bin, Y_bin = np.meshgrid(x_bin, y_bin)
X_bin = X_bin.T.ravel()
Y_bin = Y_bin.T.ravel()

#%%


df = pl.read_parquet('data_filt.parquet')
#%%

np.random.seed(42)

X = df['x'].to_numpy()
Y = df['y'].to_numpy()
U = df['u'].to_numpy()
V = df['v'].to_numpy()

n_p = 1000000 # TODO play with this parameter. It defines the max number of points which can be taken from the ensemble

idcs = np.arange(X.shape[0])
np.random.shuffle(idcs)
idcs = idcs[:n_p]

X_train = X[idcs]
Y_train = Y[idcs]
U_train = U[idcs]
V_train = V[idcs]

X = X[idcs]
Y = Y[idcs]
U = U[idcs]
V = V[idcs]

bounds = [320, 1504, 0, 1200]
x_min, x_max, y_min, y_max = bounds
scaling = max(x_max - x_min, y_max - y_min)

#%%

# ax.scatter(X, Y, c=U)

refinement_1 = np.array([
        [x_min, x_max, x_max, x_min],
        [900, 1000, 200, 300],
    ])

polygon_1 = geometry.Polygon(
    refinement_1.T
    )


refinement_2 = np.array([
        [x_min, x_max, x_max, x_min],
        [750, 650, 550, 450],
    ])

polygon_2 = geometry.Polygon(
    refinement_2.T
    )


data_stack = np.stack((X, Y))

in_box_1 = np.logical_and(
    np.logical_and(data_stack[0, :] >= refinement_1[0, :].min(),
                   data_stack[0, :] <= refinement_1[0, :].max()),
    np.logical_and(data_stack[1, :] >= refinement_1[1, :].min(),
                   data_stack[1, :] <= refinement_1[1, :].max())
    )

in_polygon_1 = np.zeros(data_stack.shape[1], dtype=bool)
in_polygon_1[in_box_1] = np.array([polygon_1.contains(geometry.Point(p)) for p in data_stack[:, in_box_1].T])


in_box_2 = np.logical_and(
    np.logical_and(data_stack[0, in_polygon_1] >= refinement_2[0, :].min(),
                   data_stack[0, in_polygon_1] <= refinement_2[0, :].max()),
    np.logical_and(data_stack[1, in_polygon_1] >= refinement_2[1, :].min(),
                   data_stack[1, in_polygon_1] <= refinement_2[1, :].max())
    )

in_polygon_2 = np.zeros(data_stack[:, in_polygon_1].shape[1], dtype=bool)
in_polygon_2[in_box_2] = np.array([polygon_2.contains(geometry.Point(p)) for p in data_stack[:, in_polygon_1][:, in_box_2].T])


#%%
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


#%%
np.random.seed(42)


fraction_out = 0.4
fraction_shear = 1.0
fraction_core = 0.2

idcs_out = np.arange(X_out_glo.shape[0])
np.random.shuffle(idcs_out)
idcs_out = idcs_out[:int(X_out_glo.shape[0] * fraction_out)]

X_out = X_out_glo[idcs_out]
Y_out = Y_out_glo[idcs_out]
U_out = U_out_glo[idcs_out]
V_out = V_out_glo[idcs_out]


idcs_shear = np.arange(X_shear_glo.shape[0])
np.random.shuffle(idcs_shear)
idcs_shear = idcs_shear[:int(X_shear_glo.shape[0] * fraction_core)]

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

print(X_core.size)
print(X_shear.size)
print(X_out.size)


#%%

refinement_rbf_1 = np.array([
        (np.array([x_min, x_max, x_max, x_min]) - x_min) / scaling,
        (np.array([900, 1000, 650, 750]) - y_min) / scaling,
    ])

poly_refinement_1 = geometry.Polygon(
    refinement_rbf_1.T
    )

refinement_rbf_2 = np.array([
        (np.array([x_min, x_max, x_max, x_min]) - x_min) / scaling,
        (np.array([450, 550, 200, 300]) - y_min) / scaling,
    ])

poly_refinement_2 = geometry.Polygon(
    refinement_rbf_2.T
    )

refinement_rbf_3 = np.array([
        (np.array([x_min, x_max, x_max, x_min]) - x_min) / scaling,
        (np.array([750, 650, 550, 450]) - y_min) / scaling,
    ])

poly_refinement_3 = geometry.Polygon(
    refinement_rbf_3.T
    )

sols_train = []

sols = []
for field in [
        U_train,
        V_train
        ]:

    SP = spicy(
        grid_point=[
            (X_train - x_min) / scaling,
            (Y_train - y_min) / scaling
            ],
        data=[field],
        basis='gauss'
        )

    SP.clustering(
        n_K=[100, 100, 300, 1000],
        Areas=[poly_refinement_1, poly_refinement_2, poly_refinement_3, None],
        r_mM=[0.05, 0.8],
        eps_l=0.9
        )


    SP.scalar_constraints()

    SP.Assembly_Regression()

    SP.Solve(K_cond=1e12)

    sols.append(
        SP.Get_Sol(
            grid=[
                (X_bin - x_min) / scaling,
                (Y_bin - y_min) / scaling
                ]
            )
        )
    sols_train.append( SP.Get_Sol(
            grid=[
                (X_train - x_min) / scaling,
                (Y_train - y_min) / scaling
                ]
            )
        )

U_sol = sols[0].reshape(n_bins_x, n_bins_y)
V_sol = sols[1].reshape(n_bins_x, n_bins_y)

U_mean_train = sols_train[0]
V_mean_train = sols_train[1]

#%%

u_train_prime = U_train - U_mean_train
v_train_prime = V_train - V_mean_train

uu_train = u_train_prime * u_train_prime
vv_train = v_train_prime * v_train_prime
uv_train = u_train_prime * v_train_prime

sols = []
for field in [
        uu_train,
        vv_train,
        uv_train
        ]:
    SP = spicy(
        grid_point=[
            (X_train - x_min) / scaling,
            (Y_train - y_min) / scaling
            ],
        data=[field],
        basis='gauss'
        )
    SP.clustering(
        n_K=[75, 75, 300, 1000],
        Areas=[poly_refinement_1, poly_refinement_2, poly_refinement_3, None],
        r_mM=[0.05, 0.8],
        eps_l=0.9
        )
    SP.scalar_constraints()
    SP.Assembly_Regression()
    SP.Solve(K_cond=1e12)
    sols.append(
        SP.Get_Sol(
            grid=[
                (X_bin.ravel() - x_min) / scaling,
                (Y_bin.ravel() - y_min) / scaling
                ]
            )
        )

uu_sol = sols[0].reshape(n_bins_x, n_bins_y)
vv_sol = sols[1].reshape(n_bins_x, n_bins_y)
uv_sol = sols[2].reshape(n_bins_x, n_bins_y)



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

fig, axes = plt.subplots(figsize=(15, 10), dpi=100, ncols=3, nrows=2, sharex=True, sharey=True)
for ax, plot, cmap, title, lim in zip(axes.ravel(), plots, cmaps, titles, lims):
    ax.imshow(plot.T, vmin=lim[0], vmax=lim[1], cmap=cmap)
    ax.set_title(title)

#%%


fill_value = np.zeros(uu_sol.size)

R_ij = np.array([
    [uu_sol.ravel(), uv_sol.ravel(), fill_value     ],
    [uv_sol.ravel(), vv_sol.ravel(), fill_value     ],
    [fill_value,      fill_value,      vv_sol.ravel()]
    ])
R_ij = np.transpose(R_ij, axes=(2, 0, 1))

k = np.sum(np.diagonal(R_ij, 0, 1, 2), axis=1) / 2

a_ij = R_ij / (2*k[:, np.newaxis, np.newaxis]) - np.diag(np.full(3, 1/3))[np.newaxis, :]

eig_vals = np.linalg.eigvals(a_ij).T
eig_vals = eig_vals - eig_vals.mean(axis=0)[np.newaxis, :]
eig_vals = np.sort(eig_vals, axis=0)[::-1, :]


II = eig_vals[0, :]**2 + eig_vals[0, :]*eig_vals[1, :] + eig_vals[1, :]**2
III = -eig_vals[0, :]*eig_vals[1, :] * (eig_vals[0, :] + eig_vals[1, :])


#%% Lumley coordinates

x_1C = np.array([2/3, -1/3, -1/3])
x_2C = np.array([1/6, 1/6, -1/3])
x_3C = np.array([0, 0, 0])

II_1C = x_1C[0]**2 + x_1C[0]*x_1C[1] + x_1C[1]**2
II_2C = x_2C[0]**2 + x_2C[0]*x_2C[1] + x_2C[1]**2
II_3C = x_3C[0]**2 + x_3C[0]*x_3C[1] + x_3C[1]**2

III_1C = -x_1C[0]*x_1C[1] * (x_1C[0] + x_1C[1])
III_2C = -x_2C[0]*x_2C[1] * (x_2C[0] + x_2C[1])
III_3C = -x_3C[0]*x_3C[1] * (x_3C[0] + x_3C[1])

n_p = 101

x_13 = np.array([
    np.linspace(0, 2/3, n_p),
    np.linspace(0, -1/3, n_p),
    np.linspace(0, -1/3, n_p),
    ])

II_13 = x_13[0, :]**2 + x_13[0, :]*x_13[1, :] + x_13[1, :]**2
III_13 = -x_13[0, :]*x_13[1, :] * (x_13[0, :] + x_13[1, :])

x_23 = np.array([
    np.linspace(0, -1/3, n_p),
    np.linspace(0, 1/6, n_p),
    np.linspace(0, 1/6, n_p)
    ])

II_23 = x_23[0, :]**2 + x_23[0, :]*x_23[1, :] + x_23[1, :]**2
III_23 = -x_23[0, :]*x_23[1, :] * (x_23[0, :] + x_23[1, :])

x_12 = np.array([
    np.linspace(2/3, 1/6, n_p),
    np.linspace(-1/3, -1/3, n_p),
    1/3 - np.linspace(2/3, 1/6, n_p)
    ])

II_12 = x_12[0, :]**2 + x_12[0, :]*x_12[1, :] + x_12[1, :]**2
III_12 = -x_12[0, :]*x_12[1, :] * (x_12[0, :] + x_12[1, :])


#%% Plot points inside
from shapely import geometry

polygon = geometry.Polygon(
    np.array([
        np.concatenate([III_13[::-1], III_23, III_12]),
        np.concatenate([II_13[::-1], II_23, II_12])
        ]).T
    )

fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
in_lumley = np.array([polygon.contains(geometry.Point(p)) for p in np.stack((III, II)).T])

invalid = np.zeros(U_sol.shape).ravel()
invalid[~in_lumley] = 1
invalid = invalid.reshape(U_sol.shape)
ax.imshow(invalid.T)



#%% Plot the Lumley triangle
lw = 3
color = 'black'

fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
ax.plot(III_13, II_13, lw=lw, c=color)
ax.plot(III_23, II_23, lw=lw, c=color)
ax.plot(III_12, II_12, lw=lw, c=color)
ax.set_xlim(-0.02, 0.08)
ax.set_ylim(-0.01, 0.35)

ax.scatter(III, II, s=10, facecolor='None', edgecolor='r')
ax.set_title('Spicy')

fig.tight_layout()
