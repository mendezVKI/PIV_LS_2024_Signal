"""
Utilities functions to deal with the basis and things like clustering, clipping etc.

Authors: Manuel Ratz
"""

import numpy as np

# these functions are used for the clustering and collocation
from sklearn.neighbors import NearestNeighbors
# Function for the k means clustering
from sklearn.cluster import MiniBatchKMeans

from scipy.stats.qmc import Halton
from ..utils._extnumpy import meshgrid_ravel

def get_shape_parameter_and_diameter(r_mM, c_k, basis):
    """
    This function clips the shape parameters of the RBFs and assigns their diameters.

    ----------------------------------------------------------------------------------------------------------------
    Parameters
    ----------
    :param c_k: 1D numpy.ndarray
        Array containing the shape parameters of the RBFs
    :param r_mM: list
        Minimum and maximum radius of the RBFs
    :param basis: str
        Type of basis function, must be c4 or Gaussian

    """
    if basis == 'gauss':
        # Set the max and min values of c_k
        c_min = 1 / (r_mM[1]) * np.sqrt(np.log(2))
        c_max = 1 / (r_mM[0]) * np.sqrt(np.log(2))
        # crop to the minimum and maximum value
        c_k[c_k < c_min] = c_min
        c_k[c_k > c_max] = c_max
        # for plotting purposes, store also the diameters
        d_k = 2 / c_k * np.sqrt(np.log(2))

    elif basis == 'c4':
        c_min = r_mM[0] / np.sqrt(1 - 0.5 ** 0.2)
        c_max = r_mM[1] / np.sqrt(1 - 0.5 ** 0.2)
        # crop to the minimum and maximum value
        c_k[c_k < c_min] = c_min
        c_k[c_k > c_max] = c_max
        # for plotting purposes, store also the diameters
        d_k = 2 * c_k * np.sqrt(1 - 0.5 ** 0.2)

    else:
        # Leave other options for future implementations.
        print('This basis is currently not implemented')
        c_k = None
        d_k = None

    return c_k, d_k

def _get_shape(collocation_points):
    neighbors = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(collocation_points)
    distances, indices = neighbors.kneighbors(collocation_points)
    sigma_level = distances[:, 1]
    sigma_level[sigma_level == 0] = np.max(sigma_level[sigma_level != 0])

    return sigma_level
