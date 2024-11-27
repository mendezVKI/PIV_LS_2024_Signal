# -*- coding: utf-8 -*-
"""
Latest update on Thu Apr 27 13:40:44 2023

@author: mendez, ratz, sperotto
"""

import numpy as np # used in all computations

# these functions are used for the clutering and collocation
from sklearn.neighbors import NearestNeighbors
# Function for the k means clusering
from sklearn.cluster import MiniBatchKMeans

# Note: there is a warning from kmeans when running on windows.
# This should fix it
import warnings
warnings.filterwarnings('ignore')

# Matplotlib for the plotting functions:
import matplotlib.pyplot as plt

# function useful for computing smallsest and largest eig:
from scipy.sparse.linalg import eigsh
# we use scipy linalg for cholesky decomposition, solving linear systems etc
from scipy import linalg

# We use this function to handle polygonal refinements areas
from shapely import geometry

# Sample method for semi-random points
from scipy.stats import qmc
from scipy.spatial.distance import cdist

from .utils._basis_Harmonic import Phi_H_2D, Phi_H_2D_x, Phi_H_2D_y, Phi_H_2D_Laplacian
from .utils._basis_Harmonic import Phi_H_3D, Phi_H_3D_x, Phi_H_3D_y, Phi_H_3D_z, Phi_H_3D_Laplacian
from .utils._basis_RBF import Phi_RBF_2D, Phi_RBF_2D_Deriv, Phi_RBF_3D, Phi_RBF_3D_Deriv
from .utils._input_checks import check_data, check_number, check_bounds
from .utils._basis_collocation import get_shape_parameter_and_diameter
from .utils._extnumpy import meshgrid_ravel, stack_to_block_matrix

class Spicy:
    """
    SPICY (Super-resolution and Pressure from Image veloCimetrY) is a software
    developed at the von Karman Institute to perform data assimilation by means
    of Radial Basis Functions (RBF). The framework works both for structured and
    unstructered data. Currently, the main application is to perform a regression
    of image velocimetry data and then solve the pressure equation. However, the framework
    can be readily extended to regression of other fields (e.g. temperature fields).

    The original article by Sperotto et al. (2022) can be found at:
    https://arxiv.org/abs/2112.12752

    YouTube channel with hands-on tutorials can be found at:
    https://www.youtube.com/@spicyVKI

    """
    # 1. Initialize the class with the data
    def __init__(self, data, points, basis='gauss', ST=None, model='laminar', verbose=2):
        """
        Initialization of an instance of the spicy class.

        :type data: list of 1D numpy.ndarray
        :param data:
            Input data for the regression or a Poisson problem.

            For a regression, it has to have specific shaps

            * If ``model`` = 'laminar' and ``data`` = [X_G, Y_G], it is [u, v], i.e. the
              components of a 2D velocity field
            * If ``model`` = 'laminar' and ``data`` = [X_G, Y_G, Z_G], it is [u, v, w], i.e. the
              components of a 3D velocity field
            * If ``model`` = 'scalar', ``data`` can be [u1, u2, ..., uN]. Multiple regressions
              can be done at once in this way and they share computations. However, they must
              share the same coordinates and the constraints have to be in the same data points.

            If the instance is to be used to solve the Poisson equation,
            this list contains the forcing term on the RHS of the Poisson equation.
            ``model`` must be 'scalar' in that case.

        :type points: list of 1D numpy.ndarray
        :param points:
            Is a list of arrays containing the points: [X_G ,Y_G] in 2D and
            [X_G, Y_G, Z_G] in 3D.

        :type basis: str
        :param basis: This defines the basis. Currently, the two options are

           - ``'gauss'``, i.e. Gaussian RBFs exp(-c_r**2*d(x))
           - ``'c4'``, i.e. C4 RBFs (1+d(x+)/c_r)**5(1-d(x+)/c_r)**5

        :type ST: list of 1D numpy.ndarray
        :param ST:
            Is a list of arrays collecting Reynolds stresses. This is empty if
            the model is 'scalar' or 'laminar'. If the model is RANSI (isotropic), it
            contains [uu']. If the model is RANSA (anisotropic), it contains [uu, vv, uv]
            in 2D and [uu, vv, ww, uv, uw, vw] in 3D.

        :type model: str
        :param model:
            Must be one of 'laminar' or 'scalar'. See input ``data`` on how to use it.

            .. versionadded:: 1.1.0

        :type verbose: int
        :param verbose:
            Sets the verbosity to print for the user

            - 0: no information printed
            - 1: updates on loops and multiple regressions
            - 2: details of each step (should only be used for a small number of regressions)

            .. versionadded:: 1.1.0

        General attributes:
            X_G, Y_G, Z_G: coordinates of the point in which the data is available
            u : function to learn or u component in case of velocity field
            v: v component in case of velocity field (absent for scalar)
            w: w component in case of velocity field (absent for scalar)

        If constraints are assigned:
            X_D, Y_D, Z_D: coordinates of the points with Dirichlet (D) conditions
            c_D: values of the D conditions

            X_N, Y_N, Z_N: coordinates of the points with Neumann (N) conditions
            n_x, n_y, n_z: normal versors where N conditions are introduced
            c_N_X, c_N_Y, c_N_Z: values of the N conditions

            X_Div, Y_Div, Z_Div: coordinates of the points with Div conditions

        If clustering is done:
            r_mM: vector collecting minimum (m) and maximum (M) radious of the RBFs
            eps_l: scalar controlling the value of an RBF at the closest RBF neighbor
            X_C, Y_C, Z_C : coordinates of the cluster centers/collocations
            c_k: shape parameters of the RBFs
            d_k: diameters of the rbfs

        If problem is assembled:
            A: matrix A in the linear system
            B: matrix B in the linear system
            b_1: vector b_1 in the linear systems
            b_2: vector b_2 in the linear system

        If computation is done:
            weights: weights of the RBF regression
            lambda: Lagrange multipliers of the RBF regression
        """

        # Check that the inputs are correct
        if not isinstance(data, list):
            raise ValueError('Input \'data\' must be a list')
        if not isinstance(points, list):
            raise ValueError('Input \'points\' must be a list')
        if not isinstance(basis, str):
            raise ValueError('Input \'basis\' must be a string')
        if not isinstance(model, str):
            raise ValueError('Input \'model\' must be a string')
        if ST is not None and not isinstance(ST, list):
            raise ValueError('Input \'st\' must be None or a list')

        # Check that 'points' and 'data' have contents of the same length
        points_data_length_flag = False
        for i in range(len(points)):
            for j in range(len(data)):
                if len(points[i]) != len(data[j]):
                    points_data_length_flag = True
        if points_data_length_flag:
            raise ValueError('Input \'data\' and \'points\' must contain lists or arrays of mutually equal length')

        # Assign verbosity
        self.verbose = verbose

        # Check that the stress tensor is not given as input, it is not implemented
        if ST is not None:
            raise NotImplementedError('RANSI/RANSA currently not implemented in 3D')

        # Assign the basis
        if basis == 'gauss' or basis == 'c4':
            self.basis = basis
        else:
            raise ValueError('Wrong basis, must be either \'gauss\' or \'c4\'')

        # Assign the model
        if model == 'laminar' or model == 'scalar':
            self.model = model
        else:
            raise ValueError('Wrong model, must be either \'laminar\' or \'scalar\'')

        # Check the length of the points to see if it is 2D or 3D
        if len(points) == 2: # 2D problem
            self.dimension = '2D'
            self.X_G = points[0]
            self.Y_G = points[1]
            self.Z_G = None  # This is for compatability with later functions
            # check the data
            if len(data) == 2 and self.model == 'laminar':  # laminar case
                self.u = data[0][:, np.newaxis]
                self.v = data[1][:, np.newaxis]
            elif len(data) != 2 and self.model == 'laminar':  # Invalid combination
                raise ValueError('When \'points\' is [X_g, Y_g] and \'model\' is laminar, \'data\' must have the'
                                 ' form: [u,v]')
            else:  # scalar case
                self.u = np.array(data).T

        elif len(points) == 3: # 3D problem
            self.dimension = '3D'
            self.X_G = points[0]
            self.Y_G = points[1]
            self.Z_G = points[2]
            if len(data) == 2 and self.model == 'laminar':  # laminar case
                self.u = data[0][:, np.newaxis]
                self.v = data[1][:, np.newaxis]
                self.w = data[2][:, np.newaxis]
            elif len(data) != 2 and self.model == 'laminar':  # Invalid combination
                raise ValueError('When \'points\' is [X_g, Y_g] and \'model\' is laminar, \'data\' must have the'
                                 ' form: [u,v]')
            else:  # scalar case
                self.u = np.array(data).T
        else:
            raise ValueError('Invalid size of input points, currently only implemented in 2D and 3D')

        # Assign the number of data points. This is the same in all cases
        self.n_p = len(self.X_G)

        # We initialize the Cholesky factorization as None. This is needed for the solver later on
        self.L_A = None

        # Initialize n_hb as 0. This is helpful, if we load weights and do evaluations without regressions
        self.n_hb = 0

        return


    # 2. Clustering (this does not depend on the model, but only on the dimension).
    def collocation(self, n_K, Areas=None, bounds=None, method='clustering', r_mM=[0.01, 0.3], eps_l=0.7):
        """
        This function defines the collocation of a set of RBFs using the multi-
        level clustering first introduced in the article. Note that we modified the slightly original formulation
        to ease the programming; see video tutorials for more.
        The function must be run before the constraint definition.

        .. versionchanged:: 1.1.0 Renamed to collocation, clustering is deprecated

        :type n_K: list
        :param n_K:
            This contains the n_k vector in eq (33) in the paper; this is the
            list of expected particles per RBF at each level. For example, if n_K=[4,10],
            it means that the clustering will try to have a first level with RBFs whose size
            seeks to embrace 4 points, while the second level seeks to embrace
            10 points, etc. The length of this vector automatically defines the
            number of levels.

        :type Areas: list
        :param Areas:
            List of the refinement regions for each clustering level. If no
            refinement is needed, then this should be a list of empty
            lists (default option). Currently not implemented in 3D.
            .. versionchanged:: 1.1.0

            Moved to keyword arguments and initialized as None.

        :

        :type r_mM: list of two float values
        :param r_mM: default=[0.01, 0.3].
            This contains the minimum and the maximum RBF's radiuses. This is
            defined as the distance from the collocation point at which the RBF
            value is 0.5.

        :type float: float
        :param eps_l: default=0.7.
            This is the value that a RBF will have at its closest neighbour. It
            is used to define the shape factor from the clustering results.

        """

        # Check the input is correct
        assert type(n_K) == list, 'Clustering levels must be given as a list'
        assert type(r_mM) == list and len(r_mM) == 2, 'r_mM must be a list of length 2'
        assert r_mM[0] < r_mM[1], 'Minimum radius must be smaller than maximum radius'
        assert eps_l < 1 and eps_l > 0, 'eps_l must be between zero and 1'
        assert method in ['clustering', 'semirandom'], 'method must be clustering or halton'

        # Define a dummy for the Areas when they are not used
        if Areas is not None:
            assert len(Areas) == len(n_K), 'Length of Areas must be the same as length of n_K'
        else:
            Areas = [None for _ in n_K]

        # we assign the clustering parameters to self
        # they are needed in the constraints to set the shape parameters for the
        # RBFs which are located at constraint points

        self.r_mM = r_mM
        self.eps_l = eps_l

        # Check if we are dealing with a 2D or a 3D case
        if self.dimension=='2D': # This is 2D

            # Number of levels
            n_l = len(n_K)

            # Loop over the number of levels
            for l in range(n_l):
                # We look for the points that belongs to the given area:
                if Areas[l]:
                    # This means a polygon object is given, so take only points
                    # inside this:
                    poly = Areas[l]
                    List = []    # prepare empty list
                    for j in range(len(self.X_G)): # fill list of points in poly
                        List.append(poly.contains(geometry.Point(self.X_G[j], self.Y_G[j])))
                    # Take only these points as data matrix
                    X_G_c=self.X_G[List]
                    Y_G_c=self.Y_G[List]
                    Data_matrix = np.column_stack((X_G_c, Y_G_c))
                    List=[] # delete the list for safety
                else: # if Areas is empty then all points should be included
                    Data_matrix = np.column_stack((self.X_G, self.Y_G))

                if method == 'clustering':
                    # Define number of clusters
                    clust = int(np.ceil(np.shape(Data_matrix)[0]/ n_K[l]))

                    # Initialize the cluster function
                    model = MiniBatchKMeans(n_clusters=clust, random_state=0)
                    # Run the clustering and return the indices (optional)
                    y_P = model.fit_predict(Data_matrix)
                    # Obtaining the centers of the points
                    collocation_points = model.cluster_centers_

                    # Get the nearest neighbour of each center
                    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(collocation_points)
                    distances, indices = nbrs.kneighbors(collocation_points)
                    sigma1 = distances[:, 1]

                    # Remove all of the clusters which either have a distance of
                    # zero to the nearest neighbor (that would be the same RBF)
                    # and the clusters with only one point in them
                    count = np.bincount(y_P, minlength=clust)
                    sigma1[sigma1 == 0] = np.amax(sigma1[sigma1 != 0])
                    sigma1[count == 1] = np.amax(sigma1)

                elif method == 'semirandom':
                    n_b = round(Data_matrix.shape[0]/n_K[l])
                    # Define sampler and collocation points in [0,1]x [0,1]
                    sampler = qmc.Halton(d=2, scramble=True, seed=42)
                    collocation_points = sampler.random(n=n_b)
                    bounds = self._extract_bounds(bounds)
                    # Define the collocation points
                    collocation_points = qmc.scale(
                        collocation_points,
                        [bounds[0], bounds[2]],
                        [bounds[1], bounds[3]]
                        )

                    # Get the nearest neighbour of each center
                    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(collocation_points)
                    distances, indices = nbrs.kneighbors(collocation_points)
                    sigma1 = distances[:, 1]
                    sigma1[sigma1 == 0] = np.amax(sigma1[sigma1 != 0])

                else:
                    pass

                # Pre-assign the collocation points
                X_C1 = collocation_points[:, 0]
                Y_C1 = collocation_points[:, 1]
                list_Index = np.array([l]*len(X_C1)) # to use also hstack

                # Assign the results to a vector of collocation points
                if l == 0: # If this is the first layer, just assign:
                    X_C = X_C1
                    Y_C = Y_C1
                    sigma = sigma1
                    l_list = list_Index
                else: # Stack onto the existing ones
                    X_C = np.hstack((X_C, X_C1))
                    Y_C = np.hstack((Y_C, Y_C1))
                    sigma = np.hstack((sigma, sigma1))
                    l_list = np.hstack((l_list,list_Index))
                if self.verbose == 2:
                    print('Clustering level ' + str(l) + ' completed')

            # Assign to the class
            self.X_C = X_C
            self.Y_C = Y_C
            # For plotting purposes, we keep track of the scale at which
            # the RBF have been places
            self.Clust_list=l_list



        elif self.dimension == '3D': # This is 3D
            # Stack the coordinates in a matrix:
            Data_matrix = np.column_stack((self.X_G, self.Y_G, self.Z_G))
            # Number of levels
            n_l = len(n_K)

            # Loop over the number of levels
            for l in range(n_l):
                if Areas[l]:
                    print('Warning: Areas currently only work in 2D')
                # Define number of clusters
                Clust = int(np.ceil(self.n_p / n_K[l]))
                # Initialize the cluster function
                model = MiniBatchKMeans(n_clusters=Clust, random_state=0)
                # Run the clustering and return the indices (optional)
                y_P = model.fit_predict(Data_matrix)
                # Obtaining the centers of the points
                Centers = model.cluster_centers_

                # Get the nearest neighbour of each center
                nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(Centers)
                distances, indices = nbrs.kneighbors(Centers)
                sigma1 = distances[:, 1]

                # Remove all of the clusters which either have a distance of
                # zero to the nearest neighbor (that would be the same RBF)
                # and the clusters with only one point in them
                count = np.bincount(y_P, minlength=collocation_points)
                sigma1[sigma1 == 0] = np.amax(sigma1[sigma1 != 0])
                sigma1[count == 1] = np.amax(sigma1)

                # Pre-assign the collocation points
                X_C1 = collocation_points[:,0]
                Y_C1 = collocation_points[:,1]
                Z_C1 = collocation_points[:,2]

                # Assign the results to a vector of collocation points
                if l == 0: # If this is the first layer, just assign:
                    X_C = X_C1
                    Y_C = Y_C1
                    Z_C = Z_C1
                    sigma = sigma1
                else: # Stack onto the existing ones
                    X_C = np.hstack((X_C, X_C1))
                    Y_C = np.hstack((Y_C, Y_C1))
                    Z_C = np.hstack((Z_C, Z_C1))
                    sigma = np.hstack((sigma, sigma1))
                if self.verbose == 2:
                    print('Clustering level ' + str(l) + ' completed')

            # Assign to the class
            self.X_C = X_C
            self.Y_C = Y_C
            self.Z_C = Z_C

        # We conclude with the computation of the shape factors. These depend
        # on the type of RBF but not whether the type is 2D or 3D.
        if self.basis =='gauss':
            # Set the max and min values of c_k
            c_min = 1/(r_mM[1])*np.sqrt(np.log(2))
            c_max = 1/(r_mM[0])*np.sqrt(np.log(2))
            # compute the c_k
            c_k = np.sqrt(-np.log(eps_l))/sigma
            # # crop to the minimum and maximum value
            c_k[c_k < c_min] = c_min
            c_k[c_k > c_max] = c_max
            # for plotting purposes, we store also the diameters
            d_k = 2/c_k*np.sqrt(np.log(2))

        elif self.basis == 'c4':
            # Set the max and min values of c_k
            c_min = r_mM[0] / np.sqrt(1 - 0.5**0.2)
            c_max = r_mM[1] / np.sqrt(1 - 0.5**0.2)
            # compute the c _k
            c_k = sigma / np.sqrt(1 - eps_l**0.2)
            # crop to the minimum and maximum value
            c_k[c_k < c_min] = c_min
            c_k[c_k > c_max] = c_max
            # for plotting purposes, we store also the diameters
            d_k = 2*c_k * np.sqrt(1 - 0.5**0.2)

        self.c_k = c_k
        self.d_k = d_k

        if self.verbose == 2:
            print(str(len(X_C)) + ' RBFs placed through clustering')

        return

    # 3. Constraints.

    # We have two sorts of constraints: scalar and vector.
    # scalar apply to model = scalar and to the poisson solvers.
    # vector apply to all the other models.

    # the scalar ones include: Dirichlet and Neuman.
    # the vector one include: Dirichlet, Neuman and Div free.

    # 3.1 Scalar constraints
    def scalar_constraints(self, DIR=[], NEU=[], extra_RBF=True):
        """
        This functions sets the boundary conditions for a scalar problem. The
        function must be run after the clustering is carried out.

        :type DIR: list of 1D numpy.ndarray
        :param DIR:
            This contains the info for the Dirichlet conditions:

            * If ``dimension`` = '2D', it has to be [X_D, Y_D, c_D1, ..., c_Dn]
            * If ``dimension`` = '3D', it has to be [X_D, Y_D, Z_D, c_D1, ..., c_Dn]

            Here X_D, Y_D, Z_D are the coordinates of the points where the value
            c_D1, ..., c_Dn are set. This allows to reuse constraints for a scalar
            regression, assuming they are set in the same points.

            .. versionchanged:: 1.1.0
                Since multiple regressions can now share computations for scalar
                regressions, the input can now accept a constraint value for
                every individual regression. The API remains the same in the case
                of regressing a single scalar

        :type NEU: list of 1D numpy.ndarray
        :param NEU:
            This contains the info for the Neumann conditions:

            * If ``dimension`` = '2D', it has to be [X_N, Y_N, n_x, n_y, c_N]
            * If ``dimension`` = '3D', it has to be [X_N, Y_N, Z_N, n_x, n_y, n_z, c_N]

            Here X_N, Y_N, Z_N are the coordinates of the points where the values
            cN1, ..., c_Nn are set for the directional derivative along the normal
            direction n_x, n_y, n_z.

            .. versionchanged:: 1.1.0
                Since multiple regressions can now share computations for scalar
                regressions, the input can now accept a constraint value for
                every individual regression. The API remains the same in the case
                of regressing a single scalar

        :type extra_RBF: bool
        :param extra_RBF: default = True
            This is a flag to put extra collocation points where a constraint is
            set. It can improve the solution of the linear system as constraints
            remove degrees of freedom.

        """

        # Check that the inputs are correct
        if self.model == 'laminar':
            raise ValueError('When model is \'laminar\', \'scalar_constraints\' cannot be called')

        # Check that DIR contains have the correct form
        check_data(data=DIR, name='DIR')
        check_data(data=NEU, name='NEU')

        # Check that the length of DIR is correct
        if len(DIR) != self.u.shape[1] + int(self.dimension[0]) and len(DIR) != 0:
            if self.dimension == '2D':
                em = ('When \'dimension\' is \'2D\' and %d quantities are regressed, \'DIR\' must be [X_D, Y_D, c_D1'
                      % self.u.shape[1])
                for i in range(self.u.shape[1] - 1):
                    em += ', c_D%d' % (i + 2)
            else:
                em = ('When \'dimension\' is \'3D\' and %d quantities are regressed, \'DIR\' must be [X_D, Y_D, Z_D, c_D1'
                      % self.u.shape[1])
                for i in range(self.u.shape[1] - 1):
                    em += ', c_D%d' % (i + 2)
            raise ValueError(em + ']')


        # Check that the length of NEU is correct
        if len(NEU) != self.u.shape[1] + 2 * int(self.dimension[0]) and len(NEU) != 0:
            if self.dimension == '2D':
                em = ('When \'dimension\' is \'2D\' and %d quantities are regressed, \'NEU\' must be [X_N, Y_N, n_x, n_y,'
                      ' c_N1') % self.u.shape[1]
                for i in range(self.u.shape[1] - 1):
                    em += ', c_N%d' % (i + 2)
            else:
                em = ('When \'dimension\' is \'3D\' and %d quantities are regressed, \'NEU\' must be [X_N, Y_N, Z_N, n_x,'
                      ' n_y, n_z, c_N1') % self.u.shape[1]
                for i in range(self.u.shape[1] - 1):
                    em += ', c_N%d' % (i + 2)
            raise ValueError(em + ']')

        # Check that extra_RBF has the correct form
        if not isinstance(extra_RBF, bool) != bool:
            raise ValueError('Input \'extra_RBF\' must be a boolean')

        # Check for Dirichlet conditions
        if not DIR:  # No Dirichlet conditions
            # Assign empty arrays so that the assembly of the system is easier
            self.n_D = 0
            self.X_D = np.array([])
            self.Y_D = np.array([])
            self.Z_D = None
            self.c_D = np.array([np.array([]) for _ in range(self.u.shape[1])]).T
            # In 3D, add the z terms
            if self.dimension == '3D':
                self.Z_D = np.array([])


        else:  # Dirichlet conditions
            # Check for a 2D or a 3D problem and assign the input values
            if self.dimension == '2D':
                self.n_D = len(DIR[0])
                self.X_D = DIR[0]
                self.Y_D = DIR[1]
                self.Z_D = None
                self.c_D = np.array(DIR[2:]).T
            elif self.dimension == '3D':
                self.n_D = len(DIR[0])
                self.X_D = DIR[0]
                self.Y_D = DIR[1]
                self.Z_D = DIR[2]
                self.c_D = np.array(DIR[3:]).T


        # Check for Neuman conditions
        if not NEU:  # No Neumann conditions
            # Assign empty arrays so that the assembly of the system is easier
            self.n_N = 0
            self.X_N = np.array([])
            self.Y_N = np.array([])
            self.Z_N = None
            self.c_N = np.array([np.array([]) for _ in range(self.u.shape[1])]).T
            self.n_x = np.array([])
            self.n_y = np.array([])
            # In 3D, add the z term
            if self.dimension == '3D':
                self.Z_N = np.array([])
                self.n_z = np.array([])

        else:  # Neumann conditions
            # Check for a 2D or a 3D problem and assign the input values
            if self.dimension == '2D':
                self.n_N = len(NEU[0])
                self.X_N = NEU[0]
                self.Y_N = NEU[1]
                self.Z_N = None
                self.n_x = NEU[2]
                self.n_y = NEU[3]
                self.c_N = np.array(NEU[4:]).T

            elif self.dimension == '3D':
                self.n_N = len(NEU[0])
                self.X_N = NEU[0]
                self.Y_N = NEU[1]
                self.Z_N = NEU[2]
                self.n_x = NEU[3]
                self.n_y = NEU[4]
                self.n_z = NEU[5]
                self.c_N = np.array(NEU[6:]).T

        # Finally, add the extra RBFs in the constraint points if desired
        if extra_RBF:
            self._add_constraint_collocations()

        # Summary output for the user
        if self.verbose == 2:
            print(str(self.n_D) + ' Dirichlet conditions assigned')
            print(str(self.n_N) + ' Neumann conditions assigned')

        return


    # 3.2 Scalar constraints
    def vector_constraints(self, DIR=[], NEU=[], DIV=[], extra_RBF=True):
        """
        # This functions sets the boundary conditions for a laminar problem. The
        function must be run after the clustering was carried out.

        Parameters
        ----------

        DIR : list of 1D numpy.ndarray, default=[]
            This contains the info for the Dirichlet conditions. There are two
            options:

            * If ``dimension`` = '2D', it has to be [X_D, Y_D, c_D_X, c_D_Y]
            * If ``dimension`` = '3D', it has to be [X_D, Y_D, Z_D, c_D_X, c_D_Y, c_D_Z]

            Here X_D, Y_D, Z_D are the coordinates of the points where the values
            c_D_X, c_D_Y, c_D_Z are set.

        NEU : list of 1D numpy.ndarray, default=[]
            This contains the info for the Dirichlet conditions. There are two
            options:

            * If ``dimension`` = '2D', it has to be [X_N, Y_N, n_x, n_y, c_N_X, c_N_Y]
            * If ``dimension`` = '3D', it has to be [X_N, Y_N, Z_N, n_x, n_y, n_z, c_N_X, c_N_Y, c_N_Z]

            Here X_N, Y_N, Z_N are the coordinates of the points where the values
            c_N_X, c_N_Y, c_N_Z are set for the directional derivative along the
            normal direction n_x, n_y, n_z.

        DIV : list of 1D numpy.ndarray, default=[]
            This contains the info for the divergence-free conditions. There
            are two options:

            * If ``dimension`` = '2D', it has to be [X_Div, Y_Div]
            * If ``dimension`` = '3D', it has to be [X_Div, Y_Div, Z_Div]

            Here X_Div, Y_Div, Z_Div are the coordinates of the points where the
            divergence-free condition is imposed.

        extra_RBF : bool, default=True
            This is a flag to put extra collocation points where a constraint is
            set. It can improve the solution of the linear system as constraints
            remove degrees of freedom
        """

        # Check that the inputs have the correct form
        if self.model == 'scalar':
            raise ValueError('When model is \'scalar\', \'vector_constriants\' cannot be called')

        # Check that DIR, NEU, DIV have the correct form
        check_data(data=DIR, name='DIR')
        check_data(data=NEU, name='NEU')
        check_data(data=DIV, name='DIV')

        # Check that the length of DIR is correct
        if len(DIR) != 2 * int(self.dimension[0]) and len(DIR) != 0:
            if self.dimension == '2D':
                em = 'When \'dimension\' is \'2D\', \'DIR\' must be [X_D, Y_D, c_D_X, c_D_Y]'
            elif self.dimension == '3D':
                em = 'When \'dimension\' is \'3D\', \'DIR\' must be [X_D, Y_D, Z_D, c_D_X, c_D_Y, c_D_Z]'
            else:
                em = 'Wrong \'dimension\' of regression, must be \'2D\' or \'3D\''
            raise ValueError(em)

        # Check that the length of NEU is correct
        if len(NEU) != 3 * int(self.dimension[0]) and len(NEU) != 0:
            if self.dimension == '2D':
                em = 'When \'dimension\' is \'2D\', \'NEU\' must be [X_N, Y_N, n_x, n_y, c_N_X, c_N_Y]'
            elif self.dimension == '3D':
                em = 'When \'dimension\' is \'3D\', \'NEU\' must be [X_N, Y_N, Z_N, n_x, n_y, n_z, c_N_X, c_N_Y, c_N_Z]'
            else:
                em = 'Wrong \'dimension\' of regression, must be \'2D\' or \'3D\''
            raise ValueError(em)

        # Check that the length of DIV is correct
        if len(DIV) != int(self.dimension[0]) and len(DIV) != 0:
            if self.dimension == '2D':
                em = 'When \'dimension\' is \'2D\', \'DIV\' must be [X_Div, Y_Div]'
            elif self.dimension == '3D':
                em = 'When \'dimension\' is \'3D\', \'DIV\' must be [X_Div, Y_Div, Z_Div]'
            else:
                em = 'Wrong \'dimension\' of regression, must be \'2D\' or \'3D\''
            raise ValueError(em)

        # Check that extra_RBF has the correct form
        if not isinstance(extra_RBF, bool):
            raise ValueError('Input \'extra_RBF\' must be a boolean')

        # Check for Dirichlet conditions
        if not DIR:  # No Dirichlet conditions
            # Still assign empty arrays so that the assembly of the system is easier
            self.n_D = 0
            self.X_D = np.array([])
            self.Y_D = np.array([])
            self.Z_D = None
            self.c_D_X = np.array([])
            self.c_D_Y = np.array([])
            # In 3D, add the z terms
            if self.dimension == '3D':
                self.Z_D = np.array([])
                self.c_D_Z = np.array([])

        else:  # Dirichlet conditions
            # Check for a 2D or a 3D problem and assign the input values
            if len(DIR) == 4 and self.dimension == '2D':
                self.n_D = len(DIR[0])
                self.X_D = DIR[0]
                self.Y_D = DIR[1]
                self.Z_D = None
                self.c_D_X = DIR[2]
                self.c_D_Y = DIR[3]

            elif len(DIR) == 6 and self.dimension == '3D':
                self.n_D = len(DIR[0])
                self.X_D = DIR[0]
                self.Y_D = DIR[1]
                self.Z_D = DIR[2]
                self.c_D_X = DIR[3]
                self.c_D_Y = DIR[4]
                self.c_D_Z = DIR[5]

        # Check for Neumann conditions
        if not NEU:  # No Neumann conditions
            # Still assign empty arrays so that the assembly of the system is easier
            self.n_N = 0
            self.X_N = np.array([])
            self.Y_N = np.array([])
            self.Z_N = None
            self.c_N_X = np.array([])
            self.c_N_Y = np.array([])
            self.n_y = np.array([])
            self.n_x = np.array([])
            # In 3D, we must add the z terms
            if self.dimension == '3D':
                self.Z_N = np.array([])
                self.c_N_Z = np.array([])
                self.n_z = np.array([])

        else:  # Neumann conditions
            # Check for a 2D or a 3D problem and assign the input values
            if self.dimension == '2D':
                self.n_N = len(NEU[0])
                self.X_N = NEU[0]
                self.Y_N = NEU[1]
                self.Z_N = None
                self.n_x = NEU[2]
                self.n_y = NEU[3]
                self.c_N_X = NEU[4]
                self.c_N_Y = NEU[5]

            elif self.dimension == '3D':
                self.n_N = len(NEU[0])
                self.X_N = NEU[0]
                self.Y_N = NEU[1]
                self.Z_N = NEU[2]
                self.n_x = NEU[3]
                self.n_y = NEU[4]
                self.n_z = NEU[5]
                self.c_N_X = NEU[6]
                self.c_N_Y = NEU[7]
                self.c_N_Z = NEU[8]

        # Check for Divergence conditions
        if not DIV:
            # Still assign empty arrays so that the assembly of the system is easier
            self.n_Div = 0
            self.X_Div = np.array([])
            self.Y_Div = np.array([])
            self.Z_Div = None
            # In 3D, add the z terms
            if self.dimension == '3D':
                self.Z_Div = np.array([])
        else:
            # Check for a 2D or a 3D problem
            if len(DIV) == 2:  # this means 2D
                self.n_Div = len(DIV[0])
                self.X_Div = DIV[0]
                self.Y_Div = DIV[1]
                self.Z_Div = None

            else:
                self.n_Div = len(DIV[0])
                self.X_Div = DIV[0]
                self.Y_Div = DIV[1]
                self.Z_Div = DIV[2]

        # Finally, add the extra RBFs in the constraint points if desired
        if extra_RBF:
            self._add_constraint_collocations()

        # Summary output for the user
        if self.verbose == 2:
            print(str(self.n_D) + ' D conditions assigned')
            print(str(self.n_N) + ' N conditions assigned')
            print(str(self.n_Div) + ' Div conditions assigned')

        return

    # 3.3 Plot the RBFs, this is just a visualization tool
    def plot_RBFs(self,l=0):
        """
        Utility function to check the spreading of the RBFs after the clustering.
        This function generates several plots. It produces no new variable in SPICY.

        :type l: int
        :param l:
            This defines the cluster level of RBF that will be visualized.
        """

        # Check if it is 2D or 3D
        if self.dimension == '2D': # 2D
            try:
                # We define the data that will be included
                X_Plot = self.X_C[np.argwhere(self.Clust_list==l)]
                Y_Plot = self.Y_C[np.argwhere(self.Clust_list==l)]
                d_K_Plot = self.d_k[np.argwhere(self.Clust_list==l)]

                fig, axs = plt.subplots(1, 2, figsize = (7, 3.5), dpi = 100)
                # First plot is the RBF distribution
                axs[0].set_title("RBF Collocation for l="+str(l))

                # Also show the data points
                if self.model == 'scalar':
                     axs[0].scatter(self.X_G, self.Y_G, c=self.u, s=10)
                elif self.model == 'laminar':
                     axs[0].scatter(self.X_G, self.Y_G, c=np.sqrt(self.u**2 + self.v**2), s=10)

                for i in range(0,len(X_Plot),1):
                    circle1 = plt.Circle((X_Plot[i], Y_Plot[i]), d_K_Plot[i]/2,
                                          fill=True,color='g',edgecolor='k',alpha=0.2)
                    axs[0].add_artist(circle1)

                # Also show the constraints if they are set
                axs[0].plot(self.X_D, self.Y_D,'ro')
                axs[0].plot(self.X_N, self.Y_N,'bs')
                if self.model == 'laminar':
                     axs[0].plot(self.X_Div, self.Y_Div, 'bd')

                # Second plot is the distribution of diameters:
                axs[1].stem(d_K_Plot)
                axs[1].set_xlabel('Basis index')
                axs[1].set_ylabel('Diameter')
                axs[1].set_title("Distribution of diameters for L="+str(l))
                fig.tight_layout()

            except:
                raise ValueError('Problems in plotting. Set constraints and cluster!')

        elif self.dimension == '3D': # 3D
            try:
                # For now, we just show the distribution of diameters, as 3D sphere
                # visualizations are very difficult
                fig, ax = plt.subplots(figsize = (5, 5), dpi = 100)
                ax.set_title("RBF Collocation")
                ax.stem(self.d_k)
                ax.set_xlabel('Basis index')
                ax.set_ylabel('Diameter')
                ax.set_title("Distribution of diameters")
                fig.tight_layout()
            except:
                raise ValueError('Problems in plotting. Set constraints and cluster!')
        return


    # 4. Assembly A, B, b_1, b_2

    # We have two sorts of assemblies: poisson and regression.
    # poisson applies to the poisson solvers.
    # regression applies to scalar and laminar regression.

    # the poisson one includes the source terms on the r.h.s..
    # the regression one inlcudes a potential penalty of a divergence free flow.

    # 4.1. Poisson solver
    def Assembly_Poisson(self, n_hb=0):
        """

        This function assembly the matrices A, B, b_1, b_2 for the Poisson problem.
        These are eqs. (31a) - (31d) in the original paper (see also video tutorial 1 for more info)

        :type n_hb: int
        :param n_hb:
            When solving the Poisson equation, global basis elements such as polynomials or series
            expansions can be of great help. This is evident if one note that the eigenfunctions of
            the Laplace operator are harmonics.
            In a non-homogeneous problem, once could homogenize the basis. This will be proposed for the next relase
            (which will align with Manuel's paper). The idea is the following: if the homogeneization is well done and
            the basis is well chosen, then we do not need constraints for these extra terms of the basis.

            For the moment, we let the user introduce the number of extra_basis.
            These will be sine and cosine bases, which are orthogonal in [-1,1].
            In 1D, they are defined as : sines_n=np.sin(2*np.pi*(n)*x); cos_n=np.cos(np.pi/2*(2*n+1)*x)
            Given n_hb, we will have that the first n_hb are sines the last n_hb will be cosines.
            This defines the basis phi_h_n, with n an index from 0 to n_hb**4 in 2D.

            In 2D, assuming separation of variables, we will take phi_h_nm=phi_n(x)*phi_m(y).
            Similarly, in 3D will be phi_nmk=phi_n(x)*phi_m(y)*phi_k(z).
            For stability purposes, the largest tolerated value at the moment is 10!.

            For an homogeneous problem, the chosen basis needs no constraints.

            !!!!!!!!!!!!!!!#### this feature is currently under development #####!!!!!!!!!!!!!!!!

        """

        assert type(n_hb) == int, 'Number of harmonic basis must be an integer'

        # Assign the number of harmonic basis functions
        self.n_hb = n_hb
        # Get the number of basis and points as we need them a couple of times
        # 2D and 3D have different bases
        if self.dimension == '2D':
            self.n_b = self.X_C.shape[0] + n_hb**4
        elif self.dimension == '3D':
            self.n_b = self.X_C.shape[0] + n_hb**6

        if self.model=='scalar':
            if self.dimension == '2D': # 2D
                # Get the rescaling factor by normalizing the r.h.s. of the source terms
                source_terms = self.u
                self.scale_U = max(np.max(source_terms), -np.max(-source_terms))
                if np.abs(self.scale_U) < 1e-10:
                    self.scale_U = 1

                ### Dirichlet constraints ###
                # Compute Phi on X_D
                Matrix_D = np.hstack((
                    Phi_H_2D(self.X_D, self.Y_D, self.n_hb),
                    Phi_RBF_2D(self.X_D, self.Y_D, self.X_C, self.Y_C, self.c_k, self.basis)
                    ))

                ### Neumann constraints ###
                # Compute Phi_x on X_N
                Matrix_Phi_RBF_2D_x, Matrix_Phi_RBF_2D_y = Phi_RBF_2D_Deriv(
                    self.X_N, self.Y_N, self.X_C, self.Y_C, self.c_k, self.basis, order=1)
                Matrix_Phi_2D_X_N_der_x = np.hstack((
                    Phi_H_2D_x(self.X_N, self.Y_N, self.n_hb),
                    Matrix_Phi_RBF_2D_x
                    ))
                # Compute Phi_y on X_N
                Matrix_Phi_2D_X_N_der_y = np.hstack((
                    Phi_H_2D_y(self.X_N, self.Y_N, self.n_hb),
                    Matrix_Phi_RBF_2D_y
                    ))
                # Compute Phi_n on X_N
                Matrix_D_N = Matrix_Phi_2D_X_N_der_x*self.n_x[:, np.newaxis] +\
                             Matrix_Phi_2D_X_N_der_y*self.n_y[:, np.newaxis]

                # Assemble B and b_2, we also rescale b_2
                self.B = np.vstack((Matrix_D, Matrix_D_N)).T
                self.b_2 = np.concatenate((self.c_D, self.c_N))

                # Compute L on X_G
                Matrix_Phi_RBF_2D_X_der_xx, Matrix_Phi_RBF_2D_X_der_yy, Phi_RBF_2D_X_der_xy
                L = np.hstack((
                    Phi_H_2D_Laplacian(self.X_G, self.Y_G, self.n_hb),
                    Phi_RBF_2D_Laplacian(self.X_G, self.Y_G, self.X_C, self.Y_C, self.c_k, self.basis)
                    ))

                # Assemble A and b_1, also rescale b_1
                self.A = 2*L.T@L
                self.b_1 = 2*L.T.dot(source_terms)

            elif self.dimension == '3D': # 3D
                # get the rescaling factor by normalizing the r.h.s. of the source terms
                source_terms = self.u
                self.scale_U = max(np.max(source_terms), -np.max(-source_terms))
                if np.abs(self.scale_U) < 1e-10:
                    self.scale_U = 1

                ### Dirichlet constraints ###
                # Compute Phi on X_D
                Matrix_D = np.hstack((
                    Phi_H_3D(self.X_D, self.Y_D, self.Z_D, self.n_hb),
                    Phi_RBF_3D(self.X_D, self.Y_D, self.Z_D,
                               self.X_C, self.Y_C, self.Z_C,
                               self.c_k, self.basis)
                    ))

                ### Neumann constraints ###
                # Compute Phi_x on X_N
                Matrix_Phi_3D_X_N_der_x = np.hstack((
                    Phi_H_3D_x(self.X_N, self.Y_N, self.Z_N, self.n_hb),
                    Phi_RBF_3D_x(self.X_N, self.Y_N, self.Z_N,
                                 self.X_C, self.Y_C, self.Z_C,
                                 self.c_k, self.basis)
                    ))
                # Compute Phi_y on X_N
                Matrix_Phi_3D_X_N_der_y = np.hstack((
                    Phi_H_3D_y(self.X_N, self.Y_N, self.Z_N, self.n_hb),
                    Phi_RBF_3D_y(self.X_N, self.Y_N, self.Z_N,
                                 self.X_C, self.Y_C, self.Z_C,
                                 self.c_k, self.basis)
                    ))
                # Compute Phi_z on X_N
                Matrix_Phi_3D_X_N_der_z = np.hstack((
                    Phi_H_3D_z(self.X_N, self.Y_N, self.Z_N, self.n_hb),
                    Phi_RBF_3D_z(self.X_N, self.Y_N, self.Z_N,
                                 self.X_C, self.Y_C, self.Z_C,
                                 self.c_k, self.basis)
                    ))

                # Compute Phi_n on X_N
                Matrix_D_N = Matrix_Phi_3D_X_N_der_x*self.n_x[:, np.newaxis] +\
                             Matrix_Phi_3D_X_N_der_y*self.n_y[:, np.newaxis] +\
                             Matrix_Phi_3D_X_N_der_z*self.n_z[:, np.newaxis]

                # Assemble B and b_2, we also rescale b_2
                self.B = np.vstack((Matrix_D, Matrix_D_N)).T
                self.b_2 = np.concatenate((self.c_D, self.c_N))

                # Compute L on X_G
                L = np.hstack((
                    Phi_H_3D_Laplacian(self.X_G, self.Y_G, self.Z_G, self.n_hb),
                    Phi_RBF_3D_Laplacian(self.X_G, self.Y_G, self.Z_G,
                                         self.X_C, self.Y_C, self.Z_C,
                                         self.c_k, self.basis)
                    ))

                # Assemble A and b_1, also rescale b_1
                self.A = 2*L.T@L
                self.b_1 = 2*L.T.dot(source_terms)

        else:
            raise NotImplementedError('Assembly_Poisson only build for the scalar function')

        return


    # 4.2. Regression
    def Assembly_Regression(self, n_hb=0, alpha_div=None, K_cond=None):
        """
        This function assembly the matrices A, B, C, D from the paper (see video tutorial 1).

        :type n_hb: int
        :param n_hb: int

            Also for a regression, the harmonic basis can improve the regression
            as they can model global trends which are similar to a low order
            polynomial. Furthermore, for homogenous problem, they automatically
            fulfill the boundary conditions.

           See the same entry in the function 'Assembly_Poisson'

        :type alpha_div: float
        :param alpha_div:
            This enables a divergence free penalty in the entire flow field.
            The higher this parameter, the more SPICY penalizes errors in the divergence-free
            condition. This is particularly important to obtain good derivatives
            for the pressure computation.

        """

        if not hasattr(self, 'c_D'):
            if self.model == 'scalar':
                self.scalar_constraints(extra_RBF=False)
            elif self.model == 'laminar':
                self.vector_constraints(extra_RBF=False)
            else:
                pass

        self.w_list = []

        # Check that the inputs have the correct form
        check_number(param=n_hb, name='n_hb', dtype=int, threshold=0, check='geq')

        # Assign the number of harmonic basis functions
        self.n_hb = n_hb
        # get the number of basis and points as they are needed them a couple of times
        # 2D and 3D have different bases
        if self.dimension == '2D':
            self.n_b = self.X_C.shape[0] + n_hb ** 4
        elif self.dimension == '3D':
            self.n_b = self.X_C.shape[0] + n_hb ** 6

        self.L_A = None

        # Loop over all the quantities to be regressed. For a laminar regression,
        # this loop will only run once. For scalar regression, the loop is executed
        # once for each quantity
        for ii in range(self.u.shape[1]):

            self._get_rescale()

            ########################################################
            ### We first build the constraint matrices B and b_2 ###
            ########################################################
            # Compute the Patch weight for the Dirichlet and Neumann points. We further compute the
            # product Phi*Omega in the Dirichlet points as this is used to build the constraint matrix
            Phi_X_D = self._compute_Phi_matrix(self.X_D, self.Y_D, self.Z_D)

            # We compute the product Phi_N*Omega to stack into the constraint matrix
            if self.dimension == '2D':
                Phi_X_N_der_x, Phi_X_N_der_y = (
                    self._compute_dxPhi_matrix(self.X_N, self.Y_N, self.Z_N)
                )
                PhiN_X_N = (
                    np.multiply(Phi_X_N_der_x, self.n_x[:, np.newaxis]) +
                    np.multiply(Phi_X_N_der_y, self.n_y[:, np.newaxis])
                )

            elif self.dimension == '3D':
                Phi_X_N_der_x, Phi_X_N_der_y, Phi_X_N_der_z = (
                    self._compute_dxPhi_matrix(self.X_N, self.Y_N, self.Z_N)
                )
                PhiN_X_N = (
                        np.multiply(Phi_X_N_der_x, self.n_x[:, np.newaxis]) +
                        np.multiply(Phi_X_N_der_y, self.n_y[:, np.newaxis]) +
                        np.multiply(Phi_X_N_der_z, self.n_z[:, np.newaxis])
                )

            # In the scalar case, we only have two matrices to stack into B and b_w
            if self.model == 'scalar':
                self.B = np.concatenate((
                    Phi_X_D,
                    PhiN_X_N
                )).T

                self.b_2 = np.concatenate((
                    self.c_D,
                    self.c_N
                ))


            # In the laminar case, we also compute the derivative matrix on the divergence points
            # b_2 depends on the dimension, so we stack it in the if-condition
            elif self.model == 'laminar':
                if self.dimension == '2D':
                    Phi_X_Div_der_x, Phi_X_Div_der_y = (
                        self._compute_dxPhi_matrix(self.X_G, self.Y_G, self.Z_G)
                    )
                    D_nabla_j = np.hstack((
                        Phi_X_Div_der_x,
                        Phi_X_Div_der_y
                    ))

                    self.b_2 = np.concatenate((
                        np.zeros(self.X_Div.shape[0]),
                        self.c_D_X,
                        self.c_D_Y,
                        self.c_N_X,
                        self.c_N_Y
                    ))

                elif self.dimension == '3D':
                    Phi_X_Div_der_x, Phi_X_Div_der_y, Phi_X_Div_der_z = (
                        self._compute_dxPhi_matrix(self.X_G, self.Y_G, self.Z_G)
                    )
                    D_nabla_j = np.hstack((
                        Phi_X_Div_der_x,
                        Phi_X_Div_der_y,
                        Phi_X_Div_der_z
                    ))

                    self.b_2 = np.concatenate((
                        np.zeros(self.X_Div.shape[0]),
                        self.c_D_X_j,
                        self.c_D_Y_j,
                        self.c_D_Z_j,
                        self.c_N_X_j,
                        self.c_N_Y_j,
                        self.c_N_Z_j
                    ))

                # B is build independent of the dimension
                self.B = np.concatenate((
                    D_nabla_j,
                    stack_to_block_matrix(self.dimension, Phi_X_D),
                    stack_to_block_matrix(self.dimension, PhiN_X_N)
                )).T

            ###############################################
            ### We close with the data matrix A and b_1 ###
            ###############################################

            # First, we compute Omega and Omega*Phi in the data points
            Phi_X = self._compute_Phi_matrix(self.X_G, self.Y_G, self.Z_G)

            # Compute the product Phi.T @ Phi
            Phi_XT_dot_Phi_X = Phi_X.T @ Phi_X

            # For the scalar case, we only have one dimension, so no further stacking
            if self.model == 'scalar':
                # Compute Phi.T@Phi
                self.A = 2 * Phi_XT_dot_Phi_X / self.scale_U

                # Assemble b_1. The data points are weighted with Omega
                self.b_1 = 2 * (
                    Phi_X.T.dot(self.u)
                ) / self.scale_U

            # For the laminar case, we have to build A as a block matrix and b_1 through stacking
            elif self.model == 'laminar':
                self.A = 2 * stack_to_block_matrix(self.dimension, Phi_XT_dot_Phi_X) / self.scale_U

                if self.dimension == '2D':
                    self.b_1 = 2 * np.concatenate((
                        Phi_X.T.dot(self.u),
                        Phi_X.T.dot(self.v)
                    )) / self.scale_U
                if self.dimension == '3D':
                    self.b_1 = 2 * np.concatenate((
                        Phi_X.T.dot(self.u),
                        Phi_X.T.dot(self.v),
                        Phi_X.T.dot(self.w)
                    )) / self.scale_U

                # Add the penalty for a divergence-free flow
                if self.alpha_div is not None:
                    self._add_divergence_penalty(alpha_div=self.alpha_div)


        # # Scalar model:
        # # Even though it is not included in the article, a scalar can also be
        # # regressed in the same way with physical constraints
        # if self.model == 'scalar':
        #     if self.dimension == '2D': # 2D
        #         # define the rescaling factor which is done based on the maximum
        #         # absolute velocity that is available in u
        #         self.scale_U = np.abs(self.u[np.argmax(np.abs(self.u))])

        #         ### Dirichlet constraints ###
        #         # Compute Phi on X_D
        #         Matrix_D = np.hstack((
        #             Phi_H_2D(self.X_D, self.Y_D, self.n_hb),
        #             Phi_RBF_2D(self.X_D, self.Y_D, self.X_C, self.Y_C, self.c_k, self.basis)
        #             ))

        #         ### Neumann constraints ###
        #         # Compute Phi_x on X_N
        #         Matrix_Phi_2D_X_N_der_x = np.hstack((
        #             Phi_H_2D_x(self.X_N, self.Y_N, self.n_hb),
        #             Phi_RBF_2D_x(self.X_N, self.Y_N,
        #                           self.X_C, self.Y_C,
        #                           self.c_k, self.basis)
        #             ))
        #         # Compute Phi_y on X_N
        #         Matrix_Phi_2D_X_N_der_y = np.hstack((
        #             Phi_H_2D_y(self.X_N, self.Y_N, self.n_hb),
        #             Phi_RBF_2D_y(self.X_N, self.Y_N,
        #                           self.X_C, self.Y_C,
        #                           self.c_k, self.basis)
        #             ))
        #         # Compute Phi_n on X_N
        #         Matrix_D_N = Matrix_Phi_2D_X_N_der_x*self.n_x[:, np.newaxis] +\
        #                      Matrix_Phi_2D_X_N_der_y*self.n_y[:, np.newaxis]

        #         # Assemble B and b_2, we also rescale b_2
        #         self.B = np.vstack((Matrix_D, Matrix_D_N)).T
        #         self.b_2 = np.concatenate((self.c_D, self.c_N))

        #         # Compute Phi on X_G
        #         Matrix_Phi_2D_X = np.hstack((
        #             Phi_H_2D(self.X_G, self.Y_G, self.n_hb),
        #             Phi_RBF_2D(self.X_G, self.Y_G, self.X_C, self.Y_C, self.c_k, self.basis)
        #             ))

        #         # Assemble A and b_1, we also rescale b_1
        #         self.A = 2*Matrix_Phi_2D_X.T.dot(Matrix_Phi_2D_X)
        #         self.b_1 = 2*Matrix_Phi_2D_X.T.dot(self.u)

        #     elif self.dimension == '3D':
        #         # define the rescaling factor which is done based on the maximum
        #         # absolute velocity that is available in u
        #         self.scale_U = np.abs(self.u[np.argmax(np.abs(self.u))])

        #         ### Dirichlet constraints ###
        #         # Compute Phi on X_D
        #         Matrix_D = np.hstack((
        #             Phi_H_3D(self.X_D, self.Y_D, self.Z_D, self.n_hb),
        #             Phi_RBF_3D(self.X_D, self.Y_D, self.Z_D,
        #                        self.X_C, self.Y_C, self.Z_C,
        #                        self.c_k, self.basis)
        #             ))

        #         ### Neumann constraints ###
        #         # Compute Phi_x on X_N
        #         Matrix_Phi_3D_X_N_der_x = np.hstack((
        #             Phi_H_3D_x(self.X_N, self.Y_N, self.Z_N, self.n_hb),
        #             Phi_RBF_3D_x(self.X_N, self.Y_N, self.Z_N,
        #                           self.X_C, self.Y_C, self.Z_C,
        #                           self.c_k, self.basis)
        #             ))
        #         # Compute Phi_y on X_N
        #         Matrix_Phi_3D_X_N_der_y = np.hstack((
        #             Phi_H_3D_y(self.X_N, self.Y_N, self.Z_N, self.n_hb),
        #             Phi_RBF_3D_y(self.X_N, self.Y_N, self.Z_N,
        #                           self.X_C, self.Y_C, self.Z_C,
        #                           self.c_k, self.basis)
        #             ))
        #         # Compute Phi_z on X_N
        #         Matrix_Phi_3D_X_N_der_z = np.hstack((
        #             Phi_H_3D_y(self.X_N, self.Y_N, self.Z_N, self.n_hb),
        #             Phi_RBF_3D_y(self.X_N, self.Y_N, self.Z_N,
        #                           self.X_C, self.Y_C, self.Z_C,
        #                           self.c_k, self.basis)
        #             ))
        #         # Compute Phi_n on X_N (equation (18))
        #         Matrix_D_N = Matrix_Phi_3D_X_N_der_x*self.n_x[:, np.newaxis] +\
        #                      Matrix_Phi_3D_X_N_der_y*self.n_y[:, np.newaxis] +\
        #                      Matrix_Phi_3D_X_N_der_z*self.n_z[:, np.newaxis]

        #         # Assemble B and b_2, we also rescale b_2
        #         self.B = np.vstack((Matrix_D, Matrix_D_N)).T
        #         self.b_2 = np.concatenate((self.c_D, self.c_N))

        #         # We compute Phi on all node points X
        #         Matrix_Phi_3D_X = np.hstack((
        #             Phi_H_3D(self.X_G, self.Y_G, self.Z_G, self.n_hb),
        #             Phi_RBF_3D(self.X_G, self.Y_G, self.Z_G,
        #                        self.X_C, self.Y_C, self.Z_C,
        #                        self.c_k, self.basis)
        #             ))
        #         # Assemble A and b_1, we also rescale b_1
        #         self.A = 2*Matrix_Phi_3D_X.T.dot(Matrix_Phi_3D_X)
        #         self.b_1 = 2*Matrix_Phi_3D_X.T.dot(self.u)

        # # Laminar model
        # elif self.model == 'laminar':
        #     # We need to check whether we are 2D or 3D laminar as this changes the assignment
        #     if self.dimension == '2D': # 2D
        #         # Define the rescaling factor which is done based on the maximum
        #         # absolute velocity that is available in u and v
        #         data = np.concatenate((self.u, self.v))
        #         self.scale_U = np.abs(data[np.argmax(np.abs(data))])

        #         ### Divergence-free constraints ###
        #         # Compute Phi_x on X_Div
        #         Matrix_Phi_2D_X_Div_der_x = np.hstack((
        #             Phi_H_2D_x(self.X_Div, self.Y_Div, self.n_hb),
        #             Phi_RBF_2D_x(self.X_Div, self.Y_Div, self.X_C, self.Y_C, self.c_k, self.basis)
        #             ))
        #         # compute the derivatives in y
        #         Matrix_Phi_2D_X_Div_der_y = np.hstack((
        #             Phi_H_2D_y(self.X_Div, self.Y_Div, self.n_hb),
        #             Phi_RBF_2D_y(self.X_Div, self.Y_Div, self.X_C, self.Y_C, self.c_k, self.basis)
        #             ))
        #         # Stack into the block structure of equation (15)
        #         Matrix_D_Div = np.hstack((Matrix_Phi_2D_X_Div_der_x, Matrix_Phi_2D_X_Div_der_y))

        #         ### Dirichlet constraints ###
        #         # Compute Phi on X_D
        #         Matrix_Phi_2D_D = np.hstack((
        #             Phi_H_2D(self.X_D, self.Y_D, self.n_hb),
        #             Phi_RBF_2D(self.X_D, self.Y_D, self.X_C, self.Y_C, self.c_k, self.basis)
        #             ))
        #         # Stack into the block structure of equation (16)
        #         Matrix_D = np.block([
        #             [Matrix_Phi_2D_D,np.zeros((self.n_D, self.n_b))],
        #             [np.zeros((self.n_D, self.n_b)), Matrix_Phi_2D_D]
        #             ])

        #         ### Neumann constraints ###
        #         # Compute Phi_x on X_N
        #         Matrix_Phi_2D_X_N_der_x = np.hstack((
        #             Phi_H_2D_x(self.X_N, self.Y_N, self.n_hb),
        #             Phi_RBF_2D_x(self.X_N, self.Y_N, self.X_C, self.Y_C, self.c_k, self.basis)
        #             ))
        #         # Compute Phi_y on X_N
        #         Matrix_Phi_2D_X_N_der_y = np.hstack((
        #             Phi_H_2D_y(self.X_N, self.Y_N, self.n_hb),
        #             Phi_RBF_2D_y(self.X_N, self.Y_N, self.X_C, self.Y_C, self.c_k, self.basis)
        #             ))
        #         # Compute Phi_n on X_N (equation (18))
        #         Matrix_Phi_N = Matrix_Phi_2D_X_N_der_x*self.n_x[:, np.newaxis] +\
        #                        Matrix_Phi_2D_X_N_der_y*self.n_y[:, np.newaxis]
        #         # Stack into the block structure of equation (17)
        #         Matrix_D_N = np.block([
        #             [Matrix_Phi_N,np.zeros((self.n_N, self.n_b))],
        #             [np.zeros((self.n_N, self.n_b)), Matrix_Phi_N]
        #             ])

        #         # Assemble B and b_2, we also rescale b_2
        #         self.B = np.vstack((Matrix_D_Div, Matrix_D, Matrix_D_N)).T
        #         self.b_2 = np.concatenate((np.zeros(self.n_Div),self.c_D_X, self.c_D_Y,
        #                                   self.c_N_X, self.c_N_Y))

        #         # Compute Phi on X_G
        #         Matrix_Phi_2D_X = np.hstack((
        #             Phi_H_2D(self.X_G, self.Y_G, self.n_hb),
        #             Phi_RBF_2D(self.X_G, self.Y_G, self.X_C, self.Y_C, self.c_k, self.basis)
        #             ))
        #         # Stack Phi.T@Phi into the block structure of equation (10)
        #         PhiT_dot_Phi = Matrix_Phi_2D_X.T.dot(Matrix_Phi_2D_X)
        #         self.A = 2*np.block([
        #             [PhiT_dot_Phi, np.zeros((self.n_b, self.n_b))],
        #             [np.zeros((self.n_b, self.n_b)), PhiT_dot_Phi]
        #             ])
        #         # compute and rescale b_1
        #         self.b_1 = 2*np.concatenate((Matrix_Phi_2D_X.T.dot(self.u), Matrix_Phi_2D_X.T.dot(self.v)))

        #         # We check if alpha_div is None or 0 (some users might give 0)
        #         # if they are not experienced so we check for both
        #         if alpha_div is not None and alpha_div != 0:
        #             # Compute Phi_x on X_G
        #             Matrix_Phi_2D_X_der_x = np.hstack((
        #                 Phi_H_2D_x(self.X_G, self.Y_G, self.n_hb),
        #                 Phi_RBF_2D_x(self.X_G, self.Y_G, self.X_C, self.Y_C, self.c_k, self.basis)
        #                 ))
        #             # Compute Phi_y on X_G
        #             Matrix_Phi_2D_X_der_y = np.hstack((
        #                 Phi_H_2D_y(self.X_G, self.Y_G, self.n_hb),
        #                 Phi_RBF_2D_y(self.X_G, self.Y_G, self.X_C, self.Y_C, self.c_k, self.basis)
        #                 ))

        #             # Compute the individual matrix products between x, y and z
        #             # For the diagonal
        #             PhiXT_dot_PhiX = Matrix_Phi_2D_X_der_x.T.dot(Matrix_Phi_2D_X_der_x)
        #             PhiYT_dot_PhiY = Matrix_Phi_2D_X_der_y.T.dot(Matrix_Phi_2D_X_der_y)
        #             # For the off-diagonal elements
        #             PhiXT_dot_PhiY = Matrix_Phi_2D_X_der_x.T.dot(Matrix_Phi_2D_X_der_y)

        #             # And we add them into the A-matrix
        #             # Diagonal
        #             self.A[self.n_b*0:self.n_b*1,self.n_b*0:self.n_b*1] += 2*alpha_div*PhiXT_dot_PhiX
        #             self.A[self.n_b*1:self.n_b*2,self.n_b*1:self.n_b*2] += 2*alpha_div*PhiYT_dot_PhiY
        #             # Upper off-diagonal elements
        #             self.A[self.n_b*0:self.n_b*1,self.n_b*1:self.n_b*2] += 2*alpha_div*PhiXT_dot_PhiY
        #             # Lower off-diagonal elements
        #             self.A[self.n_b*1:self.n_b*2,self.n_b*0:self.n_b*1] += 2*alpha_div*PhiXT_dot_PhiY.T


        #     elif self.dimension == '3D': # 3D
        #         # Define the rescaling factor which is done based on the maximum
        #         # absolute velocity that is available in u, v and w
        #         data = np.concatenate((self.u, self.v, self.w))
        #         self.scale_U = np.abs(data[np.argmax(np.abs(data))])

        #         ### Divergence-free constraints ###
        #         # Compute Phi_x on X_Div
        #         Matrix_Phi_3D_X_Div_der_x = np.hstack((
        #             Phi_H_3D_x(self.X_Div, self.Y_Div, self.Z_Div, self.n_hb),
        #             Phi_RBF_3D_x(self.X_Div, self.Y_Div, self.Z_Div,
        #                          self.X_C, self.Y_C, self.Z_C,
        #                          self.c_k, self.basis)
        #             ))
        #         # Compute Phi_y on X_Div
        #         Matrix_Phi_3D_X_Div_der_y = np.hstack((
        #             Phi_H_3D_y(self.X_Div, self.Y_Div, self.Z_Div, self.n_hb),
        #             Phi_RBF_3D_y(self.X_Div, self.Y_Div, self.Z_Div,
        #                          self.X_C, self.Y_C, self.Z_C,
        #                          self.c_k, self.basis)
        #             ))
        #         # Compute Phi_z on X_Div
        #         Matrix_Phi_3D_X_Div_der_z = np.hstack((
        #             Phi_H_3D_z(self.X_Div, self.Y_Div, self.Z_Div, self.n_hb),
        #             Phi_RBF_3D_z(self.X_Div, self.Y_Div, self.Z_Div,
        #                          self.X_C, self.Y_C, self.Z_C,
        #                          self.c_k, self.basis)
        #             ))
        #         # Stack into the block structure of equation (15)
        #         Matrix_D_Div = np.hstack((Matrix_Phi_3D_X_Div_der_x,
        #                                   Matrix_Phi_3D_X_Div_der_y,
        #                                   Matrix_Phi_3D_X_Div_der_z))

        #         ### Dirichlet constraints ###
        #         # Compute Phi on X_D
        #         Matrix_Phi_3D_D = np.hstack((
        #             Phi_H_3D(self.X_D, self.Y_D, self.Z_D, self.n_hb),
        #             Phi_RBF_3D(self.X_D, self.Y_D, self.Z_D,
        #                        self.X_C, self.Y_C, self.Z_C,
        #                        self.c_k, self.basis)
        #             ))
        #         # Stack into the block structure of equation (16)
        #         Matrix_D = np.block([
        #             [Matrix_Phi_3D_D, np.zeros((self.n_D, self.n_b)), np.zeros((self.n_D, self.n_b))],
        #             [np.zeros((self.n_D, self.n_b)), Matrix_Phi_3D_D, np.zeros((self.n_D, self.n_b))],
        #             [np.zeros((self.n_D, self.n_b)), np.zeros((self.n_D, self.n_b)), Matrix_Phi_3D_D]
        #             ])

        #         ### Neumann constraints ###
        #         # Compute Phi_x on X_N
        #         Matrix_Phi_3D_X_N_der_x = np.hstack((
        #             Phi_H_3D_x(self.X_N, self.Y_N, self.Z_N, self.n_hb),
        #             Phi_RBF_3D_x(self.X_N, self.Y_N, self.Z_N,
        #                          self.X_C, self.Y_C, self.Z_C,
        #                          self.c_k, self.basis)
        #             ))
        #         # Compute Phi_y on X_N
        #         Matrix_Phi_3D_X_N_der_y = np.hstack((
        #             Phi_H_3D_y(self.X_N, self.Y_N, self.Z_N, self.n_hb),
        #             Phi_RBF_3D_y(self.X_N, self.Y_N, self.Z_N,
        #                          self.X_C, self.Y_C, self.Z_C,
        #                          self.c_k, self.basis)
        #             ))
        #         # Compute Phi_z on X_N
        #         Matrix_Phi_3D_X_N_der_z = np.hstack((
        #             Phi_H_3D_z(self.X_N, self.Y_N, self.Z_N, self.n_hb),
        #             Phi_RBF_3D_z(self.X_N, self.Y_N, self.Z_N,
        #                          self.X_C, self.Y_C, self.Z_C,
        #                          self.c_k, self.basis)
        #             ))
        #         # Compute Phi_n on X_N (equation (18))
        #         Matrix_Phi_N = Matrix_Phi_3D_X_N_der_x*self.n_x[:, np.newaxis] +\
        #                        Matrix_Phi_3D_X_N_der_y*self.n_y[:, np.newaxis] +\
        #                        Matrix_Phi_3D_X_N_der_z*self.n_z[:, np.newaxis]
        #         # Stack into the block structure of equation (17)
        #         Matrix_D_N = np.block([
        #             [Matrix_Phi_N, np.zeros((self.n_N, self.n_b)), np.zeros((self.n_N, self.n_b))],
        #             [np.zeros((self.n_N, self.n_b)), Matrix_Phi_N, np.zeros((self.n_N, self.n_b))],
        #             [np.zeros((self.n_N, self.n_b)), np.zeros((self.n_N, self.n_b)), Matrix_Phi_N]
        #             ])

        #         # Assemble B and b_2, we also rescale b_2
        #         self.B = np.vstack((Matrix_D_Div, Matrix_D, Matrix_D_N)).T
        #         self.b_2 = np.concatenate((np.zeros(self.n_Div),
        #                                   self.c_D_X, self.c_D_Y, self.c_D_Z,
        #                                   self.c_N_X, self.c_N_Y, self.c_N_Z))

        #         # Compute Phi on X_G
        #         Matrix_Phi_3D_X = np.hstack((
        #             Phi_H_3D(self.X_G, self.Y_G, self.Z_G, self.n_hb),
        #             Phi_RBF_3D(self.X_G, self.Y_G, self.Z_G,
        #                        self.X_C, self.Y_C, self.Z_C,
        #                        self.c_k, self.basis)
        #             ))
        #         # Stack Phi.T@Phi into the block structure of equation (10)
        #         PhiT_dot_Phi = Matrix_Phi_3D_X.T.dot(Matrix_Phi_3D_X)
        #         self.A = 2*np.block([
        #             [PhiT_dot_Phi, np.zeros((self.n_b, self.n_b)), np.zeros((self.n_b, self.n_b))],
        #             [np.zeros((self.n_b, self.n_b)), PhiT_dot_Phi, np.zeros((self.n_b, self.n_b))],
        #             [np.zeros((self.n_b, self.n_b)), np.zeros((self.n_b, self.n_b)), PhiT_dot_Phi]
        #             ])
        #         # compute and rescale b_1
        #         self.b_1 = 2*np.concatenate((Matrix_Phi_3D_X.T.dot(self.u),
        #                                      Matrix_Phi_3D_X.T.dot(self.v),
        #                                      Matrix_Phi_3D_X.T.dot(self.w)))

        #         # We check if alpha_div is None or 0 (some users might give 0)
        #         # if they are not experienced so we check for both
        #         if alpha_div is not None and alpha_div != 0:
        #             # Compute Phi_x on X_G
        #             Matrix_Phi_3D_X_der_x = np.hstack((
        #                 Phi_H_3D_x(self.X_G, self.Y_G, self.Z_G, self.n_hb),
        #                 Phi_RBF_3D_x(self.X_G, self.Y_G, self.Z_G,
        #                              self.X_C, self.Y_C, self.Z_C,
        #                              self.c_k, self.basis)
        #                 ))
        #             # Compute Phi_y on X_G
        #             Matrix_Phi_3D_X_der_y = np.hstack((
        #                 Phi_H_3D_y(self.X_G, self.Y_G, self.Z_G, self.n_hb),
        #                 Phi_RBF_3D_y(self.X_G, self.Y_G, self.Z_G,
        #                              self.X_C, self.Y_C, self.Z_C,
        #                              self.c_k, self.basis)
        #                 ))
        #             # Compute Phi_z on X_G
        #             Matrix_Phi_3D_X_der_z = np.hstack((
        #                 Phi_H_3D_z(self.X_G, self.Y_G, self.Z_G, self.n_hb),
        #                 Phi_RBF_3D_z(self.X_G, self.Y_G, self.Z_G,
        #                              self.X_C, self.Y_C, self.Z_C,
        #                              self.c_k, self.basis)
        #                 ))

        #             # Compute the individual matrix products between x, y and z
        #             # For the diagonal
        #             PhiXT_dot_PhiX = Matrix_Phi_3D_X_der_x.T.dot(Matrix_Phi_3D_X_der_x)
        #             PhiYT_dot_PhiY = Matrix_Phi_3D_X_der_y.T.dot(Matrix_Phi_3D_X_der_y)
        #             PhiZT_dot_PhiZ = Matrix_Phi_3D_X_der_z.T.dot(Matrix_Phi_3D_X_der_z)
        #             # For the off-diagonal elements
        #             PhiXT_dot_PhiY = Matrix_Phi_3D_X_der_x.T.dot(Matrix_Phi_3D_X_der_y)
        #             PhiXT_dot_PhiZ = Matrix_Phi_3D_X_der_x.T.dot(Matrix_Phi_3D_X_der_z)
        #             PhiYT_dot_PhiZ = Matrix_Phi_3D_X_der_y.T.dot(Matrix_Phi_3D_X_der_z)

        #             # And we add them into the A-matrix
        #             # Diagonal
        #             self.A[self.n_b*0:self.n_b*1,self.n_b*0:self.n_b*1] += 2*alpha_div*PhiXT_dot_PhiX
        #             self.A[self.n_b*1:self.n_b*2,self.n_b*1:self.n_b*2] += 2*alpha_div*PhiYT_dot_PhiY
        #             self.A[self.n_b*2:self.n_b*3,self.n_b*2:self.n_b*3] += 2*alpha_div*PhiZT_dot_PhiZ

        #             # Upper off-diagonal elements
        #             self.A[self.n_b*0:self.n_b*1,self.n_b*1:self.n_b*2] += 2*alpha_div*PhiXT_dot_PhiY
        #             self.A[self.n_b*0:self.n_b*1,self.n_b*2:self.n_b*3] += 2*alpha_div*PhiXT_dot_PhiZ
        #             self.A[self.n_b*1:self.n_b*2,self.n_b*2:self.n_b*3] += 2*alpha_div*PhiYT_dot_PhiZ

        #             # Lower off-diagonal elements
        #             self.A[self.n_b*1:self.n_b*2,self.n_b*0:self.n_b*1] += 2*alpha_div*PhiXT_dot_PhiY.T
        #             self.A[self.n_b*2:self.n_b*3,self.n_b*0:self.n_b*1] += 2*alpha_div*PhiXT_dot_PhiZ.T
        #             self.A[self.n_b*2:self.n_b*3,self.n_b*1:self.n_b*2] += 2*alpha_div*PhiYT_dot_PhiZ.T

        # elif self.model == 'RANSI':
        #     raise NotImplementedError('RANSI currently not implemented')
        # elif self.model == 'RANSI':
        #     raise NotImplementedError('RANSI currently not implemented')
        # else:
        #     raise ValueError('No regression could be performed, check that the model is correctly set')
        return


    # 5 Solver using the Shur complement
    def Solve(self, K_cond=1e12):
        """
        This function solves the constrained quadratic problem A, B, b_1, b_2.
        The method is universal for 2D/3D problems as well as laminar/Poisson problems.

        The input parameters are the class itself and the desired condition
        number of A which is fixed based on its largest and smallest eigenvalue

        The function assigns the weights 'w' and the Lagrange multipliers
        Lambda to the class. The weights are computed for the min/max scaled problem,
        i.e. the right hand-side of the linear system is normalized. The assigned
        weights are rescaled by self.scale_U to get the real, physical quantities

        K_cond : float, default 1e12
            This is the regularization parameter. It fixes the condition number (see Video 1)
            The estimation is based such that the regularize matrix has the condition
            number k_cond. For this, we compute the max and the min eigenvalue.

        """

        # Two options:
        # 1.: We do not have constraints, then we only need to solve A*w = b_1
        # 2.: We have constraints, then B and b_2 are not empty, and we go for Schur complements

        # We follow the 9 steps from Algorithm 1 in Sperotto et al. (2022)

        # Step 1: Compute A, B, b1, b2
        # Already carried out in the Assembly_Regression function

        # No constraints (this is not mentioned in the original paper)
        if (self.B.size == 0) and (self.b_2.size == 0):
            if self.verbose == 2:
                print('Solving without constraints')

            # Step 2 + 3: Factorize the Matrix A
            self._factorize_A(K_cond=K_cond)

            # Step 9: Solve for w and append into the list of weights
            for ii in range(self.b_1.shape[1]):
                self.weights = linalg.cho_solve((self.L_A, True), self.b_1[:, ii], check_finite=False)
                self.w_list.append(self.weights)

            if self.verbose == 2:
                print('Weights computed')

        # Constraints
        elif (self.B.size != 0) and (self.b_2.size != 0):
            if self.verbose == 2:
                print('Solving without constraints')

            # Step 2 + 3: Factorize the Matrix A
            self._factorize_A(K_cond=K_cond)

            # Step 4: Solve for R. This is different to the article because of numerical stability
            R = linalg.cho_solve((self.L_A, True), self.B, check_finite=False)

            # Step 5 + 6: Factorize the Matrix A
            self._factorize_M(R=R, K_cond=K_cond)

            # Step 7 + 8: Compute b_2_star, solve the system for lambda
            for ii in range(self.b_1.shape[1]):
                b_2_star = R.T.dot(self.b_1[:, ii]) - self.b_2[:, ii]
                self.lambdas = linalg.cho_solve((self.L_M, True), b_2_star, check_finite=False)
                # print('Lambdas computed')

                # Step 9: Solve for w and append into the list of weights
                b_1_star = self.b_1[:, ii] - self.B.dot(self.lambdas)
                self.weights = linalg.cho_solve((self.L_A, True), b_1_star, check_finite=False)
                self.w_list.append(self.weights)

            if self.verbose == 2:
                print('Weights and Lambdas computed')

        else:  # This should never be reached during regular use, it is useful for debugging
            raise ValueError('b_1 or B is empty while the other is not, check your constraints!')

        return

    def _factorize_A(self, K_cond):
        """
        Function for the Cholesky factorization of A with regularization

        Parameters
        ----------
        K_cond : float
            Desired conditioning number for the matrix A. This is approximated from the largest eigenvalue.
            See video tutorials for more info

        Assigns
        -------
        L_A : 2D numpy.ndarray
            Cholesky factorization of A, overwritten onto A.

        """

        if self.L_A is None:
            # Step 2: Regularize the matrix A
            lambda_A = eigsh(self.A, 1, return_eigenvectors=False)[0]  # Largest eigenvalue
            alpha = (lambda_A - K_cond * 2.2e-16) / K_cond
            self.A = self.A + alpha * np.eye(np.shape(self.A)[0])
            if self.verbose == 2:
                print('Matrix A regularized')

            # Step 3: Cholesky Decomposition of A
            self.L_A, low = linalg.cho_factor(self.A, overwrite_a=True, check_finite=False, lower=True)
            if self.verbose == 2:
                print('Matrix A factorized')

        return

    def _factorize_M(self, R, K_cond):
        """
        Function for the Cholesky factorization of M. Computes the matrix M based on the given N
        and the constraint matrix B. Initial try is without regularization. If it fails, the
        normal regularization is applied.

        Parameters
        ----------
        R : 2D numpy.ndarray
            Intermediate quantity defined as L_A^-1 @ B (see equation 37 of Sperotto et al. (2022)).

        K_cond : float
            Desired conditioning number for the matrix M. This is approximated from the largest eigenvalue.
            See video tutorials for more info

        Assigns
        -------
        L_M : 2D numpy.ndarray
            Cholesky factorization of M, overwritten onto M.

        """

        # Step 4: prepare M
        # This computes R.T @ R. We multiply with B because of numerical stability
        M = R.T @ self.B

        # Step 5: Regularize the matrix M
        lambda_M = eigsh(M, 1, return_eigenvectors=False)[0]  # Largest eigenvalue
        alpha = (lambda_M - K_cond * 2.2e-16) / K_cond
        M = M + alpha * np.eye(np.shape(M)[0])
        if self.verbose == 2:
            print('Matrix M regularized')

        # Step 6: Cholesky Decomposition of A
        self.L_M, low = linalg.cho_factor(M, overwrite_a=True, check_finite=False, lower=True)
        if self.verbose == 2:
            print('Matrix M factorized')

        return


    # 6. Evaluate solution on arbitrary points
    def get_sol(self, points, order=0, shape=None):
        """
        This function evaluates the solution of the linear system on an arbitrary set of points.
        The function is used for both the function itself as well as the analytical derivatives.

        Parameters
        ----------

        points : list
            Contains the points at which the source term is evaluated

            * if ``dimension`` is '2D', it has to be [X_P, Y_P].
            * if ``dimension`` is '3D', it has to be [X_P, Y_P, Z_P].


        Returns
        -------

        U_sol : list of numpy.ndarray or numpy.ndarray
            The solution always depends on ``model`` = 'scalar'/'laminar' and ``dimension`` = '2D'/'3D'
            If order = 0, give the original function

            * If scalar, the solution is [U1_sol, ... Un_sol].
            * If laminar and 2D, the solution is [U_sol, V_sol]
            * If laminar and 3D, the solution is [U_sol, V_sol, W_sol]

            If the scalar solution is only [U1_sol], it is returned as U1_sol
            so that a list of length 1 does not need to be unpacked by the user.

            If order == 1, give the first derivative of all quantities

            * If 'scalar' and '2D', the output is [dU1dX, dU1dY, ..., dUndx, dUndy]
            * If 'scalar' and '3D', the output is [dU1dX, dU1dY, dU1dZ, ..., dUndx, dUndy, dUndz]
            * If 'laminar' and '2D', the solution is [dUdx, dUdY, dVdX, dVdY]
            * If 'laminar' and '3D', the solution is [dUdx, dUdY, dUdX, dVdX, dVdY, dVdX, dWdX, dWdY, dWdX]

        """

        # Check the inputs
        check_data(data=points, name='points', dimension=self.dimension)
        check_number(param=order, name='order', dtype=int, threshold=0, check='geq')

        if shape is not None:
            if len(shape) != int(self.dimension[0]):
                raise ValueError('Shape does not match dimension of the problem.')
            if np.prod(np.array(shape)) != points[0].shape:
                raise ValueError('Shape does not match the given points.')

        # points = self._normalize_points(points)

        if len(points) == 2:
            X_eval, Y_eval, Z_eval = points[0], points[1], None
        elif len(points) == 3:
            X_eval, Y_eval, Z_eval = points[0], points[1], points[2]
        else:
            pass

        # Original function
        if order == 0:

            if self.model == 'scalar':
                Sol = list(np.tile(np.zeros(X_eval.shape[0]), (self.u.shape[1], 1)))
            elif self.model == 'laminar':
                if self.dimension == '2D':
                    Sol = list(np.tile(np.zeros(X_eval.shape[0]), (2, 1)))
                elif self.dimension == '3D':
                    Sol = list(np.tile(np.zeros(X_eval.shape[0]), (3, 1)))
            else:
                return

            # Compute Omega*Phi on X_G_j
            Phi_X = self._compute_Phi_matrix(X_eval, Y_eval, Z_eval)

            # Iterate over all scalars (for a laminar problem, the loop is only executed once)
            for i in range(self.u.shape[1]):

                # Add the local solution to the global one
                if self.model == 'scalar':
                    Sol[i] = Phi_X.dot(self.w_list[i])
                if self.model == 'laminar':
                    Sol[0] = Phi_X.dot(self.w_list[i][0*self.n_b_j : 1*self.n_b_j])
                    Sol[1]  += Phi_X.dot(self.w_list[i][1*self.n_b_j : 2*self.n_b_j])
                    if self.dimension == '3D':
                        Sol[2] += Phi_X.dot(self.w_list[i][2*self.n_b_j : 3*self.n_b_j])

        if shape is None:
            if len(Sol) == 1:
                return Sol[0]
            else:
                return Sol
        else:
            if len(Sol) == 1:
                return Sol[0].reshape(shape)
            else:
                return [field.reshape(shape) for field in Sol]



    def Get_first_Derivatives(self, points):
        """
        This function evaluates the first derivative of the solution of the
        linear system on an arbitrary set of points on the points.

        :type points: list
        :param points:
            Contains the points at which the source term is evaluated
            If the model is 2D, then this has [X_P, Y_P].
            If the model is 3D, then this has [X_P, Y_P, Z_P].

        :return: dUdX, dUdY, dUdX, dVdX, dVdY, dVdX, dWdX, dWdY, dWdX.
            Depending on model = 'scalar/laminar' and type = '2D/3D'
            If scalar and 2D, the output is dUdX, dUdY
            If scalar and 3D, the output is dUdX, dUdY, dUdZ
            If laminar and 2D, the solution is dUdx, dUdY, dVdX, dVdY
            If laminar and 3D, the solution is dUdx, dUdY, dUdX, dVdX, dVdY, dVdX, dWdX, dWdY, dWdX

        """

        # Check the input is correct
        assert type(points) == list, 'points must be a list'

        # Check if the points have the correct length
        if len(points) == 2 and self.dimension == '2D': # 2D
            # Assign the points
            X_P = points[0]
            Y_P = points[1]

            # Check what model type we have
            if self.model == 'scalar': # Scalar
                # Evaluate Phi_x on the points X_P, Y_P
                Phi_x = np.hstack((
                    Phi_H_2D_x(X_P, Y_P, self.n_hb),
                    Phi_RBF_2D_x(X_P, Y_P, self.X_C, self.Y_C, self.c_k, self.basis)
                    ))
                # Evaluate Phi_y on the points X_P,Y_P
                Phi_y = np.hstack((
                    Phi_H_2D_y(X_P, Y_P, self.n_hb),
                    Phi_RBF_2D_y(X_P, Y_P, self.X_C, self.Y_C, self.c_k, self.basis)
                    ))
                # Compute dudx and dudy on the new points
                dUdX = Phi_x.dot(self.weights)
                dUdY = Phi_y.dot(self.weights)
                return dUdX, dUdY

            elif self.model == 'laminar': # Laminar
                # We do it in 2 blocks: first all derivatives in x
                # Evaluate Phi on the points X_P, Y_P
                Phi_deriv = np.hstack((
                    Phi_H_2D_x(X_P, Y_P, self.n_hb),
                    Phi_RBF_2D_x(X_P, Y_P, self.X_C, self.Y_C, self.c_k, self.basis)
                    ))
                # Compute dudx and dvdx on the new points
                dUdX = Phi_deriv.dot(self.weights[0*self.n_b:1*self.n_b])
                dVdX = Phi_deriv.dot(self.weights[1*self.n_b:2*self.n_b])

                # Then we do it again for the derivatives of y.
                # Note however, that we re-use the same variables Phi_deriv
                # to limit the memory usage. This is pretty much copy-paste.
                Phi_deriv = np.hstack((
                    Phi_H_2D_y(X_P, Y_P, self.n_hb),
                    Phi_RBF_2D_y(X_P, Y_P, self.X_C, self.Y_C, self.c_k, self.basis)
                    ))
                # Compute dudy and dvdy on the new points
                dUdY = Phi_deriv.dot(self.weights[0*self.n_b:1*self.n_b])
                dVdY = Phi_deriv.dot(self.weights[1*self.n_b:2*self.n_b])

                return dUdX, dUdY, dVdX, dVdY

        elif len(points) == 3 and self.dimension == '3D': # 3D
            # Assign the points
            X_P = points[0]
            Y_P = points[1]
            Z_P = points[2]

            # Check what model type we have
            if self.model == 'scalar': # Scalar
                # Evaluate Phi_x on the points X_P, Y_P, Z_P for dudx
                Phi_deriv = np.hstack((
                    Phi_H_3D_x(X_P, Y_P, Z_P, self.n_hb),
                    Phi_RBF_3D_x(X_P, Y_P, Z_P, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                    ))
                # Compute dudx on the new points
                dUdX = Phi_deriv.dot(self.weights)
                # Now again for the derivative on dudy
                Phi_deriv = np.hstack((
                    Phi_H_3D_y(X_P, Y_P, Z_P, self.n_hb),
                    Phi_RBF_3D_y(X_P, Y_P, Z_P, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                    ))
                # Compute dudx on the new points
                dUdY = Phi_deriv.dot(self.weights)
                # Now again for the derivative on dudz
                Phi_deriv = np.hstack((
                    Phi_H_3D_z(X_P, Y_P, Z_P, self.n_hb),
                    Phi_RBF_3D_z(X_P, Y_P, Z_P, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                    ))
                # Compute dudx on the new points
                dUdZ = Phi_deriv.dot(self.weights)
                return dUdX, dUdY, dUdZ

            elif self.model == 'laminar': # Laminar
                # We do it in 3 blocks: first all derivatives in x
                # Evaluate Phi on the points X_P, Y_P, Z_P
                Phi_deriv = np.hstack((
                     Phi_H_3D_x(X_P, Y_P, Z_P, self.n_hb),
                     Phi_RBF_3D_x(X_P, Y_P, Z_P, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                     ))
                # Compute dudx, dvdx, dwdx on the new points
                dUdX = Phi_deriv.dot(self.weights[0*self.n_b:1*self.n_b])
                dVdX = Phi_deriv.dot(self.weights[1*self.n_b:2*self.n_b])
                dWdX = Phi_deriv.dot(self.weights[2*self.n_b:3*self.n_b])

                # Then we do it again for the derivatives of y.
                # Note however, that we re-use the same variables Phi_deriv
                # to limit the memory usage. This is pretty much copy-paste.
                Phi_deriv = np.hstack((
                     Phi_H_3D_y(X_P, Y_P, Z_P, self.n_hb),
                     Phi_RBF_3D_y(X_P, Y_P, Z_P, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                     ))
                # Compute dudy, dvdy, dwdy on the new points
                dUdY = Phi_deriv.dot(self.weights[0*self.n_b:1*self.n_b])
                dVdY = Phi_deriv.dot(self.weights[1*self.n_b:2*self.n_b])
                dWdY = Phi_deriv.dot(self.weights[2*self.n_b:3*self.n_b])

                # All derivatives along z -------------------------
                Phi_deriv=np.hstack((
                     Phi_H_3D_z(X_P, Y_P, Z_P, self.n_hb),
                     Phi_RBF_3D_z(X_P, Y_P, Z_P, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                     ))
                # Compute dudz, dvdz, dwdz on the new points
                dUdZ = Phi_deriv.dot(self.weights[0*self.n_b:1*self.n_b])
                dVdZ = Phi_deriv.dot(self.weights[1*self.n_b:2*self.n_b])
                dWdZ = Phi_deriv.dot(self.weights[2*self.n_b:3*self.n_b])

                return dUdX, dUdY, dUdZ, dVdX, dVdY, dVdZ, dWdX, dWdY, dWdZ

        else:
            raise ValueError('Length of points is invalid for Type ' + self.dimension)


    # Here is a function to evaluate the forcing term on the points that are
    # used for the pressure
    def Evaluate_Source_Term(self, points, rho):
        """
        This function evaluates the source term on the right hand side of
        equation (21) in the paper (see video tutorial 1 for more info)

        :type points: list
        :param points:
            Contains the points at which the source term is evaluated
            If the model is 2D, then this has [X_P, Y_P].
            If the model is 3D, then this has [X_P, Y_P, Z_P].

        :type rho: float
        :param rho:
           Density of the fluid.

        :return: source_term
            R.h.s. of equation (21).
        """

        # Check the input is correct
        assert type(points) == list, 'points must be a list'

        # check whether it is 2D or 3D
        if len(points) == 2 and self.dimension == '2D': # 2D
            # assign the points in X and Y
            X_P = points[0]
            Y_P = points[1]
            W_u = self.weights[:self.n_b]
            W_v = self.weights[self.n_b:]

            # We compute Phi_x on X_P
            Matrix_Phi_2D_X_P_der_x = np.hstack((
                Phi_H_2D_x(X_P, Y_P, self.n_hb),
                Phi_RBF_2D_x(X_P, Y_P, self.X_C, self.Y_C, self.c_k, self.basis)
                ))
            # We compute the derivatives of the velocity field along x
            dUdX = Matrix_Phi_2D_X_P_der_x.dot(W_u)
            dVdX = Matrix_Phi_2D_X_P_der_x.dot(W_v)

            # We compute Phi_y on X_P
            Matrix_Phi_2D_X_P_der_y = np.hstack((
                Phi_H_2D_y(X_P, Y_P, self.n_hb),
                Phi_RBF_2D_y(X_P, Y_P, self.X_C, self.Y_C, self.c_k, self.basis)
                ))
            # We compute the derivatives of the velocity field along y
            dUdY = Matrix_Phi_2D_X_P_der_y.dot(W_u)
            dVdY = Matrix_Phi_2D_X_P_der_y.dot(W_v)

            #forcing term is evaluated
            source_term = -rho*(dUdX**2+2*dUdY*dVdX+dVdY**2)

        elif len(points) == 3 and self.dimension == '3D':
            # assign the points in X and Y
            X_P = points[0]
            Y_P = points[1]
            Z_P = points[2]
            W_u = self.weights[0*self.n_b:1*self.n_b]
            W_v = self.weights[1*self.n_b:2*self.n_b]
            W_w = self.weights[2*self.n_b:3*self.n_b]

            # We compute Phi_x on X_P
            Matrix_Phi_3D_X_P_der_x = np.hstack((
                Phi_H_3D_x(X_P, Y_P, Z_P, self.n_hb),
                Phi_RBF_3D_x(X_P, Y_P, Z_P, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                ))
            # We compute the derivatives of the velocity field along x
            dUdX = Matrix_Phi_3D_X_P_der_x.dot(W_u)
            dVdX = Matrix_Phi_3D_X_P_der_x.dot(W_v)
            dWdX = Matrix_Phi_3D_X_P_der_x.dot(W_w)

            # We compute Phi_y on X_P
            Matrix_Phi_3D_X_P_der_y = np.hstack((
                Phi_H_3D_y(X_P, Y_P, Z_P, self.n_hb),
                Phi_RBF_3D_y(X_P, Y_P, Z_P, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                ))
            # We compute the derivatives of the velocity field along y
            dUdY = Matrix_Phi_3D_X_P_der_y.dot(W_u)
            dVdY = Matrix_Phi_3D_X_P_der_y.dot(W_v)
            dWdY = Matrix_Phi_3D_X_P_der_y.dot(W_w)

            # We compute Phi_z on X_P
            Matrix_Phi_3D_X_P_der_z = np.hstack((
                Phi_H_3D_z(X_P, Y_P, Z_P, self.n_hb),
                Phi_RBF_3D_z(X_P, Y_P, Z_P, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                ))
            # We compute the derivatives of the velocity field along y
            dUdZ = Matrix_Phi_3D_X_P_der_z.dot(W_u)
            dVdZ = Matrix_Phi_3D_X_P_der_z.dot(W_v)
            dWdZ = Matrix_Phi_3D_X_P_der_z.dot(W_w)

            #forcing term is evaluated
            source_term = -rho*(dUdX**2+dVdY**2+dWdZ**2 + 2*dUdY*dVdX + 2*dUdZ*dWdX + 2*dVdZ*dWdY)

        else:
            raise ValueError('Length of points is invalid for Type ' + self.dimension)

        return source_term


    def Get_Pressure_Neumann(self, points, normals, rho, mu):
        """
        This function evaluates the Neumann boundary conditions for the pressure
        integration in equation (29) from the original paper (see video tutorial 1 for more info)

        :type points: list
        :param points:
            Contains the points at which the Neumann constraint is evaluated.
            If the model is 2D, then this has [X_P, Y_P].
            If the model is 3D, then this has [X_P, Y_P, Z_P].

        :type normals: list
        :param normals:
            Contains normals of the points at which the Neumann constraint is evaluated.
            If the model is 2D, then this has [n_x, n_y].
            If the model is 3D, then this has [n_x, n_y, n_z].

        :type rho: float
        :param rho:
            Density of the fluid.

        :type mu: float
        :param mu:
            Dynamic viscosity of the fluid.

        :return: P_neu
            Normal pressure in equation (29).
        """

        # Check the input is correct
        assert type(points) == list, 'points must be a list'
        assert type(normals) == list, 'normals must be a list'
        assert len(points) == len(normals), 'Length of points must be equal to the length of normals'
        # Check if we have 2D or 3D data
        if len(points) == 2 and self.dimension == '2D': # 2D
            # Assign the points
            X_N = points[0]
            Y_N = points[1]
            # Assign the normals
            n_x = normals[0]
            n_y = normals[1]
            # Assign the weights
            W_u = self.weights[:self.n_b]
            W_v = self.weights[self.n_b:]
            # Compute the matrix Phi_x on X_N
            Matrix_Phi_2D_X_N_der_x = np.hstack((
                Phi_H_2D_x(X_N, Y_N, self.n_hb),
                Phi_RBF_2D_x(X_N, Y_N, self.X_C, self.Y_C, self.c_k, self.basis)
                ))
            # Compute the derivatives along x
            dUdX = Matrix_Phi_2D_X_N_der_x.dot(W_u)
            dVdX = Matrix_Phi_2D_X_N_der_x.dot(W_v)

            # Compute the matrix Phi_y on X_N
            Matrix_Phi_2D_X_N_der_y = np.hstack((
                Phi_H_2D_y(X_N, Y_N, self.n_hb),
                Phi_RBF_2D_y(X_N, Y_N, self.X_C, self.Y_C, self.c_k, self.basis)
                ))
            # Compute the derivatives along y
            dUdY = Matrix_Phi_2D_X_N_der_y.dot(W_u)
            dVdY = Matrix_Phi_2D_X_N_der_y.dot(W_v)

            # Compute the matrix Phi on X_N
            Matrix_Phi_2D_X_N = np.hstack((
                Phi_H_2D(X_N, Y_N, self.n_hb),
                Phi_RBF_2D(X_N, Y_N, self.X_C, self.Y_C, self.c_k, self.basis)
                ))
            # Compute the velocities
            U = Matrix_Phi_2D_X_N.dot(W_u)
            V = Matrix_Phi_2D_X_N.dot(W_v)

            # Compute the Laplacian on X_N
            L_X_N = np.hstack((
                Phi_H_2D_Laplacian(X_N, Y_N, self.n_hb),
                Phi_RBF_2D_Laplacian(X_N, Y_N, self.X_C, self.Y_C, self.c_k, self.basis)
                ))
            # Compute the Laplacian for U and V
            L_U = L_X_N.dot(W_u)
            L_V = L_X_N.dot(W_v)

            # Compute the pressure normals
            P_N_x = mu*L_U - rho * (U*dUdX + V*dUdY)
            P_N_y = mu*L_V - rho * (U*dVdX + V*dVdY)

            # Multiply with the normals to get the projected pressure
            P_Neu = P_N_x * n_x + P_N_y * n_y

        elif len(points) == 3 and self.dimension == '3D':
            # Assign the points
            X_N = points[0]
            Y_N = points[1]
            Z_N = points[2]
            # Assign the normals
            n_x = normals[0]
            n_y = normals[1]
            n_z = normals[2]
            # Assign the weights
            W_u = self.weights[0*self.n_b:1*self.n_b]
            W_v = self.weights[1*self.n_b:2*self.n_b]
            W_w = self.weights[2*self.n_b:3*self.n_b]

            # Compute the matrix Phi_x on X_N
            Matrix_Phi_3D_X_N_der_x = np.hstack((
                Phi_H_3D_x(X_N, Y_N, Z_N, self.n_hb),
                Phi_RBF_3D_x(X_N, Y_N, Z_N, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                ))
            # Compute the derivatives along x
            dUdX = Matrix_Phi_3D_X_N_der_x.dot(W_u)
            dVdX = Matrix_Phi_3D_X_N_der_x.dot(W_v)
            dWdX = Matrix_Phi_3D_X_N_der_x.dot(W_w)

            # Compute the matrix Phi_y on X_N
            Matrix_Phi_3D_X_N_der_y = np.hstack((
                Phi_H_3D_y(X_N, Y_N, Z_N, self.n_hb),
                Phi_RBF_3D_y(X_N, Y_N, Z_N, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                ))
            # Compute the derivatives along y
            dUdY = Matrix_Phi_3D_X_N_der_y.dot(W_u)
            dVdY = Matrix_Phi_3D_X_N_der_y.dot(W_v)
            dWdY = Matrix_Phi_3D_X_N_der_y.dot(W_w)

            # Compute the matrix Phi_z on X_N
            Matrix_Phi_3D_X_N_der_z = np.hstack((
                Phi_H_3D_z(X_N, Y_N, Z_N, self.n_hb),
                Phi_RBF_3D_z(X_N, Y_N, Z_N, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                ))
            # Compute the derivatives along y
            dUdZ = Matrix_Phi_3D_X_N_der_z.dot(W_u)
            dVdZ = Matrix_Phi_3D_X_N_der_z.dot(W_v)
            dWdZ = Matrix_Phi_3D_X_N_der_z.dot(W_w)

            # Compute the matrix Phi on X_N
            Matrix_Phi_3D_X_N = np.hstack((
                Phi_H_3D(X_N, Y_N, Z_N, self.n_hb),
                Phi_RBF_3D(X_N, Y_N, Z_N, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                ))
            # Compute the velocities
            U = Matrix_Phi_3D_X_N.dot(W_u)
            V = Matrix_Phi_3D_X_N.dot(W_v)
            W = Matrix_Phi_3D_X_N.dot(W_w)

            # Compute the Laplacian on X_N
            L_X_N = np.hstack((
                Phi_H_3D_Laplacian(X_N, Y_N, Z_N, self.n_hb),
                Phi_RBF_3D_Laplacian(X_N, Y_N, Z_N, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                ))
            # Compute the Laplacian for U and V
            L_U = L_X_N.dot(W_u)
            L_V = L_X_N.dot(W_v)
            L_W = L_X_N.dot(W_w)

            # Compute the pressure normals
            P_N_x = mu*L_U - rho * (U*dUdX + V*dUdY + W*dUdZ)
            P_N_y = mu*L_V - rho * (U*dVdX + V*dVdY + W*dVdZ)
            P_N_z = mu*L_W - rho * (U*dWdX + V*dWdY + W*dWdZ)

            # Multiply with the normals to get the projected pressure
            P_Neu = P_N_x * n_x + P_N_y * n_y + P_N_z * n_z

        else:
            raise ValueError('Length of points is invalid for Type ' + self.dimension)

        return P_Neu



    def _add_constraint_collocations(self):
        """
        This function adds collocation points where constraints are set in 2D.
        It is called by either of the constraint functions.

        Assigns
        -------

        X_c, Y_c, (Z_c) : 1d numpy.ndarrays
            Updates of the Collocation points in 2D (3D).

        c_k, d_k : 1d numpy.ndarrays
            Updates of the shape parameter and diameter of the RBFs.

        """

        # Check the dimensions

        if self.dimension == '2D':
            # Assemble all the points where there is constraints
            X_constr = np.concatenate((self.X_N, self.X_D))
            Y_constr = np.concatenate((self.Y_N, self.Y_D))
            if self.model == 'laminar':
                X_constr = np.concatenate((X_constr, self.X_Div))
                Y_constr = np.concatenate((Y_constr, self.Y_Div))
            # Retain only the unique values
            X_unique, Y_unique = np.unique(np.column_stack((X_constr, Y_constr)), axis=0).T
            c_ks, d_ks = self._compute_ck_dk(X_unique, Y_unique)

        elif self.dimension == '3D':
            # Assemble all the points where there is constraints
            X_constr = np.concatenate((self.X_N, self.X_D))
            Y_constr = np.concatenate((self.Y_N, self.Y_D))
            Z_constr = np.concatenate((self.Z_N, self.Z_D))
            if self.model == 'laminar':
                X_constr = np.concatenate((X_constr, self.X_Div))
                Y_constr = np.concatenate((Y_constr, self.Y_Div))
                Z_constr = np.concatenate((Z_constr, self.Z_Div))
            # Retain only the unique values
            X_unique, Y_unique, Z_unique = np.unique(np.column_stack((X_constr, Y_constr, Z_constr)), axis=0).T
            c_ks, d_ks = self._compute_ck_dk(X_unique, Y_unique, Z_unique)

        # concatenate with the existing basis parameters
        self.c_k = np.concatenate((self.c_k, c_ks))
        self.d_k = np.concatenate((self.d_k, d_ks))
        self.X_C = np.concatenate((self.X_C, X_unique))
        self.Y_C = np.concatenate((self.Y_C, Y_unique))

        if self.dimension == '3D':
            self.Z_C = np.concatenate((self.Z_C, Z_unique))

        return

    def _compute_ck_dk(self, X, Y, Z=None):
        # Get the number of constraints
        n_unique = X.shape[0]
        # Initialize an empty array for the shape parameters
        c_ks = np.zeros(n_unique)

        # Loop over all constraints
        for k in range(n_unique):
            if self.dimension == '2D' and Z is None:
                # X_unique, Y_unique = points_unique
                # Get the distance to all collocation points
                dist_to_colloc = np.sqrt((self.X_C - X[k]) ** 2 +
                                         (self.Y_C - Y[k]) ** 2)
                # Get the distance to all constraints, except for itself
                dist_to_constr = np.sqrt((np.delete(X, k) - X[k]) ** 2 +
                                         (np.delete(Y, k) - Y[k]) ** 2)
            elif self.dimension == '3D' and Z is not None:
                # X_unique, Y_unique, Z_unique = points_unique
                # Get the distance to all collocation points
                dist_to_colloc = np.sqrt((self.X_C - X[k]) ** 2 +
                                         (self.Y_C - Y[k]) ** 2 +
                                         (self.Z_C - Z[k]) ** 2)
                # Get the distance to all constraints, except for itself
                dist_to_constr = np.sqrt((np.delete(X, k) - X[k]) ** 2 +
                                         (np.delete(Y, k) - Y[k]) ** 2 +
                                         (np.delete(Z, k) - Z[k]) ** 2)
            else:
                pass

            # Set the max and min values of c_k
            if self.basis == 'gauss':  # Gaussians
                c_k = np.sqrt(-np.log(self.eps_l)) / np.concatenate((dist_to_colloc, dist_to_constr))
                c_ks[k] = np.max(c_k)
            elif self.basis == 'c4':  # C4
                c_k = np.concatenate((dist_to_colloc, dist_to_constr)) / np.sqrt(1 - self.eps_l ** 0.2)
                c_ks[k] = np.min(c_k)

        # clip the shape factors and get the diameters
        c_ks, d_ks = get_shape_parameter_and_diameter(self.r_mM, c_ks, self.basis)
        return c_ks, d_ks

    def _compute_Phi_matrix(self, X_eval, Y_eval, Z_eval=None):
        if self.dimension == '2D':
            Phi_2D = np.hstack((
                Phi_H_2D(X_eval, Y_eval, self.n_hb),
                Phi_RBF_2D(X_eval, Y_eval, self.X_C, self.Y_C, self.c_k, self.basis)
            ))
            return Phi_2D

        if self.dimension == '3D':
            Phi_3D = np.hstack((
                Phi_H_3D(X_eval, Y_eval, Z_eval, self.n_hb),
                Phi_RBF_3D(X_eval, Y_eval, Z_eval,
                           self.X_C, self.Y_C, self.Z_C,
                           self.c_k, self.basis)
            ))
            return Phi_3D

    def _compute_dxPhi_matrix(self, X_eval, Y_eval, Z_eval=None):
        if self.dimension == '2D':
            Phi_RBF_2D_X_der_x, Phi_RBF_2D_X_der_y = Phi_RBF_2D_Deriv(
                X_eval, Y_eval, self.X_C, self.Y_C, self.c_k, self.basis, order=1)

            Phi_2D_X_der_x = np.hstack((
                Phi_H_2D_x(X_eval, Y_eval, self.n_hb),
                Phi_RBF_2D_X_der_x
            ))
            Phi_2D_X_der_y = np.hstack((
                Phi_H_2D_y(X_eval, Y_eval, self.n_hb),
                Phi_RBF_2D_X_der_y
            ))

            return Phi_2D_X_der_x, Phi_2D_X_der_y

        if self.dimension == '3D':
            Phi_RBF_3D_X_der_x, Phi_RBF_3D_X_der_y, Phi_RBF_3D_X_der_z = Phi_RBF_3D_Deriv(
                X_eval, Y_eval, Z_eval,
                self.X_C, self.Y_C, self.Z_C,
                self.c_k, self.basis, order=1)

            # Compute Phi_x, Phi_y, Phi_z on the evaluation and collocation points
            Phi_3D_X_der_x = np.hstack((
                Phi_H_3D_x(X_eval, Y_eval, Z_eval, self.n_hb),
                Phi_RBF_3D_X_der_x
            ))
            Phi_3D_X_der_y = np.hstack((
                Phi_H_3D_y(X_eval, Y_eval, Z_eval, self.n_hb),
                Phi_RBF_3D_X_der_y
            ))
            Phi_3D_X_der_z = np.hstack((
                Phi_H_3D_z(X_eval, Y_eval, Z_eval, self.n_hb),
                Phi_RBF_3D_X_der_z
            ))
            return Phi_3D_X_der_x, Phi_3D_X_der_y, Phi_3D_X_der_z


    def _compute_dxxPhi_matrix(self, X_eval, Y_eval, Z_eval=None):
        if self.dimension == '2D':
            Phi_RBF_2D_X_der_xx, Phi_RBF_2D_X_der_yy, Phi_RBF_2D_X_der_xy = Phi_RBF_2D_Deriv(
                X_eval, Y_eval,
                self.X_C, self.Y_C,
                self.c_k, self.basis, order=2
            )

            return Phi_RBF_2D_X_der_xx, Phi_RBF_2D_X_der_yy, Phi_RBF_2D_X_der_xy

        elif self.dimension == '3D':
            Phi_RBF_3D_X_der_xx, Phi_RBF_3D_X_der_yy, Phi_RBF_3D_X_der_zz,\
                Phi_RBF_3D_X_der_xy, Phi_RBF_3D_X_der_xz, Phi_RBF_3D_X_der_yz = Phi_RBF_3D_Deriv(
                X_eval, Y_eval, Z_eval,
                self.X_C, self.Y_C, self.Z_C,
                self.c_k, self.basis, order=2
            )
            return Phi_RBF_3D_X_der_xx, Phi_RBF_3D_X_der_yy, Phi_RBF_3D_X_der_zz,\
                Phi_RBF_3D_X_der_xy, Phi_RBF_3D_X_der_xz, Phi_RBF_3D_X_der_yz

    def _add_divergence_penalty(self, alpha_div):
        """
        Adds the divergence-free penalty of strength ``alpha_div`` to the matrix A.
        The penalty is applied in the points X_Div
        """

        if self.dimension == '2D':
            # Compute Phi_x, Phi_y on the data and collocation points
            Phi_X_Div_p_der_x, Phi_X_Div_p_der_y = (
                self._compute_dxPhi_matrix(self.X_Div, self.Y_Div)
            )

            # Compute the individual matrix products between x, y and z
            # For the diagonal
            PhiXT_dot_PhiX = Phi_X_Div_p_der_x.T.dot(Phi_X_Div_p_der_x)
            PhiYT_dot_PhiY = Phi_X_Div_p_der_y.T.dot(Phi_X_Div_p_der_y)
            # For the off-diagonal elements
            PhiXT_dot_PhiY = Phi_X_Div_p_der_x.T.dot(Phi_X_Div_p_der_y)

            # Diagonal elements
            self.A[self.n_b_j * 0: self.n_b_j * 1, self.n_b_j * 0: self.n_b_j * 1] += 2 * alpha_div * PhiXT_dot_PhiX
            self.A[self.n_b_j * 1: self.n_b_j * 2, self.n_b_j * 1: self.n_b_j * 2] += 2 * alpha_div * PhiYT_dot_PhiY
            # Upper off-diagonal elements
            self.A[self.n_b_j * 0: self.n_b_j * 1, self.n_b_j * 1: self.n_b_j * 2] += 2 * alpha_div * PhiXT_dot_PhiY
            # Lower off-diagonal elements
            self.A[self.n_b_j * 1: self.n_b_j * 2, self.n_b_j * 0: self.n_b_j * 1] += 2 * alpha_div * PhiXT_dot_PhiY.T

        elif self.dimension == '3D':
            # Compute Phi_x, Phi_y on the data and collocation points
            Phi_X_Div_p_der_x, Phi_X_Div_p_der_y, Phi_X_Div_p_der_z= (
                self._compute_dxPhi_matrix(self.X_Div, self.Y_Div, self.Z_Div)
            )

            # Compute the individual matrix products between x, y and z
            # For the diagonal
            PhiXT_dot_PhiX = Phi_X_Div_p_der_x.T.dot(Phi_X_Div_p_der_x)
            PhiYT_dot_PhiY = Phi_X_Div_p_der_y.T.dot(Phi_X_Div_p_der_y)
            PhiZT_dot_PhiZ = Phi_X_Div_p_der_z.T.dot(Phi_X_Div_p_der_z)
            # For the off-diagonal elements
            PhiXT_dot_PhiY = Phi_X_Div_p_der_x.T.dot(Phi_X_Div_p_der_y)
            PhiXT_dot_PhiZ = Phi_X_Div_p_der_x.T.dot(Phi_X_Div_p_der_z)
            PhiYT_dot_PhiZ = Phi_X_Div_p_der_y.T.dot(Phi_X_Div_p_der_z)

            # Diagonal eleemtns
            self.A[self.n_b_j * 0: self.n_b_j * 1, self.n_b_j * 0: self.n_b_j * 1] += 2 * alpha_div * PhiXT_dot_PhiX
            self.A[self.n_b_j * 1: self.n_b_j * 2, self.n_b_j * 1: self.n_b_j * 2] += 2 * alpha_div * PhiYT_dot_PhiY
            self.A[self.n_b_j * 2: self.n_b_j * 3, self.n_b_j * 2: self.n_b_j * 3] += 2 * alpha_div * PhiZT_dot_PhiZ

            # Upper off-diagonal elements
            self.A[self.n_b_j * 0: self.n_b_j * 1, self.n_b_j * 1: self.n_b_j * 2] += 2 * alpha_div * PhiXT_dot_PhiY
            self.A[self.n_b_j * 0: self.n_b_j * 1, self.n_b_j * 2: self.n_b_j * 3] += 2 * alpha_div * PhiXT_dot_PhiZ
            self.A[self.n_b_j * 1: self.n_b_j * 2, self.n_b_j * 2: self.n_b_j * 3] += 2 * alpha_div * PhiYT_dot_PhiZ

            # Lower off-diagonal elements
            self.A[self.n_b_j * 1: self.n_b_j * 2, self.n_b_j * 0: self.n_b_j * 1] += 2 * alpha_div * PhiXT_dot_PhiY.T
            self.A[self.n_b_j * 2: self.n_b_j * 3, self.n_b_j * 0: self.n_b_j * 1] += 2 * alpha_div * PhiXT_dot_PhiZ.T
            self.A[self.n_b_j * 2: self.n_b_j * 3, self.n_b_j * 1: self.n_b_j * 2] += 2 * alpha_div * PhiYT_dot_PhiZ.T

        return



    def _get_rescale(self):
        """
        This function gets the rescaling factor of the r.h.s of the linear system.
        This ensures stability during the solution process.

        Assigns
        -------

        rescale : float
            Absolute value of the largest absolute value in the data

        """

        # extract the data based on the model and dimensions
        if self.model == 'scalar':
            data = np.concatenate(self.u)
        elif self.model == 'laminar':
            if self.dimension == '2D':
                data = np.concatenate((self.u, self.v))
            elif self.dimension == '3D':
                data = np.concatenate((self.u, self.v, self.w))
            else:
                pass

        # get the maximum absolute value which ensures that the r.h.s is between -1 and 1
        self.scale_U = np.abs(data[np.argmax(np.abs(data))])
        # If the r.h.s is 0 (Laplace problem), do not rescale
        if np.abs(self.scale_U) < 1e-10:
            self.scale_U = 1

        return


    def _extract_bounds(self, bounds):

        if self.dimension == '2D':
            if bounds is None:
                x_min, x_max = self.X_G.min(), self.X_G.max()
                y_min, y_max = self.Y_G.min(), self.Y_G.max()
                self.bounds = [x_min, x_max, y_min, y_max]
            else:
                self.bounds = bounds

        if self.dimension == '3D':
            if bounds is None:
                x_min, x_max = self.X_G.min(), self.X_G.max()
                y_min, y_max = self.Y_G.min(), self.Y_G.max()
                z_min, z_max = self.Z_G.min(), self.Z_G.max()
                self.bounds = [x_min, x_max, y_min, y_max, z_min, z_max]
            else:
                self.bounds = bounds

        return self.bounds


# =============================================================================
#  Utilities functions
#  These functions are not needed/called by the user. They are simply helper
#  functions required to assemble and solve the linear systems. In the current
#  release of SPICY, these are:
#  - RBF functions and their derivatives in 2D/3D
#  - Harmonics functions and their derivatives in 2D/3D
#  - Adding collocation points in the constraints
# =============================================================================



# =============================================================================
#  RBF functions in 2D
#  Includes: Phi_RBF_2D, Phi_RBF_2D_x, Phi_RBF_2D_y, Phi_RBF_2D_Laplacian
# =============================================================================

# def Phi_RBF_2D(X_G, Y_G, X_C, Y_C, c_k, basis):
#     """
#     Get the basis matrix at the points (X_G,Y_G) from RBFs at the collocation points
#     at (X_C,Y_C), having shape factors c_k. The output is a matrix of side (n_p) x (n_c).
#     The basis can be 'c4' or 'gauss'.
#     """
#     # This is the contribution of the RBF part
#     n_b = len(X_C); n_p = len(X_G)
#     Phi_RBF = np.zeros((n_p, n_b))

#     # What comes next depends on the type of chosen RBF
#     if basis == 'gauss':
#         # Iterate over all basis elements
#         for r in range(n_b):
#             # Compute the Gaussian
#             gaussian = np.exp(-c_k[r]**2 * ((X_G-X_C[r])**2 + (Y_G-Y_C[r])**2))
#             # Assemble into matrix
#             Phi_RBF[:,r] = gaussian

#     elif basis == 'c4':
#         # Iterate over all basis elements
#         for r in range(n_b):
#             # Get distance
#             d = np.sqrt((X_G-X_C[r])**2 + (Y_G-Y_C[r])**2)
#             # Compute Phi
#             phi = (1+d/c_k[r])**5 * (1-d/c_k[r])**5
#             # Compact support
#             phi[np.abs(d) > c_k[r]] = 0
#             # Assemble into matrix
#             Phi_RBF[:,r] = phi

#     # Return the matrix
#     return Phi_RBF


# def Phi_RBF_2D_x(X_G, Y_G, X_C, Y_C, c_k, basis):
#     """
#     Get the derivative along x of the basis matrix at the points (X_G,Y_G) from
#     RBFs at the collocation points at (X_C,Y_C), having shape factors c_k. The
#     output is a matrix of side (n_p) x (n_c). The basis can be 'c4' or 'gauss'.
#     """
#     # number of bases (n_b) and points (n_p)
#     n_b = len(X_C); n_p = len(X_G)
#     # Initialize the matrix
#     Phi_RBF_x = np.zeros((n_p, n_b))

#     # What comes next depends on the type of chosen RBF
#     if basis == 'gauss':
#         # Iterate over all basis elements
#         for r in range(n_b):
#             # Compute the Gaussian
#             gaussian = np.exp(-c_k[r]**2 * ((X_G-X_C[r])**2 + (Y_G-Y_C[r])**2))
#             # Multiply with inner term and assemble into matrix
#             Phi_RBF_x[:,r] = - 2*c_k[r]**2 * (X_G-X_C[r])*gaussian

#     elif basis == 'c4':
#         # Iterate over all basis elements
#         for r in range(n_b):
#             # Get distance
#             d = np.sqrt((X_G-X_C[r])**2 + (Y_G-Y_C[r])**2)
#             # Compute derivative along x
#             phi = - 10/c_k[r]**10 * (c_k[r]+d)**4 * (c_k[r]-d)**4 * (X_G-X_C[r])
#             # Compact support
#             phi[np.abs(d) > c_k[r]] = 0
#             # Assemble into matrix
#             Phi_RBF_x[:,r] = phi

#     # Return the matrix
#     return Phi_RBF_x


# def Phi_RBF_2D_y(X_G, Y_G, X_C, Y_C, c_k, basis):
#     """
#     Get the derivative along y of the basis matrix at the points (X_G,Y_G) from
#     RBFs at the collocation points at (X_C,Y_C), having shape factors c_k. The
#     output is a matrix of side (n_p) x (n_c). The basis can be 'c4' or 'gauss'.
#     """
#     # number of bases (n_b) and points (n_p)
#     n_b = len(X_C); n_p = len(X_G)
#     # Initialize the matrix
#     Phi_RBF_y = np.zeros((n_p, n_b))

#     # What comes next depends on the type of chosen RBF
#     if basis == 'gauss':
#         # Iterate over all basis elements
#         for r in range(n_b):
#             # Compute the Gaussian
#             gaussian = np.exp(-c_k[r]**2 * ((X_G-X_C[r])**2 + (Y_G-Y_C[r])**2))
#             # Multiply with inner term and assemble into matrix
#             Phi_RBF_y[:,r] = - 2*c_k[r]**2 * (Y_G-Y_C[r])*gaussian

#     elif basis == 'c4':
#         # Iterate over all basis elements
#         for r in range(n_b):
#             # Get distance
#             d = np.sqrt((X_G-X_C[r])**2 + (Y_G-Y_C[r])**2)
#             # Compute derivative along y
#             phi = - 10/c_k[r]**10 * (c_k[r]+d)**4 * (c_k[r]-d)**4 * (Y_G-Y_C[r])
#             # Compact support
#             phi[np.abs(d) > c_k[r]] = 0
#             # Assemble into matrix
#             Phi_RBF_y[:,r] = phi

#     # Return the matrix
#     return Phi_RBF_y


# def Phi_RBF_2D_Laplacian(X_G, Y_G, X_C, Y_C, c_k, basis):
#     """
#     Get the Laplacian of the basis matrix at the points (X_G,Y_G) from
#     RBFs at the collocation points at (X_C,Y_C), having shape factors c_k. The
#     output is a matrix of side (n_p) x (n_c). The basis can be 'c4' or 'gauss'.
#     """
#     # number of bases (n_b) and points (n_p)
#     n_b = len(X_C); n_p = len(X_G)
#     # Initialize the matrix
#     Lap_RBF = np.zeros((n_p,n_b))

#     # What comes next depends on the type of chosen RBF
#     if basis == 'gauss':
#         # Iterate over all basis elements
#         for r in range(n_b):
#             # Compute the Gaussian
#             gaussian = np.exp(-c_k[r]**2*((X_G-X_C[r])**2 + (Y_G-Y_C[r])**2))
#             # Get second derivative along x and y
#             Partial_xx = 4*c_k[r]**4*(X_C[r]-X_G)**2*gaussian-2*c_k[r]**2*gaussian
#             Partial_yy = 4*c_k[r]**4*(Y_C[r]-Y_G)**2*gaussian-2*c_k[r]**2*gaussian
#             # Assemble into matrix
#             Lap_RBF[:,r] = Partial_xx+Partial_yy

#     elif basis == 'c4':
#         # Iterate over all basis elements
#         for r in range(n_b):
#             # Get distance
#             d = np.sqrt((X_G-X_C[r])**2 + (Y_G-Y_C[r])**2)
#             # Compute the prefactor in the second derivative
#             factor = 10 / c_k[r]**10 * (c_k[r] + d)**3 * (c_k[r] - d)**3
#             # Multiply with inner derivative
#             Partial_xx = factor * (8*(X_G - X_C[r])**2 - c_k[r]**2 + d**2)
#             Partial_yy = factor * (8*(Y_G - Y_C[r])**2 - c_k[r]**2 + d**2)
#             # Compute Laplacian
#             Laplacian = Partial_xx + Partial_yy
#             # Compact support
#             Laplacian[np.abs(d) > c_k[r]] = 0
#             # Assemble into matrix
#             Lap_RBF[:,r] = Laplacian

#     # Return the matrix
#     return Lap_RBF


# # =============================================================================
# #  RBF functions in 3D
# #  Includes: Phi_RBF_2D, Phi_RBF_2D_x, Phi_RBF_2D_y, Phi_RBF_2D_Laplacian
# # =============================================================================

# def Phi_RBF_3D(X_G, Y_G, Z_G, X_C, Y_C, Z_C, c_k, basis):
#     """
#     Get the basis matrix at the points (X_G,Y_G,Z_G) from RBFs at the collocation points
#     at (X_C,Y_C,Z_C), having shape factors c_k. The output is a matrix of side (n_p) x (n_c).
#     The basis can be 'c4' or 'gauss'.
#     """
#     # This is the contribution of the RBF part
#     n_b = len(X_C); n_p = len(X_G)
#     Phi_RBF = np.zeros((n_p, n_b))

#     # What comes next depends on the type of chosen RBF
#     if basis == 'gauss':
#         # Iterate over all basis elements
#         for r in range(n_b):
#             # Compute the Gaussian
#             gaussian = np.exp(-c_k[r]**2 * ((X_G-X_C[r])**2 + (Y_G-Y_C[r])**2 + (Z_G-Z_C[r])**2))
#             # Assemble into matrix
#             Phi_RBF[:,r] = gaussian

#     elif basis == 'c4':
#         # Iterate over all basis elements
#         for r in range(n_b):
#             # Get distance
#             d = np.sqrt((X_G-X_C[r])**2 + (Y_G-Y_C[r])**2 + (Z_G-Z_C[r])**2)
#             # Compute Phi
#             phi = (1+d/c_k[r])**5 * (1-d/c_k[r])**5
#             # Compact support
#             phi[np.abs(d) > c_k[r]] = 0
#             # Assemble into matrix
#             Phi_RBF[:,r] = phi

#     return Phi_RBF


# def Phi_RBF_3D_x(X_G, Y_G, Z_G, X_C, Y_C, Z_C, c_k, basis):
#     """
#     Get the derivative along x of the basis matrix at the points (X_G,Y_G,Z_G) from
#     RBFs at the collocation points at (X_C,Y_C,Z_G), having shape factors c_k. The
#     output is a matrix of side (n_p) x (n_c). The basis can be 'c4' or 'gauss'.
#     """
#     # number of bases (n_b) and points (n_p)
#     n_b = len(X_C); n_p = len(X_G)
#     # Initialize the matrix
#     Phi_RBF_x = np.zeros((n_p, n_b))

#     # What comes next depends on the type of chosen RBF
#     if basis == 'gauss':
#         # Iterate over all basis elements
#         for r in range(n_b):
#             # Compute the Gaussian
#             gaussian = np.exp(-c_k[r]**2 * ((X_G-X_C[r])**2 + (Y_G-Y_C[r])**2 + (Z_G-Z_C[r])**2))
#             # Multiply with inner term and assemble into matrix
#             Phi_RBF_x[:,r] = - 2*c_k[r]**2 * (X_G - X_C[r])*gaussian

#     elif basis == 'c4':
#         # Iterate over all basis elements
#         for r in range(n_b):
#             # Get distance
#             d = np.sqrt((X_G-X_C[r])**2 + (Y_G-Y_C[r])**2 + (Z_G-Z_C[r])**2)
#             # Compute derivative along x
#             phi = - 10/c_k[r]**10 * (c_k[r]+d)**4 * (c_k[r]-d)**4 * (X_G-X_C[r])
#             # Compact support
#             phi[np.abs(d) > c_k[r]] = 0
#             # Assemble into matrix
#             Phi_RBF_x[:,r] = phi

#     # Return the matrix
#     return Phi_RBF_x


# def Phi_RBF_3D_y(X_G, Y_G, Z_G, X_C, Y_C, Z_C, c_k, basis):
#     """
#     Get the derivative along y of the basis matrix at the points (X_G,Y_G,Z_G) from
#     RBFs at the collocation points at (X_C,Y_C,Z_G), having shape factors c_k. The
#     output is a matrix of side (n_p) x (n_c). The basis can be 'c4' or 'gauss'.
#     """
#     # number of bases (n_b) and points (n_p)
#     n_b = len(X_C); n_p = len(X_G)
#     # Initialize the matrix
#     Phi_RBF_y = np.zeros((n_p, n_b))

#     # What comes next depends on the type of chosen RBF
#     if basis == 'gauss':
#         # Iterate over all basis elements
#         for r in range(n_b):
#             # Compute the Gaussian
#             gaussian = np.exp(-c_k[r]**2 * ((X_G-X_C[r])**2 + (Y_G-Y_C[r])**2 + (Z_G-Z_C[r])**2))
#             # Multiply with inner term and assemble into matrix
#             Phi_RBF_y[:,r] = - 2*c_k[r]**2 * (Y_G-Y_C[r])*gaussian

#     elif basis == 'c4':
#         # Iterate over all basis elements
#         for r in range(n_b):
#             # Get distance
#             d = np.sqrt((X_G-X_C[r])**2 + (Y_G-Y_C[r])**2 + (Z_G-Z_C[r])**2)
#             # Compute derivative along x
#             phi = - 10/c_k[r]**10 * (c_k[r]+d)**4 * (c_k[r]-d)**4 * (Y_G-Y_C[r])
#             # Compact support
#             phi[np.abs(d) > c_k[r]] = 0
#             # Assemble into matrix
#             Phi_RBF_y[:,r] = phi

#     # Return the matrix
#     return Phi_RBF_y


# def Phi_RBF_3D_z(X_G, Y_G, Z_G, X_C, Y_C, Z_C, c_k, basis):
#     """
#     Get the derivative along z of the basis matrix at the points (X_G,Y_G,Z_G) from
#     RBFs at the collocation points at (X_C,Y_C,Z_G), having shape factors c_k. The
#     output is a matrix of side (n_p) x (n_c). The basis can be 'c4' or 'gauss'.
#     """
#     # number of bases (n_b) and points (n_p)
#     n_b = len(X_C); n_p = len(X_G)
#     # Initialize the matrix
#     Phi_RBF_z = np.zeros((n_p, n_b))

#     # What comes next depends on the type of chosen RBF
#     if basis == 'gauss':
#         # Iterate over all basis elements
#         for r in range(n_b):
#             # Compute the Gaussian
#             gaussian = np.exp(-c_k[r]**2 * ((X_G-X_C[r])**2 + (Y_G-Y_C[r])**2 + (Z_G-Z_C[r])**2))
#             # Multiply with inner term and assemble into matrix
#             Phi_RBF_z[:,r] = - 2*c_k[r]**2 * (Z_G-Z_C[r])*gaussian

#     elif basis == 'c4':
#         # Iterate over all basis elements
#         for r in range(n_b):
#             # Get distance
#             d = np.sqrt((X_G-X_C[r])**2 + (Y_G-Y_C[r])**2 + (Z_G-Z_C[r])**2)
#             # Compute derivative along x
#             phi = - 10/c_k[r]**10 * (c_k[r]+d)**4 * (c_k[r]-d)**4 * (Z_G-Z_C[r])
#             # Compact support
#             phi[np.abs(d) > c_k[r]] = 0
#             # Assemble into matrix
#             Phi_RBF_z[:,r] = phi

#     # Return the matrix
#     return Phi_RBF_z

# def Phi_RBF_3D_Laplacian(X_G, Y_G, Z_G, X_C, Y_C, Z_C, c_k, basis):
#     """
#     Get the Laplacian of the basis matrix at the points (X_G,Y_G,Z_G) from
#     RBFs at the collocation points at (X_C,Y_C,Z_C), having shape factors c_k. The
#     output is a matrix of side (n_p) x (n_c). The basis can be 'c4' or 'gauss'.
#     """
#     # number of bases (n_b) and points (n_p)
#     n_b = len(X_C); n_p = len(X_G)
#     # Initialize the matrix
#     Lap_RBF = np.zeros((n_p,n_b))

#     # What comes next depends on the type of chosen RBF
#     if basis == 'gauss':
#         # Iterate over all basis elements
#         for r in range(n_b):
#             # Compute the Gaussian
#             gaussian = np.exp(-c_k[r]**2*((X_G-X_C[r])**2 + (Y_G-Y_C[r])**2 + (Z_G-Z_C[r])**2))
#             # Get second derivative along x and y
#             Partial_xx = 4*c_k[r]**4*(X_G-X_C[r])**2*gaussian - 2*c_k[r]**2*gaussian
#             Partial_yy = 4*c_k[r]**4*(Y_G-Y_C[r])**2*gaussian - 2*c_k[r]**2*gaussian
#             Partial_zz = 4*c_k[r]**4*(Z_G-Z_C[r])**2*gaussian - 2*c_k[r]**2*gaussian
#             # Assemble into matrix
#             Lap_RBF[:,r] = Partial_xx + Partial_yy  + Partial_zz

#     elif basis == 'c4':
#         # Iterate over all basis elements
#         for r in range(n_b):
#             # Get distance
#             d = np.sqrt((X_G-X_C[r])**2 + (Y_G-Y_C[r])**2 + (Z_G-Z_C[r])**2)
#             # Compute the prefactor in the second derivative
#             factor = 10 / c_k[r]**10 * (c_k[r] + d)**3 * (c_k[r] - d)**3
#             # Multiply with inner derivative
#             Partial_xx = factor * (8*(X_G - X_C[r])**2 - c_k[r]**2 + d**2)
#             Partial_yy = factor * (8*(Y_G - Y_C[r])**2 - c_k[r]**2 + d**2)
#             Partial_zz = factor * (8*(Z_G - Z_C[r])**2 - c_k[r]**2 + d**2)
#             # Compute Laplacian
#             Laplacian = Partial_xx + Partial_yy + Partial_zz
#             # Compact support
#             Laplacian[np.abs(d) > c_k[r]] = 0
#             # Assemble into matrix
#             Lap_RBF[:,r] = Laplacian

#     # Return the matrix
#     return Lap_RBF


# def add_constraint_collocations_2D(X_constr, Y_constr, X_C, Y_C, r_mM, eps_l, basis):
#     """
#     This function adds collocation points where constraints are set in 2D.

#     ----------------------------------------------------------------------------------------------------------------
#     Parameters
#     ----------
#     :param X_constr: np.ndarray
#         X coordinates of the constraints
#     :param Y_constr: np.ndarray
#         Y coordinates of the constraints
#     :param X_C: np.ndarray
#         X coordinates of the collocation points
#     :param Y_C: np.ndarray
#         Y coordinates of the collocation points
#     :param r_mM: list
#         Minimum and maximum radius of the RBFs
#     :param eps_l: float
#         Value of the RBF at its closest neighbor
#     :param basis: str
#         Type of basis function, must be c4 or Gaussian
#     """
#     # Get the number of constraints
#     n_constr = X_constr.shape[0]
#     # Initialize an empty array for the shape parameters
#     c_ks = np.zeros(n_constr)

#     # Check the basis
#     if basis == 'gauss': # Gaussians
#         # Set the max and min values of c_k
#         c_min = 1 / (r_mM[1]) * np.sqrt(np.log(2))
#         c_max = 1 / (r_mM[0]) * np.sqrt(np.log(2))
#         # Loop over all constraints
#         for k in range(n_constr):
#             # Get the distance to all collocation points
#             dist_to_colloc = np.sqrt((X_C - X_constr[k])**2 + (Y_C - Y_constr[k])**2)
#             # Get the distance to all constraints, except for itself
#             dist_to_constr = np.sqrt((np.delete(X_constr, k) - X_constr[k])**2+\
#                                      (np.delete(Y_constr, k) - Y_constr[k])**2)
#             # Set the max and min values of c_k
#             c_k = np.sqrt(-np.log(eps_l)) / np.concatenate((dist_to_colloc, dist_to_constr))
#             # crop to the minimum and maximum value
#             c_k[c_k < c_min] = c_min
#             c_k[c_k > c_max] = c_max
#             # get the maximum value in the case of the Gaussian
#             c_ks[k] = np.max(c_k)
#         # for plotting purposes, we store also the diameters
#         d_k = 2/c_ks*np.sqrt(np.log(2))

#     elif basis == 'c4': # C4
#         c_min = r_mM[0] / np.sqrt(1 - 0.5**0.2)
#         c_max = r_mM[1] / np.sqrt(1 - 0.5**0.2)
#         for k in range(n_constr):
#             # Get the distance to all collocation points
#             dist_to_colloc = np.sqrt((X_C - X_constr[k])**2 + (Y_C - Y_constr[k])**2)
#             # Get the distance to all constraints, except for itself
#             dist_to_constr = np.sqrt((np.delete(X_constr, k) - X_constr[k])**2+\
#                                      (np.delete(Y_constr, k) - Y_constr[k])**2)
#             # Set the max and min values of c_k
#             c_k = np.concatenate((dist_to_colloc, dist_to_constr)) / np.sqrt(1 - eps_l**0.2)
#             # crop to the minimum and maximum value
#             c_k[c_k < c_min] = c_min
#             c_k[c_k > c_max] = c_max
#             # get the minimum value in the case of the c4
#             c_ks[k] = np.min(c_k)
#         # for plotting purposes, we store also the diameters
#         d_k = 2*c_ks * np.sqrt(1 - 0.5**0.2)

#     return c_ks, d_k

# def add_constraint_collocations_3D(X_constr, Y_constr, Z_constr, X_C, Y_C, Z_C, r_mM, eps_l, basis):
#     """
#     This function adds collocation points where constraints are set in 3D.

#     ----------------------------------------------------------------------------------------------------------------
#     Parameters
#     ----------
#     :param X_constr: np.ndarray
#         X coordinates of the constraints
#     :param Y_constr: np.ndarray
#         Y coordinates of the constraints
#     :param Z_constr: np.ndarray
#         Z coordinates of the constraints
#     :param X_C: np.ndarray
#         X coordinates of the collocation points
#     :param Y_C: np.ndarray
#         Y coordinates of the collocation points
#     :param Z_C: np.ndarray
#         Z coordinates of the collocation points
#     :param r_mM: list
#         Minimum and maximum radius of the RBFs
#     :param eps_l: float
#         Value of the RBF at its closest neighbor
#     :param basis: str
#         Type of basis function, must be c4 or Gaussian
#     """
#     # Get the number of constraints
#     n_constr = X_constr.shape[0]
#     # Initialize an empty array for the shape parameters
#     c_ks = np.zeros(n_constr)

#     # Check the basis
#     if basis == 'gauss': # Gaussians
#         # Set the max and min values of c_k
#         c_min = 1 / (r_mM[1]) * np.sqrt(np.log(2))
#         c_max = 1 / (r_mM[0]) * np.sqrt(np.log(2))
#         # Loop over all constraints
#         for k in range(n_constr):
#             # Get the distance to all collocation points
#             dist_to_colloc = np.sqrt((X_C - X_constr[k])**2 +\
#                                      (Y_C - Y_constr[k])**2 +\
#                                      (Z_C - Z_constr[k])**2)
#             # Get the distance to all constraints, except for itself
#             dist_to_constr = np.sqrt((np.delete(X_constr, k) - X_constr[k])**2+\
#                                      (np.delete(Y_constr, k) - Y_constr[k])**2+\
#                                      (np.delete(Z_constr, k) - Z_constr[k])**2)
#             # Set the max and min values of c_k
#             c_k = np.sqrt(-np.log(eps_l)) / np.concatenate((dist_to_colloc, dist_to_constr))
#             # crop to the minimum and maximum value
#             c_k[c_k < c_min] = c_min
#             c_k[c_k > c_max] = c_max
#             # get the maximum value in the case of the Gaussian
#             c_ks[k] = np.max(c_k)
#         # for plotting purposes, we store also the diameters
#         d_k = 2/c_ks*np.sqrt(np.log(2))

#     elif basis == 'c4': # C4
#         c_min = r_mM[0] / np.sqrt(1 - 0.5**0.2)
#         c_max = r_mM[1] / np.sqrt(1 - 0.5**0.2)
#         for k in range(n_constr):
#             # Get the distance to all collocation points
#             dist_to_colloc = np.sqrt((X_C - X_constr[k])**2 +\
#                                      (Y_C - Y_constr[k])**2 +\
#                                      (Z_C - Z_constr[k])**2)
#             # Get the distance to all constraints, except for itself
#             dist_to_constr = np.sqrt((np.delete(X_constr, k) - X_constr[k])**2+\
#                                      (np.delete(Y_constr, k) - Y_constr[k])**2+\
#                                      (np.delete(Z_constr, k) - Z_constr[k])**2)
#             # Set the max and min values of c_k
#             c_k = np.concatenate((dist_to_colloc, dist_to_constr)) / np.sqrt(1 - eps_l**0.2)
#             # crop to the minimum and maximum value
#             c_k[c_k < c_min] = c_min
#             c_k[c_k > c_max] = c_max
#             # get the minimum value in the case of the c4
#             c_ks[k] = np.min(c_k)
#         # for plotting purposes, we store also the diameters
#         d_k = 2*c_ks * np.sqrt(1 - 0.5**0.2)

#     return c_ks, d_k


    ###########################
    ### Deprecated functions ###
    ############################

    def clustering(self, n_K, Areas, r_mM, eps_l):
        self.collocation(n_K=n_K, Areas=Areas, method='clustering', r_mM=r_mM, eps_l=eps_l)

        warnings.warn('Starting from spicy version 1.2, \'clustering\' is renamed to \'collocation\'.' +
             ' To reproduce this, call \'collocation(n_K, method=\'clustering\'', FutureWarning)


    def Get_Sol(self, points):

        warnings.warn('Starting from spicy version 1.2, \'Get_Sol(points)\' is replaced.' +
             ' The new API is \'get_sol(points, order=0)\'', FutureWarning)

        return self.get_sol(points, order=0)



    def Get_first_Derivatives(self, points):
        warnings.warn('Starting from spicy version 1.2, \'Get_first_Derivatives(points)\' is replaced.' +
             ' The new API is \'get_sol(points, order=1)\'', FutureWarning)

        return self.get_sol(points, order=1)
