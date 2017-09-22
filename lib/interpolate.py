""" Classes for interpolating values.
"""
from __future__ import division, print_function, absolute_import


import itertools
import numpy as np

from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.interpolate._bsplines import make_interp_spline


class RegularGridInterpolator(object):
    """
    Interpolation on a regular grid in arbitrary dimensions.

    The data must be defined on a regular grid; the grid spacing however may be
    uneven. Linear, nearest-neighbour, and first, third and fifth order spline
    interpolation are supported. After setting up the interpolator object, the
    interpolation method (*nearest*, *linear*, *slinear*, *cubic*, and
    *quintic*) may be chosen at each evaluation. Additionally, gradients are
    provided for the spline interpolation methods.

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions.

    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.

    method : str, optional
        The method of interpolation to perform. Supported are 'nearest',
        'linear', 'slinear', 'cubic', and 'quintic'. This parameter will become
        the default for the object's
        ``__call__`` method. Default is "linear".

    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.
        Default is True (raise an exception).

    fill_value : number, optional
        If provided, the value to use for points outside of the
        interpolation domain. If None, values outside
        the domain are extrapolated. Note that gradient values will always be
        extrapolated rather than set to the fill_value if bounds_error=False
        for any points outside of the interpolation domain.
        Default is `np.nan`.

    spline_dim_error : bool, optional
        If spline_dim_error=True and an order `k` spline interpolation method
        is used, then if any dimension has fewer points than `k` + 1, an error
        will be raised. If spline_dim_error=False, then the spline interpolant
        order will be reduced as needed on a per-dimension basis. Default
        is True (raise an exception).

    Methods
    -------
    __call__
    gradient
    methods

    Notes
    -----
    Contrary to LinearNDInterpolator and NearestNDInterpolator, this class
    avoids expensive triangulation of the input data by taking advantage of the
    regular grid structure.

    If any of `points` have a dimension of size 1, linear interpolation will
    return an array of `nan` values. Nearest-neighbor interpolation will work
    as usual in this case.

    The 'slinear', 'cubic', 'quintic' methods are spline-based interpolators.
    Use of the spline interpolations allows for getting gradient
    values via the ``gradient`` method.

    Interpolation with the spline methods is expectedly slower than 'linear' or
    'nearest'. They use a different separable tensor product interpolation
    strategy, and are best used when the fitted data (the points and values
    arrays) are large relative to the number of points to be interpolated.
    As such, they are not an efficient choice for some tasks, such as
    re-sampling of image data.

    .. versionadded:: 0.14

    Examples
    --------
    Evaluate a simple example function on the points of a 3D grid:

    >>> import numpy as np
    >>> from scipy.interpolate import RegularGridInterpolator
    >>> def f(x, y, z):
    ...     return 2 * x**3 + 3 * y**2 - z
    >>> x = np.linspace(1, 4, 11)
    >>> y = np.linspace(4, 7, 22)
    >>> z = np.linspace(7, 9, 33)
    >>> data = f(*np.meshgrid(x, y, z, indexing='ij', sparse=True))

    ``data`` is now a 3D array with ``data[i,j,k] = f(x[i], y[j], z[k])``.
    Next, define an interpolating function from this data:

    >>> my_interpolating_function = RegularGridInterpolator((x, y, z), data)

    Evaluate the interpolating function at the two points
    ``(x,y,z) = (2.1, 6.2, 8.3)`` and ``(3.3, 5.2, 7.1)``:

    >>> pts = np.array([[2.1, 6.2, 8.3], [3.3, 5.2, 7.1]])
    >>> my_interpolating_function(pts)
    array([ 125.80469388,  146.30069388])

    which is indeed a close approximation to
    ``[f(2.1, 6.2, 8.3), f(3.3, 5.2, 7.1)]``.

    With the spline interpolation methods it is possible to compute smooth
    gradients for a variety of purposes, such as numerical optimization.

    To demonstrate this, let's define a function with known gradients for
    demonstration, and create grid sample axes with a variety of sizes:

    >>> from scipy.optimize import fmin_bfgs
    >>> def F(u, v, z, w):
    ...     return (u - 5.234)**2 + (v - 2.128)**2 + (z - 5.531)**2 + (w - 0.574)**2
    >>> def dF(u, v, z, w):
    ...     return 2 * (u - 5.234), 2 * (v - 2.128), 2 * (z - 5.531), 2 * (w - 0.574)
    >>> np.random.seed(0)
    >>> U = np.linspace(0, 10, 10)
    >>> V = np.random.uniform(0, 10, 10)
    >>> Z = np.random.uniform(0, 10, 10)
    >>> W = np.linspace(0, 10, 10)
    >>> V.sort(), Z.sort()
    (None, None)
    >>> points = [U, V, Z, W]
    >>> values = F(*np.meshgrid(*points, indexing='ij'))

    Now, define a random sampling point

    >>> x = np.random.uniform(1, 9, 4)

    With the ``cubic`` interpolation method, gradient information will be
    available:

    >>> interp = RegularGridInterpolator(
    ...     points, values, method="cubic", bounds_error=False, fill_value=None)

    This provides smooth interpolation values for approximating the original
    function and its gradient:

    >>> F(*x), interp(x)
    (85.842906385928046, array(85.84290638592806))
    >>> dF(*x)
    (7.1898934757242223, 10.530537027467577, -1.6783302039530898, 13.340466820583288)
    >>> interp.gradient(x)
    array([  7.18989348,  10.53053703,  -1.6783302 ,  13.34046682])

    The ``gradient`` method can conveniently be passed as an argument to any
    procedure that requires gradient information, such as
    ``scipy.optimize.fmin_bfgs``:

    >>> opt = fmin_bfgs(interp, x, fprime=interp.gradient)
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 3
             Function evaluations: 5
             Gradient evaluations: 5

    Despite the course data grid and non-homogeneous axis dimensions, the
    computed minimum matches the known solution very well:

    >>> print(opt)
    [ 5.234  2.128  5.531  0.574]

    Finally, all four interpolation methods can be compared based on the task
    of fitting a course sampling to interpolate a finer representation:

    >>> import numpy as np
    >>> from scipy.interpolate import RegularGridInterpolator as RGI
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib import rcParams
    >>> rcParams['figure.figsize'] = (10, 6)
    >>>
    >>> def F(u, v):
    ...     return u * np.cos(u * v) + v * np.sin(u * v)
    >>> fit_points = [np.linspace(0, 3, 8), np.linspace(0, 3, 8)]
    >>> values = F(*np.meshgrid(*fit_points, indexing='ij'))
    >>> test_points = [np.linspace(fit_points[0][0], fit_points[0][-1], 80), np.linspace(
    ...     fit_points[1][0], fit_points[1][-1], 80)]
    >>> ut, vt = np.meshgrid(*test_points, indexing='ij')
    >>> true_values = F(ut, vt)
    >>> pts = np.array([ut.ravel(), vt.ravel()]).T
    >>> plt.figure()
    >>> for i, method in enumerate(RGI.methods()):
    ...     plt.subplot(2, 4, i + 1)
    ...     interp = RGI(fit_points, values, method=method)
    ...     im = interp(pts).reshape(80, 80)
    ...     plt.imshow(im, interpolation='nearest')
    ...     plt.gca().axis('off')
    ...     plt.title(method)
    >>> plt.subplot(2, 4, 8)
    >>> plt.title("True values")
    >>> plt.gca().axis('off')
    >>> plt.gcf().subplots_adjust(left=0, right=1, bottom=0, top=1)
    >>> plt.tight_layout()
    >>> plt.imshow(true_values, interpolation='nearest')
    >>> plt.show()


    As expected, the spline interpolations are closest to the
    true values, though are more expensive to compute than with `linear` or
    `nearest`.

    The computed gradient fields can also be visualized:

    >>> import numpy as np
    >>> from scipy.interpolate import RegularGridInterpolator as RGI
    >>> from matplotlib import pyplot as plt
    >>> from matplotlib import rcParams
    >>> rcParams['figure.figsize'] = (6, 3)
    >>> n = 30
    >>> fit_points = [np.linspace(0, 3, 8), np.linspace(0, 3, 8)]
    >>> values = F(*np.meshgrid(*fit_points, indexing='ij'))
    >>> test_points = [np.linspace(0, 3, n), np.linspace(0, 3, n)]
    >>> ut, vt = np.meshgrid(*test_points, indexing='ij')
    >>> true_values = F(ut, vt)
    >>> pts = np.array([ut.ravel(), vt.ravel()]).T
    >>> interp = RGI(fit_points, values, method='cubic')
    >>> im = interp(pts).reshape(n, n)
    >>> gradient = interp.gradient(pts).reshape(n, n, 2)
    >>> plt.figure()
    >>> plt.subplot(121)
    >>> plt.title("cubic fit")
    >>> plt.imshow(im[::-1], interpolation='nearest')
    >>> plt.gca().axis('off')
    >>> plt.gcf().subplots_adjust(left=0, right=1, bottom=0, top=1)
    >>> plt.tight_layout()
    >>> plt.subplot(122)
    >>> plt.title('gradients')
    >>> plt.gca().axis('off')
    >>> plt.gcf().subplots_adjust(left=0, right=1, bottom=0, top=1)
    >>> plt.tight_layout()
    >>> plt.quiver(ut, vt, gradient[:, :, 0], gradient[:, :, 1], width=0.01)
    >>> plt.show()


    Higher-dimensional gradient field predictions and visualizations can be
    done the same way:

    >>> import numpy as np
    >>> from scipy.interpolate import RegularGridInterpolator
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import axes3d
    >>> from matplotlib import rcParams
    >>> rcParams['figure.figsize'] = (7, 6)
    >>> # set up 4D test problem
    >>> fig = plt.figure()
    >>> ax = fig.gca(projection='3d')
    >>> ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
    >>> n = 10
    >>> pts = [np.linspace(-1, 1, n), np.linspace(-1, 1, n), np.linspace(-1, 1, n)]
    >>> x, y, z = np.meshgrid(*pts, indexing='ij')
    >>> voxels = np.array([x.ravel(), y.ravel(), z.ravel()]).T
    >>> values = np.sin(x) * y**2 - np.cos(z)
    >>> # interpolate the created 4D data
    >>> interp = RegularGridInterpolator(pts, values, method='cubic')
    >>> gradient = interp.gradient(voxels).reshape(n, n, n, 3)
    >>> u, v, w = gradient[:, :, :, 0], gradient[:, :, :, 1], gradient[:, :, :, 2]
    >>> # Plot the predicted gradient field
    >>> fig.tight_layout()
    >>> ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)
    >>> plt.show()

    """

    # the nearest and linear interpolators in this class are based on code
    # originally programmed by Johannes Buchner,
    # see https://github.com/JohannesBuchner/regulargrid

    @staticmethod
    def _interp_methods():
        """Method-specific settings for interpolation and for testing."""
        interpolator_configs = {
            "slinear": 1,
            "cubic": 3,
            "quintic": 5,
        }

        spline_interps = interpolator_configs.keys()
        all_methods = ['nearest', 'linear'] + list(spline_interps)

        return spline_interps, all_methods, interpolator_configs

    @staticmethod
    def methods():
        """Return a list of valid interpolation method names."""
        return ['nearest', 'linear', 'slinear', 'cubic', 'quintic']

    def __init__(self, points, values, method="linear", bounds_error=True,
                 fill_value=np.nan, spline_dim_error=True):

        configs = RegularGridInterpolator._interp_methods()
        self._spline_methods, self._all_methods, self._interp_config = configs
        if method not in self._all_methods:
            all_m = ', '.join(['"' + m + '"' for m in self._all_methods])
            raise ValueError('Method "%s" is not defined. Valid methods are '
                             '%s.' % (method, all_m))
        self.method = method
        self.bounds_error = bounds_error

        if not hasattr(values, 'ndim'):
            # allow reasonable duck-typed values
            values = np.asarray(values)

        if len(points) > values.ndim:
            raise ValueError("There are %d point arrays, but values has %d "
                             "dimensions" % (len(points), values.ndim))

        if hasattr(values, 'dtype') and hasattr(values, 'astype'):
            if not np.issubdtype(values.dtype, np.inexact):
                values = values.astype(float)

        self.fill_value = fill_value
        if fill_value is not None:
            fill_value_dtype = np.asarray(fill_value).dtype
            if (hasattr(values, 'dtype') and not
                    np.can_cast(fill_value_dtype, values.dtype,
                                casting='same_kind')):
                raise ValueError("fill_value must be either 'None' or "
                                 "of a type compatible with values")

        self._ki = []
        for i, p in enumerate(points):
            n_p = len(p)
            if not np.all(np.diff(p) > 0.):
                raise ValueError("The points in dimension %d must be strictly "
                                 "ascending" % i)
            if not np.asarray(p).ndim == 1:
                raise ValueError("The points in dimension %d must be "
                                 "1-dimensional" % i)
            if not values.shape[i] == n_p:
                raise ValueError("There are %d points and %d values in "
                                 "dimension %d" % (len(p), values.shape[i], i))
            if method in self._spline_methods:
                k = self._interp_config[method]
                self._ki.append(k)
                if n_p <= k:
                    if not spline_dim_error:
                        self._ki[-1] = n_p - 1
                    else:
                        raise ValueError("There are %d points in dimension %d,"
                                         " but method %s requires at least %d "
                                         "points per "
                                         "dimension."
                                         "" % (n_p, i, method, k + 1))

        if method in self._spline_methods:
            if np.iscomplexobj(values[:]):
                raise ValueError("method '%s' does not support complex values."
                                 " Use 'linear' or 'nearest'." % method)
        self.grid = tuple([np.asarray(p) for p in points])
        self.values = values
        self._xi = None
        self._all_gradients = None
        self._spline_dim_error = spline_dim_error
        self._gmethod = None

    def __call__(self, xi, method=None, compute_gradients=True):
        """
        Interpolation at coordinates

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at

        method : str, optional
            The method of interpolation to perform. Supported are 'nearest',
            'linear', 'slinear', 'cubic', and 'quintic'. Default is None,
            which will use the method defined at
            the construction of the interpolation object instance.

        computer_gradients : bool, optional
            If a spline interpolation method is chosen, this determines
            whether gradient calculations should be made and cached.
            Default is True.
        """
        # cache latest evaluation point for gradient method's use later
        self._xi = xi

        method = self.method if method is None else method
        if method not in self._all_methods:
            all_m = ', '.join(['"' + m + '"' for m in self._all_methods])
            raise ValueError('Method "%s" is not defined. Valid methods are '
                             '%s.' % (method, all_m))

        ndim = len(self.grid)
        self.ndim = ndim
        xi = _ndim_coords_from_arrays(xi, ndim=ndim)
        if xi.shape[-1] != len(self.grid):
            raise ValueError("The requested sample points xi have dimension "
                             "%d, but this RegularGridInterpolator has "
                             "dimension %d" % (xi.shape[1], ndim))

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                if not np.logical_and(np.all(self.grid[i][0] <= p),
                                      np.all(p <= self.grid[i][-1])):
                    raise ValueError("One of the requested xi is out of bounds"
                                     " in dimension %d" % i)

        indices, norm_distances, out_of_bounds = self._find_indices(xi.T)
        if method == "linear":
            result = self._evaluate_linear(indices,
                                           norm_distances,
                                           out_of_bounds)
        elif method == "nearest":
            result = self._evaluate_nearest(indices,
                                            norm_distances,
                                            out_of_bounds)

        elif method in self._spline_methods:
            if np.iscomplexobj(self.values[:]):
                raise ValueError("method '%s' does not support complex values."
                                 " Use 'linear' or 'nearest'." % method)
            ki = self._ki
            if method != self.method:
                # re-validate dimensions vs spline order

                ki = []
                for i, p in enumerate(self.grid):
                    n_p = len(p)
                    k = self._interp_config[method]
                    ki.append(k)
                    if n_p <= k:
                        if not self._spline_dim_error:
                            ki[-1] = n_p - 1
                        else:
                            raise ValueError("There are %d points in dimension"
                                             " %d, but method %s requires at "
                                             "least % d points per dimension."
                                             "" % (n_p, i, method, k + 1))

            interpolator = make_interp_spline
            result = self._evaluate_splines(self.values[:].T,
                                            xi,
                                            indices,
                                            interpolator,
                                            method,
                                            ki,
                                            compute_gradients=compute_gradients)

        if not self.bounds_error and self.fill_value is not None:
            result[out_of_bounds] = self.fill_value

        return result.reshape(xi_shape[:-1] +
                              self.values.shape[ndim:])

    def _evaluate_linear(self, indices, norm_distances, out_of_bounds):
        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,) * (self.values.ndim - len(indices))

        # find relevant values
        # each i and i+1 represents a edge
        edges = itertools.product(*[[i, i + 1] for i in indices])
        values = 0.
        for edge_indices in edges:
            weight = 1.
            for ei, i, yi in zip(edge_indices, indices, norm_distances):
                weight *= np.where(ei == i, 1 - yi, yi)
            values += np.asarray(self.values[edge_indices]) * weight[vslice]
        return values

    def _evaluate_nearest(self, indices, norm_distances, out_of_bounds):
        idx_res = []
        for i, yi in zip(indices, norm_distances):
            idx_res.append(np.where(yi <= .5, i, i + 1))
        return self.values[idx_res]

    def _evaluate_splines(self, data_values, xi, indices, interpolator, method,
                          ki, compute_gradients=True):
        """Convenience method for separable regular grid interpolation."""
        # for spline based methods

        # requires floating point input
        xi = xi.astype(np.float)

        # ensure xi is 2D list of points to evaluate
        if xi.ndim == 1:
            xi = xi.reshape((1, xi.size))
        m, n = xi.shape

        # create container arrays for output and gradients
        result = np.empty(m)
        if compute_gradients:
            all_gradients = np.empty_like(xi)

        # Non-stationary procedure: difficult to vectorize this part entirely
        # into numpy-level operations. Unfortunately this requires explicit
        # looping over each point in xi.

        # can at least vectorize the first pass across all points in the
        # last variable of xi
        i = n - 1
        first_values, first_derivs = self._do_spline_fit(interpolator,
                                                         self.grid[i],
                                                         data_values,
                                                         xi[:, i],
                                                         ki[i],
                                                         compute_gradients)

        # the rest of the dimensions have to be on a per point-in-xi basis
        for j, x in enumerate(xi):
            gradient = np.empty_like(x)
            values = data_values[:]

            # Main process: Apply 1D interpolate in each dimension
            # sequentially, starting with the last dimension. These are then
            # "folded" into the next dimension in-place.
            for i in reversed(range(1, n)):
                if i == n - 1:
                    values = first_values[j]
                    if compute_gradients:
                        local_derivs = first_derivs[j]
                else:
                    # Interpolate and collect gradients for each 1D in this
                    # last dimensions. This collapses each 1D sequence into a
                    # scalar.
                    values, local_derivs = self._do_spline_fit(interpolator,
                                                               self.grid[i],
                                                               values,
                                                               x[i],
                                                               ki[i],
                                                               compute_gradients)

                # Chain rule: to compute gradients of the output w.r.t. xi
                # across the dimensions, apply interpolation to the collected
                # gradients. This is equivalent to multiplication by
                # dResults/dValues at each level.
                if compute_gradients:
                    gradient[i] = self._evaluate_splines(local_derivs,
                                                         x[: i],
                                                         indices,
                                                         interpolator,
                                                         method,
                                                         ki,
                                                         compute_gradients=False)

            # All values have been folded down to a single dimensional array
            # compute the final interpolated results, and gradient w.r.t. the
            # first dimension
            output_value, gradient[0] = self._do_spline_fit(interpolator,
                                                            self.grid[0],
                                                            values,
                                                            x[0],
                                                            ki[0],
                                                            compute_gradients)

            if compute_gradients:
                all_gradients[j] = gradient
            result[j] = output_value

        # Cache the computed gradients for return by the gradient method
        if compute_gradients:
            self._all_gradients = all_gradients
            # indicate what method was used to compute these
            self._gmethod = method
        return result

    def _do_spline_fit(self, interpolator, x, y, pt, k, compute_gradients):
        """Do a single interpolant call, and compute a gradient if needed."""
        interp_kwargs = {'k': k, 'axis': 0}
        local_interp = interpolator(x, y, **interp_kwargs)
        values = local_interp(pt)
        local_derivs = None
        if compute_gradients:
            local_derivs = local_interp(pt, 1)
        return values, local_derivs

    def _find_indices(self, xi):
        # find relevant edges between which xi are situated
        indices = []
        # compute distance to lower edge in unity units
        norm_distances = []
        # check for out of bounds xi
        out_of_bounds = np.zeros((xi.shape[1]), dtype=bool)
        # iterate through dimensions
        for x, grid in zip(xi, self.grid):
            i = np.searchsorted(grid, x) - 1
            i[i < 0] = 0
            i[i > grid.size - 2] = grid.size - 2
            indices.append(i)
            norm_distances.append((x - grid[i]) /
                                  (grid[i + 1] - grid[i]))
            if not self.bounds_error:
                out_of_bounds += x < grid[0]
                out_of_bounds += x > grid[-1]
        return indices, norm_distances, out_of_bounds

    def gradient(self, xi, method=None):
        """Return the computed gradients at the specified point.

        The gradients are computed as the interpolation itself is performed,
        but are cached and returned separately by this method.

        If the point for evaluation differs from the point used to produce
        the currently cached gradient, the interpolation is re-performed in
        order to return the correct gradient.

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at

        method : str, optional
            The method of interpolation to perform. Supported are 'slinear',
            'cubic', and 'quintic'. Default is None, which will use the method
            defined at the construction of the interpolation object instance.

        Returns
        -------
        gradient : ndarray of shape (..., ndim)
            gradient vector of the gradients of the interpolated values with
            respect to each value in xi
        """
        # Determine if the needed gradients have been cached already

        if not method:
            method = self.method
        if method not in self._spline_methods:
            raise ValueError("method '%s' does not support gradient"
                             " calculations. " % method)

        if (self._xi is None) or \
                (not np.array_equal(xi, self._xi)) or \
                (method != self._gmethod):
            # if not, compute the interpolation to get the gradients
            self.__call__(xi, method=method)
        gradients = self._all_gradients
        gradients = gradients.reshape(np.asarray(xi).shape)
        return gradients
