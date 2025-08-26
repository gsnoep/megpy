"""
created by gsnoep on 13 August 2022, with contributions from aho

Module for tracing contour lines for level in field on a y, x grid.
"""
import numpy as np

from scipy import integrate, interpolate, optimize, spatial

from .utils import *

def intersect1d(x, y, y_val, kind='l'):
    if kind == 'l':
        # find sign changes (roots likely here)
        sign_changes = np.where(np.diff(np.sign(y - y_val)))[0]
        
        # linear interpolation function
        def f_interp(x_val):
            return np.interp(x_val, x, y)

    elif kind == 's':
        # fit a spline through the data
        spline = interpolate.CubicSpline(x,y)
        
        # find sign changes in the spline (roots likely here)
        sign_changes = np.where(np.diff(np.sign(spline(x) - y_val)))[0]

        # spline interpolation function
        def f_interp(x_val):
            return spline(x_val)

    else:
        raise ValueError("kind must be 'l' for linear or 's' for spline")
    
    x_values = [optimize.brentq(lambda x_val: f_interp(x_val) - y_val, x[i], x[i+1]) for i in sign_changes]

    return np.array(x_values)

def intersect2d(x, y, field, level, axis=0, indices=None, kind='l'):
    """
    Vectorized calculation of intersections of a 2D array with a level along rows (axis = 1) or columns (axis = 0).

    Parameters:
    - x, y: 1D arrays of x- and y-coordinates.
    - field: 2D array of values.
    - level: Scalar value to find intersections with.
    - axis: 0 to interpolate along y, 1 to interpolate along x.
    - indices: Array of row or column indices to process (default: all).
    - kind: Interpolation type ('l' for linear, 's' for spline).

    Returns:
    - x_values, y_values: Arrays of x- and y-coordinates of intersections.
    """
    # input validation
    x, y, field = np.asarray(x), np.asarray(y), np.asarray(field)
    if x.ndim != 1 or y.ndim != 1 or field.ndim != 2:
        raise ValueError("x and y must be 1D, field must be 2D")
    if not (x.size == field.shape[1] and y.size == field.shape[0]):
        raise ValueError("x and y must match field dimensions")
    if axis not in [0, 1]:
        raise ValueError("axis must be 0 or 1")
    if kind not in ['l', 's']:
        raise ValueError("kind must be 'l' (linear) or 's' (spline)")

    # specify indices
    indices = np.arange(field.shape[1] if axis == 0 else field.shape[0]) if indices is None else np.asarray(indices)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("indices must be a non-empty 1D array")

    # transpose data if axis=1
    if axis == 1:
        field = field.T
        x, y = y, x

    # select data and coordinates
    data = field[:, indices]
    interp_coords = y
    fixed_coords = x[indices]

    # find sign changes
    sign_changes = np.diff(np.sign(data - level), axis=0) != 0
    rows, cols = np.where(sign_changes)

    # check if any intersections are found
    if len(rows) == 0:
        return np.array([]), np.array([])

    # linear interpolation
    if kind == 'l':
        x0 = interp_coords[rows]
        x1 = interp_coords[rows + 1]
        y0 = data[rows, cols]
        y1 = data[rows + 1, cols]

        with np.errstate(divide='ignore', invalid='ignore'):
            t = (level - y0) / (y1 - y0)
            mask = (y1 != y0) & (t >= 0) & (t <= 1)
            interp_values = x0 + t * (x1 - x0)

        x_values = fixed_coords[cols][mask]
        y_values = interp_values[mask]

    # spline interpolation
    elif kind == 's':
        x_values, y_values = [], []
        unique_cols = np.unique(cols)
        for col_idx in unique_cols:
            idx = indices[col_idx]
            slice_data = data[:, col_idx]
            row_indices = rows[cols == col_idx]

            # create spline
            spline = interpolate.CubicSpline(interp_coords, slice_data)

            # process each sign change
            for i in row_indices:
                c0, c1 = interp_coords[i], interp_coords[i + 1]
                try:
                    root = optimize.brentq(lambda x: spline(x) - level, c0, c1)
                    x_values.append(fixed_coords[col_idx])
                    y_values.append(root)
                except ValueError:
                    continue
    
    x_values = np.array(x_values)
    y_values = np.array(y_values)

    # account for axis
    if axis == 1:
        x_values, y_values = y_values, x_values

    return x_values, y_values

def sort2d(x, y, ref_point=None, threshold=None, start='farthest', metric='euclidean', x_point=False):
    """
    Sort 2D points based on nearest-neighbor traversal, splitting into separate segments
    when the next nearest neighbor is further than the threshold.

    Parameters:
    - x, y: 1D arrays of x- and y-coordinates.
    - ref_point: Reference point for starting (default: mean of coordinates).
    - threshold: Distance threshold for segment splitting (default: 2 * median distance).
    - start: 'farthest' or 'closest' to ref_point (default: 'farthest').
    - metric: Distance metric for cdist (default: 'euclidean').

    Returns:
    - segments: List of tuples, each containing (x_coords, y_coords) for a segment.
    """

    # input validation
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1D arrays of equal length")
    coordinates = np.column_stack((x, y))
    
    # set default threshold based on median distance between consecutive points
    if threshold is None:
        distances = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
        median_distance = np.median(distances) if len(distances) > 0 else 1.0
        threshold = 2 * median_distance
    
    if start not in ['farthest', 'closest']:
        raise ValueError("start must be 'farthest' or 'closest'")
    
    ref_point = np.mean(coordinates, axis=0) if ref_point is None else np.asarray(ref_point)

    # compute pairwise distances
    dist_matrix = spatial.distance.cdist(coordinates, coordinates, metric=metric)
    
    n = len(coordinates)
    used = np.zeros(n, dtype=bool)
    segments = []

    while not all(used):
        # find starting point for a segment
        l2_norm_ref = np.linalg.norm(coordinates - ref_point, axis=1)
        l2_norm_ref[used] = np.inf if start == 'closest' else -np.inf
        i_start = np.argmin(l2_norm_ref) if start == 'closest' else np.argmax(l2_norm_ref)
        
        # initialize new segment
        segment_indices = [i_start]
        used[i_start] = True
        
        # construct segment using nearest neighbor
        while True:
            current_idx = segment_indices[-1]
            dists = dist_matrix[current_idx].copy()
            dists[used] = np.inf
            dists[dists == 0] = np.inf  # exclude duplicate points (zero distance)
            next_idx = np.argmin(dists)
            
            # check if all points are used or if the next point is too far
            if dists[next_idx] == np.inf or dists[next_idx] > threshold:
                break
                
            segment_indices.append(next_idx)
            used[next_idx] = True
        
        # store segment coordinates
        segment_coords = coordinates[segment_indices]
        segments.append((segment_coords[:, 0], segment_coords[:, 1]))
    
    # close contour segments when the end is close to the start
    if segments:
        #if x_point:
            # case 1, segment self-intersects, a coordinate is repeated later in the sequence, 
            # implementation: 1. check if any point in a segment is duplicated, 2. split the segment into two, 3. connect the part of the segment before the first occurrence to the part after the second occurrence (including this point) and leave the part between the two points in place. 
            # case 2, segments share points, but don't connect, 
            
        # extract first and last points for all segments
        first_points = np.array([seg[0][0] for seg in segments])
        last_points = np.array([seg[0][-1] for seg in segments])
        first_coords = np.array([np.array([seg[0][0], seg[1][0]]) for seg in segments])
        last_coords = np.array([np.array([seg[0][-1], seg[1][-1]]) for seg in segments])
        
        # compute distances between first and last points
        distances = np.linalg.norm(last_coords - first_coords, axis=1)
        
        # identify segments to close: distance <= threshold and not identical
        to_close = (distances <= threshold) & (first_points != last_points)
        
        # update segments: close those that meet the condition
        segments = [
            (np.append(seg[0], seg[0][0]) if to_close[i] else seg[0],
             np.append(seg[1], seg[1][0]) if to_close[i] else seg[1])
            for i, seg in enumerate(segments)
        ]
    
    return segments

def find_x_point(x, y, z, level):
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    diff = np.abs((z - level)) / (dx*dy)**2

    # Find indices of points where diff is close to its minimum
    threshold = 1.05 * np.min(diff)
    indices = np.where(diff < threshold)
    approx_nulls = list(zip(indices[1], indices[0]))  # (col, row) pairs

    interp_psi = interpolate.RectBivariateSpline(y, x, z, kx=3, ky=3)  # Cubic spline

    def grad_psi(x, interp):
        """
        Compute gradient at point x = [r_val, z_val]
        """
        r_val, z_val = x
        dpsi_dr = interp.ev(z_val, r_val, dx=1, dy=0)
        dpsi_dz = interp.ev(z_val, r_val, dx=0, dy=1)
        return np.array([dpsi_dr, dpsi_dz])

    def objective(x, interp):
        """
        Minimize |∇ψ|^2
        """
        grad = grad_psi(x, interp)
        return np.sum(grad**2)

    # Refine each approximate null
    exact_rz = []
    for col, row in approx_nulls:
        r_guess = x[col]
        z_guess = y[row]
        x0 = [r_guess, z_guess]
        
        # Minimize |∇ψ|^2 starting from approximate point
        result = optimize.minimize(objective, x0, args=(interp_psi,), method='L-BFGS-B',
                        bounds=[(x.min(), x.max()), (y.min(), y.max())])
        
        if result.success and result.fun < 1e-6:  # Check if gradient is near zero
            exact_rz.append((result.x[0], result.x[1]))
    
    return np.array(exact_rz)

def contour(x, y, field, level, kind='l', ref_point=None, x_point=False):
    # compute the difference field
    diff = field - level

    # identify intersections
    rows = np.where(np.any(np.diff(np.sign(diff), axis=1) != 0, axis=1))[0]
    cols = np.where(np.any(np.diff(np.sign(diff), axis=0) != 0, axis=0))[0]

    # compute intersections
    x_rows, y_rows = intersect2d(x, y, field, level, axis=1, indices=rows, kind=kind)
    x_cols, y_cols = intersect2d(x, y, field, level, axis=0, indices=cols, kind=kind)

    # concatenate coordinates
    if x_rows.size > 0 or x_cols.size > 0:
        x_coordinates = np.concatenate([x_rows, x_cols])
        y_coordinates = np.concatenate([y_rows, y_cols])

        dx = x[1] - x[0]
        dy = y[1] - y[0]

        threshold = np.sqrt((1.5 * dx)**2 + (1.5 * dy)**2)

        if x_point:
            x_points = find_x_point(x, y, field,level)

            # Compute grid spacing and radius
            radius = np.sqrt((1. * dx)**2 + (1. * dy)**2)

            for _x_point in x_points:
                distances = np.sqrt((x_coordinates-_x_point[0])**2+(y_coordinates-_x_point[1])**2)
                mask = distances > radius
                x_coordinates = np.concatenate([x_coordinates[mask],np.repeat(_x_point[0],2)])
                y_coordinates = np.concatenate([y_coordinates[mask],np.repeat(_x_point[1],2)])

        contours = sort2d(x_coordinates, y_coordinates, ref_point, threshold, x_point=x_point)
        #contours = [(x_coordinates, y_coordinates)]

    else:
        x_coordinates = np.array([])
        y_coordinates = np.array([])
        
        contours = [(x_coordinates, y_coordinates)]

    return contours

def contour_center(c):
    """Find the geometric center of a contour trace c given by c['X'], c['Y'].

    Args:
        `c` (dict): description of the contour, c={['X'],['Y'],['label'],...}, where:
                    - X,Y: the contour coordinates, 
                    - label is the normalised contour level label np.sqrt((level - center)/(threshold - center)).

    Returns:
        (dict): the contour with the extrema information added
    """

    # close the contour if not closed
    c_ = copy.deepcopy(c)
    if c_['X'][-1] != c_['X'][0] or c_['Y'][-1] != c_['Y'][0]:
        c_['X'] = np.append(c_['X'],c_['X'][0])
        c_['Y'] = np.append(c_['Y'],c_['Y'][0])

    # find the average elevation (midplane) of the contour by computing the vertical centroid [Candy PPCF 51 (2009) 105009]
    c['Y0'] = integrate.trapezoid(c_['X']*c_['Y'],c_['Y'])/integrate.trapezoid(c_['X'],c_['Y'])

    # find the extrema of the contour in the radial direction at the average elevation
    c = contour_extrema(c)

    # compute the minor and major radii of the contour at the average elevation
    c['r'] = (c['X_out']-c['X_in'])/2
    c['X0'] = (c['X_out']+c['X_in'])/2
    #c['X0'] = integrate.trapezoid(c['X']*c['Y'],c['X'])/integrate.trapezoid(c['Y'],c['X'])

    return c

def contour_extrema(c):
    """Find the (true) extrema in X and Y of a contour trace c given by c['X'], c['Y'].

    Args:
        `c` (dict): description of the contour, c={['X'],['Y'],['Y0'],['label'],...}, where:
                    - X,Y: the contour coordinates, 
                    - Y0 (optional): is the average elevation,
                    - label is the normalised contour level label np.sqrt((level - center)/(threshold - center)).

    Returns:
        (dict): the contour with the extrema information added
    """
    # restack R_fs and Z_fs to get a continuous midplane outboard trace
    X_in = c['X'][int(len(c['Y'])/2)-int(0.1*len(c['Y'])):int(len(c['Y'])/2)+int(0.1*len(c['Y']))]
    X_out = np.hstack((c['X'][int(0.9*len(c['Y'])):],c['X'][:int(0.1*len(c['Y']))]))
    
    Y_in = c['Y'][int(len(c['Y'])/2)-int(0.1*len(c['Y'])):int(len(c['Y'])/2)+int(0.1*len(c['Y']))]
    Y_out = np.hstack((c['Y'][int(0.9*len(c['Y'])):],c['Y'][:int(0.1*len(c['Y']))]))

    # find the approximate(!) extrema in Y of the contour
    Y_max = np.max(c['Y'])
    Y_min = np.min(c['Y'])

    # check if the midplane of the contour is provided
    if 'Y0' not in c:
        c['Y0'] = Y_min+((Y_max-Y_min)/2)

    # find the extrema in X of the contour at the midplane
    c['X_out'] = np.interp(c['Y0'],Y_out,X_out)
    c['X_in'] = np.interp(c['Y0'],Y_in,X_in)

    # in case level is out of bounds in these interpolations
    if np.isnan(c['X_out']) or np.isinf(c['X_out']):
        # restack X to get continuous trace on right side
        X_ = np.hstack((c['X'][np.argmin(c['X']):],c['X'][:np.argmin(c['X'])]))
        # take the derivative of X_
        dX_ = np.gradient(X_,edge_order=2)
        # find X_out by interpolating the derivative of X to 0.
        dX_out =  dX_[np.argmax(dX_):np.argmin(dX_)]
        X_out = X_[np.argmax(dX_):np.argmin(dX_)]
        c['X_out'] = float(interpolate.interp1d(dX_out,X_out,bounds_error=False)(0.))
    if np.isnan(c['X_in']) or np.isinf(c['X_in']):
        dX = np.gradient(c['X'],edge_order=2)
        dX_in =  dX[np.argmin(dX):np.argmax(dX)]
        X_in = c['X'][np.argmin(dX):np.argmax(dX)]
        c['X_in'] = float(interpolate.interp1d(dX_in,X_in,bounds_error=False)(0.))

    # generate filter lists that take a representative slice of the max and min of the contour coordinates around the approximate Y_max and Y_min
    alpha = (0.9+0.075*c['label']**2) # magic to ensure just enough points are 
    max_filter = [z > alpha*(Y_max-c['Y0']) for z in c['Y']-c['Y0']]
    min_filter = [z < alpha*(Y_min-c['Y0']) for z in c['Y']-c['Y0']]

    # patch for the filter lists in case the filter criteria results in < 7 points (minimum of required for 5th order fit + 1)
    i_Y_max = np.argmax(c['Y'])
    i_Y_min = np.argmin(c['Y'])

    if np.array(max_filter).sum() < 7:
        for i in range(i_Y_max-3,i_Y_max+4):
            if c['Y'][i] >= (np.min(c['Y'])+np.max(c['Y']))/2:
                max_filter[i] = True

    if np.array(min_filter).sum() < 7:
        for i in range(i_Y_min-3,i_Y_min+4):
            if c['Y'][i] <= (np.min(c['Y'])+np.max(c['Y']))/2:
                min_filter[i] = True

    # fit the max and min slices of the contour, compute the gradient of these fits and interpolate to zero to find X_Ymax, Y_max, X_Ymin and Y_min
    X_Ymax_fit = np.linspace(c['X'][max_filter][-1],c['X'][max_filter][0],5000)
    try:
        Y_max_fit = interpolate.UnivariateSpline(c['X'][max_filter][::-1],c['Y'][max_filter][::-1],k=5)(X_Ymax_fit)
    except:
        Y_max_fit = np.poly1d(np.polyfit(c['X'][max_filter][::-1],c['Y'][max_filter][::-1],5))(X_Ymax_fit)
    Y_max_fit_grad = np.gradient(Y_max_fit,X_Ymax_fit)

    X_Ymax = interpolate.interp1d(Y_max_fit_grad,X_Ymax_fit,bounds_error=False)(0.)
    Y_max = interpolate.interp1d(X_Ymax_fit,Y_max_fit,bounds_error=False)(X_Ymax)

    X_Ymin_fit = np.linspace(c['X'][min_filter][-1],c['X'][min_filter][0],5000)
    try:
        Y_min_fit = interpolate.UnivariateSpline(c['X'][min_filter],c['Y'][min_filter],k=5)(X_Ymin_fit)
    except:
        Y_min_fit = np.poly1d(np.polyfit(c['X'][min_filter][::-1],c['Y'][min_filter][::-1],5))(X_Ymin_fit)
    Y_min_fit_grad = np.gradient(Y_min_fit,X_Ymin_fit)

    X_Ymin = interpolate.interp1d(Y_min_fit_grad,X_Ymin_fit,bounds_error=False)(0.)
    Y_min = interpolate.interp1d(X_Ymin_fit,Y_min_fit,bounds_error=False)(X_Ymin)
    
    i_X_max = np.argmax(c['X'])
    i_X_min = np.argmin(c['X'])

    X_max = c['X'][i_X_max]
    Y_Xmax = c['Y'][i_X_max]

    X_min = c['X'][i_X_min]
    Y_Xmin = c['Y'][i_X_min]

    c.update({'X_Ymax':float(X_Ymax),'Y_max':float(Y_max),
            'X_Ymin':float(X_Ymin),'Y_min':float(Y_min),
            'X_min':float(X_min),'Y_Xmin':float(Y_Xmin),
            'X_max':float(X_max),'Y_Xmax':float(Y_Xmax)})

    return c
