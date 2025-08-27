"""
created by gsnoep on 13 August 2022, with contributions from aho

Module for tracing contour lines for level in field on a y,x grid.
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
    - x_point: Boolean flag to activate the sorting considerations when x-points are present.

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

            # check if the path length is optimal near the start and swap if needed
            if len(segment_indices) == 3:
                
                # compare dist(0,2) and dist(1,2)
                dist_0_2 = dist_matrix[segment_indices[0], segment_indices[2]]
                dist_1_2 = dist_matrix[segment_indices[1], segment_indices[2]]
                
                # swap indices in segment_indices if dist_0_2 < dist_1_2
                if dist_0_2 < dist_1_2:
                    segment_indices[0], segment_indices[1] = segment_indices[1], segment_indices[0]
        
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

def find_nulls(x, y, field, threshold=1e-12):
    """
    Find null points in a 2D field where the field gradients are zero.
    This is a vectorized 2D adaptation of the method described in https://doi.org/10.1063/1.2756751.

    Args:
        x (numpy.ndarray): 1D array of x-coordinates (shape: (nx,)).
        y (numpy.ndarray): 1D array of y-coordinates (shape: (ny,)).
        field (numpy.ndarray): 2D array of field values (shape: (ny, nx)).
        threshold (float, optional): Numerical threshold for null point calculations. Defaults to 1e-12.

    Returns:
        numpy.ndarray: Array of shape (n, 2) containing x and y coordinates of null points.
    """

    # compute gradients
    ddzfield, ddrfield = np.gradient(field, y, x)
    
    # grid dimensions
    nz, nr = field.shape
    nz, nr = nz - 1, nr - 1
    
    # create meshgrid indices for cells
    i, j = np.indices((nz, nr))
    i, j = i.ravel(), j.ravel()
    
    # extract corner values for all cells
    v1_00 = ddrfield[:-1, :-1].ravel()
    v1_01 = ddrfield[:-1, 1:].ravel()
    v1_10 = ddrfield[1:, :-1].ravel()
    v1_11 = ddrfield[1:, 1:].ravel()
    
    v2_00 = ddzfield[:-1, :-1].ravel()
    v2_01 = ddzfield[:-1, 1:].ravel()
    v2_10 = ddzfield[1:, :-1].ravel()
    v2_11 = ddzfield[1:, 1:].ravel()
    
    # filter cells where both ddrfield and ddzfield can cross zero
    v1_min = np.minimum.reduce([v1_00, v1_01, v1_10, v1_11])
    v1_max = np.maximum.reduce([v1_00, v1_01, v1_10, v1_11])
    v2_min = np.minimum.reduce([v2_00, v2_01, v2_10, v2_11])
    v2_max = np.maximum.reduce([v2_00, v2_01, v2_10, v2_11])
    
    valid_cells = ~((v1_min > 0) | (v1_max < 0) | (v2_min > 0) | (v2_max < 0))
    i, j = i[valid_cells], j[valid_cells]
    v1_00, v1_01, v1_10, v1_11 = v1_00[valid_cells], v1_01[valid_cells], v1_10[valid_cells], v1_11[valid_cells]
    v2_00, v2_01, v2_10, v2_11 = v2_00[valid_cells], v2_01[valid_cells], v2_10[valid_cells], v2_11[valid_cells]
    
    # bilinear coefficients for v1(s, t) = a1 + b1 t + c1 s + d1 s t
    a1 = v1_00
    b1 = v1_01 - v1_00
    c1 = v1_10 - v1_00
    d1 = v1_00 - v1_01 - v1_10 + v1_11
    
    # bilinear coefficients for v2(s, t)
    a2 = v2_00
    b2 = v2_01 - v2_00
    c2 = v2_10 - v2_00
    d2 = v2_00 - v2_01 - v2_10 + v2_11
    
    # solve quadratic equation: A t^2 + B t + C = 0
    A = b2 * d1 - d2 * b1
    B = a2 * d1 + b2 * c1 - c2 * b1 - d2 * a1
    C = a2 * c1 - c2 * a1
    
    # make linear and quadratic masks
    linear_mask = np.abs(A) < threshold
    quadratic_mask = ~linear_mask
    
    # linear case: solve B t + C = 0 for t
    linear_valid = linear_mask & (np.abs(B) >= threshold)
    t_linear = -C[linear_valid] / B[linear_valid]
    
    # quadratic case: solve A t^2 + B t + C = 0 for t
    disc = B[quadratic_mask]**2 - 4 * A[quadratic_mask] * C[quadratic_mask]
    valid_quad = disc >= 0
    sqrt_disc = np.sqrt(disc[valid_quad])
    A_quad = A[quadratic_mask][valid_quad]
    B_quad = B[quadratic_mask][valid_quad]
    
    t1 = (-B_quad + sqrt_disc) / (2 * A_quad)
    t2 = (-B_quad - sqrt_disc) / (2 * A_quad)
    
    # combine solution branches
    t_all = np.concatenate([
        t_linear,
        t1, t2
    ])
    idx_linear = np.where(linear_valid)[0]
    idx_quad = np.where(quadratic_mask)[0][valid_quad]
    idx_all = np.concatenate([
        idx_linear,
        idx_quad, idx_quad
    ])
    
    # filter t values in [0, 1]
    t_mask = (t_all >= 0) & (t_all <= 1)
    t_all = t_all[t_mask]
    idx_all = idx_all[t_mask]
    
    # compute s for each valid t
    den = c1[idx_all] + d1[idx_all] * t_all
    num = a1[idx_all] + b1[idx_all] * t_all
    den_mask = np.abs(den) >= threshold
    s_all = np.zeros_like(t_all)
    s_all[den_mask] = -num[den_mask] / den[den_mask]
    
    # filter s values in [0, 1]
    s_mask = (s_all >= 0) & (s_all <= 1)
    t_all = t_all[s_mask]
    s_all = s_all[s_mask]
    idx_all = idx_all[s_mask]
    
    # map to physical coordinates
    x0 = x[j[idx_all]]
    x1 = x[j[idx_all] + 1]
    y0 = y[i[idx_all]]
    y1 = y[i[idx_all] + 1]
    
    x_null = x0 + t_all * (x1 - x0)
    y_null = y0 + s_all * (y1 - y0)
    null_points = np.vstack((x_null, y_null)).T

    return null_points

def null_classifier(nulls, x, y, field, level=None, atol=1e-3, delta=1e-5, eigtol=1e-10):
    """
    Classify null points as O-points (local minima/maxima) or X-points (saddle points) based on the
    Hessian matrix of the field.

    Args:
        nulls (numpy.ndarray): Array of null points with shape (n, 2), where each row is [x, y].
        x (numpy.ndarray): 1D array of x-coordinates (shape: (nx,)).
        y (numpy.ndarray): 1D array of y-coordinates (shape: (ny,)).
        field (numpy.ndarray): 2D array of field values (shape: (ny, nx)).
        level (float, optional): Field value to filter null points. If provided, only null points
            where the field value is within `atol` of `level` are returned. Defaults to None.
        atol (float, optional): Absolute tolerance for level filtering. Defaults to 1e-3.
        delta (float, optional): Step size for finite difference calculations of second derivatives.
            Defaults to 1e-5.
        eigtol (float, optional): Eigenvalue tolerance for classifying points. Defaults to 1e-10.

    Returns:
        dict: Dictionary with keys 'o-points' and 'x-points', each containing a numpy.ndarray of
            shape (m, 2) with [x, y] coordinates of classified points.
    """
    # create interpolator for level filtering and second derivatives
    interpolator = interpolate.RegularGridInterpolator((y, x), field, method='cubic', bounds_error=False, fill_value=np.nan)

    # swap to (y, x) for interpolator
    points = np.array([[_y, _x] for _x, _y in nulls])

    if len(nulls) == 0:
        return {'o-points': [], 'x-points': []}
    
    # points for second derivatives
    points_x_plus = points + np.array([0, delta])
    points_x_minus = points - np.array([0, delta])
    points_y_plus = points + np.array([delta, 0])
    points_y_minus = points - np.array([delta, 0])
    points_xy_plus = points + np.array([delta, delta])
    points_xy_minus = points + np.array([delta, -delta])
    points_yx_plus = points + np.array([-delta, delta])
    points_yx_minus = points + np.array([-delta, -delta])
    
    # evaluate field interpolator at all points
    val = interpolator(points)
    val_x_plus = interpolator(points_x_plus)
    val_x_minus = interpolator(points_x_minus)
    val_y_plus = interpolator(points_y_plus)
    val_y_minus = interpolator(points_y_minus)
    val_xy_plus = interpolator(points_xy_plus)
    val_xy_minus = interpolator(points_xy_minus)
    val_yx_plus = interpolator(points_yx_plus)
    val_yx_minus = interpolator(points_yx_minus)
    
    # compute second derivatives using finite differences
    d2dx2_field = (val_x_plus - 2 * val + val_x_minus) / (delta**2)
    d2dy2_field = (val_y_plus - 2 * val + val_y_minus) / (delta**2)
    d2dxdy_field = (val_xy_plus - val_xy_minus - val_yx_plus + val_yx_minus) / (4 * delta**2)
    
    # form Hessian matrix: shape (n_points, 2, 2)
    hessian = np.array([
        [d2dx2_field, d2dxdy_field],
        [d2dxdy_field, d2dy2_field]
    ]).transpose(2, 0, 1)  # shape: (n_points, 2, 2)
    
    # compute eigenvalues for all Hessians
    lamba = np.linalg.eigvals(hessian).real  # shape: (n_points, 2)
    eig1, eig2 = lamba[:, 0], lamba[:, 1]
    
    # classify points
    is_o_point = ((eig1 > eigtol) & (eig2 > eigtol)) | ((eig1 < -eigtol) & (eig2 < -eigtol))
    is_x_point = ((eig1 > eigtol) & (eig2 < -eigtol)) | ((eig1 < -eigtol) & (eig2 > eigtol))
    
    # Extract O-points and X-points
    o_points = nulls[is_o_point]
    x_points = nulls[is_x_point]

    # filter nulls by level if provided
    if level is not None:
        _o_points = np.array([[_y, _x] for _x, _y in o_points])
        _x_points = np.array([[_y, _x] for _x, _y in x_points])
        o_values = interpolator(_o_points)
        x_values = interpolator(_x_points)
        o_mask = np.abs(o_values - level) <= atol
        x_mask = np.abs(x_values - level) <= atol
        o_points = o_points[o_mask]
        x_points = x_points[x_mask]
    
    return {'o-points': o_points, 'x-points': x_points}

def find_x_points(x, y, field, level=None, atol=1e-3):
    """
    Find X-points (saddle points) in a 2D field.

    Args:
        x (numpy.ndarray): 1D array of x-coordinates (shape: (nx,)).
        y (numpy.ndarray): 1D array of y-coordinates (shape: (ny,)).
        field (numpy.ndarray): 2D array of field values (shape: (ny, nx)).
        level (float, optional): Field value to filter X-points. If provided, only X-points
            where the field value is within `atol` of `level` are returned. Defaults to None.
        atol (float, optional): Absolute tolerance for level filtering. Defaults to 1e-3.

    Returns:
        numpy.ndarray: Array of shape (m, 2) containing [x, y] coordinates of X-points.
    """
    nulls = find_nulls(x, y, field)
    return null_classifier(nulls, x, y, field, level=level, atol=atol)['x-points']

def find_o_points(x, y, field, level=None, atol=1e-3):
    """
    Find O-points (local minima or maxima) in a 2D vector field.

    Args:
        x (numpy.ndarray): 1D array of x-coordinates (shape: (nx,)).
        y (numpy.ndarray): 1D array of y-coordinates (shape: (ny,)).
        field (numpy.ndarray): 2D array of field values (shape: (ny, nx)).
        level (float, optional): Field value to filter O-points. If provided, only O-points
            where the field value is within `atol` of `level` are returned. Defaults to None.
        atol (float, optional): Absolute tolerance for level filtering. Defaults to 1e-3.

    Returns:
        numpy.ndarray: Array of shape (m, 2) containing [x, y] coordinates of O-points.
    """
    nulls = find_nulls(x, y, field)
    return null_classifier(nulls, x, y, field, level=level, atol=atol)['o-points']

def find_null_points(x, y, field, level=None, atol=1e-3):
    """
    Find and classify all null points (O-points and X-points) in a 2D vector field.

    Args:
        x (numpy.ndarray): 1D array of x-coordinates (shape: (nx,)).
        y (numpy.ndarray): 1D array of y-coordinates (shape: (ny,)).
        field (numpy.ndarray): 2D array of field values (shape: (ny, nx)).
        level (float, optional): Field value to filter null points. If provided, only null points
            where the field value is within `atol` of `level` are returned. Defaults to None.
        atol (float, optional): Absolute tolerance for level filtering. Defaults to 1e-3.

    Returns:
        dict: Dictionary with keys 'o-points' and 'x-points', each containing a numpy.ndarray of
            shape (m, 2) with [x, y] coordinates of classified points.
    """
    nulls = find_nulls(x, y, field)
    return null_classifier(nulls, x, y, field, level=level, atol=atol)

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
            x_points = find_x_points(x, y, field, level)

            # Compute grid spacing and radius
            radius = np.sqrt((1. * dx)**2 + (1. * dy)**2)

            for _x_point in x_points:
                distances = np.sqrt((x_coordinates-_x_point[0])**2+(y_coordinates-_x_point[1])**2)
                mask = distances > radius
                x_coordinates = np.concatenate([x_coordinates[mask],np.repeat(_x_point[0],2)])
                y_coordinates = np.concatenate([y_coordinates[mask],np.repeat(_x_point[1],2)])
            
            threshold = np.sqrt((2 * dx)**2 + (2 * dy)**2)

        contours = sort2d(x_coordinates, y_coordinates, ref_point, threshold, x_point=x_point)

    else:
        x_coordinates = np.array([])
        y_coordinates = np.array([])
        
        contours = [(x_coordinates, y_coordinates)]

    return contours

def contour_center(c):
    """
    Find the geometric center of a contour trace c given by c['X'], c['Y'].

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
    """
    Find the (true) extrema in X and Y of a contour trace c given by c['X'], c['Y'].

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
