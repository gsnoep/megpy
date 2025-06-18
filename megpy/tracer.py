"""
created by gsnoep on 13 August 2022, with contributions from aho

Module for tracing closed(!) contour lines in Z on a Y,X grid.
"""
import numpy as np

from scipy import integrate, interpolate, optimize, spatial
from itertools import product

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

def intersect2d(x, y, z, indices, level, kind='l', axis='rows'):
    """
    Compute intersections of Z with level along rows or columns.
    
    Parameters:
    - x, y: 1D arrays of x- and y-coordinates.
    - Z: 2D array of values.
    - indices: Array of row or column indices to process.
    - level: Scalar value to find intersections with.
    - kind: Interpolation type ('l' for linear, 's' for spline).
    - axis: 'rows' to interpolate along X, 'cols' to interpolate along Y.

    Returns:
    - x_values, y_values: Arrays of x- and y-coordinates of intersections (sorted if sort=True).
    """
    # check if any rows/columns are selected
    if len(indices) == 0:
        return np.array([]), np.array([])
    
    # set data and coordinates based on axis
    if axis == 'rows':
        data = z[indices, :]  # shape: (len(indices), Z.shape[1])
        interp_coords = x
        fixed_coords = y[indices]
        slice_axis = 1
    elif axis == 'cols':
        data = z[:, indices]  # shape: (Z.shape[0], len(indices))
        interp_coords = y
        fixed_coords = x[indices]
        slice_axis = 0
    else:
        raise ValueError("axis must be 'rows' or 'cols'")

    # find sign changes
    sign_changes = np.diff(np.sign(data - level), axis=slice_axis) != 0
    rows, cols = np.where(sign_changes)

    # check if any intersections were found
    if len(rows) == 0:
        return np.array([]), np.array([])
    
    # linear interpolation
    if kind == 'l':
        # set coordinates based on axis
        if axis == 'rows':
            row_idx = indices[rows]
            x0 = interp_coords[cols]
            x1 = interp_coords[cols + 1]
            y0 = z[row_idx, cols]
            y1 = z[row_idx, cols + 1]
        else:
            col_idx = indices[cols]
            x0 = interp_coords[rows]
            x1 = interp_coords[rows + 1]
            y0 = z[rows, col_idx]
            y1 = z[rows + 1, col_idx]
        
        # interpolate
        mask = y1 != y0
        t = np.zeros_like(y0)
        t[mask] = (level - y0[mask]) / (y1[mask] - y0[mask])
        interp_values = x0 + t * (x1 - x0)

        # process results
        if axis == 'rows':
            x_values = interp_values[mask]
            y_values = fixed_coords[rows][mask]
        else:
            x_values = fixed_coords[cols][mask]
            y_values = interp_values[mask]
    
    # spline interpolation  
    elif kind == 's':
        def process_slice(i, sign_changes_idx):
            if axis == 'rows':
                idx = indices[i]
                slice_data = z[idx, :]
                slice_coords = fixed_coords[i]
            else:
                idx = indices[i]
                slice_data = z[:, idx]
                slice_coords = fixed_coords[i]
            
            spline = interpolate.CubicSpline(interp_coords, slice_data)
            x_vals, y_vals = [], []
            for i_sign in sign_changes_idx:
                c0, c1 = interp_coords[i_sign], interp_coords[i_sign + 1]
                try:
                    root = optimize.brentq(lambda x: spline(x) - level, c0, c1)
                    if axis == 'rows':
                        x_vals.append(root)
                        y_vals.append(slice_coords)
                    else:
                        x_vals.append(slice_coords)
                        y_vals.append(root)
                except ValueError:
                    continue
            return x_vals, y_vals
        
        if axis == 'rows':
            unique_idx = np.unique(rows)
            results = [process_slice(i, cols[rows == i]) for i in range(len(indices)) if i in unique_idx]
        else:
            unique_idx = np.unique(cols)
            results = [process_slice(i, rows[cols == i]) for i in range(len(indices)) if i in unique_idx]
        
        x_values = np.concatenate([_x for _x, _ in results]) if results else np.array([])
        y_values = np.concatenate([_y for _, _y in results]) if results else np.array([])
    
    else:
        raise ValueError("kind must be 'l' (linear) or 's' (spline)")
    
    return x_values, y_values

def sort2d(x, y, centroid=None, start='farthest', metric='euclidean'): 
    coordinates = np.column_stack((x, y))  
    n = len(coordinates)

    if centroid is None:
        centroid = np.mean(coordinates, axis=0)

    # find starting point
    l2_norm_centroid = np.linalg.norm(coordinates - centroid, axis=1)
    i_start = np.argmax(l2_norm_centroid) if start == 'farthest' else np.argmin(l2_norm_centroid)

    # compute pairwise distances
    dist_matrix = spatial.distance.cdist(coordinates, coordinates, metric=metric)

    # find nearest-neighbor sorting indices
    sorted_indices = np.zeros(n, dtype=int)
    used = np.zeros(n, dtype=bool)
    sorted_indices[0] = i_start
    used[i_start] = True

    for i in range(1, n):
        dists = dist_matrix[sorted_indices[i-1]]
        dists[used] = np.inf
        sorted_indices[i] = np.argmin(dists)
        used[sorted_indices[i]] = True

    # apply sorting indices
    sorted_coords = coordinates[sorted_indices]
    x_coordinates, y_coordinates = sorted_coords[:, 0], sorted_coords[:, 1] 
    return x_coordinates, y_coordinates

def segment2d(x, y, centroid=None, threshold=None, sort=True):
    coordinates = np.column_stack((x, y))
    distances = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)

    if threshold is None:
        median_distance = np.median(distances)
        threshold = 2 * median_distance
    
    # segment the contours
    segment_indices = np.where(distances > threshold)[0] + 1
    if segment_indices.size > 0:
        x_segments = np.array_split(x, segment_indices)
        y_segments = np.array_split(y, segment_indices)
    else:
        x_segments = [x]
        y_segments = [y]
    segments = list(zip(x_segments, y_segments))

    endpoints = [] 
    dist = []
    for i, (segment_x, segment_y) in enumerate(segments):
        # close contour segments when the end is close to the start
        segment = np.column_stack((segment_x, segment_y))
        if np.linalg.norm(segment[-1] - segment[0], axis=0) <= threshold:
            segment = np.vstack((segment, segment[0]))
            segments[i] = (segment[:, 0], segment[:, 1])

        if sort:
            if centroid is None:
                centroid = np.array([np.mean(x),np.mean(y)])
            mean_dist = np.mean(np.linalg.norm(segment - centroid, axis=1))
            dist.append(mean_dist)
        
        # collect the endpoints of the segments
        start = (segments[i][0][0], segments[i][0][1])
        end = (segments[i][-1][0], segments[i][-1][1])
        endpoints += [start, end]
    
    endpoints = np.array(endpoints)
    # TODO: check pairwise if the endpoints are close to each other and merge segments within a threshold distance

    # sort the segments by distance from the centroid
    if sort:
        i_sorted = np.argsort(dist)
        segments = [segments[i] for i in i_sorted]

    return segments

def extract_segments(x_contours, y_contours, x_bounds, y_bounds, dx, dy, ref_point=None):
    # convert bounds to bounding boxes
    x_boxes = list(zip(x_bounds[0::2], x_bounds[1::2]))
    y_boxes = list(zip(y_bounds[0::2], y_bounds[1::2]))

    x_contours = np.asarray(x_contours)
    y_contours = np.asarray(y_contours)

    # check for NaN or Inf in contour data
    if np.isnan(x_contours).any() or np.isnan(y_contours).any() or np.isinf(x_contours).any() or np.isinf(y_contours).any():
        print("Warning: Contour data contains NaN or Inf values")

    # split the contour coordinates into separate segments based on the bounding boxes
    segments = []
    for (x_min, x_max), (y_min, y_max) in product(x_boxes, y_boxes):
        mask = (
            (x_contours >= x_min) & (x_contours <= x_max) & 
            (y_contours >= y_min) & (y_contours <= y_max)
        )

        if np.any(mask): # exclude empty bounding boxes
            segments.append((x_contours[mask], y_contours[mask]))
        
        # print for debugging
            #print(f"Found points for box: x[{x_min}, {x_max}], y[{y_min}, {y_max}]")
        #else:
            #print(f"No points found for box: x[{x_min}, {x_max}], y[{y_min}, {y_max}]")
    
    #for i_seg, segment in enumerate(segments):
    #    if np.abs(segment[0][-1]-segment[0][0]) < dx or np.abs(segment[1][-1]-segment[1][0]):
    #        segments[i_seg] = (np.concatenate((segment[0], [segment[0][0]])),np.concatenate((segment[1], [segment[1][0]])))
    
    # sort the contour segments by distance from a reference point, if one is provided
    if len(segments) > 1 and ref_point:
        distance = []
        for segment_x, segment_y in segments:
            points = np.column_stack((segment_x, segment_y))
            mean_dist = np.mean(np.linalg.norm(points - ref_point, axis=1))
            distance.append(mean_dist)
        # sort the indices
        i_sorted = np.argsort(distance)
        
        return [segments[i] for i in i_sorted]
    else:
        return segments

def find_x_point(X,Y,Z,level):
    dX = X[1] - X[0]
    dY = Y[1] - Y[0]

    diff = np.abs((Z - level)) / (dX*dY)**2

    rows = np.where(np.any(diff < 1.05*np.min(diff), axis=1))[0]
    cols = np.where(np.any(diff < 1.05*np.min(diff), axis=0))[0]
    approx_nulls = list(product(cols, rows))  # (col, row) pairs

    interp_psi = interpolate.RectBivariateSpline(Y, X, Z, kx=3, ky=3)  # Cubic spline

    def grad_psi(x, interp):
        """Compute gradient at point x = [X, Y]"""
        X_val, Y_val = x
        dpsi_dX = interp.ev(Y_val, X_val, dx=1, dy=0)  # ∂ψ/∂R
        dpsi_dY = interp.ev(Y_val, X_val, dx=0, dy=1)  # ∂ψ/∂Z
        return np.array([dpsi_dX, dpsi_dY])

    def objective(x, interp):
        """Minimize |∇ψ|^2"""
        grad = grad_psi(x, interp)
        return np.sum(grad**2)

    # Refine each approximate null
    exact_rz = []
    for col, row in approx_nulls:
        r_guess = X[col]
        z_guess = Y[row]
        x0 = [r_guess, z_guess]
        
        # Minimize |∇ψ|^2 starting from approximate point
        result = optimize.minimize(objective, x0, args=(interp_psi,), method='L-BFGS-B',
                        bounds=[(X.min(), X.max()), (Y.min(), Y.max())])
        
        if result.success and result.fun < 1e-6:  # Check if gradient is near zero
            exact_rz.append((result.x[0], result.x[1]))
    return exact_rz

def contour(X, Y, Z, level, kind='l', ref_point=None, x_point=False):
    # compute the difference field
    diff = Z - level

    # identify intersections
    rows = np.where(np.any(np.diff(np.sign(diff), axis=1) != 0, axis=1))[0]
    cols = np.where(np.any(np.diff(np.sign(diff), axis=0) != 0, axis=0))[0]

    # compute intersections
    x_rows, y_rows = intersect2d(X, Y, Z, rows, level, kind=kind, axis='rows')
    x_cols, y_cols = intersect2d(X, Y, Z, cols, level, kind=kind, axis='cols')

    # concatenate coordinates
    if x_rows.size > 0 or x_cols.size > 0:
        x_coordinates = np.concatenate([x_rows, x_cols])
        y_coordinates = np.concatenate([y_rows, y_cols])

        dX = X[1] - X[0]
        dY = Y[1] - Y[0]

        if not x_point:
            threshold = np.sqrt((1.5 * dX)**2 + (1.5 * dY)**2)
        else:
            threshold = np.sqrt((2. * dX)**2 + (2. * dY)**2)

        x_coordinates, y_coordinates = sort2d(x_coordinates, y_coordinates, centroid=ref_point)
        contours = segment2d(x_coordinates, y_coordinates, centroid=ref_point, threshold=threshold)

    else:
        x_coordinates = np.array([])
        y_coordinates = np.array([])
        
        contours = [(x_coordinates, y_coordinates)]
    
    c = {'X':contours[0][0],'Y':contours[0][1],'level':level}
    # compute a normalised level label for the contour level
    c['label'] = level
    # find the contour center quantities and add them to the contour dict
    c.update(contour_center(c))

    # zipsort the contour from 0 - 2 pi
    c['theta_XY'] = arctan2pi(c['Y'] - c['Y0'], c['X'] - c['X0'])
    c['theta_XY'], c['X'], c['Y'] = zipsort(c['theta_XY'], c['X'], c['Y'])

    # close the contour
    c['theta_XY'] = np.append(c['theta_XY'],c['theta_XY'][0])
    c['X'] = np.append(c['X'],c['X'][0])
    c['Y'] = np.append(c['Y'],c['Y'][0])

    return c
    #return contours

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
    #print('bah')
    # restack R_fs and Z_fs to get a continuous midplane outboard trace
    X_out = np.hstack((c['X'][int(0.9*len(c['Y'])):],c['X'][:int(0.1*len(c['Y']))]))
    Y_out = np.hstack((c['Y'][int(0.9*len(c['Y'])):],c['Y'][:int(0.1*len(c['Y']))]))

    X_in = c['X'][int(len(c['Y'])/2)-int(0.1*len(c['Y'])):int(len(c['Y'])/2)+int(0.1*len(c['Y']))]
    Y_in = c['Y'][int(len(c['Y'])/2)-int(0.1*len(c['Y'])):int(len(c['Y'])/2)+int(0.1*len(c['Y']))]

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
