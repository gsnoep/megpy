"""
created by gsnoep on 13 August 2022, with contributions from aho

Module for tracing contour lines for level in field on a y,x grid.
"""
import numpy as np

from scipy import integrate, interpolate, optimize, spatial

from .utils import *

# tracing methods
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
                dist_0_2 = dist_matrix[segment_indices[0], segment_indices[2]]
                dist_1_2 = dist_matrix[segment_indices[1], segment_indices[2]]
                if dist_0_2 < dist_1_2:
                    segment_indices[0], segment_indices[1] = segment_indices[1], segment_indices[0]
        
        # store segment coordinates
        segments.append(coordinates[segment_indices])
    
    if segments:
        if x_point:
            def find_shared_coords(segment_1, segment_2):
                if len(segment_1) == 0 or len(segment_2) == 0:
                    return None
                
                # find common rows using broadcasting
                matches = np.all(segment_1[:, np.newaxis, :] == segment_2[np.newaxis, :, :], axis=2)
                
                # Get indices where rows match
                i_shared_1, i_shared_2 = np.where(matches)

                return  segment_1[i_shared_1], i_shared_1, i_shared_2
            
            def aligned_concat(segment_1, segment_2, threshold):
                if len(segment_1) == 0 or len(segment_2) == 0:
                    return False
                
                segment_1 = np.asarray(segment_1)
                segment_2 = np.asarray(segment_2)
                
                ends = np.array([segment_1[0], segment_1[-1], segment_2[0], segment_2[-1]])
                dists = np.linalg.norm(np.array([ends[1] - ends[2], ends[3] - ends[0], ends[0] - ends[2], ends[1] - ends[3]]), axis=1)
                
                if np.any(dists <= threshold):
                    min_idx = np.argmin(dists)
                    if min_idx == 0: 
                        return np.vstack([segment_1, segment_2])
                    elif min_idx == 1: 
                        return np.vstack([segment_2, segment_1])
                    elif min_idx == 2: 
                        return np.vstack([segment_1[::-1], segment_2])
                    else: 
                        return np.vstack([segment_1, segment_2[::-1]])
                return False

            # handle sorting exceptions around x-points
            merged_segments = []
            processed = set()

            for i in range(len(segments)):
                if i in processed:
                    continue

                intersection = False
                segment_i = segments[i]

                # case 1: segment self-intersects
                unique_coords, counts = np.unique(segment_i, axis=0, return_counts=True)
                if np.any(counts > 1):
                    # get self-intersection indices
                    intersection_coords = unique_coords[counts > 1]
                    matches = np.where(np.any(np.all(segment_i[:, None, :] == intersection_coords[None, :, :], axis=-1), axis=1))[0]
                    i_split = sorted([_i for _i in matches if _i not in [0, len(segment_i) - 1]])

                    # split segments
                    if i_split:
                        split_segments = [_segment for _segment in np.split(segment_i, i_split, axis=0) if len(_segment) > 1]

                        # merge ends
                        if len(i_split) == 2:
                            split_segments = [np.vstack([split_segments[0],split_segments[-1]]),split_segments[1]]
                            merged_segments.extend(split_segments)
                        
                        else:
                            # identify open and closed segments
                            mask_closed_segments = np.array([np.linalg.norm(_segment[-1]-_segment[0],axis=0)<=threshold for _segment in split_segments])
                            closed_split_segments = [split_segments[i] for i in np.where(mask_closed_segments)[0]]
                            open_split_segments = [split_segments[i] for i in np.where(~mask_closed_segments)[0]]

                            # try to merge the two open line segments closest to ref_point
                            open_norm_ref_distances = [np.min(np.linalg.norm(_segment - ref_point, axis=1)) for _segment in open_split_segments]
                            i_open_norm_ref = np.argsort(open_norm_ref_distances)
                            trial_loop = aligned_concat(open_split_segments[i_open_norm_ref[0]], open_split_segments[i_open_norm_ref[1]], threshold)

                            # check if the ends of the merged segments meet within threshold and update open/closed lists
                            if np.linalg.norm(trial_loop[-1]-trial_loop[0],axis=0)<=threshold:
                                closed_split_segments.append(trial_loop)
                                open_split_segments = [open_split_segments[i] for i in np.where(~mask_closed_segments)[0] if i in i_open_norm_ref[2:]]
                            
                            open_segments = np.vstack(open_split_segments)
                            for _intersect in intersection_coords:
                                if _intersect not in open_segments:
                                    distances = np.linalg.norm(open_segments-_intersect,axis=1)
                                    i_close = np.where(distances <= threshold)[0]
                                    if np.any(i_close):
                                        open_segments = np.insert(open_segments,i_close[np.argmin(distances[i_close])]+1,_intersect,axis=0)

                            # append segments to merged
                            merged_segments.append(open_segments)
                            merged_segments.extend(closed_split_segments)

                        # continue with next segments
                        intersection = True
                        processed.add(i)
                        continue

                # case 2: segment intersects with other segments
                for j in range(i + 1, len(segments)):
                    if j in processed:
                        continue

                    segment_j = segments[j]

                    # find shared coordinates with other segments
                    intersection_coords, i_split_i, i_split_j = find_shared_coords(segment_i, segment_j)

                    # check if there are one or more shared coordinates and the shared coordinate is not the intersection between two closed loops
                    if len(i_split_i) >= 1 and ((np.linalg.norm(segment_i[0] - segment_i[-1]) > threshold) and (np.linalg.norm(segment_j[0] - segment_j[-1]) > threshold)):
                        # split segments at shared point(s)
                        segments_i = [_segment for _segment in np.split(segment_i, np.sort(i_split_i)) if len(_segment) > 1]
                        segments_j = [_segment for _segment in np.split(segment_j, np.sort(i_split_j)) if len(_segment) > 1]

                        # reverse segments order to account for sorting of i_split_i/j
                        if len(i_split_i) > 1 and (i_split_i[-1] < i_split_i[0]):
                            segments_i = segments_i[::-1]
                        if len(i_split_j) > 1 and (i_split_j[-1] < i_split_j[0]):
                            segments_j = segments_j[::-1]

                        # try to concatenate corresponding split segments (assuming an even number of segments)
                        for k in range(len(segments_i)):
                            # collect the relevant intersection coordinates, l can at most be k-1
                            l = min(k, len(intersection_coords) - 1)
                            _intersection = intersection_coords[l]

                            # ensure the intersection coordinates are present in at least on of the segments
                            if not np.any(np.all(_intersection == segments_i[k], axis=1)) and not np.any(np.all(_intersection == segments_j[k], axis=1)):
                                segments_i[k] = np.vstack((segments_i[k],_intersection))

                            # try to align and merge the segments
                            merged_segment = aligned_concat(segments_i[k], segments_j[k], threshold)
                            if merged_segment is not False:
                                merged_segments.append(merged_segment)
                            else:
                                # if aligning and merging fails, add split segments individually
                                merged_segments.append(segments_i[k])
                                merged_segments.append(segments_j[k])

                        # update processed segments
                        intersection = True
                        processed.add(i)
                        processed.add(j)
                        break

                # if no self-intersection or shared points detected, add the original segment
                if not intersection:
                    merged_segments.append(segments[i])
                    processed.add(i)

            # update segments after splitting/merging
            segments = merged_segments
        
        # sort segments based on minimum distance to the reference point
        if ref_point is not None:
            # compute the minimum distance to the reference point for all segments
            ref_dist = [np.min(np.linalg.norm(seg-ref_point, axis=1)) for seg in segments]
            segments = [segments[i] for i in np.argsort(ref_dist)]
    
    segments = [(seg[:,0],seg[:,1]) for seg in segments]

    return segments

# null point detection
def find_nulls(x, y, field, threshold=1e-6):
    """
    Find null points in a 2D scalar field where gradients are zero, using a partially vectorized approach.
    Adapted from https://arxiv.org/abs/0706.0521 with bicubic interpolation and numerical root-finding.

    Args:
        x (numpy.ndarray): 1D array of x-coordinates (shape: (nx,)).
        y (numpy.ndarray): 1D array of y-coordinates (shape: (ny,)).
        field (numpy.ndarray): 2D array of field values (shape: (ny, nx)).
        threshold (float, optional): Numerical threshold for null point calculations. Defaults to 1e-6.

    Returns:
        numpy.ndarray: Array of shape (n, 2) containing x and y coordinates of null points.
    """

    # list to collect null points
    null_points = []

    # compute gradients
    ddyfield, ddxfield = np.gradient(field, y, x, edge_order=2)
    
    # grid dimensions
    ny, nx = field.shape
    
    # pad gradients and coordinates
    ddx_pad = np.pad(ddxfield, 1, mode='edge')
    ddy_pad = np.pad(ddyfield, 1, mode='edge')
    
    x_left = x[0] - (x[1] - x[0]) if len(x) > 1 else x[0]
    x_right = x[-1] + (x[-1] - x[-2]) if len(x) > 1 else x[-1]
    x_pad = np.pad(x, 1, mode='constant', constant_values=(x_left, x_right))
    
    y_left = y[0] - (y[1] - y[0]) if len(y) > 1 else y[0]
    y_right = y[-1] + (y[-1] - y[-2]) if len(y) > 1 else y[-1]
    y_pad = np.pad(y, 1, mode='constant', constant_values=(y_left, y_right))

    # vectorized cell filtering
    jj, ii = np.meshgrid(np.arange(nx-1), np.arange(ny-1))
    jj, ii = jj.ravel(), ii.ravel()
    
    # extract 4x4 local grids for gradients
    v1_locals = np.array([ddx_pad[i:i+4, j:j+4] for i, j in zip(ii, jj)])
    v2_locals = np.array([ddy_pad[i:i+4, j:j+4] for i, j in zip(ii, jj)])
    
    # extract 2x2 field values for filtering
    f_locals = np.array([field[i:i+2, j:j+2] for i, j in zip(ii, jj)])
    f_max_abs = np.max(np.abs(f_locals), axis=(1, 2))
    
    # filter cells where gradients cross zero and field magnitude is sufficient
    v1_min = np.min(v1_locals, axis=(1, 2))
    v1_max = np.max(v1_locals, axis=(1, 2))
    v2_min = np.min(v2_locals, axis=(1, 2))
    v2_max = np.max(v2_locals, axis=(1, 2))
    
    valid_cells = (v1_min <= 0) & (v1_max >= 0) & (v2_min <= 0) & (v2_max >= 0) & (f_max_abs >= threshold)
    valid_idx = np.where(valid_cells)[0]
    
    if not valid_idx.size:
        return np.empty((0, 2))

    # extract valid cell indices and local grids
    ii_valid = ii[valid_idx]
    jj_valid = jj[valid_idx]
    v1_locals = v1_locals[valid_idx]
    v2_locals = v2_locals[valid_idx]

    # process each valid cell
    for i, j, v1_local, v2_local in zip(ii_valid, jj_valid, v1_locals, v2_locals):
        # local coordinates for interpolation
        x_local = x_pad[j:j+4]
        y_local = y_pad[i:i+4]

        # create bicubic interpolators (swap x, y order)
        interp_v1 = interpolate.RectBivariateSpline(x_local, y_local, v1_local.T, kx=3, ky=3, s=0)
        interp_v2 = interpolate.RectBivariateSpline(x_local, y_local, v2_local.T, kx=3, ky=3, s=0)

        # cell boundaries
        x0, x1 = x[j], x[j + 1]
        y0, y1 = y[i], y[i + 1]

        # function for fsolve
        def interpolator(st):
            xp = x0 + st[0] * (x1 - x0)
            yp = y0 + st[1] * (y1 - y0)
            return np.array([interp_v1(xp, yp, grid=False), interp_v2(xp, yp, grid=False)])

        # start from initial guess at cell center
        init = np.array([0.5, 0.5])
        sol, infodict, ier, mesg = optimize.fsolve(interpolator, init, full_output=True)
        if ier == 1 and np.all(sol >= 0) and np.all(sol <= 1):
            residual = np.linalg.norm(interpolator(sol))
            if residual < threshold:
                xp = x0 + sol[0] * (x1 - x0)
                yp = y0 + sol[1] * (y1 - y0)
                null_points.append([xp, yp])

    # convert to array and remove duplicates
    if null_points:
        null_points = np.array(null_points)
        sort_idx = np.lexsort((null_points[:, 1], null_points[:, 0]))
        null_points = null_points[sort_idx]
        diff = np.diff(null_points, axis=0)
        mask = np.linalg.norm(diff, axis=1) > threshold
        null_points = null_points[np.concatenate(([True], mask))]
    else:
        null_points = np.empty((0, 2))

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
    interpolator = interpolate.RegularGridInterpolator((x, y), field.T, method='cubic', bounds_error=False, fill_value=np.nan)

    if len(nulls) == 0:
        return {'o-points': [], 'x-points': []}
    
    # points for second derivatives
    points_x_plus = nulls + np.array([delta, 0])
    points_x_minus = nulls - np.array([delta, 0])
    points_y_plus = nulls + np.array([0, delta])
    points_y_minus = nulls - np.array([0, delta])
    points_xy_plus = nulls + np.array([delta, delta])
    points_xy_minus = nulls + np.array([delta, -delta])
    points_yx_plus = nulls + np.array([-delta, delta])
    points_yx_minus = nulls + np.array([-delta, -delta])
    
    # evaluate field interpolator at all points
    val = interpolator(nulls)
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
        o_values = interpolator(o_points)
        x_values = interpolator(x_points)
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

# contour methods
def contour(x, y, field, level, kind='l', ref_point=None, x_point=False):
    # compute the difference field
    diff = field - level

    # identify intersections
    rows = np.where(np.any(np.diff(np.sign(diff), axis=1) != 0, axis=1))[0]
    cols = np.where(np.any(np.diff(np.sign(diff), axis=0) != 0, axis=0))[0]

    # compute intersections
    if len(rows)>0:
        x_rows, y_rows = intersect2d(x, y, field, level, axis=1, indices=rows, kind=kind)
    else:
        x_rows, y_rows = np.array([]), np.array([])
    if len(cols)>0:
        x_cols, y_cols = intersect2d(x, y, field, level, axis=0, indices=cols, kind=kind)
    else:
        x_cols, y_cols = np.array([]), np.array([])

    # concatenate coordinates
    if x_rows.size > 0 or x_cols.size > 0:
        x_coordinates = np.concatenate([x_rows, x_cols])
        y_coordinates = np.concatenate([y_rows, y_cols])

        # get grid spacing (assuming equidistant grid)
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        # set nearest neighbor threshold
        threshold = np.sqrt((1.5 * dx)**2 + (1.5 * dy)**2)

        # handle x-points in the contour
        if x_point:
            x_points = find_x_points(x, y, field, level)

            if np.any(x_points):
                # compute elimination radius
                radius = np.sqrt((.85 * dx)**2 + (.85 * dy)**2)

                # increase threshold
                threshold = np.sqrt((2 * dx)**2 + (2 * dy)**2)

                # eliminate points inside radius around an x-point to avoid jagged x-point approaches
                x_distances = np.sqrt((x_coordinates[:, None] - x_points[:, 0])**2 + (y_coordinates[:, None] - x_points[:, 1])**2)
                mask = np.all(x_distances > radius, axis=1)
                x_coordinates = np.concatenate([x_coordinates[mask], np.repeat(x_points[:, 0], 2)])
                y_coordinates = np.concatenate([y_coordinates[mask], np.repeat(x_points[:, 1], 2)])
            else:
                x_point = False

        # apply nearest neighbor sorting to contour coordinates
        contours = sort2d(x_coordinates, y_coordinates, ref_point, threshold, x_point=x_point)

        # extract first and last points for all contours
        first_coords = np.array([np.array([seg[0][0], seg[1][0]]) for seg in contours])
        last_coords = np.array([np.array([seg[0][-1], seg[1][-1]]) for seg in contours])
        
        # compute distances between first and last points
        distances = np.linalg.norm(last_coords - first_coords, axis=1)

        # define domain edges
        x_edges = [x[0], x[-1]]
        y_edges = [y[0], y[-1]]

        # check if endpoints are not on domain edges
        not_on_x_edge_first = ~np.isin(first_coords[:, 0], x_edges)
        not_on_x_edge_last = ~np.isin(last_coords[:, 0], x_edges)
        not_on_y_edge_first = ~np.isin(first_coords[:, 1], y_edges)
        not_on_y_edge_last = ~np.isin(last_coords[:, 1], y_edges)

        # check whether to close the contour segments or not
        to_close = (distances <= threshold) & not_on_x_edge_first & not_on_y_edge_first & not_on_x_edge_last & not_on_y_edge_last

        # sort the closed contour around the reference point
        if ref_point:
            contours[0] = sorted(contours[0], key=lambda p: np.mod(np.atan2(p[1]-ref_point[1], p[0]-ref_point[0]),2*np.pi))
        
        # update contours: close those that meet the condition
        contours = [
            (np.append(seg[0], seg[0][0]) if to_close[i] else seg[0],
             np.append(seg[1], seg[1][0]) if to_close[i] else seg[1])
            for i, seg in enumerate(contours)
        ]

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
