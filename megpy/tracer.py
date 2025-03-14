"""
created by gsnoep on 13 August 2022

Module for tracing closed(!) contour lines in Z on a Y,X grid.
"""
import numpy as np
from scipy import interpolate,integrate
import matplotlib.pyplot as plt
from operator import itemgetter
from .utils import *

from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq,minimize
from itertools import product

def find_inters1d(x, y, level, kind='l'):
    if kind == 'l':
        # find sign changes (roots likely here)
        sign_changes = np.where(np.diff(np.sign(y - level)))[0]
        
        # linear interpolation function
        def slice_interp(x_val):
            return np.interp(x_val, x, y)

    elif kind == 's':
        # fit a spline through the data
        spline = interpolate.CubicSpline(x,y)
        
        # find sign changes in the spline (roots likely here)
        sign_changes = np.where(np.diff(np.sign(spline(x) - level)))[0]

        # spline interpolation function
        def slice_interp(x_val):
            return spline(x_val)

    else:
        raise ValueError("kind must be 'l' for linear or 's' for spline")
    
    intersection_points = [brentq(lambda x_val: slice_interp(x_val) - level, x[i], x[i+1]) for i in sign_changes]

    return np.array(intersection_points)

def find_x_point(X,Y,Z,level):
    dX = X[1] - X[0]
    dY = Y[1] - Y[0]

    diff = np.abs((Z - level)) / (dX*dY)**2

    rows = np.where(np.any(diff < 1.05*np.min(diff), axis=1))[0]
    cols = np.where(np.any(diff < 1.05*np.min(diff), axis=0))[0]
    approx_nulls = list(product(cols, rows))  # (col, row) pairs

    interp_psi = RectBivariateSpline(Y, X, Z, kx=3, ky=3)  # Cubic spline

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
        result = minimize(objective, x0, args=(interp_psi,), method='L-BFGS-B',
                        bounds=[(X.min(), X.max()), (Y.min(), Y.max())])
        
        if result.success and result.fun < 1e-6:  # Check if gradient is near zero
            exact_rz.append((result.x[0], result.x[1]))
    return exact_rz

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

def contour(X, Y, Z, level, kind='l',x_point=False):
    # compute the difference field
    diff = Z - level

    # identify intersections from sign changes in diff
    rows = np.where(np.any((diff[:, :-1] * diff[:, 1:]) < 0, axis=1))[0]
    cols = np.where(np.any((diff[:-1, :] * diff[1:, :]) < 0, axis=0))[0]

    if np.any(rows):
        # compute the coordinates of the intersections in the identified rows
        x_fs_rows = [find_inters1d(X, Z[row, :], level, kind=kind) for row in rows]
        y_fs_rows = [[Y[row]] * len(x_fs_rows[i_row]) for i_row,row in enumerate(rows)]

        # compute the vertical edges of the segment bounding boxes
        dY = Y[1]-Y[0]
        rows_breaks = np.where(np.diff(rows) > 1)[0]
        y_bounds = np.sort(np.concatenate([[Y[rows[0]]-dY],
                                            Y[rows[rows_breaks]]+dY,
                                            Y[rows[rows_breaks+1]]-dY,
                                           [Y[rows[-1]]+dY]]))
    else:
        x_fs_rows = []
        y_fs_rows = []
        y_bounds = []
    
    if np.any(cols):
        # compute the coordinates of the intersections in the identified columns
        y_fs_cols = [find_inters1d(Y, Z[:, col], level, kind=kind) for col in cols]
        x_fs_cols = [[X[col]] * len(y_fs_cols[i_col]) for i_col,col in enumerate(cols)]
        
        # compute the horizontal edges of the segment bounding boxes
        cols_breaks = np.where(np.diff(cols) > 1)[0]
        dX = X[1]-X[0]
        x_bounds = np.sort(np.concatenate([[X[cols[0]]-dX],
                                            X[cols[cols_breaks]]+dX,
                                            X[cols[cols_breaks+1]]-dX,
                                           [X[cols[-1]]+dX]]))
    else:
        x_fs_cols = []
        y_fs_cols = []
        x_bounds = []

    if np.any(rows) or np.any(cols):
        # concatenate the computed coordinates
        x_coordinates = np.concatenate(x_fs_rows + x_fs_cols)
        y_coordinates = np.concatenate(y_fs_rows + y_fs_cols)

        # sort all the computed coordinates by polar angle
        contours = np.column_stack((x_coordinates, y_coordinates))
        center = np.mean(contours, axis=0)
        angles = np.arctan2(contours[:, 1] - center[1], contours[:, 0] - center[0])
        sorted_indices = np.argsort(np.mod(angles, 2 * np.pi))
        x_coordinates, y_coordinates = contours[sorted_indices, 0], contours[sorted_indices, 1]
    else:
        print(f'Warning: No contour points found for level: {level} !')
        return
    
    # split all coordinates into separate segments
    segments = extract_segments(x_coordinates,y_coordinates,x_bounds,y_bounds,dX,dY,(np.mean(X),np.mean(Y)))

    if x_point:
        # find the x-point coordinates
        xp_coordinates = find_x_point(X,Y,Z,level)

        y_xp = np.array([float(xp_coordinates[i][1]) for i in range(len(xp_coordinates))])

        # sort y_xp to ensure consistent ordering (smallest to largest)
        y_xp = np.sort(y_xp)

        # compute the mean of the segment y-values once
        y_mean = np.mean(segments[0][1])

        # set default boundaries
        lower_bound = np.min([y_xp[0], np.min(segments[0][1])]) if len(y_xp) > 0 else np.min(segments[0][1])
        upper_bound = np.max([y_xp[-1], np.max(segments[0][1])]) if len(y_xp) > 0 else np.max(segments[0][1])

        # adjust bounds based on position relative to mean
        if len(y_xp) == 1:
            # single point case: mask above or below based on mean
            if y_xp[0] > y_mean:
                upper_bound = y_xp[0]
            else:
                lower_bound = y_xp[0]
        elif len(y_xp) == 2:
            # two-point case: use the two y_xp values directly as bounds
            lower_bound = y_xp[0]
            upper_bound = y_xp[1]

        # create the mask using the bounds
        mask = (segments[0][1] >= lower_bound) & (segments[0][1] <= upper_bound)
        
        segments = [(segments[0][0][mask],segments[0][1][mask])]
    
    c = {'X':segments[0][0],'Y':segments[0][1],'level':level}
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
    #return segments

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
