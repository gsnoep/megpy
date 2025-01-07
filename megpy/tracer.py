"""
created by gsnoep on 13 August 2022

Module for tracing closed(!) contour lines in Z on a Y,X grid.
"""
import numpy as np
from scipy import interpolate,integrate
import matplotlib.pyplot as plt
from operator import itemgetter
from .utils import *

def check_less(one, two):
    return np.all(one < np.nanmax(two))

def check_more(one, two):
    return np.all(one > np.nanmin(two))

def f_bnd(data, sign='>', early=0, verbose=0):
    found = (len(data) <= 0)
    val = np.nanmin(data) if sign == '>' else np.nanmax(data)
    ii = 0
    jj = 0
    while (not found):
        if sign == '>':
            if data[ii] >= val:
                val = data[ii]
                jj = 0
            else:
                jj += 1
        elif sign == '<':
            if data[ii] <= val:
                val = data[ii]
                jj = 0
            else:
                jj += 1
        ii += 1
        if ii >= len(data):
            found = True
        if early > 0 and jj == early:
            found = True
    if verbose > 0:
        print(data, ii, jj)
    return ii - jj

def find_dual_value(x, y, split_index, level, threshold, method='normal', sign='>', tol=0, tracer_diag='none'):

    lower_index = split_index - f_bnd(y[:split_index+1][::-1], sign=sign, early=tol)
    if lower_index <= 0:
        lower_index = None
    upper_index = split_index + 1 + f_bnd(y[split_index:], sign=sign, early=tol)
    if upper_index >= len(y):
        upper_index = None

    # chop x,y in two parts to separate two solutions
    x_lower = x[lower_index:split_index+1][::-1]
    x_upper = x[split_index:upper_index]
    y_lower = y[lower_index:split_index+1][::-1]
    y_upper = y[split_index:upper_index]

    # if the normal method provides spikey contour traces, bound the interpolation domain and extrapolate the intersection
    if method == 'bounded_extrapolation':
        lower_mask = (y_lower >= threshold) if sign == '>' else (y_lower <= threshold)
        upper_mask = (y_upper >= threshold) if sign == '<' else (y_upper <= threshold)
        x_lower = x_lower[lower_mask]
        x_upper = x_upper[upper_mask]
        y_lower = y_lower[lower_mask]
        y_upper = y_upper[upper_mask]

    # interpolate the x coordinate to either side
    #print(y_lower[0], y_upper[-1])
    xl_out = interpolate.interp1d(y_lower,x_lower,bounds_error=False,fill_value='extrapolate')(level)[0]
    xu_out = interpolate.interp1d(y_upper,x_upper,bounds_error=False,fill_value='extrapolate')(level)[0]

    # diagnostic plot to check interpolation issues for rows
    if tracer_diag == 'interp':
        plt.figure()
        #plt.title('Y:{}'.format(third_axis))
        plt.plot(y_lower,x_lower,'b-',label='lower')
        plt.plot(y_upper,x_upper,'r-',label='upper')
        plt.axvline(level,ls='dashed',color='black')
        plt.legend()

    return xl_out, xu_out

def contour(X,Y,Z=None,level=None,threshold=None,i_center=None,interp_method='normal',tracer_diag='none',symmetrise=False):
    """Find the X,Y trace of a closed(!) contour in Z.

    Args:
        `X` (array): vector of X grid.
        `Y` (array): vector of Y grid.
        `Z` (array): a Y,X map on which to trace the contour.
        `level` (float): the Z value of the contour to be traced.
        `threshold` (float): the Z value of the last closed contour.
        `i_center` (list, array or tuple, optional): the indexes of the approximate location of the lowest/highest level in Z to speed up the tracing calculation in a loop.
        interp_method (str, optional): interpolation method, if set to 'bounded_extrapolation' fill_value = 'extrapolate' is used. Defaults to 'normal' (which means no extrapolation).
        tracer_diag (str, optional): _description_. Defaults to 'none'.
        symmetrise (bool, optional): _description_. Defaults to False.

    Returns:
        dict: a dict containing the X, Y, level (and center + extrema) values of the contour trace in Z.
    """
    with np.errstate(divide='ignore'):

        lvl = np.array([level]) if not isinstance(level, np.ndarray) else level
        thr = np.array([threshold]) if not isinstance(threshold, np.ndarray) else threshold
        dX = np.mean(np.abs(np.diff(X)))
        dY = np.mean(np.abs(np.diff(Y)))

        # define storage for contour coordinates
        XY_contour = {'left':{'top':[],'bottom':[]},'right':{'top':[],'bottom':[]}}

        # take/find the approximate location of the contour center on the Z map
        if i_center is not None:
            i_xcenter = i_center[0]
            i_ycenter = i_center[1]
        else:
            i_xcenter = int(len(X)/2)
            i_ycenter = int(len(Y)/2)
        i_xcenter_pad = int(0.3 * len(X))
        i_ycenter_pad = int(0.3 * len(Y))

        center = Z[i_ycenter,i_xcenter]

        # find the vertical extrema of the threshold contour at the xcenter
        search_tol = 3
        if threshold > center:
            #f_threshold = np.min
            f_split = np.argmin
            #f_bnd = np.argmax
            f_sgn = '>'
            f_check = check_more
        elif threshold < center:
            #f_threshold = np.max
            f_split = np.argmax
            #f_bnd = np.argmin
            f_sgn = '<'
            f_check = check_less
        i_ZY_xcenter_split = f_split(Z[i_ycenter-i_ycenter_pad:i_ycenter+i_ycenter_pad+1,i_xcenter].flatten()) + i_ycenter - i_ycenter_pad
        i_ZY_xcenter_bottom_max = i_ZY_xcenter_split - f_bnd(Z[:i_ZY_xcenter_split+1,i_xcenter].flatten()[::-1], sign=f_sgn, early=3)
        i_ZY_xcenter_top_max = i_ZY_xcenter_split + f_bnd(Z[i_ZY_xcenter_split:,i_xcenter].flatten(), sign=f_sgn, early=3) + 1
        if i_ZY_xcenter_bottom_max <= 0:
            i_ZY_xcenter_bottom_max = None
        if i_ZY_xcenter_top_max >= len(Y):
            i_ZY_xcenter_top_max = None

        Y_xcenter_bottom = Y[i_ZY_xcenter_bottom_max:i_ZY_xcenter_split+1].flatten()
        Y_xcenter_top = Y[i_ZY_xcenter_split:i_ZY_xcenter_top_max].flatten()
        ZY_xcenter_bottom = Z[i_ZY_xcenter_bottom_max:i_ZY_xcenter_split+1,i_xcenter].flatten()
        ZY_xcenter_top = Z[i_ZY_xcenter_split:i_ZY_xcenter_top_max,i_xcenter].flatten()

        '''# patch for rare cases where the Z map becomes flat below the threshold values (e.g. some ESCO EQDSKs)
        if not np.any(ZY_xcenter_bottom >= threshold):
            dZY_xcenter_bottom_dr = np.gradient(ZY_xcenter_bottom)
            if np.any(dZY_xcenter_bottom_dr==0.):
                i_dZY_xcenter_bottom_dr = np.where(dZY_xcenter_bottom_dr==0.)[0][-1]+1
                i_ZY_xcenter_bottom_max += i_dZY_xcenter_bottom_dr
                ZY_xcenter_bottom = ZY_xcenter_bottom[i_dZY_xcenter_bottom_dr:]

        # patch for rare cases where the Z map becomes flat below the threshold values (e.g. some ESCO EQDSKs)
        if not np.any(ZY_xcenter_top >= threshold):
            dZY_xcenter_top_dr = np.gradient(ZY_xcenter_top)
            if np.any(dZY_xcenter_top_dr==0.):
                i_dZY_xcenter_top_dr = np.where(dZY_xcenter_top_dr==0.)[0][0]-1
                i_ZY_xcenter_top_max = i_ZY_xcenter_split+i_dZY_xcenter_top_dr
                ZY_xcenter_top = ZY_xcenter_top[:i_dZY_xcenter_top_dr]'''

        Y_search_max = interpolate.interp1d(ZY_xcenter_top,Y_xcenter_top,bounds_error=False,fill_value='extrapolate')(thr)[0] + 3.0 * dY
        Y_search_min = interpolate.interp1d(ZY_xcenter_bottom,Y_xcenter_bottom,bounds_error=False,fill_value='extrapolate')(thr)[0] - 3.0 * dY

        i_ZX_ycenter_split = f_split(Z[i_ycenter,i_xcenter-i_xcenter_pad:i_xcenter+i_xcenter_pad+1].flatten()) + i_xcenter - i_xcenter_pad
        i_ZX_ycenter_left_max = i_ZX_ycenter_split - f_bnd(Z[i_ycenter,:i_ZX_ycenter_split+1].flatten()[::-1], sign=f_sgn, early=3)
        i_ZX_ycenter_right_max = i_ZX_ycenter_split + f_bnd(Z[i_ycenter,i_ZX_ycenter_split:].flatten(), sign=f_sgn, early=3) + 1
        if i_ZX_ycenter_left_max <= 0:
            i_ZX_ycenter_left_max = None
        if i_ZX_ycenter_right_max >= len(X):
            i_ZX_ycenter_right_max = None

        X_ycenter_left = X[i_ZX_ycenter_left_max:i_ZX_ycenter_split+1].flatten()
        X_ycenter_right = X[i_ZX_ycenter_split:i_ZX_ycenter_right_max].flatten()
        ZX_ycenter_left = Z[i_ycenter,i_ZX_ycenter_left_max:i_ZX_ycenter_split+1].flatten()
        ZX_ycenter_right = Z[i_ycenter,i_ZX_ycenter_split:i_ZX_ycenter_right_max].flatten()

        '''# patch for rare cases where the Z map becomes flat below the threshold values (e.g. some ESCO EQDSKs)
        if not np.any(ZX_ycenter_bottom >= threshold):
            dZX_ycenter_bottom_dr = np.gradient(ZX_ycenter_bottom)
            if np.any(dZX_ycenter_bottom_dr==0.):
                i_dZX_ycenter_bottom_dr = np.where(dZX_ycenter_bottom_dr==0.)[0][-1]+1
                i_ZX_ycenter_bottom_max += i_dZX_ycenter_bottom_dr
                ZX_ycenter_bottom = ZX_ycenter_bottom[i_dZX_ycenter_bottom_dr:]

        # patch for rare cases where the Z map becomes flat below the threshold values (e.g. some ESCO EQDSKs)
        if not np.any(ZX_ycenter_top >= threshold):
            dZX_ycenter_top_dr = np.gradient(ZX_ycenter_top)
            if np.any(dZX_ycenter_top_dr==0.):
                i_dZX_ycenter_top_dr = np.where(dZX_ycenter_top_dr==0.)[0][0]-1
                i_ZX_ycenter_top_max = i_ZX_ycenter_split+i_dZX_ycenter_top_dr
                ZX_ycenter_top = ZX_ycenter_top[:i_dZX_ycenter_top_dr]'''

        X_search_max = interpolate.interp1d(ZX_ycenter_right,X_ycenter_right,bounds_error=False,fill_value='extrapolate')(thr)[0] + 3.0 * dY
        X_search_min = interpolate.interp1d(ZX_ycenter_left,X_ycenter_left,bounds_error=False,fill_value='extrapolate')(thr)[0] - 3.0 * dY

        # diagnostic plot to check interpolation issues for the vertical bounds
        if tracer_diag == 'bounds':
            plt.figure()
            plt.plot(Y[i_ZY_xcenter_split:i_ZY_xcenter_top_max],ZY_xcenter_top,'.-')
            plt.plot(Y[i_ZY_xcenter_bottom_max:i_ZY_xcenter_split],ZY_xcenter_bottom,'.-')
            plt.plot(X[i_ZX_ycenter_split:i_ZX_ycenter_top_max],ZX_ycenter_top,'.-')
            plt.plot(X[i_ZX_ycenter_bottom_max:i_ZX_ycenter_split],ZX_ycenter_bottom,'.-')
            plt.axhline(threshold,ls='dashed',color='black')
            plt.show()

        # set the starting coordinates for the contour tracing algorithm
        i, j = i_ycenter-1, i_ycenter
        k, l = i_xcenter-1, i_xcenter

        # while the level intersects with the current Z slice gather the intersection coordinates
        do_j = True
        while (do_j and j < Z.shape[0] and f_check(lvl, Z[j,:])):
            if Y[j] <= Y_search_max:
                try:
                    # find the split in the Y slice of Z
                    j_ZX_slice_split = f_split(Z[j,i_xcenter-i_xcenter_pad:i_xcenter+i_xcenter_pad+1].flatten()) + i_xcenter - i_xcenter_pad
                    X_top_left, X_top_right = find_dual_value(X.flatten(), Z[j,:].flatten(), j_ZX_slice_split, lvl, thr, interp_method, f_sgn, search_tol)

                    # insert the coordinates into the contour trace dict if not nan (bounds error) and order properly for merging later
                    if np.isfinite(X_top_left):
                        theta_X_top_left = np.arctan2(Y[j] - Y[i_ycenter], X_top_left - X[i_xcenter])
                        if theta_X_top_left < 0.0:
                            theta_X_top_left = 2.0 * np.pi + theta_X_top_left
                        XY_contour['left']['top'].append((X_top_left,Y[j],theta_X_top_left))
                        if X_top_left < (X_search_min + 3.0 * dX):
                            X_search_min = X_top_left - 3.0 * dX
                    if np.isfinite(X_top_right):
                        theta_X_top_right = np.arctan2(Y[j] - Y[i_ycenter], X_top_right - X[i_xcenter])
                        if theta_X_top_right < 0.0:
                            theta_X_top_right = 2.0 * np.pi + theta_X_top_right
                        XY_contour['right']['top'].append((X_top_right,Y[j],theta_X_top_right))
                        if X_top_right > (X_search_max - 3.0 * dX):
                            X_search_max = X_top_right + 3.0 * dX
                    #print('1', Y[j], X_top_left, X_top_right, theta_X_top_left, theta_X_top_right)
                except:
                    pass
                # update the slice coordinates
                j += 1
            else:
                do_j = False

        # while the level intersects with the current Z slice gather the intersection coordinates
        do_k = True
        while (do_k and k >= 0 and f_check(lvl, Z[:,k])):
            # interpolate the X coordinate of the bottom half of the contour on both the right and left
            if X[k] >= X_search_min:
                try:
                    # split the current column of the Z map in top and bottom
                    # find the split and the extrema of the Y slice of Z
                    k_ZY_slice_split = f_split(Z[i_ycenter-i_ycenter_pad:i_ycenter+i_ycenter_pad+1,k].flatten()) + i_ycenter - i_ycenter_pad
                    Y_bottom_left, Y_top_left = find_dual_value(Y.flatten(), Z[:,k].flatten(), k_ZY_slice_split, lvl, thr, interp_method, f_sgn, search_tol)

                    # insert the coordinates into the contour trace dict if not nan (bounds error) and order properly for merging later
                    if np.isfinite(Y_bottom_left):
                        theta_Y_bottom_left = np.arctan2(Y_bottom_left - Y[i_ycenter], X[k] - X[i_xcenter])
                        if theta_Y_bottom_left < 0.0:
                            theta_Y_bottom_left = 2.0 * np.pi + theta_Y_bottom_left
                        XY_contour['left']['bottom'].append((X[k],Y_bottom_left,theta_Y_bottom_left))
                        # update the higher vertical bound if it is higher than the vertical bound found at the magnetic axis
                        if Y_bottom_left < (Y_search_min + 3.0 * dY):
                            Y_search_min = Y_bottom_left - 3.0 * dY
                    if np.isfinite(Y_top_left):
                        theta_Y_top_left = np.arctan2(Y_top_left - Y[i_ycenter], X[k] - X[i_xcenter])
                        if theta_Y_top_left < 0.0:
                            theta_Y_top_left = 2.0 * np.pi + theta_Y_top_left
                        XY_contour['left']['top'].append((X[k],Y_top_left,theta_Y_top_left))
                        # update the lower vertical bound if it is lower than the vertical bound found at the magnetic axis
                        if Y_top_left > (Y_search_max - 3.0 * dY):
                            Y_search_max = Y_top_left + 3.0 * dY
                    #print('1', X[k], Y_bottom_left, Y_top_left, theta_Y_bottom_left, theta_Y_top_left)
                except:
                    pass
                # update the slice coordinates
                k -= 1
            else:
                do_k = False

        # while the level intersects with the current Z slice gather the intersection coordinates
        do_i = True
        while (do_i and i >= 0 and f_check(lvl, Z[i,:])):
            # interpolate the R coordinate of the bottom half of the contour on both the right and left
            if Y[i] >= Y_search_min:
                try:
                    # find the split and the extrema of the Y slice of Z
                    i_ZX_slice_split = f_split(Z[i,i_xcenter-i_xcenter_pad:i_xcenter+i_xcenter_pad+1].flatten()) + i_xcenter - i_xcenter_pad
                    X_bottom_left, X_bottom_right = find_dual_value(X.flatten(), Z[i,:].flatten(), i_ZX_slice_split, lvl, thr, interp_method, f_sgn, search_tol)

                    if np.isfinite(X_bottom_left):
                        theta_X_bottom_left = np.arctan2(Y[i] - Y[i_ycenter], X_bottom_left - X[i_xcenter])
                        if theta_X_bottom_left < 0.0:
                            theta_X_bottom_left = 2.0 * np.pi + theta_X_bottom_left
                        XY_contour['left']['bottom'].append((X_bottom_left,Y[i],theta_X_bottom_left))
                        if X_bottom_left < (X_search_min + 3.0 * dX):
                            X_search_min = X_bottom_left - 3.0 * dX
                    if np.isfinite(X_bottom_right):
                        theta_X_bottom_right = np.arctan2(Y[i] - Y[i_ycenter], X_bottom_right - X[i_xcenter])
                        if theta_X_bottom_right < 0.0:
                            theta_X_bottom_right = 2.0 * np.pi + theta_X_bottom_right
                        XY_contour['right']['bottom'].append((X_bottom_right,Y[i],theta_X_bottom_right))
                        if X_bottom_right > (X_search_max - 3.0 * dX):
                            X_search_max = X_bottom_right + 3.0 * dX
                    #print('1', Y[i], X_bottom_left, X_bottom_right, theta_X_bottom_left, theta_X_bottom_right)
                except:
                    pass
                # update the slice coordinates
                i -= 1
            else:
                do_i = False

        # while the level intersects with the current Z slice gather the intersection coordinates
        do_l = True
        while (do_l and l < Z.shape[1] and f_check(lvl, Z[:,l])):
            if X[l] <= X_search_max:
                try:
                    # find the split and the extrema of the X slice of Z
                    l_ZY_slice_split = f_split(Z[i_ycenter-i_ycenter_pad:i_ycenter+i_ycenter_pad+1,l].flatten()) + i_ycenter - i_ycenter_pad
                    Y_bottom_right, Y_top_right = find_dual_value(Y.flatten(), Z[:,l].flatten(), l_ZY_slice_split, lvl, thr, interp_method, f_sgn, search_tol)

                    # insert the coordinates into the contour trace dict if not nan (bounds error) and order properly for merging later
                    if np.isfinite(Y_bottom_right):
                        theta_Y_bottom_right = np.arctan2(Y_bottom_right - Y[i_ycenter], X[l] - X[i_xcenter])
                        if theta_Y_bottom_right < 0.0:
                            theta_Y_bottom_right = 2.0 * np.pi + theta_Y_bottom_right
                        XY_contour['right']['bottom'].append((X[l],Y_bottom_right,theta_Y_bottom_right))
                        # update the higher vertical bound if it is higher than the vertical bound found at the magnetic axis
                        if Y_bottom_right < (Y_search_min + 3.0 * dY):
                            Y_search_min = Y_bottom_right - 3.0 * dY
                    if np.isfinite(Y_top_right):
                        theta_Y_top_right = np.arctan2(Y_top_right - Y[i_ycenter], X[l] - X[i_xcenter])
                        if theta_Y_top_right < 0.0:
                            theta_Y_top_right = 2.0 * np.pi + theta_Y_top_right
                        XY_contour['right']['top'].append((X[l],Y_top_right,theta_Y_top_right))
                        # update the lower vertical bound if it is lower than the vertical bound found at the magnetic axis
                        if Y_top_right > (Y_search_max - 3.0 * dY):
                            Y_search_max = Y_top_right + 3.0 * dY
                    #print('1', X[l], Y_bottom_right, Y_top_right, theta_Y_bottom_right, theta_Y_top_right)
                except:
                    pass
                # update the slice coordinates
                l += 1
            else:
                do_l = False

        ### GO THROUGH LOOP AGAIN IN ANOTHER ORDER TO CATCH MISSING POINTS DUE TO INADEQUATE SEARCH RANGES!

        # while the level intersects with the current Z slice gather the intersection coordinates
        do_i = True
        while (do_i and i >= 0 and f_check(lvl, Z[i,:])):
            # interpolate the R coordinate of the bottom half of the contour on both the right and left
            if Y[i] >= Y_search_min:
                try:
                    # find the split and the extrema of the Y slice of Z
                    i_ZX_slice_split = f_split(Z[i,i_xcenter-i_xcenter_pad:i_xcenter+i_xcenter_pad+1].flatten()) + i_xcenter - i_xcenter_pad
                    X_bottom_left, X_bottom_right = find_dual_value(X.flatten(), Z[i,:].flatten(), i_ZX_slice_split, lvl, thr, interp_method, f_sgn, search_tol)

                    if np.isfinite(X_bottom_left):
                        theta_X_bottom_left = np.arctan2(Y[i] - Y[i_ycenter], X_bottom_left - X[i_xcenter])
                        if theta_X_bottom_left < 0.0:
                            theta_X_bottom_left = 2.0 * np.pi + theta_X_bottom_left
                        XY_contour['left']['bottom'].append((X_bottom_left,Y[i],theta_X_bottom_left))
                        if X_bottom_left < (X_search_min + 3.0 * dX):
                            X_search_min = X_bottom_left - 3.0 * dX
                    if np.isfinite(X_bottom_right):
                        theta_X_bottom_right = np.arctan2(Y[i] - Y[i_ycenter], X_bottom_right - X[i_xcenter])
                        if theta_X_bottom_right < 0.0:
                            theta_X_bottom_right = 2.0 * np.pi + theta_X_bottom_right
                        XY_contour['right']['bottom'].append((X_bottom_right,Y[i],theta_X_bottom_right))
                        if X_bottom_right > (X_search_max - 3.0 * dX):
                            X_search_max = X_bottom_right + 3.0 * dX
                    #print('2', Y[i], X_bottom_left, X_bottom_right, theta_X_bottom_left, theta_X_bottom_right)
                except:
                    pass
                # update the slice coordinates
                i -= 1
            else:
                do_i = False

        # while the level intersects with the current Z slice gather the intersection coordinates
        do_k = True
        while (do_k and k >= 0 and f_check(lvl, Z[:,k])):
            # interpolate the X coordinate of the bottom half of the contour on both the right and left
            if X[k] >= X_search_min:
                try:
                    # split the current column of the Z map in top and bottom
                    # find the split and the extrema of the Y slice of Z
                    k_ZY_slice_split = f_split(Z[i_ycenter-i_ycenter_pad:i_ycenter+i_ycenter_pad+1,k].flatten()) + i_ycenter - i_ycenter_pad
                    Y_bottom_left, Y_top_left = find_dual_value(Y.flatten(), Z[:,k].flatten(), k_ZY_slice_split, lvl, thr, interp_method, f_sgn, search_tol)

                    # insert the coordinates into the contour trace dict if not nan (bounds error) and order properly for merging later
                    if np.isfinite(Y_bottom_left):
                        theta_Y_bottom_left = np.arctan2(Y_bottom_left - Y[i_ycenter], X[k] - X[i_xcenter])
                        if theta_Y_bottom_left < 0.0:
                            theta_Y_bottom_left = 2.0 * np.pi + theta_Y_bottom_left
                        XY_contour['left']['bottom'].append((X[k],Y_bottom_left,theta_Y_bottom_left))
                        # update the higher vertical bound if it is higher than the vertical bound found at the magnetic axis
                        if Y_bottom_left < (Y_search_min + 3.0 * dY):
                            Y_search_min = Y_bottom_left - 3.0 * dY
                    if np.isfinite(Y_top_left):
                        theta_Y_top_left = np.arctan2(Y_top_left - Y[i_ycenter], X[k] - X[i_xcenter])
                        if theta_Y_top_left < 0.0:
                            theta_Y_top_left = 2.0 * np.pi + theta_Y_top_left
                        XY_contour['left']['top'].append((X[k],Y_top_left,theta_Y_top_left))
                        # update the lower vertical bound if it is lower than the vertical bound found at the magnetic axis
                        if Y_top_left > (Y_search_max - 3.0 * dY):
                            Y_search_max = Y_top_left + 3.0 * dY
                    #print('2', X[k], Y_bottom_left, Y_top_left, theta_Y_bottom_left, theta_Y_top_left)
                except:
                    pass
                # update the slice coordinates
                k -= 1
            else:
                do_k = False

        # while the level intersects with the current Z slice gather the intersection coordinates
        do_j = True
        while (do_j and j < Z.shape[0] and f_check(lvl, Z[j,:])):
            if Y[j] <= Y_search_max:
                try:
                    # find the split in the Y slice of Z
                    j_ZX_slice_split = f_split(Z[j,i_xcenter-i_xcenter_pad:i_xcenter+i_xcenter_pad+1].flatten()) + i_xcenter - i_xcenter_pad
                    X_top_left, X_top_right = find_dual_value(X.flatten(), Z[j,:].flatten(), j_ZX_slice_split, lvl, thr, interp_method, f_sgn, search_tol)

                    # insert the coordinates into the contour trace dict if not nan (bounds error) and order properly for merging later
                    if np.isfinite(X_top_left):
                        theta_X_top_left = np.arctan2(Y[j] - Y[i_ycenter], X_top_left - X[i_xcenter])
                        if theta_X_top_left < 0.0:
                            theta_X_top_left = 2.0 * np.pi + theta_X_top_left
                        XY_contour['left']['top'].append((X_top_left,Y[j],theta_X_top_left))
                        if X_top_left < (X_search_min + 3.0 * dX):
                            X_search_min = X_top_left - 3.0 * dX
                    if np.isfinite(X_top_right):
                        theta_X_top_right = np.arctan2(Y[j] - Y[i_ycenter], X_top_right - X[i_xcenter])
                        if theta_X_top_right < 0.0:
                            theta_X_top_right = 2.0 * np.pi + theta_X_top_right
                        XY_contour['right']['top'].append((X_top_right,Y[j],theta_X_top_right))
                        if X_top_right > (X_search_max - 3.0 * dX):
                            X_search_max = X_top_right + 3.0 * dX
                    #print('2', Y[j], X_top_left, X_top_right, theta_X_top_left, theta_X_top_right)
                except:
                    pass
                # update the slice coordinates
                j += 1
            else:
                do_j = False

        # while the level intersects with the current Z slice gather the intersection coordinates
        do_l = True
        while (do_l and l < Z.shape[1] and f_check(lvl, Z[:,l])):
            if X[l] <= X_search_max:
                try:
                    # find the split and the extrema of the X slice of Z
                    l_ZY_slice_split = f_split(Z[i_ycenter-i_ycenter_pad:i_ycenter+i_ycenter_pad+1,l].flatten()) + i_ycenter - i_ycenter_pad
                    Y_bottom_right, Y_top_right = find_dual_value(Y.flatten(), Z[:,l].flatten(), l_ZY_slice_split, lvl, thr, interp_method, f_sgn, search_tol)

                    # insert the coordinates into the contour trace dict if not nan (bounds error) and order properly for merging later
                    if np.isfinite(Y_bottom_right):
                        theta_Y_bottom_right = np.arctan2(Y_bottom_right - Y[i_ycenter], X[l] - X[i_xcenter])
                        if theta_Y_bottom_right < 0.0:
                            theta_Y_bottom_right = 2.0 * np.pi + theta_Y_bottom_right
                        XY_contour['right']['bottom'].append((X[l],Y_bottom_right,theta_Y_bottom_right))
                        # update the higher vertical bound if it is higher than the vertical bound found at the magnetic axis
                        if Y_bottom_right < (Y_search_min + 3.0 * dY):
                            Y_search_min = Y_bottom_right - 3.0 * dY
                    if np.isfinite(Y_top_right):
                        theta_Y_top_right = np.arctan2(Y_top_right - Y[i_ycenter], X[l] - X[i_xcenter])
                        if theta_Y_top_right < 0.0:
                            theta_Y_top_right = 2.0 * np.pi + theta_Y_top_right
                        XY_contour['right']['top'].append((X[l],Y_top_right,theta_Y_top_right))
                        # update the lower vertical bound if it is lower than the vertical bound found at the magnetic axis
                        if Y_top_right > (Y_search_max - 3.0 * dY):
                            Y_search_max = Y_top_right + 3.0 * dY
                    #print('2', X[l], Y_bottom_right, Y_top_right, theta_Y_bottom_right, theta_Y_top_right)
                except:
                    pass
                # update the slice coordinates
                l += 1
            else:
                do_l = False

        # collate in order of theta, counterclockwise from outer midplane
        XY_contour = sorted(XY_contour['right']['top']+XY_contour['left']['top']+XY_contour['left']['bottom']+XY_contour['right']['bottom'], key=itemgetter(2))
        X_contour = [x for x,y,t in XY_contour]
        Y_contour = [y for x,y,t in XY_contour]
        T_contour = [t for x,y,t in XY_contour]
        if X_contour[0] != X_contour[-1]:
            X_contour.append(X_contour[0])
        if Y_contour[0] != Y_contour[-1]:
            Y_contour.append(Y_contour[0])
        if T_contour[0] != T_contour[-1]:
            T_contour.append(T_contour[0])

        dist_limit = 1.2 * np.sqrt(dX ** 2 + dY ** 2)
        d_contour = np.sqrt(np.diff(X_contour) ** 2 + np.diff(Y_contour) ** 2)
        itr = 0
        while itr < len(d_contour):
            if d_contour[itr] > dist_limit:
                X_contour.pop(itr+1)
                Y_contour.pop(itr+1)
                T_contour.pop(itr+1)
                d_contour = np.sqrt(np.diff(X_contour) ** 2 + np.diff(Y_contour) ** 2)
            else:
                itr += 1
        XY_contour = zip(X_contour, Y_contour, T_contour)
        #print(XY_contour)
        XY_contour = list(dict.fromkeys(XY_contour))

        #try:
            # find the extrema of the traced coordinates
            #Y_X_min = min(XY_contour,key=itemgetter(0)) # coordinates for min(X)
            #Y_X_max = max(XY_contour,key=itemgetter(0)) # coordinates for max(X)
            #X_Y_min = min(XY_contour,key=itemgetter(1)) # coordinates for min(Y)
            #X_Y_max = max(XY_contour,key=itemgetter(1)) # coordinates for max(Y)

        # split all the contour coordinates in four sequential quadrants, all sorted by increasing X
        XY_right_top = [(x,y) for x,y,t in XY_contour if (t >= 0.0) and (t < 0.5 * np.pi)] #if x>=X_Y_max[0] and y>Y_X_max[1]]
        XY_left_top = [(x,y) for x,y,t in XY_contour if (t >= 0.5 * np.pi) and (t < np.pi)] #if x<X_Y_max[0] and y>=Y_X_min[1]]
        XY_right_bottom = [(x,y) for x,y,t in XY_contour if (t >= np.pi) and (t < 1.5 * np.pi)] #if x>X_Y_min[0] and y<=Y_X_max[1]]
        XY_left_bottom = [(x,y) for x,y,t in XY_contour if (t >= 1.5 * np.pi) and (t < 2.0 * np.pi)] #if x<=X_Y_min[0] and y<Y_X_min[1]]

        # diagnostic plot for checking the sorting and glueing of the contour
        if tracer_diag == 'trace':
            #for x in [X_Y_min,X_Y_max,Y_X_min,Y_X_max]:
            #    plt.plot(*zip(x),'x')
            plt.plot(*zip(*sorted(XY_right_top)),'.-',label='right, Top')
            plt.plot(*zip(*sorted(XY_left_top)),'.-',label='left, Top')
            plt.plot(*zip(*sorted(XY_right_bottom)),'.-',label='right, Bottom')
            plt.plot(*zip(*sorted(XY_left_bottom)),'.-',label='left, Bottom')
            plt.plot(*zip(*XY_contour))
            plt.axis('scaled')
            plt.legend()
            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')
            plt.show()

        # merge the complete contour trace, starting at the right Xmax,Y_Xmax, 
        #XY_contour = XY_right_top[::-1]+XY_left_top[::-1]+XY_left_bottom+XY_right_bottom

        # separate the R and Z coordinates in separate vectors
        X_ = np.array([x for x,y,t in XY_contour])
        Y_ = np.array([y for x,y,t in XY_contour])

        contour_ = {'X':X_,'Y':Y_,'level':lvl[0]}

        # diagnostic plot for checking the complete contour trace, e.g. in combination with the contour extrema fitting
        if tracer_diag == 'contour': # and np.abs(lvl - center) > (0.7 * np.abs(thr - center)):
            #plt.figure()
            plt.plot(contour_['X'],contour_['Y'],'b-')
            if ((center < thr) and (lvl < thr)) or ((center > thr) and (lvl > thr)):
                plt.contour(X,Y,Z,lvl,linestyles='dashed',colors='purple')
            plt.axis('scaled')
            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')
            plt.title(f'psi = {lvl[0]:.4f}')
            plt.show()

        if symmetrise:
            R_sym = (contour_['X']+contour_['X'][::-1])/2
            Z_sym = (contour_['Y']-contour_['Y'][::-1])/2+integrate.trapezoid(contour_['X']*contour_['Y'],contour_['Y'])/integrate.trapezoid(contour_['X'],contour_['Y'])
            contour_['X_sym']=R_sym
            contour_['Y_sym']=Z_sym

        # compute a normalised level label for the contour level
        contour_['label'] = np.sqrt((lvl - center)/(thr - center))[0]
        # find the contour center quantities and add them to the contour dict
        contour_.update(contour_center(contour_,tracer_diag=tracer_diag))

        # zipsort the contour from 0 - 2 pi
        contour_['theta_XY'] = arctan2pi(contour_['Y'] - contour_['Y0'], contour_['X'] - contour_['X0'])
        contour_['theta_XY'], contour_['X'], contour_['Y'] = zipsort(contour_['theta_XY'], contour_['X'], contour_['Y'])

        # close the contour
        contour_['theta_XY'] = np.append(contour_['theta_XY'],contour_['theta_XY'][0])
        contour_['X'] = np.append(contour_['X'],contour_['X'][0])
        contour_['Y'] = np.append(contour_['Y'],contour_['Y'][0])

        #plt.show()
        return contour_
        #except ValueError:
        #    print('tracer.contour: No contour could be traced at the requested level!')

def contour_center(c,tracer_diag='none'):
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
    c = contour_extrema(c,tracer_diag=tracer_diag)

    # compute the minor and major radii of the contour at the average elevation
    c['r'] = (c['X_out']-c['X_in'])/2
    c['X0'] = (c['X_out']+c['X_in'])/2
    #c['X0'] = integrate.trapezoid(c['X']*c['Y'],c['X'])/integrate.trapezoid(c['Y'],c['X'])

    return c

def contour_extrema(c,tracer_diag='none'):
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
    c['X_out'] = float(interpolate.interp1d(Y_out,X_out,bounds_error=False)(c['Y0']))
    c['X_in'] = float(interpolate.interp1d(Y_in,X_in,bounds_error=False)(c['Y0']))

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

    if tracer_diag =='fs':
        # diagnostic plots
        print(len(c['X'][max_filter]),len(c['X'][min_filter]))

        plt.plot(c['X'][max_filter],c['Y'][max_filter],'r.')
        plt.plot(c['X'][min_filter],c['Y'][min_filter],'r.')
        plt.plot(X_Ymax_fit,Y_max_fit,'g-')
        plt.plot(X_Ymin_fit,Y_min_fit,'g-')
        plt.axis('equal')
        plt.show()

    c.update({'X_Ymax':float(X_Ymax),'Y_max':float(Y_max),
              'X_Ymin':float(X_Ymin),'Y_min':float(Y_min),
              'X_min':float(X_min),'Y_Xmin':float(Y_Xmin),
              'X_max':float(X_max),'Y_Xmax':float(Y_Xmax)})

    return c

def _contour_extrema(c,tracer_diag='none'):
    """Find the (true) extrema in X and Y of a contour trace c given by c['X'], c['Y'].

    Args:
        `c` (dict): description of the contour, c={['X'],['Y'],['Y0'],['label'],...}, where:
                    - X,Y: the contour coordinates, 
                    - Y0 (optional): is the average elevation,
                    - label is the normalised contour level label np.sqrt((level - center)/(threshold - center)).

    Returns:
        (dict): the contour with the extrema information added
    """
    n = 16
    zpad = 2 if len(c['Y']) <= n else 3
    zmid = int(len(c['Y']) / 2)

    # restack R_fs and Z_fs to get a continuous midplane outboard trace
    X_out = np.hstack((c['X'][-zpad:],c['X'][1:zpad+1]))
    Y_out = np.hstack((c['Y'][-zpad:],c['Y'][1:zpad+1]))

    X_in = c['X'][zmid-zpad:zmid+zpad+1][::-1]
    Y_in = c['Y'][zmid-zpad:zmid+zpad+1][::-1]

    # find the approximate(!) extrema in Y of the contour
    i_Y_max = np.argmax(c['Y'])
    i_Y_min = np.argmin(c['Y'])

    # check if the midplane of the contour is provided
    if 'Y0' not in c:
        Y_max = c['Y'][i_Y_max]
        Y_min = c['Y'][i_Y_min]
        c['Y0'] = float(Y_min+((Y_max-Y_min)/2))
    Y0 = np.array([c['Y0']])

    # find the extrema in X of the contour at the midplane
    X_out = interpolate.interp1d(Y_out,X_out,bounds_error=False)(Y0)[0]
    X_in = interpolate.interp1d(Y_in,X_in,bounds_error=False)(Y0)[0]

    # in case Y0 is out of bounds in these interpolations
    if not np.isfinite(X_out):
        # restack X to get continuous trace on right side
        X_ = np.hstack((c['X'][np.argmin(c['X']):],c['X'][1:np.argmin(c['X'])]))
        # take the derivative of X_
        dX_ = np.gradient(X_,edge_order=2)
        # find X_out by interpolating the derivative of X to 0.
        dX_vec_out =  dX_[np.argmax(dX_):np.argmin(dX_)]
        X_vec_out = X_[np.argmax(dX_):np.argmin(dX_)]
        X_out = interpolate.interp1d(dX_vec_out,X_vec_out,bounds_error=False)(np.array([0.0]))[0]
    if not np.isfinite(X_in):
        dX = np.gradient(c['X'],edge_order=2)
        dX_vec_in =  dX[np.argmin(dX):np.argmax(dX)]
        X_vec_in = c['X'][np.argmin(dX):np.argmax(dX)]
        X_in = interpolate.interp1d(dX_vec_in,X_vec_in,bounds_error=False)(np.array([0.0]))[0]

    # generate filter lists that take a representative slice of the max and min of the contour coordinates around the approximate Y_max and Y_min
    #alpha = (0.9+0.075*c['label']**2) # magic to ensure just enough points are 
    #max_filter = [z > alpha*(Y_max-c['Y0']) for z in c['Y']-c['Y0']]
    #min_filter = [z < alpha*(Y_min-c['Y0']) for z in c['Y']-c['Y0']]
    #max_filter = c['Y'] > (alpha * Y_max + Y0 * (1.0 - alpha))
    #min_filter = c['Y'] < (alpha * Y_min + Y0 * (1.0 - alpha))

    # patch for the filter lists in case the filter criteria results in < 7 points (minimum of required for 5th order fit + 1)
    order = 5 if len(c['Y']) > n else 3
    mpad = int((order + 1) / 2)
    l = 2 * n
    while (l < len(c['Y'])):
        mpad += 1
        l += n
    X_max_vec = c['X'][i_Y_max-mpad:i_Y_max+mpad+1]
    Y_max_vec = c['Y'][i_Y_max-mpad:i_Y_max+mpad+1]
    X_min_vec = c['X'][i_Y_min-mpad:i_Y_min+mpad+1]
    Y_min_vec = c['Y'][i_Y_min-mpad:i_Y_min+mpad+1]
    #if np.count_nonzero(max_filter) < (order + 2):
    #    for i in range(i_Y_max-mpad,i_Y_max+mpad+1):
    #        if c['Y'][i] >= (np.min(c['Y'])+np.max(c['Y']))/2:
    #            max_filter[i] = True
    #if np.count_nonzero(min_filter) < (order + 2):
    #    for i in range(i_Y_min-mpad,i_Y_min+mpad+1):
    #        if c['Y'][i] <= (np.min(c['Y'])+np.max(c['Y']))/2:
    #            min_filter[i] = True

    # fit the max and min slices of the contour, compute the gradient of these fits and interpolate to zero to find X_Ymax, Y_max, X_Ymin and Y_min
    X_Ymax_fit = np.linspace(np.nanmin(X_max_vec),np.nanmax(X_max_vec),5000)
    try:
        Y_max_fit = interpolate.UnivariateSpline(X_max_vec[::-1],Y_max_vec[::-1],k=order)(X_Ymax_fit)
    except:
        Y_max_fit = np.poly1d(np.polyfit(X_max_vec[::-1],Y_max_vec[::-1],order))(X_Ymax_fit)
    Y_max_fit_grad = np.gradient(Y_max_fit,X_Ymax_fit)

    X_Ymax = interpolate.interp1d(Y_max_fit_grad,X_Ymax_fit,bounds_error=False)(np.array([0.0]))[0]
    Y_max = interpolate.interp1d(X_Ymax_fit,Y_max_fit,bounds_error=False)(np.array([X_Ymax]))[0]

    X_Ymin_fit = np.linspace(np.nanmin(X_min_vec),np.nanmax(X_min_vec),5000)
    try:
        Y_min_fit = interpolate.UnivariateSpline(X_min_vec,Y_min_vec,k=order)(X_Ymin_fit)
    except:
        Y_min_fit = np.poly1d(np.polyfit(X_min_vec,Y_min_vec,order))(X_Ymin_fit)
    Y_min_fit_grad = np.gradient(Y_min_fit,X_Ymin_fit)

    X_Ymin = interpolate.interp1d(Y_min_fit_grad,X_Ymin_fit,bounds_error=False)(np.array([0.0]))[0]
    Y_min = interpolate.interp1d(X_Ymin_fit,Y_min_fit,bounds_error=False)(np.array([X_Ymin]))[0]

    if tracer_diag =='fs':
        # diagnostic plots
        plt.plot(X_max_vec,Y_max_vec,'r.')
        plt.plot(X_min_vec,Y_min_vec,'r.')
        plt.plot(X_Ymax_fit,Y_max_fit,'g-')
        plt.plot(X_Ymin_fit,Y_min_fit,'g-')
        #plt.axis('equal')
        #plt.show()

    c.update({'X_out': X_out, 'X_in': X_in, 'X_Ymax': X_Ymax, 'Y_max': Y_max, 'X_Ymin': X_Ymin, 'Y_min': Y_min})
    #print(c['level'], X_out, X_in, X_Ymax, X_Ymin, Y_max, Y_min)

    return c
