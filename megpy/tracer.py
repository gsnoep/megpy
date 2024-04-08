"""
created by gsnoep on 13 August 2022

Module for tracing closed(!) contour lines in Z on a Y,X grid.
"""
import numpy as np
from scipy import interpolate,integrate
import matplotlib.pyplot as plt
from operator import itemgetter
from .utils import *

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
        # define storage for contour coordinates
        XY_contour = {'left':{'top':[],'bottom':[]},'right':{'top':[],'bottom':[]}}

        # take/find the approximate location of the contour center on the Z map
        if i_center is not None:
            i_xcenter = i_center[0]
            i_ycenter = i_center[1]
        else:
            i_xcenter = int(len(X)/2)
            i_ycenter = int(len(Y)/2)

        center = Z[i_ycenter,i_xcenter]

        # find the vertical extrema of the threshold contour at the xcenter
        if threshold > center:
            f_threshold = np.min
            f_split = np.argmin
            f_bnd = np.argmax
            f_sgn = '>'
        elif threshold < center:
            f_threshold = np.max
            f_split = np.argmax
            f_bnd = np.argmin
            f_sgn = '<'
        i_ZY_xcenter_split = f_split(Z[:,i_xcenter])
        i_ZY_xcenter_bottom_max = f_bnd(Z[:i_ZY_xcenter_split,i_xcenter])
        i_ZY_xcenter_top_max = i_ZY_xcenter_split+f_bnd(Z[i_ZY_xcenter_split:,i_xcenter])

        ZY_xcenter_bottom = Z[i_ZY_xcenter_bottom_max:i_ZY_xcenter_split,i_xcenter]
        ZY_xcenter_top = Z[i_ZY_xcenter_split:i_ZY_xcenter_top_max,i_xcenter]

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

        Y_max = interpolate.interp1d(ZY_xcenter_top,Y[i_ZY_xcenter_split:i_ZY_xcenter_top_max],bounds_error=False,fill_value='extrapolate')(threshold)
        Y_min = interpolate.interp1d(ZY_xcenter_bottom,Y[i_ZY_xcenter_bottom_max:i_ZY_xcenter_split],bounds_error=False,fill_value='extrapolate')(threshold)

        i_ZX_ycenter_split = f_split(Z[i_ycenter,:])
        i_ZX_ycenter_bottom_max = f_bnd(Z[i_ycenter,:i_ZX_ycenter_split])
        i_ZX_ycenter_top_max = i_ZY_xcenter_split+f_bnd(Z[i_ycenter,i_ZX_ycenter_split:])

        ZX_ycenter_bottom = Z[i_ycenter,i_ZX_ycenter_bottom_max:i_ZX_ycenter_split]
        ZX_ycenter_top = Z[i_ycenter,i_ZX_ycenter_split:i_ZX_ycenter_top_max]

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

        X_max = interpolate.interp1d(ZX_ycenter_top,X[i_ZX_ycenter_split:i_ZX_ycenter_top_max],bounds_error=False,fill_value='extrapolate')(threshold)
        X_min = interpolate.interp1d(ZX_ycenter_bottom,X[i_ZX_ycenter_bottom_max:i_ZX_ycenter_split],bounds_error=False,fill_value='extrapolate')(threshold)

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
        i, j = i_ycenter, i_ycenter
        k, l = i_xcenter, i_xcenter

        # while the level intersects with the current Z slice gather the intersection coordinates
        while (eval('level' + f_sgn + 'f_threshold(Z[i])') and i < Z.shape[0]-1):
            if Y[i] <= Y_max:
                try:
                    # find the split in the Y slice of Z
                    i_ZX_slice_split = f_split(Z[i])

                    # chop the Z and X slices in two parts to separate the left and right
                    ZX_slice_top_left = Z[i,:i_ZX_slice_split]
                    ZX_slice_top_right = Z[i,i_ZX_slice_split:]

                    # interpolate the X coordinate of the top half left and right
                    if interp_method == 'normal':
                        X_top_left = float(interpolate.interp1d(ZX_slice_top_left,X[i_ZX_slice_split-len(ZX_slice_top_left):i_ZX_slice_split],bounds_error=False)(level))
                        X_top_right = float(interpolate.interp1d(ZX_slice_top_right,X[i_ZX_slice_split:i_ZX_slice_split+len(ZX_slice_top_right)],bounds_error=False)(level))
                    # if the normal method provides spikey contour traces, bound the interpolation domain and extrapolate the intersection
                    elif interp_method == 'bounded_extrapolation':
                        ZX_slice_top_left = ZX_slice_top_left[ZX_slice_top_left<=threshold]
                        ZX_slice_top_right = ZX_slice_top_right[ZX_slice_top_right<=threshold]

                        X_top_left = float(interpolate.interp1d(ZX_slice_top_left,X[i_ZX_slice_split-len(ZX_slice_top_left):i_ZX_slice_split][ZX_slice_top_left<=threshold],bounds_error=False,fill_value='extrapolate')(level))
                        X_top_right = float(interpolate.interp1d(ZX_slice_top_right,X[i_ZX_slice_split:i_ZX_slice_split+len(ZX_slice_top_right)][ZX_slice_top_right<=threshold],bounds_error=False,fill_value='extrapolate')(level))

                    # diagnostic plot to check interpolation issues for rows
                    if tracer_diag == 'interp':
                        plt.figure()
                        plt.title('Y:{}'.format(Y[i]))
                        plt.plot(ZX_slice_top_left,X[i_ZX_slice_split-len(ZX_slice_top_left):i_ZX_slice_split],'b-',label='left, top')
                        plt.plot(ZX_slice_top_right,X[i_ZX_slice_split:i_ZX_slice_split+len(ZX_slice_top_right)],'r-',label='right, top')
                        plt.axvline(level,ls='dashed',color='black')
                        plt.legend()

                    # insert the coordinate pairs into the contour trace dict if not nan (bounds error) and order properly for merging later
                    if not np.isnan(X_top_left):
                        XY_contour['left']['top'].append((X_top_left,Y[i]))
                        if X_top_left < X_min:
                            X_min = X_top_left
                    if not np.isnan(X_top_right):
                        XY_contour['right']['top'].append((X_top_right,Y[i]))
                        if X_top_right > X_max:
                            X_max = X_top_right
                except:
                    pass

            # update the slice coordinates
            if i < Z.shape[0]-1:
                i+=1

        # while the level intersects with the current Z slice gather the intersection coordinates
        while (eval('level' + f_sgn + 'f_threshold(Z[:,k])') and k < Z.shape[1]-1):
            if X[k] <= X_max:
                try:
                    # find the split and the extrema of the X slice of Z
                    k_ZY_slice_split = f_split(Z[:,k])
                    k_ZY_slice_bottom_max = f_bnd(Z[:k_ZY_slice_split,k])
                    k_ZY_slice_top_max = k_ZY_slice_split+f_bnd(Z[k_ZY_slice_split:,k])

                    # chop the Z slices in two parts to separate the top and bottom halves
                    ZY_slice_bottom_right = Z[k_ZY_slice_bottom_max:k_ZY_slice_split+1,k]
                    ZY_slice_top_right = Z[k_ZY_slice_split:k_ZY_slice_top_max,k]
                    
                    # interpolate the Y coordinate for the right top and bottom
                    if interp_method == 'normal':
                        Y_bottom_right = float(interpolate.interp1d(ZY_slice_bottom_right,Y[k_ZY_slice_bottom_max:k_ZY_slice_split+1],bounds_error=False)(level))
                        Y_top_right = float(interpolate.interp1d(ZY_slice_top_right,Y[k_ZY_slice_split:k_ZY_slice_top_max],bounds_error=False)(level))
                    elif interp_method == 'bounded_extrapolation':
                        Y_slice_bottom_right = Y[k_ZY_slice_bottom_max:k_ZY_slice_split+1][ZY_slice_bottom_right<=threshold]
                        Y_slice_top_right = Y[k_ZY_slice_split:k_ZY_slice_top_max][ZY_slice_top_right<=threshold]
                        ZY_slice_bottom_right = ZY_slice_bottom_right[ZY_slice_bottom_right<=threshold]
                        ZY_slice_top_right = ZY_slice_top_right[ZY_slice_top_right<=threshold]

                        Y_bottom_right = float(interpolate.interp1d(ZY_slice_bottom_right,Y_slice_bottom_right,bounds_error=False,fill_value='extrapolate')(level))
                        Y_top_right = float(interpolate.interp1d(ZY_slice_top_right,Y_slice_top_right,bounds_error=False,fill_value='extrapolate')(level))

                    # diagnostic plot to check interpolation issues for columns
                    if tracer_diag == 'interp':
                        plt.figure()
                        plt.title('X:{}'.format(X[k]))
                        plt.plot(Z[k_ZY_slice_bottom_max:k_ZY_slice_split+1,k],Y[k_ZY_slice_bottom_max:k_ZY_slice_split+1],'b--',label='right, bottom')
                        plt.plot(ZY_slice_top_right,Y[k_ZY_slice_split:k_ZY_slice_top_max],'r--',label='right, top')
                        plt.axvline(level,ls='dashed',color='black')
                        plt.legend()

                    if not np.isnan(Y_bottom_right):
                        XY_contour['right']['bottom'].append((X[k],Y_bottom_right))
                        # update the higher vertical bound if it is higher than the vertical bound found at the magnetic axis
                        if Y_bottom_right < Y_min:
                            Y_min = Y_bottom_right
                    if not np.isnan(Y_top_right):
                        XY_contour['right']['top'].append((X[k],Y_top_right))
                        # update the lower vertical bound if it is lower than the vertical bound found at the magnetic axis
                        if Y_top_right > Y_max:
                            Y_max = Y_top_right
                except:
                    pass

            # update the slice coordinates
            if k < Z.shape[1]-1:
                k+=1

        # while the level intersects with the current Z slice gather the intersection coordinates
        while (eval('level' + f_sgn + 'f_threshold(Z[j])') and j > 0):
            # interpolate the R coordinate of the bottom half of the contour on both the right and left
            if Y[j] >= Y_min:
                try:
                    # find the split and the extrema of the Y slice of Z
                    j_ZX_slice_split = f_split(Z[j])

                    ZX_slice_bottom_left = Z[j,:j_ZX_slice_split]
                    ZX_slice_bottom_right = Z[j,j_ZX_slice_split:]

                    # interpolate the X coordinate of the bottom half left and right
                    if interp_method == 'normal':
                        X_bottom_left = float(interpolate.interp1d(ZX_slice_bottom_left,X[j_ZX_slice_split-len(ZX_slice_bottom_left):j_ZX_slice_split],bounds_error=False)(level))
                        X_bottom_right = float(interpolate.interp1d(ZX_slice_bottom_right,X[j_ZX_slice_split:j_ZX_slice_split+len(ZX_slice_bottom_right)],bounds_error=False)(level))
                    elif interp_method == 'bounded_extrapolation':
                        ZX_slice_bottom_left = ZX_slice_bottom_left[ZX_slice_bottom_left<=threshold]
                        ZX_slice_bottom_right = ZX_slice_bottom_right[ZX_slice_bottom_right<=threshold]
                        X_bottom_left = float(interpolate.interp1d(ZX_slice_bottom_left,X[j_ZX_slice_split-len(ZX_slice_bottom_left):j_ZX_slice_split][ZX_slice_bottom_left<=threshold],bounds_error=False,fill_value='extrapolate')(level))
                        X_bottom_right = float(interpolate.interp1d(ZX_slice_bottom_right,X[j_ZX_slice_split:j_ZX_slice_split+len(ZX_slice_bottom_right)][ZX_slice_bottom_right<=threshold],bounds_error=False,fill_value='extrapolate')(level))

                    # diagnostic plot to check interpolation issues for rows
                    if tracer_diag == 'interp':
                        plt.figure()
                        plt.title('Y:{}'.format(Y[j]))
                        plt.plot(ZX_slice_bottom_left,X[j_ZX_slice_split-len(ZX_slice_bottom_left):j_ZX_slice_split],'b-',label='left, bottom')
                        plt.plot(ZX_slice_bottom_right,X[j_ZX_slice_split:j_ZX_slice_split+len(ZX_slice_bottom_right)],'r-',label='right, bottom')
                        plt.axvline(level,ls='dashed',color='black')
                        plt.legend()

                    if not np.isnan(X_bottom_left):
                        XY_contour['left']['bottom'].append((X_bottom_left,Y[j]))
                        if X_bottom_left < X_min:
                            X_min = X_bottom_left
                    if not np.isnan(X_bottom_right):
                        XY_contour['right']['bottom'].append((X_bottom_right,Y[j]))
                        if X_bottom_right > X_max:
                            X_max = X_bottom_right
                except:
                    pass
            # update the slice coordinates
            if j > 0:
                j-=1

        # while the level intersects with the current Z slice gather the intersection coordinates
        while (eval('level' + f_sgn + 'f_threshold(Z[:,l])') and l > 0):
            # interpolate the X coordinate of the bottom half of the contour on both the right and left
            if X[l] >= X_min:
                try:
                    # split the current column of the Z map in top and bottom
                    # find the split and the extrema of the Y slice of Z
                    l_ZY_slice_split = f_split(Z[:,l])
                    l_ZY_slice_bottom_max = f_bnd(Z[:l_ZY_slice_split,l])
                    l_ZY_slice_top_max = l_ZY_slice_split+f_bnd(Z[l_ZY_slice_split:,l])

                    ZY_slice_bottom_left = Z[l_ZY_slice_bottom_max:l_ZY_slice_split+1,l]
                    ZY_slice_top_left = Z[l_ZY_slice_split:l_ZY_slice_top_max,l]

                    # interpolate the Y coordinate for the left top and bottom
                    if interp_method == 'normal':
                        Y_bottom_left = float(interpolate.interp1d(ZY_slice_bottom_left,Y[l_ZY_slice_bottom_max:l_ZY_slice_split+1],bounds_error=False)(level))
                        Y_top_left = float(interpolate.interp1d(ZY_slice_top_left,Y[l_ZY_slice_split:l_ZY_slice_top_max],bounds_error=False)(level))
                    elif interp_method == 'bounded_extrapolation':
                        Y_slice_bottom_left = Y[l_ZY_slice_bottom_max:l_ZY_slice_split+1][ZY_slice_bottom_left<=threshold]
                        Y_slice_top_left = Y[l_ZY_slice_split:l_ZY_slice_top_max][ZY_slice_top_left<=threshold]
                        ZY_slice_bottom_left = ZY_slice_bottom_left[ZY_slice_bottom_left<=threshold]
                        ZY_slice_top_left = ZY_slice_top_left[ZY_slice_top_left<=threshold]

                        Y_bottom_left = float(interpolate.interp1d(ZY_slice_bottom_left,Y_slice_bottom_left,bounds_error=False,fill_value='extrapolate')(level))
                        Y_top_left = float(interpolate.interp1d(ZY_slice_top_left,Y_slice_top_left,bounds_error=False,fill_value='extrapolate')(level))

                    # diagnostic plot to check interpolation issues for columns
                    if tracer_diag == 'interp':
                        plt.figure()
                        plt.title('X:{}'.format(X[l]))
                        plt.plot(Z[l_ZY_slice_bottom_max:l_ZY_slice_split+1,l],Y[l_ZY_slice_bottom_max:l_ZY_slice_split+1],'b--',label='left, bottom')
                        plt.plot(ZY_slice_top_left,Y[l_ZY_slice_split:l_ZY_slice_top_max],'r--',label='left, top')
                        plt.axvline(level,ls='dashed',color='black')
                        plt.legend()

                    if not np.isnan(Y_bottom_left):
                        XY_contour['left']['bottom'].append((X[l],Y_bottom_left))
                        # update the higher vertical bound if it is higher than the vertical bound found at the magnetic axis
                        if Y_bottom_left < Y_min:
                            Y_min = Y_bottom_left
                    if not np.isnan(Y_top_left):
                        XY_contour['left']['top'].append((X[l],Y_top_left))
                        # update the lower vertical bound if it is lower than the vertical bound found at the magnetic axis
                        if Y_top_left > Y_max:
                            Y_max = Y_top_left
                except:
                    pass

            # update the slice coordinates
            if l > 0:
                l-=1

        # collate all the traced coordinates of the contour and sort by increasing X
        XY_contour = sorted(XY_contour['right']['top']+XY_contour['left']['top']+XY_contour['left']['bottom']+XY_contour['right']['bottom'])
        #print(XY_contour)
        XY_contour = list(dict.fromkeys(XY_contour))

        try:
            # find the extrema of the traced coordinates
            Y_X_min = min(XY_contour) # coordinates for min(X)
            Y_X_max = max(XY_contour) # coordinates for max(X)
            X_Y_min = min(XY_contour,key=itemgetter(1)) # coordinates for min(Y)
            X_Y_max = max(XY_contour,key=itemgetter(1)) # coordinates for max(Y)

            # split all the contour coordinates in four sequential quadrants, all sorted by increasing X
            XY_right_top = [(x,y) for x,y in XY_contour if x>=X_Y_max[0] and y>Y_X_max[1]]
            XY_left_top = [(x,y) for x,y in XY_contour if x<X_Y_max[0] and y>=Y_X_min[1]]
            XY_right_bottom = [(x,y) for x,y in XY_contour if x>X_Y_min[0] and y<=Y_X_max[1]]
            XY_left_bottom = [(x,y) for x,y in XY_contour if x<=X_Y_min[0] and y<Y_X_min[1]]

            # diagnostic plot for checking the sorting and glueing of the contour
            if tracer_diag == 'trace':
                for x in [X_Y_min,X_Y_max,Y_X_min,Y_X_max]:
                    plt.plot(*zip(x),'x')
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
            XY_contour = XY_right_top[::-1]+XY_left_top[::-1]+XY_left_bottom+XY_right_bottom

            # separate the R and Z coordinates in separate vectors
            X_ = np.array([x for x,y in XY_contour])
            Y_ = np.array([y for x,y in XY_contour])

            contour_ = {'X':X_,'Y':Y_,'level':level}

            # diagnostic plot for checking the complete contour trace, e.g. in combination with the contour extrema fitting
            if tracer_diag == 'contour':
                #plt.figure()
                plt.plot(contour_['X'],contour_['Y'],'b-')
                if ((center < threshold) and (level < threshold)) or ((center > threshold) and (level > threshold)):
                    plt.contour(X,Y,Z,[level],linestyles='dashed',colors='purple')
                plt.axis('scaled')
                plt.xlabel('X [m]')
                plt.ylabel('Y [m]')
                plt.show()

            if symmetrise:
                R_sym = (contour_['X']+contour_['X'][::-1])/2
                Z_sym = (contour_['Y']-contour_['Y'][::-1])/2+integrate.trapz(contour_['X']*contour_['Y'],contour_['Y'])/integrate.trapz(contour_['X'],contour_['Y'])
                contour_['X_sym']=R_sym
                contour_['Y_sym']=Z_sym

            # compute a normalised level label for the contour level
            contour_['label'] = np.sqrt((level - center)/(threshold - center))
            # find the contour center quantities and add them to the contour dict
            contour_.update(contour_center(contour_,tracer_diag=tracer_diag))

            # zipsort the contour from 0 - 2 pi
            contour_['theta_XY'] = arctan2pi(contour_['Y']-contour_['Y0'],contour_['X']-contour_['X0'])
            contour_['theta_XY'], contour_['X'], contour_['Y'] = zipsort(contour_['theta_XY'], contour_['X'], contour_['Y'])

            # close the contour
            contour_['theta_XY'] = np.append(contour_['theta_XY'],contour_['theta_XY'][0])
            contour_['X'] = np.append(contour_['X'],contour_['X'][0])
            contour_['Y'] = np.append(contour_['Y'],contour_['Y'][0])

            #plt.show()
            return contour_
        except ValueError:
            print('tracer.contour: No contour could be traced at the requested level!')

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
        if c['X'][-1] != c['X'][0] or c['Y'][-1] != c['Y'][0]:
            c['X'] = np.append(c['X'],c['X'][0])
            c['Y'] = np.append(c['Y'],c['Y'][0])

        # find the average elevation (midplane) of the contour by computing the vertical centroid [Candy PPCF 51 (2009) 105009]
        c['Y0'] = integrate.trapz(c['X']*c['Y'],c['Y'])/integrate.trapz(c['X'],c['Y'])

        # find the extrema of the contour in the radial direction at the average elevation
        c = contour_extrema(c,tracer_diag=tracer_diag)

        # compute the minor and major radii of the contour at the average elevation
        c['r'] = (c['X_out']-c['X_in'])/2
        c['X0'] = (c['X_out']+c['X_in'])/2
        #c['X0'] = integrate.trapz(c['X']*c['Y'],c['X'])/integrate.trapz(c['Y'],c['X'])

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

    if tracer_diag =='fs':
        # diagnostic plots
        plt.plot(c['X'][max_filter],c['Y'][max_filter],'r.')
        plt.plot(c['X'][min_filter],c['Y'][min_filter],'r.')
        plt.plot(X_Ymax_fit,Y_max_fit,'g-')
        plt.plot(X_Ymin_fit,Y_min_fit,'g-')
        #plt.axis('equal')
        #plt.show()

    c.update({'X_Ymax':float(X_Ymax),'Y_max':float(Y_max),'X_Ymin':float(X_Ymin),'Y_min':float(Y_min)})

    return c