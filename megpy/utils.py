"""
created by gsnoep on 11 August 2022

A collection of general numerical or Python utilities useful across the package
"""

import numpy as np
import os
import copy
import pickle
import h5py

def number(x):
    """Check if x is actually a (real) number type (int,float).

    Args:
        `x` (any): the value to be checked for being a number.

    Returns:
        int,float: returns `x` typed as either int or float.
    """
    try:
        return int(x)
    except ValueError:
        return float(x)

def find(val, arr,n=1):
    """Find the n closest values in an array.

    Args:
        `val` (int,float,str): the sought after value.
        `arr` (ndarray,list): the array to search in for val.
        `n` (int, optional): the number of closest values in arr to be returned. Defaults to 1.

    Returns:
        int,list: depending on `n` either an int for the index of the closest found value, or a list of the `n` indexes of the closest found indexes is returned.
    """
    if isinstance(arr,list):
        arr_ = np.array(arr)
    else:
        arr_ = arr
    if n == 1:
        try:
            return np.argsort(np.abs(arr_-val))[0]
        except:
            pass
    else:
        return list(np.argsort(np.abs(arr_-val)))[:n]

def arctan2pi(y,x):
    """Compute the element-wise arctan(y/x) choosing the quadrant correctly and reformating the result to be bound between 0 and 2 * pi

    Args:
        y (array): y-coordinates, MUST be real-valued!
        x (array): x-coordinates, MUST be real-values!

    Raises:
        ValueError: if y-coordinates are not all real-valued
        ValueError: if x-coordinates are not all real-valued

    Returns:
        thetha (array): element-wise arctan(y/x) bounded between 0 and 2 * pi
    """
    if np.all(np.isreal(y)) and np.all(np.isreal(x)):
        # calculate the angle
        theta = np.arctan2(y,x)
        # fix the arctan asymptote jump
        for ind in np.flatnonzero(np.abs(np.diff(theta)) > np.pi):
            theta[ind+1:] += -1*np.sign(np.diff(theta)[ind])*2*np.pi
        # make sure the resulting angle array is between 0 and 2 pi
        theta[theta<0.] += 2*np.pi
        theta[theta>2*np.pi] -= 2*np.pi

        return theta
    else:
        if not np.all(np.isreal(y)):
            raise ValueError('y-coordinates are not all real, check your inputs!')
        if not np.all(np.isreal(x)):
            raise ValueError('x-coordinates are not all real, check your inputs!')

def arcsin2pi(x,bounds_error=False):
    """Compute the element-wise arcsin(x) reformating the result to be bound between 0 and 2 * pi

    Args:
        x (array): x-coordinates, MUST be real-values between -1 and 1!

    Raises:
        ValueError: if x-coordinates are not -1 <= x <= 1

    Returns:
        thetha (array): element-wise arcsin(x) bounded between 0 and 2 * pi
    """
    if (np.any(x<-1) or np.any(x>1)):
        if bounds_error:
            raise ValueError('arcsin2pi: x does not fit the input bounds requirement: -1 <= x <= 1 !')
        else:
            x[x>1] = 1.
            x[x<-1] = -1.

    # calculate the angle
    theta = np.arcsin(x)

    # fix the asymptote jumps
    theta[find(1,x):find(-1,x)] = (-np.arcsin(x)+np.pi)[find(1,x):find(-1,x)]
    theta[find(-1,x):] = (np.arcsin(x)+2*np.pi)[find(-1,x):]

    return theta

def arccos2pi(x,bounds_error=False):
    """Compute the element-wise arccos(x) reformating the result to be bound between 0 and 2 * pi

    Args:
        x (array): x-coordinates, MUST be real-values between -1 and 1!

    Raises:
        ValueError: if x-coordinates are not -1 <= x <= 1

    Returns:
        thetha (array): element-wise arcsin(x) bounded between 0 and 2 * pi
    """
    if (np.any(x<-1) or np.any(x>1)):
        if bounds_error:
            raise ValueError('arcsin2pi: x does not fit the input bounds requirement: -1 <= x <= 1 !')
        else:
            x[x>1] = 1.
            x[x<-1] = -1.

    # calculate the angle
    theta = np.arccos(x)

    # fix the asymptote jumps
    theta[:find(1,x)] = (-np.arccos(x)+2*np.pi)[:find(1,x)]
    theta[find(-1,x):] = (-np.arccos(x)+2*np.pi)[find(-1,x):]

    return theta

def read_file(path='./',file=None,mode='r'):
    """Read the contents of a file to a list of lines, with automatic path validity checks.

    Args:
        `path` (str): path to the file. Defaults to './'.
        `file` (str): filename. Defaults to None.
        `mode` (str, optional): reading mode of open(). Defaults to 'r', 'rb' also possible.

    Raises:
        `ValueError`: if `path` is not a valid path
        `ValueError`: if `path+file` is not a valid path

    Returns:
        `list`: a list of the lines read from the file
    """
    # check if the provided path exists
    if os.path.isdir(path):
        # check if the provided file name exists in the output path
        if os.path.isfile(path+file):
            # read the file contents into a list of strings of the lines
            with open(path+file,mode) as f:
                lines = f.readlines()
        else:
            raise ValueError('The file {}{} does not exist!'.format(path,file))
    else:
        raise ValueError('{} is not a valid path to a directory!'.format(path))

    return lines

def autotype(value):
    """Automatically types any string value input to bool, int, float or str.
    Useful when reading mixed type values from a text file.

    Args:
        `value` (str): a value that needs to be re-typed

    Returns:
        bool,int,float,str: typed value
    """
    # first check if value is a bool to prevent int(False)=0 or int(True)=1
    if not isinstance(value,bool):
        # then try int
        try:
            value = int(value)
        except:
            # then try float
            try:
                value = float(value)
            # if not float then perhaps a bool string (from Fortran)
            except:
                if value in ['.T.','.t.','.true.','T','t','True','true','y','Y','yes','Yes']:
                    value = True
                elif value in ['.F.','.f.','.false.','F','f','False','false','n','N','no','No']:
                    value = False
                # no other likely candidates, just strip whitespace and return string
                else:
                    value = str(value.strip("'"))
    # return the bool
    else:
        value = bool(value)
    return value

def list_to_array(object):
    """Convert any list in the object to a ndarray.
    Includes recursive check for dict as input to convert any list in the dict to ndarray, assuming fully unconnected dict!

    Args:
        `object` (list,dict): the object containing one or more list that needs to be converted to an array

    Returns:
        ndarray,dict: `object` containing the converted arrays
    """
    if isinstance(object,dict):
        #print('found dict instead of list, rerouting...')
        for key in object.keys():
            object[key] = list_to_array(object[key])
    elif isinstance(object,list):
        # check if any value in the list is a str
        str_check = [isinstance(value,str) for value in object]
        array_check = [isinstance(value,float) for value in object]
        # if not any strings in the list convert to ndarray
        if all(array_check) and not any(str_check):
            #print('converting list to array...')
            object = np.array(object)
        elif any(str_check) and 'list-of-arrays' in object:
            object.remove('list-of-arrays')
            for index in range(0,len(object)):
                object[index] = np.array(object[index])

    return object

def array_to_list(object):
    """Convert any ndarray into a (list of) list(s).
    Includes recursive check for dict as input to convert any ndarray in the dict to list, assuming fully unconnected dict!

    Args:
        object (ndarray, dict): the object containing one or more arrays that needs to be converted to a list

    Returns:
        list,dict: the object containing the converted lists
    """
    if isinstance(object,np.ndarray):
        #print('converting array to list...')
        object = object.tolist()
    elif isinstance(object,dict):
        #print('found dict instead of array, rerouting...')
        for key in object.keys():
            object[key] = array_to_list(object[key])
    elif isinstance(object,list):
        list_of_arrays = False
        for index in range(0,len(object)):
            if isinstance(object[index],np.ndarray):
                list_of_arrays = True
                object[index] = object[index].tolist()
        if list_of_arrays:     
            object.append('list-of-arrays')
    
    return object

def merge_trees(source,target):
    """Merge two dictionaries, key by key append values to the target dictionary.

    Args:
        source (dict): dict containing key,value pairs to be appended.
        target (dict): dict to append the values to.
    """
    if isinstance(source,dict) and isinstance(target,dict):
        for key in source.keys():
            if key not in target.keys():
                target.update({key:copy.deepcopy(source[key])})
            elif isinstance(source[key],(float,int,str,np.ndarray,bool)) and isinstance(target[key],(float,int,str,np.ndarray,bool)):
                target.update({key:[target[key]]})
                target[key].append(source[key])
            elif isinstance(source[key],(float,int,str,list,np.ndarray,bool)) and isinstance(target[key],list):
                target[key].append(source[key])
            elif isinstance(source[key],dict):
                merge_trees(source[key],target[key])
    if isinstance(source,list) and isinstance(target,dict):
        for key in source:
            if key not in target.keys():
                target.update({key:[]})

def zipsort(theta,x,y):
    """Sort two coordinates x,y as function of the polar angle theta between them.

    Args:
        theta (array): array containing polar angles.
        x (array): array containing x-coordinates.
        y (array): array containing y-coordinates.

    Returns:
        theta_sorted (array), x_sorted (array), y_sorted (array): Tuple of sorted arrays.
    """
    if not isinstance(theta,np.ndarray):
        theta = np.array(theta)
    # sort 2D coordinates x,y by their polar angle theta to ensure theta_param=[0,2pi]
    #theta_xy_sorted = sorted(zip(theta,zip(x,y)))
    #theta_sorted = np.array(list(list(zip(*theta_xy_sorted))[0]))
    #x_sorted = np.array(list(list(zip(*list(zip(*theta_xy_sorted))[1]))[0]))
    #y_sorted = np.array(list(list(zip(*list(zip(*theta_xy_sorted))[1]))[1]))
    i_theta = theta.argsort()
    theta_sorted = theta[i_theta]
    x_sorted = x[i_theta]
    y_sorted = y[i_theta]

    return theta_sorted,x_sorted,y_sorted

def interpolate_periodic(x_in,y_in,x_out):
    """Interpolate data on a periodic basis between 0 and 2*pi

    Args:
        x_in (array): input periodic basis (sorted between 0 and 2*pi)
        y_in (array): periodic data (sorted on an ascending basis between 0 and 2*pi)
        x_out (array): output basis

    Returns:
        array: interpolated data
    """
    # extend the periodic data one period before and one beyond
    x_extended = np.hstack(((x_in-2*np.pi),x_in,(x_in+2*np.pi)))
    y_extended = np.hstack((y_in,y_in,y_in))
    
    # interpolate to [0,2*pi]
    y_out = np.interp(x_out,x_extended,y_extended,period=2*np.pi)
    return y_out

def read_pickle(f_path):
    try:
        with open(f_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except (pickle.UnpicklingError, FileNotFoundError) as error:
        print("Error loading pickle file:", error)
        return None

def read_hdf5(f_path):
    try:
        data = h5py.File(f_path, 'r')
        return data
    except (OSError, FileNotFoundError) as error:
        print("Error loading HDF5 file:", error)
        return None
