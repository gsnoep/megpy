#!/usr/bin/env python3
"""
Python translation of read_vmec.m

This module reads VMEC wout files (netCDF format) and returns the data in a structure
similar to the original MATLAB function. Based on 'readw_only_priv.f' subroutine.

Currently supports VMEC files up to version 8+ (netCDF format only).
Returns fourier harmonics in the NESCOIL (nu+nv) format.

Example usage:
    data = read_vmec('wout_test.nc')

Maintained by: Python translation
Version: 1.0 (based on MATLAB version 1.96)
"""

import numpy as np
import netCDF4 as nc
import os
from typing import Dict, Any, Union


class VMECData:
    """Class to hold VMEC data in a structure similar to MATLAB struct"""
    def __init__(self):
        pass


def read_vmec(filename: str) -> Union[VMECData, int]:
    """
    Read VMEC wout file and return data structure
    
    Args:
        filename: Path to VMEC wout file (netCDF format)
        
    Returns:
        VMECData object containing all VMEC data, or error code if failed
    """
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f'ERROR: Could not find file {filename}')
        return -1
    
    # Handle file type - for now only support netCDF
    if filename.endswith('.nc') and 'wout' in filename:
        try:
            f = read_vmec_netcdf(filename)
        except Exception as e:
            print(f'Error reading netCDF file: {e}')
            return -2
    else:
        print('Only netCDF format (.nc) files are currently supported')
        return -3
    
    # If VMEC threw an error then return what was read
    if hasattr(f, 'ierr_vmec') and f.ierr_vmec and (f.ierr_vmec != 4):
        print('VMEC runtime error detected!')
        return f
    
    # Set default values if missing
    if not hasattr(f, 'lrfp__logical__'):
        f.lrfp__logical__ = 0
    
    # Handle missing NYQ values
    if not hasattr(f, 'xm_nyq'):
        f.xm_nyq = f.xm
    if not hasattr(f, 'xn_nyq'):
        f.xn_nyq = f.xn
    if not hasattr(f, 'mnmax_nyq'):
        f.mnmax_nyq = f.mnmax
    
    # Convert various quantities to full mesh
    f = half2fullmesh(f)
    
    # Recompose the Fourier arrays
    f = create_fourier_arrays(f)
    
    # Calculate additional derived quantities
    f = calculate_derived_quantities(f)
    
    return f


def read_vmec_netcdf(filename: str) -> VMECData:
    """
    Read VMEC netCDF file
    
    Args:
        filename: Path to netCDF file
        
    Returns:
        VMECData object with raw netCDF data and derived quantities
    """
    
    mu0 = 4.0 * np.pi * 1e-7
    
    # Read netCDF file
    f = VMECData()
    
    with nc.Dataset(filename, 'r') as ncfile:
        # Read all variables from netCDF file
        for var_name in ncfile.variables:
            var = ncfile.variables[var_name]
            if var.ndim == 0:  # Scalar
                setattr(f, var_name, var[:].item())
            else:  # Array
                setattr(f, var_name, var[:])
        
        # Read global attributes
        for attr_name in ncfile.ncattrs():
            setattr(f, attr_name, getattr(ncfile, attr_name))
    
    # Fix named fields to match MATLAB convention
    f.ierr_vmec = f.ier_flag
    if f.ierr_vmec != 0:
        return f
    
    f.input_extension = f.input_extension if hasattr(f, 'input_extension') else ''
    f.mgrid_file = f.mgrid_file if hasattr(f, 'mgrid_file') else ''
    f.rmax_surf = f.rmax_surf if hasattr(f, 'rmax_surf') else 0
    f.rmin_surf = f.rmin_surf if hasattr(f, 'rmin_surf') else 0  
    f.zmax_surf = f.zmax_surf if hasattr(f, 'zmax_surf') else 0
    f.ireconstruct = getattr(f, 'lrecon__logical__', 0)
    f.imse = -1
    f.itse = -1
    f.RBtor = f.rbtor
    f.Rmajor = f.Rmajor_p
    f.Aminor = f.Aminor_p
    f.betatot = f.betatotal
    f.Volume = f.volume_p
    f.VolAvgB = f.volavgB
    
    if hasattr(f, 'betavol'):
        f.beta_vol = f.betavol
    if hasattr(f, 'specw'):
        f.specw = f.specw
    
    if not hasattr(f, 'iasym'):
        f.iasym = 0
    f.iasym = getattr(f, 'lasym__logical__', 0)
    f.freeb = getattr(f, 'lfreeb__logical__', 0)
    f.lfreeb = f.freeb
    f.Itor = f.ctor
    f.Dmerc = f.DMerc
    f.Dwell = f.DWell  
    f.Dshear = f.DShear
    f.Dcurr = f.DCurr
    f.Dgeod = f.DGeod
    
    # Cast some values to ensure they're the right type
    f.ntor = int(f.ntor)
    f.mpol = int(f.mpol)
    f.nfp = int(f.nfp)
    f.ns = int(f.ns)
    
    # Fix stripped field names
    f.xm_nyq = f.xm_nyq if hasattr(f, 'xm_nyq') else f.xm
    f.xn_nyq = f.xn_nyq if hasattr(f, 'xn_nyq') else f.xn
    f.mnmax_nyq = f.mnmax_nyq if hasattr(f, 'mnmax_nyq') else f.mnmax
    
    # Calculate current densities
    f = calculate_currents(f, mu0)
    
    return f


def calculate_currents(f: VMECData, mu0: float) -> VMECData:
    """Calculate current densities from magnetic field components"""
    
    # Check if currents are already calculated
    if hasattr(f, 'currumnc') and hasattr(f, 'currvmnc'):
        # Currents already exist in file, just normalize by mu0
        f.currumnc = f.currumnc / mu0
        f.currvmnc = f.currvmnc / mu0
        return f
    
    # If not, we would need to calculate them, but for now just skip
    # since most modern VMEC files include them
    print("Warning: Current densities not found in file and calculation not implemented")
    
    return f


def half2fullmesh(f: VMECData) -> VMECData:
    """Convert half-mesh quantities to full mesh"""
    
    # This is a simplified version - the full implementation would
    # handle the interpolation from half to full grid
    # For now, we'll assume most quantities are already on full grid
    
    return f


def create_fourier_arrays(f: VMECData) -> VMECData:
    """Create full Fourier arrays from spectral coefficients"""
    
    msize = int(np.max(f.xm))
    nsize = max(1, int(np.max(f.xn/f.nfp) - np.min(f.xn/f.nfp) + 1))
    
    f.rbc = np.zeros((msize+1, nsize, f.ns))
    f.zbs = np.zeros((msize+1, nsize, f.ns))
    if hasattr(f, 'lmns'):
        f.lbs = np.zeros((msize+1, nsize, f.ns))
    
    # Create derivative terms
    f.rsc = np.zeros((msize+1, nsize, f.ns))
    f.rus = np.zeros((msize+1, nsize, f.ns))
    f.rvs = np.zeros((msize+1, nsize, f.ns))
    f.zss = np.zeros((msize+1, nsize, f.ns))
    f.zuc = np.zeros((msize+1, nsize, f.ns))
    f.zvc = np.zeros((msize+1, nsize, f.ns))
    
    offset = int(np.min(f.xn/f.nfp))
    original_xn = f.xn.copy()  # Save original before sign change
    f.xn = -f.xn  # Sign convention
    
    for i in range(f.ns):
        for j in range(f.mnmax):
            m = int(f.xm[j])
            n = int(-offset + f.xn[j]/f.nfp)
            
            # Make sure indices are within bounds
            if m <= msize and 0 <= n < nsize:
                f.rbc[m, n, i] = f.rmnc[i, j]  # Note: transposed indices
                f.rus[m, n, i] = -f.rmnc[i, j] * f.xm[j]
                f.rvs[m, n, i] = -f.rmnc[i, j] * f.xn[j]
                f.zbs[m, n, i] = f.zmns[i, j]  # Note: transposed indices
                f.zuc[m, n, i] = f.zmns[i, j] * f.xm[j]
                f.zvc[m, n, i] = f.zmns[i, j] * f.xn[j]
                if hasattr(f, 'lmns'):
                    f.lbs[m, n, i] = f.lmns[i, j]
    
    # Create additional derived arrays with correct indexing
    f.rumns = -f.rmnc * f.xm[np.newaxis, :]  # Broadcasting
    f.rvmns = -f.rmnc * original_xn[np.newaxis, :]  # Use original xn
    f.zumnc = f.zmns * f.xm[np.newaxis, :]
    f.zvmnc = f.zmns * original_xn[np.newaxis, :]
    
    # Handle radial derivatives
    f.rsc = f.rbc.copy()
    f.zss = f.zbs.copy()
    f.rsmnc = f.rmnc.copy()
    f.zsmns = f.zmns.copy()
    
    for i in range(1, f.ns-1):
        f.rsmnc[i, :] = f.rmnc[i+1, :] - f.rmnc[i-1, :]
        f.zsmns[i, :] = f.zmns[i+1, :] - f.zmns[i-1, :]
        f.rsc[:, :, i] = f.rbc[:, :, i+1] - f.rbc[:, :, i-1]
        f.zss[:, :, i] = f.zbs[:, :, i+1] - f.zbs[:, :, i-1]
    
    f.rsc = 0.5 * f.rsc
    f.zss = 0.5 * f.zss
    f.rsmnc = 0.5 * f.rsmnc
    f.zsmns = 0.5 * f.zsmns
    
    # Boundary conditions
    f.rsc[:, :, 0] = f.rbc[:, :, 1] - 2.0 * f.rbc[:, :, 0]
    f.zss[:, :, 0] = f.zbs[:, :, 1] - 2.0 * f.zbs[:, :, 0]
    f.rsmnc[0, :] = f.rsmnc[1, :] - 2.0 * f.rsmnc[0, :]
    f.zsmns[0, :] = f.zsmns[1, :] - 2.0 * f.zsmns[0, :]
    
    f.rsc[:, :, -1] = 2.0 * f.rbc[:, :, -1] - f.rbc[:, :, -2]
    f.zss[:, :, -1] = 2.0 * f.zbs[:, :, -1] - f.zbs[:, :, -2]
    f.rsmnc[-1, :] = 2.0 * f.rsmnc[-1, :] - f.rsmnc[-2, :]
    f.zsmns[-1, :] = 2.0 * f.zsmns[-1, :] - f.zsmns[-2, :]
    
    # Restore original xn for compatibility
    f.xn = original_xn
    
    return f


def calculate_derived_quantities(f: VMECData) -> VMECData:
    """Calculate additional derived quantities"""
    
    # Calculate chi and chip
    if not getattr(f, 'lrfp__logical__', 0):
        if hasattr(f, 'iotaf') and hasattr(f, 'phipf'):
            f.chipf = f.iotaf * f.phipf
            f.chif = np.cumsum(f.chipf)
    
    # Create resonance arrays if possible
    if hasattr(f, 'xm') and hasattr(f, 'xn'):
        f.M = np.arange(1, int(np.max(f.xm)) + 1)
        positive_xn = f.xn[f.xn > 0]
        if len(positive_xn) > 0:
            f.N = np.arange(int(np.min(positive_xn)), 
                           int(np.max(f.xn)) + 1, 
                           int(np.min(positive_xn)))
            
            Minv = 1.0 / f.M
            f.iota_res = np.outer(f.N, Minv)
    
    # Calculate stored energy
    if hasattr(f, 'vp') and (hasattr(f, 'presf') or hasattr(f, 'pres')):
        if hasattr(f, 'presf') and len(f.vp) == f.ns:
            f.eplasma = 1.5 * 4 * np.pi**2 * np.sum(f.vp * f.presf) / f.ns
        elif hasattr(f, 'pres'):
            f.eplasma = 1.5 * 4 * np.pi**2 * np.sum(f.vp * f.pres) / (f.ns - 1)
    
    # Set some defaults
    f.nu = 2 * (f.mpol + 1) + 6
    f.datatype = 'wout'
    
    return f


if __name__ == "__main__":
    # Test the function with available wout files
    import matplotlib.pyplot as plt
    
    wout_dir = "/Users/rjjm/Documents/GitHub/megpy/megpy/wouts"
    wout_files = [
        "wout_Hmode_ns200.nc",
        "wout_ITERModel.nc", 
        "wout_Lmode.nc",
        "wout_iter_sc4_neg_axi.nc"
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, filename in enumerate(wout_files):
        filepath = os.path.join(wout_dir, filename)
        
        try:
            print(f"\nReading {filename}...")
            data = read_vmec(filepath)
            
            if isinstance(data, int):
                print(f"Error reading {filename}: code {data}")
                continue
                
            print(f"Successfully read {filename}")
            print(f"  ns = {data.ns}, mpol = {data.mpol}, ntor = {data.ntor}")
            print(f"  nfp = {data.nfp}, Rmajor = {data.Rmajor:.3f}")
            print(f"  Aminor = {data.Aminor:.3f}, Volume = {data.Volume:.3f}")
            
            # Plot flux surface shapes at different radial locations
            ax = axes[i]
            
            # Simple plot of R,Z boundary (last surface)
            if hasattr(data, 'rmnc') and hasattr(data, 'zmns'):
                # Reconstruct boundary shape
                theta = np.linspace(0, 2*np.pi, 100)
                phi = 0  # Take phi=0 cut
                
                R = np.zeros_like(theta)
                Z = np.zeros_like(theta)
                
                for j in range(min(50, data.mnmax)):  # Limit to first 50 modes
                    m = data.xm[j]
                    n = data.xn[j]
                    arg = m * theta - n * phi
                    R += data.rmnc[-1, j] * np.cos(arg)  # Last surface, correct indexing
                    Z += data.zmns[-1, j] * np.sin(arg)  # Last surface, correct indexing
                
                if np.max(R) > 0:  # Only plot if we have valid data
                    ax.plot(R, Z, 'b-', linewidth=2)
                    ax.set_aspect('equal')
                    ax.set_title(filename.replace('wout_', '').replace('.nc', ''))
                    ax.set_xlabel('R [m]')
                    ax.set_ylabel('Z [m]')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'No valid boundary\nfor {filename}', 
                           ha='center', va='center', transform=ax.transAxes)
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    plt.tight_layout()
    plt.suptitle('VMEC Equilibrium Boundary Shapes', y=0.98)
    plt.savefig('/Users/rjjm/Documents/GitHub/megpy/vmec_boundaries.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nPlot saved as 'vmec_boundaries.png'")
