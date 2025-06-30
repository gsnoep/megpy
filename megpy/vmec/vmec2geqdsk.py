#!/usr/bin/env python3
"""
Python translation of vmec2geqdsk.m

This function writes a GEQDSK file from a VMEC wout file. The user must
pass a VMEC data structure. Options include passing of a vessel data
structure and file name to output the data. Note that axisymmetry is
assumed non-axisymmetric equilibria will have their cross section output
at the phi=0 plane.

Example usage:
    vmec_data = read_vmec('wout.test')
    ves_data = read_vessel('vessel.dat')  # Optional
    vmec2geqdsk(vmec_data, ves_data, 'gtemp.0000')

Created by: S. Lazerson
Machine translation by: Claude Sonnet 4.0
Version: 1.0
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import PchipInterpolator
from read_vmec import read_vmec, VMECData
from typing import Optional, Union
import datetime


def cfunct(theta, zeta, rmnc, xm, xn):
    """
    Cosine Fourier Transform - MATLAB version translation
    """
    # Convert masked arrays to regular arrays
    rmnc = np.asarray(rmnc)
    xm = np.asarray(xm)
    xn = np.asarray(xn)
    
    ns = rmnc.shape[0]
    theta = np.atleast_1d(theta)
    zeta = np.atleast_1d(zeta) 
    lt = len(theta)
    lz = len(zeta)
    
    # Create mode x angle arrays
    mt = np.outer(xm, theta)  # (mn, lt)
    nz = np.outer(xn, zeta)   # (mn, lz)
    
    # Create Trig Arrays
    cosmt = np.cos(mt)  # (mn, lt)
    sinmt = np.sin(mt)  # (mn, lt)
    cosnz = np.cos(nz)  # (mn, lz)
    sinnz = np.sin(nz)  # (mn, lz)
    
    # Calculate the transform
    f = np.zeros((ns, lt, lz))
    for k in range(ns):
        # MATLAB: rmn=repmat(rmnc(:,k),[1 lt]);
        # This creates a (mn, lt) array where each row is replicated lt times
        rmn = np.repeat(rmnc[k, :][:, np.newaxis], lt, axis=1)  # (mn, lt)
        
        # MATLAB: f(k,:,:)=(rmn.*cosmt)'*cosnz-(rmn.*sinmt)'*sinnz;
        # (rmn.*cosmt)' is (lt, mn), cosnz is (mn, lz), result is (lt, lz)
        f[k, :, :] = (rmn * cosmt).T @ cosnz - (rmn * sinmt).T @ sinnz
        
    return f


def sfunct(theta, zeta, zmns, xm, xn):
    """
    Sine Fourier Transform - MATLAB version translation
    """
    # Convert masked arrays to regular arrays
    zmns = np.asarray(zmns)
    xm = np.asarray(xm)
    xn = np.asarray(xn)
    
    ns = zmns.shape[0]
    theta = np.atleast_1d(theta)
    zeta = np.atleast_1d(zeta)
    lt = len(theta)
    lz = len(zeta)
    
    # Create mode x angle arrays
    mt = np.outer(xm, theta)  # (mn, lt)
    nz = np.outer(xn, zeta)   # (mn, lz)
    
    # Create Trig Arrays
    cosmt = np.cos(mt)  # (mn, lt)
    sinmt = np.sin(mt)  # (mn, lt)
    cosnz = np.cos(nz)  # (mn, lz)
    sinnz = np.sin(nz)  # (mn, lz)
    
    # Calculate the transform
    f = np.zeros((ns, lt, lz))
    for k in range(ns):
        # MATLAB: zmn=repmat(zmns(:,k),[1 lt]);
        zmn = np.repeat(zmns[k, :][:, np.newaxis], lt, axis=1)  # (mn, lt)
        
        # MATLAB: f(k,:,:)=(zmn.*sinmt)'*cosnz+(zmn.*cosmt)'*sinnz;
        f[k, :, :] = (zmn * sinmt).T @ cosnz + (zmn * cosmt).T @ sinnz
        
    return f


def vmec2geqdsk(vmec_data: VMECData, ves_data: Optional[dict] = None, filename: str = 'junk.0000'):
    """
    Writes a WOUT data structure to GEQDSK File
    
    This function writes a GEQDSK file from a VMEC wout file. The user must
    pass a VMEC data structure. Options include passing of a vessel data
    structure and file name to output the data. Note that axisymmetry is
    assumed non-axisymmetric equilibria will have their cross section output
    at the phi=0 plane.
    
    Args:
        vmec_data: VMEC data structure from read_vmec
        ves_data: Optional vessel data structure
        filename: Output filename for GEQDSK file
    """
    
    if vmec_data is None:
        return
    
    # Set default filename if not provided
    if not filename:
        filename = 'junk.0000'
    
    nx = vmec_data.ns
    nz = vmec_data.ns
    ntheta = 360
    theta = np.linspace(0, 2*np.pi, ntheta)
    zeta = 0.0
    
    # Transform the data
    r = cfunct(theta, zeta, vmec_data.rmnc, vmec_data.xm, vmec_data.xn)
    z = sfunct(theta, zeta, vmec_data.zmns, vmec_data.xm, vmec_data.xn)
    
    if hasattr(vmec_data, 'iasym') and vmec_data.iasym:
        r = r + sfunct(theta, zeta, vmec_data.rmns, vmec_data.xm, vmec_data.xn)
        z = z + cfunct(theta, zeta, vmec_data.zmnc, vmec_data.xm, vmec_data.xn)
    
    phin = vmec_data.phi / vmec_data.phi[-1]
    
    # Create chirz array
    chirz = np.zeros((vmec_data.ns, ntheta, 1))
    for i in range(vmec_data.ns):
        chirz[i, :, 0] = vmec_data.chif[i]
    
    # Reshape for interpolation
    r2 = r.reshape(-1, 1)
    z2 = z.reshape(-1, 1)
    chirz_flat = chirz.reshape(-1, 1)
    
    # Grid for interpolation
    r1 = 0.9 * vmec_data.rmin_surf
    r2_max = 1.1 * vmec_data.rmax_surf
    z1 = 1.1 * np.min(z[-1, :, 0])
    z2_max = 1.1 * np.max(z[-1, :, 0])
    
    xgrid = np.linspace(r1, r2_max, nx)
    zgrid = np.linspace(z1, z2_max, nz)
    xgrid_2d, zgrid_2d = np.meshgrid(xgrid, zgrid)
    
    # Interpolate using griddata (equivalent to TriScatteredInterp)
    points = np.column_stack((r2.flatten(), z2.flatten()))
    values = chirz_flat.flatten()
    psixz = griddata(points, values, (xgrid_2d, zgrid_2d), method='linear', fill_value=0.0)
    psixz = np.nan_to_num(psixz, nan=0.0)
    
    # Write the file
    with open(filename, 'w') as fid:
        # Header line
        fid.write(f"{'  VMEC':<10s}")
        fid.write(f"{datetime.datetime.now().strftime('%m/%d/%Y'):>10s}")
        fid.write(f"{'  #000001':>10s}")
        fid.write(f"{'  0000ms':>10s}")
        fid.write(f"{1:4d} {nx:4d} {nz:4d}\n")
        
        # Line 2 - Grid bounds and parameters
        vals = [vmec_data.rmax_surf * 1.1, vmec_data.zmax_surf * 1.1, vmec_data.rmin_surf * 0.9, vmec_data.rmax_surf, 0.0]
        for val in vals:
            if val >= 0:
                fid.write(f" {val:14.9E}")
            else:
                fid.write(f"{val:15.9E}")
        fid.write('\n')
        
        # Line 3 - Magnetic axis and flux values
        vals = [r[0, 0, 0], z[0, 0, 0], vmec_data.chif[0], vmec_data.chif[-1], vmec_data.b0]
        for val in vals:
            if val >= 0:
                fid.write(f" {val:14.9E}")
            else:
                fid.write(f"{val:15.9E}")
        fid.write('\n')
        
        # Line 4 - Current and flux values
        vals = [vmec_data.Itor, vmec_data.phi[-1], 0.0, r[0, 0, 0], 0.0]
        for val in vals:
            if val >= 0:
                fid.write(f" {val:14.9E}")
            else:
                fid.write(f"{val:15.9E}")
        fid.write('\n')
        
        # Line 5 - Additional parameters
        vals = [z[0, 0, 0], 0.0, vmec_data.phi[-1], r[-1, 0, 0], z[-1, 0, 0]]
        for val in vals:
            if val >= 0:
                fid.write(f" {val:14.9E}")
            else:
                fid.write(f"{val:15.9E}")
        fid.write('\n')
        
        # Flux function on uniform flux grid (sf)
        sf_vals = PchipInterpolator(phin, vmec_data.phi)(np.linspace(0, 1, nx))
        write_array_to_file(fid, sf_vals)
        fid.write('\n')
        
        # Pressure on uniform flux grid (sp)
        sp_vals = PchipInterpolator(phin, vmec_data.presf)(np.linspace(0, 1, nx))
        write_array_to_file(fid, sp_vals)
        fid.write('\n')
        
        # FF' on uniform flux grid (sffp) - same as pressure for VMEC
        write_array_to_file(fid, sp_vals)
        fid.write('\n')
        
        # PP' on uniform flux grid (spp)
        spp_vals = PchipInterpolator(phin, np.gradient(vmec_data.presf))(np.linspace(0, 1, nx))
        write_array_to_file(fid, spp_vals)
        fid.write('\n')
        
        # Flux function on (R,Z) grid (psirz)
        write_array_to_file(fid, psixz.flatten())
        fid.write('\n')
        
        # q profile (1/iota) on uniform flux grid
        q_vals = PchipInterpolator(phin, 1.0/vmec_data.iotaf)(np.linspace(0, 1, nx))
        write_array_to_file(fid, q_vals)
        fid.write('\n')
        
        # Number of boundary and limiter points
        vals = [float(ntheta), float(ntheta)]
        for val in vals:
            if val >= 0:
                fid.write(f" {val:14.9E}")
            else:
                fid.write(f"{val:15.9E}")
        fid.write('\n')
        
        # Boundary points - R coordinates
        write_array_to_file(fid, r[-1, :, 0])
        fid.write('\n')
        
        # Boundary points - Z coordinates  
        write_array_to_file(fid, z[-1, :, 0])
        fid.write('\n')
        
        # Limiter points
        if ves_data is None:
            # Use boundary as limiter if no vessel data
            write_array_to_file(fid, r[-1, :, 0])
            fid.write('\n')
            write_array_to_file(fid, z[-1, :, 0])
            fid.write('\n')
        else:
            # Use vessel data if provided
            dex = ves_data['coords'][3, :] == 1
            r_v = ves_data['coords'][0, dex]
            z_v = ves_data['coords'][1, dex]
            write_array_to_file(fid, r_v)
            fid.write('\n')
            write_array_to_file(fid, z_v)
            fid.write('\n')


def write_array_to_file(fid, arr):
    """
    Write array to file in FORTRAN format (5 values per line, scientific notation)
    Matches MATLAB formatting exactly - space before positive numbers, no extra space before negative
    """
    arr = np.atleast_1d(arr)
    for i, val in enumerate(arr):
        if i % 5 == 0 and i > 0:
            fid.write('\n')
        # MATLAB uses % 15.9E which adds space before positive numbers but not negative
        if val >= 0:
            fid.write(f" {val:14.9E}")
        else:
            fid.write(f"{val:15.9E}")


if __name__ == "__main__":
    # Test the function with available wout files
    
    # Test with both wout files
    wout_files = [
        "../wouts/wout_iter_sc4_neg_axi.nc",
        "../wouts/wout_ITERModel.nc"
    ]
    
    output_files = [
        "EQDSK_HMODE_PYTHON",
        "EQDSK_ITER_PYTHON",
        "EQDSK_ITERModel_PYTHON",
        "EQDSK_LMODE_PYTHON"
    ]
    
    for wout_file, output_file in zip(wout_files, output_files):
        try:
            print(f"Processing {wout_file}...")
            vmec_data = read_vmec(wout_file)
            if isinstance(vmec_data, int):
                print(f"Error reading {wout_file}")
                continue
                
            print(f"Successfully loaded {wout_file}")
            print(f"rmnc shape: {vmec_data.rmnc.shape}")
            print(f"zmns shape: {vmec_data.zmns.shape}")
            
            # Test cfunct and sfunct first
            theta = np.linspace(0, 2*np.pi, 360)
            zeta = 0.0
            
            print("Testing cfunct...")
            r = cfunct(theta, zeta, vmec_data.rmnc, vmec_data.xm, vmec_data.xn)
            print(f"cfunct result shape: {r.shape}")
            
            print("Testing sfunct...")
            z = sfunct(theta, zeta, vmec_data.zmns, vmec_data.xm, vmec_data.xn)
            print(f"sfunct result shape: {z.shape}")
            
            print("Calling vmec2geqdsk...")
            vmec2geqdsk(vmec_data, filename=output_file)
            print(f"Created {output_file}")
            
        except Exception as e:
            import traceback
            print(f"Error processing {wout_file}: {e}")
            traceback.print_exc()
