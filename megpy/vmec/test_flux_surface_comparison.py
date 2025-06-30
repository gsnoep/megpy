#!/usr/bin/env python3
"""
Test script to compare flux surface reconstruction between 
Python read_vmec and existing vmec2plot functionality
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf_file
import os
import sys

# Import our Python read_vmec function
sys.path.append('/Users/rjjm/Documents/GitHub/megpy')
from read_vmec import read_vmec


def reconstruct_flux_surface_vmec2plot_style(filename, iradius=-1, ntheta=100, nzeta=1):
    """
    Reconstruct flux surface using the same method as vmec2plot.py
    This is the reference implementation to compare against
    """
    
    # Read netCDF file using scipy (same as vmec2plot)
    try:
        f = netcdf_file(filename, 'r', mmap=False)
    except Exception as e:
        print(f"Error reading NetCDF file: {e}")
        return None, None, None
    
    # Get variables
    ns = f.variables['ns'][()]
    nfp = f.variables['nfp'][()]
    xn = f.variables['xn'][()]
    xm = f.variables['xm'][()]
    rmnc = f.variables['rmnc'][()]
    zmns = f.variables['zmns'][()]
    
    # Check for non-axisymmetric components
    lasym = f.variables['lasym__logical__'][()]
    if lasym == 1:
        rmns = f.variables['rmns'][()]
        zmnc = f.variables['zmnc'][()]
    else:
        rmns = 0 * rmnc
        zmnc = 0 * rmnc
    
    f.close()
    
    # Set up coordinate grids
    theta = np.linspace(0, 2*np.pi, num=ntheta)
    zeta = np.linspace(0, 2*np.pi/nfp, num=nzeta, endpoint=False)
    
    # Use last surface if iradius is -1
    if iradius == -1:
        iradius = ns - 1
    
    nmodes = len(xn)
    
    # Reconstruct surface
    R = np.zeros((ntheta, nzeta))
    Z = np.zeros((ntheta, nzeta))
    
    for itheta in range(ntheta):
        for izeta in range(nzeta):
            for imode in range(nmodes):
                angle = xm[imode]*theta[itheta] - xn[imode]*zeta[izeta]
                R[itheta, izeta] += (rmnc[iradius, imode]*np.cos(angle) + 
                                   rmns[iradius, imode]*np.sin(angle))
                Z[itheta, izeta] += (zmns[iradius, imode]*np.sin(angle) + 
                                   zmnc[iradius, imode]*np.cos(angle))
    
    return R, Z, {'ns': ns, 'nfp': nfp, 'nmodes': nmodes}


def reconstruct_flux_surface_read_vmec(data, iradius=-1, ntheta=100, nzeta=1):
    """
    Reconstruct flux surface using our Python read_vmec data
    """
    
    # Use last surface if iradius is -1
    if iradius == -1:
        iradius = data.ns - 1
    
    # Set up coordinate grids
    theta = np.linspace(0, 2*np.pi, num=ntheta)
    zeta = np.linspace(0, 2*np.pi/data.nfp, num=nzeta, endpoint=False)
    
    # Reconstruct surface
    R = np.zeros((ntheta, nzeta))
    Z = np.zeros((ntheta, nzeta))
    
    for itheta in range(ntheta):
        for izeta in range(nzeta):
            for imode in range(data.mnmax):
                angle = data.xm[imode]*theta[itheta] - data.xn[imode]*zeta[izeta]
                R[itheta, izeta] += data.rmnc[iradius, imode]*np.cos(angle)
                Z[itheta, izeta] += data.zmns[iradius, imode]*np.sin(angle)
                
                # Add non-axisymmetric terms if present
                if hasattr(data, 'rmns') and hasattr(data, 'zmnc'):
                    R[itheta, izeta] += data.rmns[iradius, imode]*np.sin(angle)
                    Z[itheta, izeta] += data.zmnc[iradius, imode]*np.cos(angle)
    
    return R, Z


def compare_flux_surfaces(filename, surfaces_to_test=None, ntheta=100):
    """
    Compare flux surface reconstruction between vmec2plot method and read_vmec method
    """
    
    print(f"\n{'='*60}")
    print(f"Testing flux surface reconstruction for: {os.path.basename(filename)}")
    print(f"{'='*60}")
    
    # Load data with our read_vmec function
    try:
        data = read_vmec(filename)
        if isinstance(data, int):
            print(f"❌ Error reading file with read_vmec: code {data}")
            return
        print("✅ Successfully loaded data with read_vmec")
    except Exception as e:
        print(f"❌ Exception with read_vmec: {e}")
        return
    
    # Default surfaces to test
    if surfaces_to_test is None:
        surfaces_to_test = [data.ns//4, data.ns//2, 3*data.ns//4, data.ns-1]
    
    # Initialize comparison results
    max_errors = []
    mean_errors = []
    
    for i, iradius in enumerate(surfaces_to_test):
        print(f"\n📍 Testing surface {iradius+1}/{data.ns} (s = {iradius/(data.ns-1):.3f})")
        
        # Reconstruct with vmec2plot method (reference)
        try:
            R_ref, Z_ref, info = reconstruct_flux_surface_vmec2plot_style(
                filename, iradius, ntheta, nzeta=1)
            if R_ref is None:
                print("❌ Failed to reconstruct with vmec2plot method")
                continue
        except Exception as e:
            print(f"❌ Exception with vmec2plot method: {e}")
            continue
        
        # Reconstruct with read_vmec method
        try:
            R_new, Z_new = reconstruct_flux_surface_read_vmec(
                data, iradius, ntheta, nzeta=1)
        except Exception as e:
            print(f"❌ Exception with read_vmec method: {e}")
            continue
        
        # Take phi=0 slice for comparison
        R_ref_slice = R_ref[:, 0]
        Z_ref_slice = Z_ref[:, 0]
        R_new_slice = R_new[:, 0]
        Z_new_slice = Z_new[:, 0]
        
        # Calculate errors
        R_error = np.abs(R_new_slice - R_ref_slice)
        Z_error = np.abs(Z_new_slice - Z_ref_slice)
        
        max_R_error = np.max(R_error)
        max_Z_error = np.max(Z_error)
        mean_R_error = np.mean(R_error)
        mean_Z_error = np.mean(Z_error)
        
        max_error = max(max_R_error, max_Z_error)
        mean_error = max(mean_R_error, mean_Z_error)
        
        max_errors.append(max_error)
        mean_errors.append(mean_error)
        
        print(f"   R errors: max = {max_R_error:.2e} m, mean = {mean_R_error:.2e} m")
        print(f"   Z errors: max = {max_Z_error:.2e} m, mean = {mean_Z_error:.2e} m")
        
        # Status based on error magnitude
        if max_error < 1e-12:
            status = "✅ PERFECT"
        elif max_error < 1e-10:
            status = "✅ EXCELLENT"
        elif max_error < 1e-8:
            status = "✅ VERY GOOD"
        elif max_error < 1e-6:
            status = "⚠️  ACCEPTABLE"
        else:
            status = "❌ POOR"
        
        print(f"   Overall: {status} (max error: {max_error:.2e} m)")
    
    # Summary statistics
    if max_errors:
        overall_max_error = np.max(max_errors)
        overall_mean_error = np.mean(max_errors)
        
        print(f"\n📊 SUMMARY for {os.path.basename(filename)}:")
        print(f"   Maximum error across all surfaces: {overall_max_error:.2e} m")
        print(f"   Average maximum error: {overall_mean_error:.2e} m")
        
        if overall_max_error < 1e-10:
            print("   🏆 EXCELLENT agreement between methods!")
        elif overall_max_error < 1e-8:
            print("   ✅ Very good agreement between methods")
        elif overall_max_error < 1e-6:
            print("   ⚠️  Acceptable agreement, minor differences")
        else:
            print("   ❌ Significant differences detected!")
        
        return overall_max_error, overall_mean_error
    
    return None, None


def plot_comparison(filename, iradius=-1, ntheta=200):
    """
    Create visual comparison plot of flux surface reconstruction
    """
    
    # Load data
    data = read_vmec(filename)
    if isinstance(data, int):
        print(f"Error loading data: {data}")
        return
    
    if iradius == -1:
        iradius = data.ns - 1
    
    # Reconstruct surfaces
    R_ref, Z_ref, _ = reconstruct_flux_surface_vmec2plot_style(filename, iradius, ntheta)
    R_new, Z_new = reconstruct_flux_surface_read_vmec(data, iradius, ntheta)
    
    if R_ref is None:
        print("Failed to reconstruct reference surface")
        return
    
    # Take phi=0 slice
    R_ref_slice = R_ref[:, 0]
    Z_ref_slice = Z_ref[:, 0]
    R_new_slice = R_new[:, 0]
    Z_new_slice = Z_new[:, 0]
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Overlay of both methods
    axes[0].plot(R_ref_slice, Z_ref_slice, 'r-', linewidth=2, label='vmec2plot (reference)')
    axes[0].plot(R_new_slice, Z_new_slice, 'b--', linewidth=2, label='read_vmec')
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('R [m]')
    axes[0].set_ylabel('Z [m]')
    axes[0].set_title(f'Flux Surface Comparison\n{os.path.basename(filename)}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Difference in R
    theta = np.linspace(0, 2*np.pi, ntheta)
    R_diff = R_new_slice - R_ref_slice
    axes[1].plot(theta, R_diff, 'g-', linewidth=2)
    axes[1].set_xlabel('θ [rad]')
    axes[1].set_ylabel('ΔR [m]')
    axes[1].set_title('R Coordinate Difference')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Difference in Z
    Z_diff = Z_new_slice - Z_ref_slice
    axes[2].plot(theta, Z_diff, 'm-', linewidth=2)
    axes[2].set_xlabel('θ [rad]')
    axes[2].set_ylabel('ΔZ [m]')
    axes[2].set_title('Z Coordinate Difference')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print error statistics
    max_R_error = np.max(np.abs(R_diff))
    max_Z_error = np.max(np.abs(Z_diff))
    print(f"\nError statistics for surface {iradius+1}/{data.ns}:")
    print(f"  Max R error: {max_R_error:.2e} m ({max_R_error*1e9:.3f} nm)")
    print(f"  Max Z error: {max_Z_error:.2e} m ({max_Z_error*1e9:.3f} nm)")
    
    return fig


def run_comprehensive_test():
    """
    Run comprehensive test on all available wout files
    """
    
    wout_dir = "/Users/rjjm/Documents/GitHub/megpy/megpy/wouts"
    wout_files = [
        "wout_Hmode_ns200.nc",
        "wout_ITERModel.nc", 
        "wout_Lmode.nc",
        "wout_iter_sc4_neg_axi.nc"
    ]
    
    print("🧪 COMPREHENSIVE FLUX SURFACE RECONSTRUCTION TEST")
    print("=" * 80)
    print("Comparing Python read_vmec against vmec2plot reference implementation")
    print("=" * 80)
    
    all_max_errors = []
    all_mean_errors = []
    
    for filename in wout_files:
        filepath = os.path.join(wout_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"⚠️  File not found: {filename}")
            continue
        
        try:
            max_error, mean_error = compare_flux_surfaces(filepath)
            if max_error is not None:
                all_max_errors.append(max_error)
                all_mean_errors.append(mean_error)
        except Exception as e:
            print(f"❌ Exception testing {filename}: {e}")
            continue
    
    # Overall summary
    if all_max_errors:
        overall_max = np.max(all_max_errors)
        overall_mean = np.mean(all_max_errors)
        
        print(f"\n🏁 FINAL RESULTS:")
        print(f"=" * 40)
        print(f"Files tested: {len(all_max_errors)}")
        print(f"Worst maximum error: {overall_max:.2e} m")
        print(f"Average maximum error: {overall_mean:.2e} m")
        
        if overall_max < 1e-10:
            print("🏆 OVERALL VERDICT: EXCELLENT - Python read_vmec matches vmec2plot perfectly!")
        elif overall_max < 1e-8:
            print("✅ OVERALL VERDICT: VERY GOOD - Python read_vmec is highly accurate")
        elif overall_max < 1e-6:
            print("⚠️  OVERALL VERDICT: ACCEPTABLE - Minor differences present")
        else:
            print("❌ OVERALL VERDICT: ISSUES DETECTED - Significant differences found")
        
        print(f"=" * 40)
    else:
        print("❌ No successful tests completed")


if __name__ == "__main__":
    # Run the comprehensive test
    run_comprehensive_test()
    
    # Create a detailed plot for one case
    print("\n📊 Creating detailed comparison plot for one case...")
    wout_file = "../wouts/wout_iter_sc4_neg_axi.nc"
    if os.path.exists(wout_file):
        fig = plot_comparison(wout_file)
        if fig:
            plt.savefig('flux_surface_comparison.png', 
                       dpi=150, bbox_inches='tight')
            plt.show()
            print("Plot saved as 'flux_surface_comparison.png'")
    else:
        print("Reference file not found for plotting")
