#!/usr/bin/env python3
"""
Comprehensive EQDSK plotting script combining all visualization methods:
1. Simple built-in plots (using megpy's native methods)
2. Custom 2D matplotlib plots  
3. Advanced 3D matplotlib visualization
4. Interactive 3D PyVista visualization

This script provides a unified interface to visualize magnetic equilibrium data
with progressive complexity and interactivity options.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from megpy.equilibrium import Equilibrium

# Check for PyVista availability
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
    print("PyVista is available for advanced 3D visualization")
except ImportError:
    PYVISTA_AVAILABLE = False
    print("PyVista not available. Install with: pip install pyvista")

def plot_simple_builtin(eq):
    """
    Method 1: Simple plots using custom implementation to handle decreasing flux levels
    
    Args:
        eq: Equilibrium object with loaded data
    
    Returns:
        tuple: (flux_fig, derived_fig)
    """
    print("\n=== Method 1: Simple Built-in Plots ===")
    
    # Create custom flux surface plot to handle decreasing levels
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    
    # Create meshgrid for contour plotting
    R_mesh, Z_mesh = np.meshgrid(eq.derived['R'], eq.derived['Z'])
    
    # Plot psi contours - ensure levels are increasing
    psi_min = min(eq.raw['simag'], eq.raw['sibry'])
    psi_max = max(eq.raw['simag'], eq.raw['sibry'])
    levels = np.linspace(psi_min, psi_max, 15)
    
    contours = ax1.contour(R_mesh, Z_mesh, eq.derived['psirz'], levels=levels, colors='blue', alpha=0.7)
    contourf = ax1.contourf(R_mesh, Z_mesh, eq.derived['psirz'], levels=levels, alpha=0.3, cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax1)
    cbar.set_label('ψ [Wb/rad]')
    
    # Plot boundary if available
    if 'rbbbs' in eq.raw and 'zbbbs' in eq.raw:
        ax1.plot(eq.raw['rbbbs'], eq.raw['zbbbs'], 'r-', linewidth=2, label='LCFS')
        ax1.legend()
    
    # Plot magnetic axis
    ax1.plot(eq.raw['rmaxis'], eq.raw['zmaxis'], 'ro', markersize=8, label='Magnetic Axis')
    
    ax1.set_xlabel('R [m]')
    ax1.set_ylabel('Z [m]')
    ax1.set_title('Flux Surfaces (Custom Implementation)')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    plt.suptitle("EQDSK - Flux Surfaces (Custom Method)")
    
    # Plot derived quantities using built-in method (this should work)
    try:
        fig2 = eq.plot_derived(figsize=(12, 8))
        plt.suptitle("EQDSK - Plasma Profiles (Built-in Method)")
    except Exception as e:
        print(f"Warning: Could not create derived quantities plot: {e}")
        # Create a simple substitute plot
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        ax2.text(0.5, 0.5, f"Profiles plot unavailable\nError: {e}", 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Plasma Profiles (Error)")
    
    return fig1, fig2

def plot_custom_2d_matplotlib(eq):
    """
    Method 2: Custom 2D matplotlib plots with detailed analysis
    
    Args:
        eq: Equilibrium object with loaded data
    
    Returns:
        matplotlib.figure.Figure: 2D analysis figure
    """
    print("\n=== Method 2: Custom 2D Matplotlib Analysis ===")
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('EQDSK Equilibrium Analysis (Custom 2D)', fontsize=16)
    
    # Plot 1: Flux surfaces (psi contours)
    ax1 = axes[0, 0]
    R_mesh, Z_mesh = np.meshgrid(eq.derived['R'], eq.derived['Z'])
    
    # Plot psi contours - ensure levels are increasing
    psi_min = min(eq.raw['simag'], eq.raw['sibry'])
    psi_max = max(eq.raw['simag'], eq.raw['sibry'])
    levels = np.linspace(psi_min, psi_max, 20)
    contours = ax1.contour(R_mesh, Z_mesh, eq.derived['psirz'], levels=levels, colors='blue', alpha=0.7)
    ax1.contourf(R_mesh, Z_mesh, eq.derived['psirz'], levels=levels, alpha=0.3, cmap='viridis')
    
    # Plot boundary if available
    if 'rbbbs' in eq.raw and 'zbbbs' in eq.raw:
        ax1.plot(eq.raw['rbbbs'], eq.raw['zbbbs'], 'r-', linewidth=2, label='LCFS')
    
    # Plot limiter if available
    if 'rlim' in eq.raw and 'zlim' in eq.raw:
        ax1.plot(eq.raw['rlim'], eq.raw['zlim'], 'k-', linewidth=2, label='Limiter')
    
    # Mark magnetic axis
    ax1.plot(eq.raw['rmaxis'], eq.raw['zmaxis'], 'ro', markersize=8, label='Magnetic Axis')
    
    ax1.set_xlabel('R [m]')
    ax1.set_ylabel('Z [m]')
    ax1.set_title('Flux Surfaces (ψ contours)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Safety factor q profile
    ax2 = axes[0, 1]
    ax2.plot(eq.derived['rho_pol'], eq.raw['qpsi'], 'b-', linewidth=2)
    ax2.set_xlabel('ρ_pol')
    ax2.set_ylabel('q (Safety Factor)')
    ax2.set_title('Safety Factor Profile')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Pressure profile
    ax3 = axes[1, 0]
    ax3.plot(eq.derived['rho_pol'], eq.raw['pres'], 'g-', linewidth=2)
    ax3.set_xlabel('ρ_pol')
    ax3.set_ylabel('Pressure [Pa]')
    ax3.set_title('Pressure Profile')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Toroidal field function F = RB_tor
    ax4 = axes[1, 1]
    ax4.plot(eq.derived['rho_pol'], eq.raw['fpol'], 'm-', linewidth=2)
    ax4.set_xlabel('ρ_pol')
    ax4.set_ylabel('F = RB_tor [T⋅m]')
    ax4.set_title('Toroidal Field Function')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_3d_matplotlib(eq, n_surfaces=8, n_points_toroidal=60):
    """
    Method 3: Advanced 3D matplotlib visualization with progressive toroidal coverage
    
    Args:
        eq: Equilibrium object with loaded data
        n_surfaces: Number of flux surfaces to plot
        n_points_toroidal: Number of points in toroidal direction
    
    Returns:
        matplotlib.figure.Figure: 3D matplotlib figure
    """
    print("\n=== Method 3: Advanced 3D Matplotlib Visualization ===")
    
    # Calculate delta automatically
    if n_surfaces > 1:
        delta = np.pi/2 / (n_surfaces - 1)
    else:
        delta = 0
    
    print(f"Using matplotlib for 3D visualization with δ = {delta:.3f} rad ({delta*180/np.pi:.1f}°)")
    print(f"Creating {n_surfaces} flux surfaces with varying toroidal coverage...")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract flux surface data
    R_mesh, Z_mesh = np.meshgrid(eq.derived['R'], eq.derived['Z'])
    
    # Define flux surface levels (from outer boundary to magnetic axis)
    psi_min = min(eq.raw['simag'], eq.raw['sibry'])  
    psi_max = max(eq.raw['simag'], eq.raw['sibry'])
    psi_levels = np.linspace(psi_max, psi_min, n_surfaces)  # From boundary to axis
    
    # Color map for different surfaces
    colors = plt.cm.plasma(np.linspace(0, 1, len(psi_levels)))
    
    surfaces_plotted = 0
    
    for i, psi_level in enumerate(psi_levels):
        # Use matplotlib contour to find flux surface
        fig_temp = plt.figure()
        ax_temp = fig_temp.add_subplot(111)
        cs = ax_temp.contour(R_mesh, Z_mesh, eq.derived['psirz'], levels=[psi_level])
        
        # Extract contour paths
        try:
            if hasattr(cs, 'allsegs') and len(cs.allsegs) > 0 and len(cs.allsegs[0]) > 0:
                segments = cs.allsegs[0]
                if segments:
                    # Find the longest segment (main flux surface)
                    longest_segment = max(segments, key=len)
                    R_contour = longest_segment[:, 0]
                    Z_contour = longest_segment[:, 1]
                    
                    # Close the contour if not already closed
                    if len(R_contour) > 2 and (R_contour[0] != R_contour[-1] or Z_contour[0] != Z_contour[-1]):
                        R_contour = np.append(R_contour, R_contour[0])
                        Z_contour = np.append(Z_contour, Z_contour[0])
                    
                    # Define toroidal angle range for this surface
                    phi_min = -np.pi/2 - i * delta
                    phi_max = np.pi/2 + i * delta
                    
                    # Create toroidal angles
                    phi = np.linspace(phi_min, phi_max, n_points_toroidal)
                    
                    # Create 3D surface coordinates
                    PHI, R_CONTOUR = np.meshgrid(phi, R_contour)
                    Z_CONTOUR = np.tile(Z_contour, (len(phi), 1)).T
                    
                    # Convert to Cartesian coordinates
                    X_3d = R_CONTOUR * np.cos(PHI)
                    Y_3d = R_CONTOUR * np.sin(PHI)
                    
                    # Calculate normalized flux coordinate for labeling
                    rho_pol = np.sqrt(abs((psi_level - eq.raw["simag"])/(eq.raw["sibry"] - eq.raw["simag"])))
                    
                    # Plot surface with reduced opacity to show vessel
                    surface = ax.plot_surface(X_3d, Y_3d, Z_CONTOUR, 
                                            alpha=0.6, 
                                            color=colors[i],
                                            linewidth=0,
                                            antialiased=True)
                    
                    # Add surface edges for better visualization
                    ax.plot_wireframe(X_3d, Y_3d, Z_CONTOUR, 
                                    alpha=0.2, 
                                    color=colors[i],
                                    linewidth=0.3,
                                    rstride=6, cstride=6)
                    
                    surfaces_plotted += 1
                    
                    print(f"  Surface {surfaces_plotted}/{n_surfaces}: ρ_pol = {rho_pol:.2f}, φ = [{phi_min:.2f}, {phi_max:.2f}] rad ({phi_min*180/np.pi:.0f}° to {phi_max*180/np.pi:.0f}°)")
                    
        except Exception as e:
            print(f"  Error processing surface {i+1}: {e}")
        
        plt.close(fig_temp)
    
    # Add actual vessel boundary (limiter) if available
    if 'rlim' in eq.raw and 'zlim' in eq.raw and eq.raw['rlim'] is not None:
        # Quarter revolution: 0 to π/2 (90 degrees)
        phi_vessel = np.linspace(0, np.pi/2, n_points_toroidal//2)
        R_vessel = eq.raw['rlim']
        Z_vessel = eq.raw['zlim']
        
        # Close vessel boundary if not already closed
        if len(R_vessel) > 1 and (R_vessel[0] != R_vessel[-1] or Z_vessel[0] != Z_vessel[-1]):
            R_vessel = np.append(R_vessel, R_vessel[0])
            Z_vessel = np.append(Z_vessel, Z_vessel[0])
        
        # Create vessel surface - plot as wireframe only for visibility
        PHI_V, R_V = np.meshgrid(phi_vessel, R_vessel)
        Z_V = np.tile(Z_vessel, (len(phi_vessel), 1)).T
        
        X_vessel = R_V * np.cos(PHI_V)
        Y_vessel = R_V * np.sin(PHI_V)
        
        # Plot vessel as prominent wireframe
        ax.plot_wireframe(X_vessel, Y_vessel, Z_V, 
                         alpha=0.9, color='darkslategray',
                         linewidth=3, rstride=3, cstride=3)
        
        # Add vessel poloidal cross-sections at quarter revolution angles
        vessel_angles = [0, np.pi/6, np.pi/3, np.pi/2]
        for i, phi_key in enumerate(vessel_angles):
            X_cross = R_vessel * np.cos(phi_key)
            Y_cross = R_vessel * np.sin(phi_key)
            ax.plot(X_cross, Y_cross, Z_vessel, 'k-', 
                   linewidth=4, alpha=1.0, 
                   label='Vessel Wall' if i == 0 else "")
        
        print(f"  Added vessel wall with {len(R_vessel)} points (quarter revolution: 0° to 90°)")
    
    # Mark magnetic axis (full toroidal coverage with enhanced visibility)
    phi_axis = np.linspace(-np.pi, np.pi, 100)
    X_axis = eq.raw['rmaxis'] * np.cos(phi_axis)
    Y_axis = eq.raw['rmaxis'] * np.sin(phi_axis)
    Z_axis = np.full_like(phi_axis, eq.raw['zmaxis'])
    ax.plot(X_axis, Y_axis, Z_axis, 'r-', linewidth=4, 
           label='Magnetic Axis', alpha=0.9)
    
    # Add axis point markers at key angles for better visibility
    key_angles = [0, np.pi/2, np.pi, -np.pi/2]
    for phi_mark in key_angles:
        X_mark = eq.raw['rmaxis'] * np.cos(phi_mark)
        Y_mark = eq.raw['rmaxis'] * np.sin(phi_mark)
        Z_mark = eq.raw['zmaxis']
        ax.scatter([X_mark], [Y_mark], [Z_mark], 
                  color='red', s=50, alpha=0.8)
    
    # Set labels and title
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('3D Toroidal Flux Surfaces with Vessel Wall (Matplotlib)\nOuter: LCFS (±π/2), Inner: extended coverage, Innermost: magnetic axis (full torus)')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Set equal aspect ratio
    def set_axes_equal_3d(ax):
        """Set 3D plot axes to equal scale"""
        extents = np.array([getattr(ax, f'get_{dim}lim')() for dim in 'xyz'])
        centers = np.mean(extents, axis=1)
        max_extent = np.max(np.abs(extents[:, 1] - extents[:, 0]))
        for center, dim in zip(centers, 'xyz'):
            getattr(ax, f'set_{dim}lim')(center - max_extent/2, center + max_extent/2)
    
    set_axes_equal_3d(ax)
    
    # Improve viewing angle
    ax.view_init(elev=20, azim=45)
    
    print(f"\nSuccessfully created 3D matplotlib plot with {surfaces_plotted} flux surfaces")
    
    plt.tight_layout()
    return fig

def create_flux_surface_mesh(R_contour, Z_contour, phi_min, phi_max, n_points_toroidal=60):
    """
    Create a PyVista mesh for a single flux surface (for Method 4)
    
    Args:
        R_contour: R coordinates of the flux surface contour
        Z_contour: Z coordinates of the flux surface contour
        phi_min: Minimum toroidal angle
        phi_max: Maximum toroidal angle
        n_points_toroidal: Number of points in toroidal direction
    
    Returns:
        pv.StructuredGrid: PyVista mesh for the flux surface
    """
    # Create toroidal angles
    phi = np.linspace(phi_min, phi_max, n_points_toroidal)
    
    # Create 3D surface coordinates
    PHI, R_CONTOUR = np.meshgrid(phi, R_contour, indexing='ij')
    Z_CONTOUR = np.tile(Z_contour, (len(phi), 1))
    
    # Convert to Cartesian coordinates
    X_3d = R_CONTOUR * np.cos(PHI)
    Y_3d = R_CONTOUR * np.sin(PHI)
    
    # Create PyVista structured grid
    grid = pv.StructuredGrid(X_3d, Y_3d, Z_CONTOUR)
    
    return grid

def plot_3d_pyvista(eq, n_surfaces=8, n_points_toroidal=60):
    """
    Method 4: Interactive 3D PyVista visualization with advanced rendering
    
    Args:
        eq: Equilibrium object with loaded data
        n_surfaces: Number of flux surfaces to plot
        n_points_toroidal: Number of points in toroidal direction
    
    Returns:
        pv.Plotter: PyVista plotter object (or None if PyVista unavailable)
    """
    print("\n=== Method 4: Interactive 3D PyVista Visualization ===")
    
    if not PYVISTA_AVAILABLE:
        print("PyVista not available. Cannot create interactive 3D visualization.")
        print("To install PyVista, run: pip install pyvista")
        return None
    
    # Calculate delta automatically
    if n_surfaces > 1:
        delta = np.pi/2 / (n_surfaces - 1)
    else:
        delta = 0
    
    print(f"Using PyVista for 3D visualization with δ = {delta:.3f} rad ({delta*180/np.pi:.1f}°)")
    print(f"Creating {n_surfaces} flux surfaces with varying toroidal coverage...")
    
    # Create PyVista plotter
    plotter = pv.Plotter(window_size=(1200, 800))
    plotter.set_background('white')
    
    # Extract flux surface data
    R_mesh, Z_mesh = np.meshgrid(eq.derived['R'], eq.derived['Z'])
    
    # Define flux surface levels (from outer boundary to magnetic axis)
    psi_min = min(eq.raw['simag'], eq.raw['sibry'])  
    psi_max = max(eq.raw['simag'], eq.raw['sibry'])
    psi_levels = np.linspace(psi_max, psi_min, n_surfaces)  # From boundary to axis
    
    # Color map for different surfaces
    colors = plt.cm.plasma(np.linspace(0, 1, len(psi_levels)))
    
    surfaces_plotted = 0
    
    for i, psi_level in enumerate(psi_levels):
        # Use matplotlib contour to find flux surface
        fig_temp = plt.figure()
        ax_temp = fig_temp.add_subplot(111)
        cs = ax_temp.contour(R_mesh, Z_mesh, eq.derived['psirz'], levels=[psi_level])
        
        # Extract contour paths
        try:
            if hasattr(cs, 'allsegs') and len(cs.allsegs) > 0 and len(cs.allsegs[0]) > 0:
                segments = cs.allsegs[0]
                if segments:
                    # Find the longest segment (main flux surface)
                    longest_segment = max(segments, key=len)
                    R_contour = longest_segment[:, 0]
                    Z_contour = longest_segment[:, 1]
                    
                    # Close the contour if not already closed
                    if len(R_contour) > 2 and (R_contour[0] != R_contour[-1] or Z_contour[0] != Z_contour[-1]):
                        R_contour = np.append(R_contour, R_contour[0])
                        Z_contour = np.append(Z_contour, Z_contour[0])
                    
                    # Define toroidal angle range for this surface
                    phi_min = -np.pi/2 - i * delta
                    phi_max = np.pi/2 + i * delta
                    
                    # Create mesh for this flux surface
                    mesh = create_flux_surface_mesh(R_contour, Z_contour, phi_min, phi_max, n_points_toroidal)
                    
                    # Calculate normalized flux coordinate for coloring
                    rho_pol = np.sqrt(abs((psi_level - eq.raw["simag"])/(eq.raw["sibry"] - eq.raw["simag"])))
                    
                    # Convert matplotlib color to RGB
                    color_rgb = colors[i][:3]  # Remove alpha channel
                    
                    # Add surface to plotter
                    plotter.add_mesh(mesh, 
                                   color=color_rgb,
                                   opacity=1.0,
                                   smooth_shading=True,
                                   name=f'flux_surface_{i}')
                    
                    # Add wireframe for better definition
                    plotter.add_mesh(mesh, 
                                   color=color_rgb,
                                   style='wireframe',
                                   line_width=1,
                                   opacity=1.0,
                                   name=f'flux_wireframe_{i}')
                    
                    surfaces_plotted += 1
                    
                    print(f"  Surface {surfaces_plotted}/{n_surfaces}: ρ_pol = {rho_pol:.2f}, φ = [{phi_min:.2f}, {phi_max:.2f}] rad ({phi_min*180/np.pi:.0f}° to {phi_max*180/np.pi:.0f}°)")
                    
        except Exception as e:
            print(f"  Error processing surface {i+1}: {e}")
        
        plt.close(fig_temp)
    
    # Add vessel boundary (limiter) if available
    if 'rlim' in eq.raw and 'zlim' in eq.raw and eq.raw['rlim'] is not None:
        # Quarter revolution: 0 to π/2 (90 degrees)
        phi_vessel = np.linspace(0, np.pi/2, n_points_toroidal//2)
        R_vessel = eq.raw['rlim']
        Z_vessel = eq.raw['zlim']
        
        # Close vessel boundary if not already closed
        if len(R_vessel) > 1 and (R_vessel[0] != R_vessel[-1] or Z_vessel[0] != Z_vessel[-1]):
            R_vessel = np.append(R_vessel, R_vessel[0])
            Z_vessel = np.append(Z_vessel, Z_vessel[0])
        
        # Create vessel mesh
        vessel_mesh = create_flux_surface_mesh(R_vessel, Z_vessel, 0, np.pi/2, n_points_toroidal//2)
        
        # Add vessel as wireframe
        plotter.add_mesh(vessel_mesh,
                        color='gray',
                        style='wireframe',
                        line_width=3,
                        opacity=1.0,
                        name='vessel_wall')
        
        # Add vessel cross-sections
        vessel_angles = [0, np.pi/6, np.pi/3, np.pi/2]
        for i, phi_key in enumerate(vessel_angles):
            X_cross = R_vessel * np.cos(phi_key)
            Y_cross = R_vessel * np.sin(phi_key)
            Z_cross = Z_vessel
            
            # Create line points
            points = np.column_stack((X_cross, Y_cross, Z_cross))
            
            # Create polydata for line
            line = pv.lines_from_points(points, close=True)
            plotter.add_mesh(line, 
                           color='black',
                           line_width=4,
                           name=f'vessel_cross_{i}')
        
        print(f"  Added vessel wall with {len(R_vessel)} points (quarter revolution: 0° to 90°)")
    
    # Add magnetic axis
    phi_axis = np.linspace(-np.pi, np.pi, 200)
    X_axis = eq.raw['rmaxis'] * np.cos(phi_axis)
    Y_axis = eq.raw['rmaxis'] * np.sin(phi_axis)
    Z_axis = np.full_like(phi_axis, eq.raw['zmaxis'])
    
    # Create magnetic axis line
    axis_points = np.column_stack((X_axis, Y_axis, Z_axis))
    axis_line = pv.lines_from_points(axis_points)
    plotter.add_mesh(axis_line,
                    color='red',
                    line_width=6,
                    name='magnetic_axis')
    
    # Add axis point markers
    key_angles = [0, np.pi/2, np.pi, -np.pi/2]
    for phi_mark in key_angles:
        X_mark = eq.raw['rmaxis'] * np.cos(phi_mark)
        Y_mark = eq.raw['rmaxis'] * np.sin(phi_mark)
        Z_mark = eq.raw['zmaxis']
        
        point = pv.PolyData([X_mark, Y_mark, Z_mark])
        plotter.add_mesh(point,
                        color='red',
                        point_size=15,
                        render_points_as_spheres=True,
                        name=f'axis_marker_{phi_mark:.1f}')
    
    # Set up the scene
    plotter.add_axes(xlabel='X [m]', ylabel='Y [m]', zlabel='Z [m]')
    plotter.add_title('3D Toroidal Flux Surfaces with Vessel Wall (PyVista)', font_size=16)
    
    # Set camera position for good viewing angle
    plotter.camera_position = 'iso'
    plotter.camera.elevation = 20
    plotter.camera.azimuth = 45
    
    print(f"\nSuccessfully created PyVista 3D plot with {surfaces_plotted} flux surfaces")
    
    return plotter

def main():
    """Main function with interactive menu for choosing visualization method"""
    import sys
    
    # Check for command line argument
    if len(sys.argv) > 1:
        eqdsk_path = sys.argv[1]
    else:
        # Default path
        eqdsk_path = '/Users/rjjm/Documents/GitHub/megpy/megpy/vmec/TRUE_EQDSK_FILE'
    
    print("=" * 70)
    print("COMPREHENSIVE EQDSK VISUALIZATION TOOLKIT")
    print("=" * 70)
    print(f"EQDSK file: {eqdsk_path}")
    print("=" * 70)
    print("Loading EQDSK file...")
    
    # Create equilibrium object and read the file
    eq = Equilibrium(verbose=True)
    
    try:
        # Read the EQDSK file and add derived quantities
        eq.read_geqdsk(f_path=eqdsk_path, add_derived=True)
        print("Successfully loaded EQDSK file!")
        
        # Print some basic information
        print(f"\nEquilibrium Information:")
        print(f"Grid size: {eq.raw['nw']} x {eq.raw['nh']}")
        print(f"Magnetic axis: R = {eq.raw['rmaxis']:.3f} m, Z = {eq.raw['zmaxis']:.3f} m")
        print(f"Psi axis: {eq.raw['simag']:.6f}")
        print(f"Psi boundary: {eq.raw['sibry']:.6f}")
        print(f"Toroidal field at axis: {eq.raw['bcentr']:.3f} T")
        print(f"Plasma current: {eq.raw['current']:.0f} A")
        
        # Interactive menu
        print("\n" + "=" * 70)
        print("VISUALIZATION OPTIONS:")
        print("=" * 70)
        print("1. Simple built-in plots (flux surfaces + profiles)")
        print("2. Custom 2D matplotlib analysis")
        print("3. Advanced 3D matplotlib visualization")
        print("4. Interactive 3D PyVista visualization")
        print("5. All methods (comprehensive comparison)")
        print("6. Exit")
        
        while True:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                # Method 1: Simple built-in plots
                fig1, fig2 = plot_simple_builtin(eq)
                fig1.savefig('/Users/rjjm/Documents/GitHub/megpy/flux_surfaces_builtin.png', dpi=300, bbox_inches='tight')
                fig2.savefig('/Users/rjjm/Documents/GitHub/megpy/profiles_builtin.png', dpi=300, bbox_inches='tight')
                print("Plots saved as 'flux_surfaces_builtin.png' and 'profiles_builtin.png'")
                plt.show()
                
            elif choice == "2":
                # Method 2: Custom 2D matplotlib
                fig = plot_custom_2d_matplotlib(eq)
                fig.savefig('/Users/rjjm/Documents/GitHub/megpy/equilibrium_2d_custom.png', dpi=300, bbox_inches='tight')
                print("Plot saved as 'equilibrium_2d_custom.png'")
                plt.show()
                
            elif choice == "3":
                # Method 3: 3D matplotlib
                fig = plot_3d_matplotlib(eq, n_surfaces=8, n_points_toroidal=60)
                fig.savefig('/Users/rjjm/Documents/GitHub/megpy/equilibrium_3d_matplotlib.png', dpi=300, bbox_inches='tight')
                print("Plot saved as 'equilibrium_3d_matplotlib.png'")
                plt.show()
                
            elif choice == "4":
                # Method 4: PyVista 3D
                plotter = plot_3d_pyvista(eq, n_surfaces=8, n_points_toroidal=60)
                if plotter is not None:
                    print("Showing interactive PyVista 3D plot...")
                    print("Note: Use the PyVista interface to save screenshots if needed")
                    plotter.show()
                
            elif choice == "5":
                # All methods
                print("\n" + "=" * 70)
                print("GENERATING ALL VISUALIZATIONS...")
                print("=" * 70)
                
                # Method 1
                fig1, fig2 = plot_simple_builtin(eq)
                fig1.savefig('/Users/rjjm/Documents/GitHub/megpy/comprehensive_flux_surfaces.png', dpi=300, bbox_inches='tight')
                fig2.savefig('/Users/rjjm/Documents/GitHub/megpy/comprehensive_profiles.png', dpi=300, bbox_inches='tight')
                
                # Method 2
                fig_2d = plot_custom_2d_matplotlib(eq)
                fig_2d.savefig('/Users/rjjm/Documents/GitHub/megpy/comprehensive_2d_analysis.png', dpi=300, bbox_inches='tight')
                
                # Method 3
                fig_3d = plot_3d_matplotlib(eq, n_surfaces=8, n_points_toroidal=60)
                fig_3d.savefig('/Users/rjjm/Documents/GitHub/megpy/comprehensive_3d_matplotlib.png', dpi=300, bbox_inches='tight')
                
                print("\nStatic plots saved with 'comprehensive_' prefix")
                
                # Show matplotlib plots
                plt.show()
                
                # Method 4 (interactive)
                plotter = plot_3d_pyvista(eq, n_surfaces=8, n_points_toroidal=60)
                if plotter is not None:
                    print("Showing interactive PyVista 3D plot...")
                    plotter.show()
                
            elif choice == "6":
                print("Exiting...")
                break
                
            else:
                print("Invalid choice. Please enter 1-6.")
        
    except Exception as e:
        print(f"Error loading or plotting EQDSK file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
