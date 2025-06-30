# Here we reproduce the poloidal coordinates from the paper:
# "Ideal MHD Stability Calculations in Axisymmetric Toroidal Coordinate Systems"
# by Grimm, Dewar, and Manickam (1981). [ https://www.sciencedirect.com/science/article/pii/002199918390116X?ref=cra_js_challenge&fr=RR-1 ]

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


E = 0.5
R0 = 1.0
i=2
j=0

# calculate the poloidal flux function psi
def psi(R,Z,R0=R0,E=E):
    """
    Calculate the poloidal flux function psi in a cylindrical coordinate system.
    """
    term1 = (R**2 - R0**2)**2
    term2 = E**2 * R**2 * Z**2
    return term1 + term2


# calculate gradient of psi
def grad_psi(R, Z, R0=R0, E=E):
    """
    Calculate the gradient of the poloidal flux function psi.
    """
    dpsi_dR = 2 * (R**2 - R0**2) * 2 * R + 2 * E**2 * R * Z**2
    dpsi_dZ = 2 * E**2 * R**2 * Z
    return np.array([dpsi_dR, dpsi_dZ])


# get contour list (R,Z) for a given psi value
def get_contour(psi_value, R0=R0, E=E):
    """
    Get the contour (R,Z) for a given psi value.
    """
    R = np.linspace(0, 2*R0, 100)
    Z = np.linspace(-2*R0, 2*R0, 100)
    R, Z = np.meshgrid(R, Z)
    
    # Calculate psi for the grid
    psi_grid = psi(R, Z, R0, E)
    
    # Find the contour where psi equals the specified value
    R = np.linspace(0, 2*R0, 2000)
    Z = np.linspace(-2*R0, 2*R0, 2000)
    R, Z = np.meshgrid(R, Z)
    psi_grid = psi(R, Z, R0, E)
    contour = plt.contour(R, Z, psi_grid, levels=[psi_value])
    contour_data = contour.get_paths()[0].vertices
    plt.close()  # Close the plot to avoid displaying it
    return contour_data


def compute_poloidal_theta(psi_value=0.5, i=i, j=j, R0=R0, E=E):
    """
    Compute the poloidal angle theta along the contour of constant psi.
    Returns theta, R, Z arrays.
    """
    c = get_contour(psi_value, R0=R0, E=E)
    R = c[:, 0]
    Z = c[:, 1]

    grad_psi_norm = np.linalg.norm(grad_psi(R, Z, R0=R0, E=E), axis=0)

    Z_left = np.roll(Z, 1)
    Z_right = np.roll(Z, -1)
    R_left = np.roll(R, 1)
    R_right = np.roll(R, -1)
    ds = np.sqrt((R_left - R_right)**2 + (Z_left - Z_right)**2)

    integrand = grad_psi_norm**(j-1)/R**(i-1)
    alpha = 2 * np.pi / np.sum(integrand * ds)

    J = R**i/(alpha * grad_psi_norm**j)
    dtheta_ds = R/(J * grad_psi_norm)
    theta = np.cumsum(dtheta_ds * ds)
    return theta, R, Z

# plot mesh of poloidal coordinates for a number of psi values
# (i.e. plot R,Z, also showing constant theta contours)
psis = np.linspace(0.1, 0.5, 10)
plt.figure(figsize=(10, 8))
Rs = []
Zs = []
thetas = []
for psi_value in psis:
    theta, R, Z = compute_poloidal_theta(psi_value=psi_value, i=i, j=j, R0=R0, E=E)
    Rs.append(R)
    Zs.append(Z)
    thetas.append(theta)


# plot R(theta) and Z(theta) for each psi in a 2x1 subplot
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
for i, psi_value in enumerate(psis):
    axs[0].plot(thetas[i], Rs[i], label=f'psi={psi_value:.2f}')
    axs[1].plot(thetas[i], Zs[i], label=f'psi={psi_value:.2f}')
axs[0].set_title('R(theta) for different psi values')
axs[0].set_xlabel('Theta')
axs[0].set_ylabel('R')
axs[1].set_title('Z(theta) for different psi values')
axs[1].set_xlabel('Theta')
axs[1].set_ylabel('Z')
axs[0].legend()
axs[1].legend()
plt.tight_layout()
plt.show()