from .tracer import contour

import numpy as np
import xarray as xr
from scipy import integrate,interpolate
from scipy.optimize import least_squares

from .tracer import *
from .utils import *

class FluxSurface():
    
    def __init__(self):
        self.n_theta = 360
        self.n_harmonics = 5 #12
        self.theta = np.linspace(-1,1,self.n_theta) * np.pi

        self.data = xr.DataArray()

        # flux surface
        self.R = np.array([])
        self.Z = np.array([])
        self.theta_RZ = np.array([])

        # geometry scalar values
        self.R0 = 0.
        self.Z0 = 0.
        self.r = 0.
        self.R_centroid = 0.
        self.Z_centroid = 0.
        
        # physics scalar values
        self.fpol = 0.
        self.psi = 0.
        self.phi = 0.

        # parameterization
        self.shape = np.array([])
        self.tolerance = 2.23e-16

        #TODO: add local field calculations
        self.Btor = self.fpol / self.R
        self.Bpol = 0.
        pass
        return
    
    def compute_bpol(self,grid_R,grid_Z,grid_Bpol,R_fs,Z_fs):
        R_mesh, Z_mesh = np.meshgrid(grid_R,grid_Z)

        # interpolate Bpol and Btor
        #B_pol_fs = interpolate.griddata(np.column_stack((Z_mesh.flatten(), R_mesh.flatten())), grid_Bpol.flatten(), (Z_fs, R_fs), method='cubic')
        B_pol_fs = interpolate.RectBivariateSpline(grid_Z, grid_R, grid_Bpol, kx=3, ky=3, s=0)(Z_fs, R_fs, grid=False)

        return B_pol_fs
    
    #def from_eq(self,eq,x_fs=None,x_label='rho_tor',x_point=False):

    #    return

    def from_tracer(self,grid_R,grid_Z,field,level,k='l',m_axis=None,xpoint=False):
        fs = contour(grid_R,grid_Z,field,level,kind=k,ref_point=m_axis,x_point=xpoint)

        self.R = fs['X']
        self.Z = fs['Y']
        self.theta_RZ = fs['theta_XY']

        self.Z_max = fs['Y_max']
        self.Z_min = fs['Y_min']
        
        self.R0 = fs['X0']
        self.Z0 = fs['Y0']
        self.r = fs['r']
        #self.R_centroid = fs['Xc']
        #self.Z_centroid = fs['Yc']

        return

    def from_RZ(self,R,Z):
        self.R = R
        self.Z = Z
        self.theta_RZ = np.mod(np.arctan2(self.Z-np.mean(self.Z),self.R-np.mean(self.R)),2*np.pi)

        fs = contour_center({'X':self.R, 'Y':self.Z, 'theta_XY':self.theta_RZ})

        self.R0 = fs['X0']
        self.Z0 = fs['Y0']
        self.r = fs['r']

        self.Zmax = fs['Y_max']
        self.Zmin = fs['Y_min']

        self.kappa = ((self.Zmax-self.Zmin)/2)/self.r

        #self.R_centroid = fs['Xc']
        #self.Z_centroid = fs['Yc']

        return

    def from_mxh(self,shape=None,shape_deriv=None,dpsidr=None):
        if shape is None:
            shape = self.shape
        self.R, self.Z, self.theta_ref = self.mxh(shape,self.theta)

        """if shape_deriv is not None and dpsidr is not None:
            dRdtheta, dZdtheta, dRdr, dZdr, J_r = self.mxh_jr(shape,shape_deriv,self.theta,self.R,return_deriv=True)

            # compute the Mercier-Luc arclength derivative and |grad r|
            dl_dtheta = np.sqrt(dRdtheta**2 + dZdtheta**2)
            grad_r_norm = (self.R/J_r)*dl_dtheta

            # Poloidal magnetic flux density
            self.Bpol_param = (dpsidr / self.R) * grad_r_norm"""

        return

    def from_turnbull(self,shape=None):
        if shape is None:
            shape = self.shape
        self.R, self.Z, self.theta_ref = self.turnbull(shape,self.theta)

        return
    
    # convert to analytical flux surface
    def to_mxh(self,optimize=True):
        self.kappa = ((self.Zmax-self.Zmin)/2)/self.r
        #self.theta = arcsin2pi(np.clip((self.Z-self.Z0)/(self.kappa*self.r),-1,1))
        self.theta = np.arcsin(np.clip((self.Z-self.Z0)/(self.kappa*self.r),-1,1))
        
        if optimize:
            # set initial condition and bounds
            self.param_initial = [0.,0.,0.,1.,0.]+list(np.zeros(2*self.n_harmonics))
            self.param_bounds = [[0.,-np.inf,0.,0.,-2*np.pi]+list(-np.inf*np.ones(2*self.n_harmonics)),[np.inf,np.inf,np.inf,np.inf,2*np.pi]+list(np.inf*np.ones(2*self.n_harmonics))]

            self.R_ref = self.R - self.R0
            self.Z_ref = self.Z - self.Z0

            def cost(shape):
                # compute the flux-surface parameterization for a given shape set `params`
                self.R_param, self.Z_param, self.theta_ref = self.mxh(shape, self.theta, norm=True)
                
                # define the cost function
                L1_norm = np.abs(np.array([self.R_param,self.Z_param])-np.array([self.R_ref,self.Z_ref])).flatten()
                L2_norm = np.sqrt((self.R_param-self.R_ref)**2+(self.Z_param-self.Z_ref)**2)

                return self.n_theta*np.hstack((L1_norm,L2_norm))
            
            lsq = least_squares(cost, 
                                self.param_initial, 
                                bounds=self.param_bounds, 
                                ftol=self.tolerance, 
                                xtol=self.tolerance, 
                                gtol=self.tolerance, 
                                loss='soft_l1', 
                                verbose=0)
            self.shape = lsq['x']

        else:
            # solve for polar angles
            theta_cont = arcsin2pi(np.clip((self.Z - self.Z0) / (self.r * self.kappa), -1, 1))
            theta_r_cont = arccos2pi(np.clip(((self.R - self.R0) / self.r), -1, 1))

            theta_r_cont = theta_r_cont - theta_cont
            theta_r_cont[-1] = theta_r_cont[0]

            self.theta = theta_cont

            # fourier decomposition
            c_n, s_n = np.zeros(self.n_harmonics+1), np.zeros(self.n_harmonics+1)

            def f_theta_r(theta):
                return np.interp(theta, theta_cont, theta_r_cont)
            
            for i in np.arange(self.n_harmonics+1):
                s_n[i] = float(integrate.quad(f_theta_r,0,2*np.pi, weight="sin", wvar=i,limit=1000)[0]/np.pi)
                c_n[i] = float(integrate.quad(f_theta_r,0,2*np.pi, weight="cos", wvar=i,limit=1000)[0]/np.pi)
            
            c_n[0] /= 2

            self.shape = np.array([self.R0,self.Z0,self.r,self.kappa,c_n[0]]+list(np.array([[c_n[i],s_n[i]] for i in range(1,self.n_harmonics+1)]).flatten()))

            self.R_param, self.Z_param, self.theta_ref = self.mxh(self.shape, self.theta, norm=True)

        return

    def to_turnbull(self,):
        # set initial condition and bounds
        self.param_initial = [0.,0.,0.,1.,0.,0.]
        self.param_bounds = ([0.,-np.inf,0.,0.,-1,-0.5],[np.inf,np.inf,np.inf,np.inf,1,0.5])

        def cost(shape):
            # compute the flux-surface parameterization for a given shape set `params`
            self.R_param, self.Z_param, self.theta_ref = self.turnbull(shape, self.theta, norm=True)

            self.R_ref = np.array(interpolate_periodic(self.theta_RZ, self.R, self.theta_ref)) - self.R0
            self.Z_ref = np.array(interpolate_periodic(self.theta_RZ, self.Z, self.theta_ref)) - self.Z0

            # define the cost function
            L1_norm = np.abs(np.array([self.R_param,self.Z_param])-np.array([self.R_ref,self.Z_ref])).flatten()
            L2_norm = np.sqrt((self.R_param-self.R_ref)**2+(self.Z_param-self.Z_ref)**2)

            return self.n_theta*np.hstack((L1_norm,L2_norm))
        
        lsq = least_squares(cost, 
                            self.param_initial, 
                            bounds=self.param_bounds, 
                            ftol=self.tolerance, 
                            xtol=self.tolerance, 
                            gtol=self.tolerance, 
                            loss='soft_l1', 
                            verbose=0)
        self.shape = lsq['x']

        return
    
    # analytical flux surface descriptions
    def mxh(self,shape,theta=None,norm=False):
        """Compute MXH flux-surface parameterization given a set of shape parameters and a theta-grid.

        Args:
            shape (array): 1D array or list containing the MXH shape parameters [R0,Z0,r,kappa,c_0,c_1,s_1,...,c_n,s_n].
            theta (array): 1D array containing the theta-grid.

        Returns:
            - R_param (array): 1D array containing the radial flux-surface parameterization coordinate sorted by theta_ref [0,2*pi].
            - Z_param (array): 1D array containing the vertical flux-surface parameterization coordinate sorted by theta_ref [0,2*pi].
            - theta_ref (array): 1D array containing the poloidal angle between radial and vertical flux-surface parameterization coordinates sorted ascending.
        """
        # flux-surface coordinate parameterization from [Arbon PPCF 63 (2020)]

        if theta is None:
            theta = self.theta

        # extract bounding box values
        shape[:4] = [self.R0,self.Z0,self.r,self.kappa]

        # set tilt
        c_0 = shape[4]

        # compute poloidal angle harmonic composition 
        theta_R = theta + c_0
        N = int((len(shape)-5)/2)
        for n in range(1,N+1):
            c_n = shape[5 + (n-1)*2]
            s_n = shape[6 + (n-1)*2]
            theta_R += c_n * np.cos(n * theta) + s_n * np.sin(n * theta)

        # compute flux surface R, Z coordinates
        R_param = self.R0 + self.r * np.cos(theta_R)
        Z_param = self.Z0 + self.kappa * self.r * np.sin(theta)

        theta_ref = np.mod(np.arctan2(Z_param-self.Z0, R_param-self.R0), 2*np.pi)

        if norm:
            R_param-=self.R0
            Z_param-=self.Z0

        return R_param, Z_param, theta_ref
    
    def mxh_jr(self,shape,shape_deriv,theta,R,return_deriv=True):
        """Compute MXH flux-surface parameterization Jacobian given sets of shape and shape derivative parameters and a theta-grid.

        Args:
            shape (array): 1D array or list containing the MXH shape parameters [R0,Z0,r,kappa,c_0,c_1,s_1,...,c_n,s_n].
            shape_deriv (array): 1D array or list containing the MXH shape derivative parameters [dR0dr,dZ0dr,s_kappa,rdc_0dr,rdc_1dr,rds_1dr,...,rdc_ndr,rds_ndr].
            theta (array): 1D array containing the theta-grid.
            R (array): 1D array containing the radial flux-surface coordinate as output by mxh().
            return_deriv (bool, optional): Switch to return the radial and poloidal derivatives in addition to the Jacobian or not. Defaults to True.

        Returns:
            dRdtheta (array, optional): 1D array containing the poloidal derivative of the radial flux-surface coordinate.
            dZdtheta (array, optional): 1D array containing the poloidal derivative of the vertical flux-surface coordinate.
            dRdr (array, optional): 1D array containing the radial derivative of the radial flux-surface coordinate.
            dZdr (array, optional): 1D array containing the radial derivative of the vertical flux-surface coordinate.
            J_r (array): 1D array containing the Jacobian for the Miller parameterization.
        """
        # set tilt
        c_0 = shape[4]

        # compute poloidal angle harmonic composition
        theta_R = theta + c_0
        dtheta_Rdtheta = np.ones_like(theta)
        N = int((len(shape)-5)/2)
        for n in range(1,N+1):
            c_n = shape[5 + (n-1)*2]
            s_n = shape[6 + (n-1)*2]
            theta_R +=  c_n * np.cos(n * theta) + s_n * np.sin(n * theta)
            dtheta_Rdtheta += (-n * c_n * np.sin(n * theta)) + (n * s_n * np.cos(n * theta))
        
        # compute the derivatives for the Jacobian
        [dR0dr,dZ0dr,s_kappa,dc_0dr] = shape_deriv[:4]
        dtheta_Rdr = dc_0dr * np.ones_like(theta)
        M = int((len(shape_deriv)-4)/2)
        for m in range(1,M+1):
            dc_mdr = shape_deriv[4 + (m-1)*2]
            ds_mdr = shape_deriv[5 + (m-1)*2]
            dtheta_Rdr += dc_mdr * np.cos(m * theta) + ds_mdr * np.sin(m * theta)

        dRdtheta = - self.r * np.sin(theta_R) * dtheta_Rdtheta
        dZdtheta = self.kappa * self.r * np.cos(theta)

        dRdr = dR0dr + np.cos(theta_R) - np.sin(theta_R) * dtheta_Rdr
        dZdr = dZ0dr + self.kappa * (s_kappa + 1) * np.sin(theta)

        # compute Mercier-Luc arclength derivative and |grad r|
        J_r = R * (dRdr * dZdtheta - dRdtheta * dZdr)

        if return_deriv:
            return dRdtheta, dZdtheta, dRdr, dZdr, J_r
        else:
            return J_r

    def turnbull(self,shape,theta=None,norm=False):
        """Compute Turnbull-Miller flux-surface parameterization given a set of shape parameters and a theta-grid.

        Args:
            shape (array): 1D array or list containing the Turnbull-Miller shape parameters [R0,Z0,r,kappa,delta,zeta].
            theta (array): 1D array containing the theta-grid.
            norm (bool, optional): Normalize the parameterized flux-surface coordinates by the center coordinates. Defaults to False.

        Returns:
            - R_param (array): 1D array containing the radial flux-surface parameterization coordinate sorted by theta_ref [0,2*pi].
            - Z_param (array): 1D array containing the vertical flux-surface parameterization coordinate sorted by theta_ref [0,2*pi].
            - theta_ref (array): 1D array containing the poloidal angle between radial and vertical flux-surface parameterization coordinates sorted ascending.
        """
        # flux-surface coordinate parameterization from [Turnbull PoP 6 (1999)]

        if theta is None:
            theta = self.theta

        R0, Z0, r, kappa, delta, zeta = shape
        
        with np.errstate(invalid='ignore'):
            x = np.arcsin(delta)
        
        # compute poloidal angle shape modifications
        theta_R = theta + x * np.sin(theta)
        theta_Z = theta + zeta * np.sin(2 * theta)

        # compute flux surface R, Z coordinates
        R_param = R0 + r * np.cos(theta_R)
        Z_param = Z0 + kappa * r * np.sin(theta_Z)
        
        theta_ref = np.mod(np.arctan2(Z_param-Z0,R_param-R0), 2*np.pi)

        if norm:
            R_param-=self.R0
            Z_param-=self.Z0
        
        return R_param, Z_param, theta_ref
    
    def turnbull_jr(self,shape,shape_deriv,theta,R,return_deriv=True):
        """Compute Turnbull-Miller flux-surface parameterization Jacobian given sets of shape and shape derivative parameters and a theta-grid.

        Args:
            shape (array): 1D array or list containing the Turnbull-Miller shape parameters [R0,Z0,r,kappa,delta,zeta].
            shape_deriv (array): 1D array or list containing the Turnbull-Miller shape derivative parameters [dR0dr,dZ0dr,s_kappa,s_delta,s_zeta].
            theta (array): 1D array containing the theta-grid.
            R (array): 1D array containing the radial flux-surface coordinate as output by turnbull().
            return_deriv (bool, optional): Switch to return the radial and poloidal derivatives in addition to the Jacobian or not. Defaults to True.

        Returns:
            dRdtheta (array, optional): 1D array containing the poloidal derivative of the radial flux-surface coordinate.
            dZdtheta (array, optional): 1D array containing the poloidal derivative of the vertical flux-surface coordinate.
            dRdr (array, optional): 1D array containing the radial derivative of the radial flux-surface coordinate.
            dZdr (array, optional): 1D array containing the radial derivative of the vertical flux-surface coordinate.
            J_r (array): 1D array containing the Jacobian for the Miller parameterization.
        """
        # define the parameters
        [R0,Z0,r,kappa,delta,zeta] = shape
        [dR0dr,dZ0dr,s_kappa,s_delta,s_zeta] = shape_deriv
        with np.errstate(invalid='ignore'):
            x = np.arcsin(delta)
        theta_R = theta + x * np.sin(theta)
        dtheta_Rdtheta = 1 + x * np.cos(theta)
        theta_Z = theta + zeta * np.sin(2 * theta)
        dtheta_Zdtheta = 1 + 2 * zeta * np.cos(2 * theta)

        # compute the derivatives for the Jacobian
        dRdtheta = - r * np.sin(theta_R) * dtheta_Rdtheta
        dZdtheta = kappa * r * np.cos(theta_Z) * dtheta_Zdtheta
        dRdr = dR0dr + np.cos(theta_R) - s_delta * np.sin(theta) * np.sin(theta_R)
        dZdr = dZ0dr + kappa * ((s_kappa + 1) * np.sin(theta_Z) + s_zeta * np.sin(2 * theta) * np.cos(theta_Z))

        # compute Mercier-Luc arclength derivative and |grad r|
        J_r = R * (dRdr * dZdtheta - dRdtheta * dZdr)

        if return_deriv:
            return dRdtheta, dZdtheta, dRdr, dZdr, J_r
        else:
            return J_r

    def toq9(self,shape,theta=None,norm=False):
        if theta is None:
            theta = self.theta

        R0, Z0, r, kappa, delta, zeta_in, zeta_out = shape
        
        with np.errstate(invalid='ignore'):
            x = np.arcsin(delta)

        #TODO fix this
        theta_in = theta[np.where((theta>=0.5*np.pi)&(theta<=1.5*np.pi))]
        theta_out = np.hstack((theta[np.where((theta>=0.5*np.pi)&(theta<=1.5*np.pi))],))
        
        # compute poloidal angle shape modifications
        theta_R = theta + x * np.sin(theta)
        theta_Z = theta + zeta_in * np.sin(2 * theta_in) + zeta_out * np.sin(2 * theta_out)

        # compute flux surface R, Z coordinates
        R_param = R0 + r * np.cos(theta_R)
        Z_param = Z0 + kappa * r * np.sin(theta_Z)
        
        theta_ref = np.mod(np.arctan2(Z_param-Z0,R_param-R0), 2*np.pi)

        if norm:
            R_param-=self.R0
            Z_param-=self.Z0
        
        return R_param, Z_param, theta_ref