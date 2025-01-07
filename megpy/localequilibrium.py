"""
created by gsnoep on 11 August 2022, extract_analytic_shape method adapted from 'extract_miller_from_eqdsk.py' by dtold
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

from scipy import interpolate
from scipy.optimize import least_squares
from sys import stdout

from .utils import *

class LocalEquilibrium():
    """Class to handle any and all methods related to local magnetic equilibrium parameterization.
        - parametrize a set of flux-surface coordinates 
        - extract analytic shape parameters as described by T Luce
        - extract FFT coefficients for a Fourier expansion of the local equilibrium
        - print magnetic geometry input parameters for several microturbulence codes (GENE, TGLF, ...)
    """
    def __init__(self,param,equilibrium,x_loc,x_label='rho_tor',n_x=9,n_theta='default',n_harmonics=1,analytic_shape=False,opt_bpol=False,opt_deriv=False,diag_lsq=0,verbose=True):
        self._params = {'miller':{
                             'param':self.miller,
                             'param_jr':self.miller_jr,
                             'param_initial':[0.,0.,0.,1.,0.],
                             'param_bounds':([0.,-np.inf,0.,0.,-np.inf],np.inf),
                             'param_labels':['R0','Z0','r','kappa','delta'],
                             'deriv_initial':[0.,0.,0.,0.],
                             'deriv_bounds':[-np.inf,np.inf],
                             'deriv_labels':['dR0dr','dZ0dr','s_kappa','s_delta'],
                         },
                         'turnbull':{
                             'param':self.turnbull,
                             'param_jr':self.turnbull_jr,
                             'param_initial':[0.,0.,0.,1.,0.,0.],
                             'param_bounds':([0.,-np.inf,0.,0.,-1,-0.5],[np.inf,np.inf,np.inf,np.inf,1,0.5]),
                             'param_labels':['R0','Z0','r','kappa','delta','zeta'],
                             'deriv_initial':[0.,0.,0.,0.,0.],
                             'deriv_bounds':[-np.inf,np.inf],
                             'deriv_labels':['dR0dr','dZ0dr','s_kappa','s_delta','s_zeta'],
                         },
                         'turnbull_tilt':{
                             'param':self.turnbull_tilt,
                             'param_jr':self.turnbull_tilt_jr,
                             'param_initial':[0.,0.,0.,1.,0.,0.,0.],
                             'param_bounds':([0.,-np.inf,0.,0.,-np.inf,-np.inf,-np.inf],np.inf),
                             'param_labels':['R0','Z0','r','kappa','delta','zeta','tilt'],
                             'deriv_initial':[0.,0.,0.,0.,0.,0.],
                             'deriv_bounds':[-np.inf,np.inf],
                             'deriv_labels':['dR0dr','dZ0dr','s_kappa','s_delta','s_zeta','s_tilt'],
                         },
                         'fourier':{
                             'param':self.fourier,
                             'param_jr':self.fourier_jr,
                             'param_initial':list(np.zeros(2))+([1,0.,0.,1]+[0.,0.,0.,0.]*(n_harmonics-1)),
                             'param_bounds':[[0,-np.inf]+[-1]*(4*n_harmonics),[np.inf,np.inf]+[1]*(4*n_harmonics)],
                             'param_labels':['R0','Z0']+[label for sublist in [['aR_{}'.format(n),'bR_{}'.format(n),'aZ_{}'.format(n),'bZ_{}'.format(n)] for n in range(1,n_harmonics+1)] for label in sublist],
                             #'param_labels':['aR_0','aZ_0']+[label for sublist in [['aR_{}'.format(n),'bR_{}'.format(n),'aZ_{}'.format(n),'bZ_{}'.format(n)] for n in range(1,n_harmonics+1)] for label in sublist],
                             'deriv_initial':list(np.ones(2+4*n_harmonics)),
                             'deriv_bounds':[-np.inf,np.inf],
                             'deriv_labels':['d{}dr'.format(label) for label in ['R0','Z0']]+['d{}dr'.format(label) for sublist in [['aR_{}'.format(n),'bR_{}'.format(n),'aZ_{}'.format(n),'bZ_{}'.format(n)] for n in range(1,n_harmonics+1)] for label in sublist],
                         },
                         'miller_general':{
                             'param':self.miller_general,
                             'param_jr':self.miller_general_jr,
                             'param_initial':([0.5,0.5]+[0.,0.]*(n_harmonics-1)),
                             'param_bounds':[-np.inf,np.inf],
                             'param_labels':['cN_{}'.format(n) for n in range(0,n_harmonics)]+['sN_{}'.format(n) for n in range(0,n_harmonics)],
                             'deriv_initial':list(np.ones(2*n_harmonics)),
                             'deriv_bounds':[-np.inf,np.inf],
                             'deriv_labels':['dcN_{}dr'.format(n) for n in range(0,n_harmonics)]+['dsN_{}dr'.format(n) for n in range(0,n_harmonics)],
                         },
                         'mxh':{
                             'param':self.mxh,
                             'param_jr':self.mxh_jr,
                             'param_initial':[0.,0.,0.,1.,0.]+list(np.zeros(2*n_harmonics)),
                             'param_bounds':[[0.,-np.inf,0.,0.,-2*np.pi]+list(-np.inf*np.ones(2*n_harmonics)),[np.inf,np.inf,np.inf,np.inf,2*np.pi]+list(np.inf*np.ones(2*n_harmonics))],
                             'param_labels':['R0','Z0','r','kappa','c_0']+[label for sublist in [['c_{}'.format(n),'s_{}'.format(n),] for n in range(1,n_harmonics+1)] for label in sublist],
                             'deriv_initial':list(np.ones(4+2*n_harmonics)),
                             'deriv_bounds':[-np.inf,np.inf],
                             'deriv_labels':['dR0dr','dZ0dr','s_kappa','rdc_0dr']+[label for sublist in [['rdc_{}dr'.format(n),'rds_{}dr'.format(n),] for n in range(1,n_harmonics+1)] for label in sublist],
                         }
        }

        self.verbose = verbose
        self.tolerance = 2.23e-16
        self.x_loc = x_loc
        self.x_label = x_label

        # copy the equilibrium
        self.eq = copy.deepcopy(equilibrium)

        # initialize the LocalEquilibrium parametrization instance attributes
        self._param = param
        self.param = self._params[param]['param']
        self.param_jr = self._params[param]['param_jr']
        self.param_initial = self._params[param]['param_initial']
        self.param_bounds = self._params[param]['param_bounds']
        self.param_labels = self._params[param]['param_labels']
        self.deriv_initial = self._params[param]['deriv_initial']
        self.deriv_bounds = self._params[param]['deriv_bounds']
        self.deriv_labels = self._params[param]['deriv_labels']

        # generate the radial grid
        self.n_x = n_x
        # use case 1: standalone CLI call
        if self.n_x > 1:
            # if n_x is not odd, make it odd
            if not self.n_x & 0x1:
                self.n_x = self.n_x-1
                if verbose:
                    print('Provided n_x is not odd, setting x-grid size to {}'.format(self.n_x))
            # set the fraction of x_loc used set the min/max of the x_grid
            loc_frac = 0.005
            x_list = [(self.x_loc-loc_frac*self.x_loc)+i*(loc_frac*self.x_loc/int((self.n_x-1)/2)) for i in range(0,self.n_x)]
            self.x_grid = [x for x in x_list if 0. <= x <= 1.]
        # use case 2: integrated in Equilibrium API
        else:
            self.x_grid = [self.x_loc]
        
         # extract flux-surfaces
        self.eq.fluxsurfaces = {}
        self.eq.add_fluxsurfaces(x=self.x_grid,x_label=x_label,analytic_shape=analytic_shape,incl_B=[x==self.x_loc for x in self.x_grid],verbose=self.verbose)
        # add equilibrium sub-tree for fit data
        self.eq.fluxsurfaces['fit_geo'] = {}

        # generate the poloidal grid
        theta_min = 0
        theta_max = 2*np.pi
        if isinstance(n_theta,int):
            self.n_theta = n_theta
            self.theta = np.linspace(theta_min,theta_max,self.n_theta)
            if self._param == 'miller_general':
                self.theta = np.linspace(0,2,self.n_theta)*np.pi
        elif not (isinstance(n_theta,int) or isinstance(n_theta,str)):
            raise ValueError('Invalid n_theta input!')

        # optimize the shape parameters
        opt_timing = 0.
        if self.verbose:
            print('Optimising parameterization fit of fluxsurfaces...')
        for i_x_loc,xfs in enumerate(self.x_grid):
            with np.errstate(divide='ignore',invalid='ignore'):
                # gather all the flux-surface quantities from the equilibrium
                self.fs = {}
                for key in set(['R','Z','R0','Z0','theta_RZ','r']+self.param_labels):
                    if key in self.eq.fluxsurfaces:
                        quantity = copy.deepcopy(self.eq.fluxsurfaces[key][i_x_loc])
                        self.fs.update({key:quantity})
                if analytic_shape:
                    self.fs['miller_geo'] = {}
                    for key in self.eq.fluxsurfaces['miller_geo']:
                        quantity = copy.deepcopy(self.eq.fluxsurfaces['miller_geo'][key][i_x_loc])
                        self.fs['miller_geo'].update({key:quantity})
                # set the default n_theta (interpolating only once in between the traced flux-surface grid points)
                if n_theta == 'default':
                    self.n_theta = 2 * len(self.eq.fluxsurfaces['R'][self.x_grid.index(self.x_loc)])
                    self.theta = np.linspace(theta_min,theta_max,self.n_theta)
                    if self._param == 'miller_general':
                        self.theta = np.linspace(0,2,self.n_theta)*np.pi

                # modifications for MXH and Fourier
                if self._param == 'mxh':
                    self.fs['R0'] = (np.max(self.fs['R'][:-1])+np.min(self.fs['R'][:-1]))/2
                    self.fs['Z0'] = (np.max(self.fs['Z'][:-1])+np.min(self.fs['Z'][:-1]))/2
                    self.fs['r'] = (np.max(self.fs['R'][:-1])-np.min(self.fs['R'][:-1]))/2
                    self.fs['kappa'] = ((np.max(self.fs['Z'][:-1])-np.min(self.fs['Z'][:-1]))/2)/self.fs['r']

                    self.theta = arcsin2pi((self.fs['Z'][:-1]-self.fs['Z0'])/(self.fs['kappa']*self.fs['r']))
                    self.n_theta = len(self.theta)
                if self._param == 'miller_general':
                    fluxsurface = {'R':self.eq.fluxsurfaces['R'][self.x_grid.index(self.x_loc)],'Z':self.eq.fluxsurfaces['Z'][self.x_grid.index(self.x_loc)],'theta_RZ':self.eq.fluxsurfaces['theta_RZ'][self.x_grid.index(self.x_loc)]}
                    cN,sN = self.extract_fft_shape(fluxsurface,
                                                   self.eq.fluxsurfaces['R0'][self.x_grid.index(self.x_loc)],
                                                   self.eq.fluxsurfaces['Z0'][self.x_grid.index(self.x_loc)],
                                                   self.theta,
                                                   n_harmonics)
                    for i_key,key in enumerate(self.param_labels):
                        self.fs.update({key:copy.deepcopy((list(cN)+list(sN))[i_key])})
                
                # array-ify the flux-surface data
                list_to_array(self.fs)

                # set the initial shape condition for the optimization routine
                if i_x_loc >= 1 and self._param in ['fourier','miller_general','mxh']:
                    # set the previous optimization solution as the initial condition
                    self.param_initial = copy.deepcopy(self.params)
                else:
                    # check if there are values for the shape parameters that can be used as initial condition
                    for i_key,key in enumerate(self.param_labels):
                        if key in self.fs:
                            self.param_initial[i_key] = copy.deepcopy(self.fs[key])
                        elif ('miller_geo' in self.fs) and (key in self.fs['miller_geo']):
                            self.param_initial[i_key] = copy.deepcopy(self.fs['miller_geo'][key])
                    if self._param == 'miller_general':
                        self.param_initial = list(cN)+list(sN)

                time0 = time.time()
                # compute the optimized shape parameters
                lsq = least_squares(self.cost_param, 
                                            self.param_initial, 
                                            bounds=self.param_bounds, 
                                            ftol=self.tolerance, 
                                            xtol=self.tolerance, 
                                            gtol=self.tolerance, 
                                            loss='soft_l1', 
                                            verbose=diag_lsq)
                self.params = lsq['x']
                opt_timing += time.time()-time0
                #print('Optimization time pp:{}'.format(opt_timing))

                # add the final parameterized and interpolated 
                params_keys = ['theta', 'R_param', 'Z_param', 'theta_ref', 'R_ref', 'Z_ref']
                if self._param in ['mxh']:
                    params_values = [self.theta, self.R_param+self.fs['R0'], self.Z_param+self.fs['Z0'], self.theta_ref, self.R_ref+self.fs['R0'], self.Z_ref+self.fs['Z0']]
                else:
                    params_values = [self.theta, self.R_param+self.params[0], self.Z_param+self.params[1], self.theta_ref, self.R_ref+self.params[0], self.Z_ref+self.params[1]]
                
                for i_key,key in enumerate(params_keys):
                    self.fs.update({key:copy.deepcopy(params_values[i_key])})

                # if the current flux-surface is the one at self.x_loc, set the shape parameters
                if xfs == self.x_loc:
                    self.shape = copy.deepcopy(self.params)
                    self.lsq = copy.deepcopy(lsq)

                # label and add the optimized shape parameters to the flux-surface dict
                for i_key, key in enumerate(self.param_labels):
                    self.fs.update({key:self.params[i_key]})
                
                for key in ['R','Z','theta_RZ','miller_geo']:
                    if key in self.fs:
                        del self.fs[key]

                merge_trees(self.fs,self.eq.fluxsurfaces['fit_geo'])

                if self.verbose:
                    # print a progress %
                    stdout.write('\r {}% completed'.format(round(100*(find(xfs,self.x_grid)+1)/len(self.x_grid))))
                    stdout.flush()
        
        list_to_array(self.eq.fluxsurfaces['fit_geo'])
        
        if self.verbose:
            stdout.write('\n')
            print('Optimization time:{:.3f}s / flux-surface'.format(opt_timing/len(self.x_grid)))
            print('Total fitting time:{:.3f}s'.format(opt_timing))

        # re-set the LocalEquilibrium state variables to the x_loc values
        self.fs = {}
        for key in self.eq.fluxsurfaces:
            if isinstance(self.eq.fluxsurfaces[key],list) or isinstance(self.eq.fluxsurfaces[key],np.ndarray):
                self.fs.update({key:copy.deepcopy(self.eq.fluxsurfaces[key][self.x_grid.index(self.x_loc)])})
            else:
                self.fs.update({key:copy.deepcopy(self.eq.fluxsurfaces[key])})
        self.fs.update({'s':(self.fs['fit_geo']['r']*np.gradient(np.log(self.eq.fluxsurfaces['q']),self.fs['fit_geo']['r'],edge_order=2))[self.x_grid.index(self.x_loc)]})
        self.fs.update({'Bref_miller': (self.eq.fluxsurfaces['fpol']/self.fs['fit_geo']['R0'])[self.x_grid.index(self.x_loc)]})
        self.fs.update({'B_unit': ((self.fs['q']/self.fs['fit_geo']['r'])*np.gradient(self.eq.fluxsurfaces['psi'],self.fs['fit_geo']['r'],edge_order=2))[self.x_grid.index(self.x_loc)]})
        
        if self.verbose:
            print('Number of points on reference flux-surface: {}'.format(len(self.fs['R'])))
        
        self.theta = copy.deepcopy(self.eq.fluxsurfaces['fit_geo']['theta'][self.x_grid.index(self.x_loc)])
        self.n_theta = len(self.theta)
        self.R_param, self.Z_param, self.theta_ref = self.param(self.shape, np.append(self.theta,self.theta[0]), norm=False)
        self.Bt_param = interpolate.interp1d(self.eq.derived['psi'],self.eq.derived['fpol'],bounds_error=False)(self.eq.fluxsurfaces['psi'][self.x_grid.index(self.x_loc)])/(self.R_param[:-1])

        # compute the Hessian estimate from the least squares fit Jacobian
        H = self.lsq['jac'].T @ self.lsq['jac']
        try:
            H_inv = np.linalg.inv(H)
        except:
            H_inv = np.nan # in case the Hessian is singular

        # compute the covariance matrix, where N_DOF = 2*n_theta - n_shape, 2*n_theta since both R and Z are minimized on
        self.shape_cov = H_inv * np.sum(self.lsq['fun'])/((2*self.n_theta)-len(self.shape))

        # interpolate the actual flux-surface contour to the theta basis
        if self._param != 'mxh':
            self.R_ref = self.eq.fluxsurfaces['fit_geo']['R_ref'][self.x_grid.index(self.x_loc)]
            self.R_ref = np.append(self.R_ref,self.R_ref[0])
            self.Z_ref = self.eq.fluxsurfaces['fit_geo']['Z_ref'][self.x_grid.index(self.x_loc)]
            self.Z_ref = np.append(self.Z_ref,self.Z_ref[0])
        else:
            self.R_ref = self.eq.fluxsurfaces['R'][self.x_grid.index(self.x_loc)]
            self.Z_ref = self.eq.fluxsurfaces['Z'][self.x_grid.index(self.x_loc)]
        self.Bt_ref  = interpolate.interp1d(self.eq.derived['psi'],self.eq.derived['fpol'],bounds_error=False)(self.eq.fluxsurfaces['psi'][self.x_grid.index(self.x_loc)])/(self.R_ref[:-1])

        # compute flux-surface chi^2
        observed = np.sqrt(self.R_param**2+self.Z_param**2)
        expected = np.sqrt(self.R_ref**2+self.Z_ref**2)
        self.chi_2 = np.sum((observed-expected)**2/expected)
        # compute (modified) reduced chi^2 statistic (in percent), with N_DOF = n_theta-n_shape, if chi_2_nu > 0.005, typically the flux-surface fit starts deviating significantly
        self.chi_2_nu = np.sqrt(np.sum(((observed-expected)/expected)**2)/((self.n_theta-len(self.shape))**2))*100

        # compute the self-consistent shape derivative parameters
        self.dxdr = np.gradient(self.eq.fluxsurfaces[x_label],self.eq.fluxsurfaces['fit_geo']['r'],edge_order=2)
        self.dpsidr = np.abs(self.dxdr*np.gradient(self.eq.fluxsurfaces['psi'],np.array(self.eq.fluxsurfaces[x_label])))[self.x_grid.index(self.x_loc)]
        self.Bunit = (self.eq.fluxsurfaces['q']/self.eq.fluxsurfaces['fit_geo']['r'])[self.x_grid.index(self.x_loc)]*self.dpsidr

        self.fs.update({'dxdr':self.dxdr[self.x_grid.index(self.x_loc)]})

        self.shape_deriv = []
        s_deriv = {}
        if 'kappa' in self.param_labels or 'kappa' in self.fs:
            s_deriv.update({'s_kappa':self.eq.fluxsurfaces['fit_geo']['r']*self.dxdr*np.gradient(np.log(self.eq.fluxsurfaces['fit_geo']['kappa']),np.array(self.eq.fluxsurfaces[x_label]),edge_order=2)})
        if 'delta' in self.param_labels:
            s_deriv.update({'s_delta':(self.eq.fluxsurfaces['fit_geo']['r']/np.sqrt(1-self.eq.fluxsurfaces['fit_geo']['delta']**2))*self.dxdr*np.gradient(self.eq.fluxsurfaces['fit_geo']['delta'],np.array(self.eq.fluxsurfaces[x_label]),edge_order=2)})
        
        for i_key,key in enumerate(self.deriv_labels):
            key_param = None
            if key in s_deriv.keys():
                self.eq.fluxsurfaces['fit_geo'][key] = s_deriv[key]
            else:
                if 'dr' in key:
                    key_param = (key.split('dr')[0]).split('d')[1]
                elif 's_' in key:
                    key_param = key.split('s_')[1]
                if param not in ['fourier','miller_general'] and key_param not in ['R0','Z0']:
                    self.eq.fluxsurfaces['fit_geo'][key] = self.eq.fluxsurfaces['fit_geo']['r']*self.dxdr*np.gradient(self.eq.fluxsurfaces['fit_geo'][key_param],np.array(self.eq.fluxsurfaces[x_label]),edge_order=2)
                else:
                    self.eq.fluxsurfaces['fit_geo'][key] = self.dxdr*np.gradient(np.array(self.eq.fluxsurfaces['fit_geo'][key_param]),np.array(self.eq.fluxsurfaces[x_label]),edge_order=2)

            self.shape_deriv.append(self.eq.fluxsurfaces['fit_geo'][key][self.x_grid.index(self.x_loc)])

        self.dRdtheta_param, self.dZdtheta_param, self.dRdr_param, self.dZdr_param, self.J_r_param = self.param_jr(self.shape,self.shape_deriv,self.theta,self.R_param[:-1],return_deriv=True)
        self.Bp_param = self.param_bpol(self.param_jr, self.shape, self.shape_deriv, self.theta, self.R_param[:-1], self.dpsidr)
        if self._param != 'mxh':
            self.Bp_ref = np.array(interpolate.interp1d(self.eq.fluxsurfaces['theta_RZ'][self.x_grid.index(self.x_loc)][:-1],self.eq.fluxsurfaces['Bpol'][self.x_grid.index(self.x_loc)][:-1],bounds_error=False,fill_value='extrapolate')(self.theta_ref[:-1]))
        else:
            self.Bp_ref = self.eq.fluxsurfaces['Bpol'][self.x_grid.index(self.x_loc)][:-1]
        self.shape_deriv_ref = copy.deepcopy(self.shape_deriv)
        
        # add shape_analytic and shape_deriv_analytic for the contour-averaged analytic (Turnbull-)Miller shape parameters
        if analytic_shape:
            if self.verbose:
                print('Computing analytical Miller geometry quantities...')
            if self._param in ['miller','turnbull']:
                label_analytic = self._param
                param_analytic = self.param
                jr_analytic = self.param_jr
            else:
                label_analytic = 'turnbull'
                param_analytic = self.turnbull
                jr_analytic = self.turnbull_jr
            self.shape_analytic = []
            for key in self._params[label_analytic]['param_labels']:
                if '0' in key:
                    key = key.replace('0','o')
                if key in self.eq.derived:
                    self.shape_analytic.append(self.eq.derived[key][self.x_grid.index(self.x_loc)])
                elif key in self.eq.derived['miller_geo']:
                    self.shape_analytic.append(self.eq.derived['miller_geo'][key][self.x_grid.index(self.x_loc)])
            self.shape_deriv_analytic = []
            for key in self._params[label_analytic]['deriv_labels']:
                if '0' in key:
                    key = key.replace('0','o')
                if key in self.eq.derived:
                    self.shape_deriv_analytic.append(self.eq.derived[key][self.x_grid.index(self.x_loc)])
                elif key in self.eq.derived['miller_geo']:
                    self.shape_deriv_analytic.append(self.eq.derived['miller_geo'][key][self.x_grid.index(self.x_loc)])
                        
            self.R_geo, self.Z_geo, self.theta_ref_geo = param_analytic(self.shape_analytic, np.append(self.theta,self.theta[0]), norm=False)
            self.Bt_geo = interpolate.interp1d(self.eq.derived['psi'],self.eq.derived['fpol'],bounds_error=False)(self.eq.fluxsurfaces['psi'][self.x_grid.index(self.x_loc)])/(self.R_geo[:-1])

            self.R_ref_geo = np.array(interpolate_periodic(self.eq.fluxsurfaces['theta_RZ'][self.x_grid.index(self.x_loc)], self.eq.fluxsurfaces['R'][self.x_grid.index(self.x_loc)], self.theta_ref_geo))
            self.Z_ref_geo = np.array(interpolate_periodic(self.eq.fluxsurfaces['theta_RZ'][self.x_grid.index(self.x_loc)], self.eq.fluxsurfaces['Z'][self.x_grid.index(self.x_loc)], self.theta_ref_geo))
            self.Bt_ref_geo  = interpolate.interp1d(self.eq.derived['psi'],self.eq.derived['fpol'],bounds_error=False)(self.eq.fluxsurfaces['psi'][self.x_grid.index(self.x_loc)])/(self.R_ref_geo[:-1])

            self.Bp_geo = self.param_bpol(jr_analytic,self.shape_analytic, self.shape_deriv_analytic, self.theta, self.R_geo[:-1], self.dpsidr)#,method='analytic')
            self.Bp_ref_geo = np.array(interpolate_periodic(self.eq.fluxsurfaces['theta_RZ'][self.x_grid.index(self.x_loc)][:-1],self.eq.fluxsurfaces['Bpol'][self.x_grid.index(self.x_loc)][:-1],self.theta_ref_geo[:-1]))

        # optimize shape_derive based on L1 Bpol distance
        if opt_bpol:
            print('Optimising Bpol parameterization fit...')
            self.deriv_initial = copy.deepcopy(self.shape_deriv)

            self.shape_deriv = least_squares(self.cost_bpol, 
                                            self.deriv_initial, 
                                            bounds=self.deriv_bounds, 
                                            ftol=self.tolerance, 
                                            xtol=self.tolerance, 
                                            gtol=self.tolerance, 
                                            loss='soft_l1', 
                                            verbose=diag_lsq)['x']
            
            for i_key,key in enumerate(self.deriv_labels):
                self.eq.fluxsurfaces['fit_geo'][key+'_opt'] = copy.deepcopy(self.shape_deriv[i_key])
        
        # optimize shape_derive based on L1 Bpol, dRdr, dZdr, dRdtheta, dZdtheta distances
        if opt_deriv:
            print('Optimising derivative parameterization fits...')
            self.deriv_initial = copy.deepcopy(self.shape_deriv)

            # compute the reference flux-surface contour gradients on the theta basis
            dR_refdr, dR_refdtheta = np.gradient(np.array(self.eq.fluxsurfaces['fit_geo']['R_ref']),self.eq.fluxsurfaces['fit_geo']['r'],self.theta)
            self.dR_refdr, self.dR_refdtheta = dR_refdr[self.x_grid.index(self.x_loc)], dR_refdtheta[self.x_grid.index(self.x_loc)]

            dZ_refdr, dZ_refdtheta = np.gradient(np.array(self.eq.fluxsurfaces['fit_geo']['Z_ref']),self.eq.fluxsurfaces['fit_geo']['r'],self.theta)
            self.dZ_refdr, self.dZ_refdtheta = dZ_refdr[self.x_grid.index(self.x_loc)], dZ_refdtheta[self.x_grid.index(self.x_loc)]

            self.Jr_ref = self.R_ref[:-1]*(self.dR_refdr*self.dZ_refdtheta-self.dR_refdtheta*self.dZ_refdr)
            self.dldtheta_ref = np.sqrt(self.dR_refdtheta**2+self.dZ_refdtheta**2)

            self.shape_deriv = least_squares(self.cost_deriv, 
                                            self.deriv_initial, 
                                            bounds=self.deriv_bounds, 
                                            ftol=self.tolerance, 
                                            xtol=self.tolerance, 
                                            gtol=self.tolerance, 
                                            loss='soft_l1', 
                                            verbose=diag_lsq)['x']
            
            self.Bp_param_opt = self.param_bpol(self.param_jr, self.shape, self.shape_deriv, self.theta, self.R_param[:-1], self.dpsidr)
            
            for i_key,key in enumerate(self.deriv_labels):
                self.eq.fluxsurfaces['fit_geo'][key+'_opt'] = copy.deepcopy(self.shape_deriv[i_key])

            self.Bp_ref_opt = np.array(interpolate.interp1d(self.eq.fluxsurfaces['theta_RZ'][self.x_grid.index(self.x_loc)][:-1],self.eq.fluxsurfaces['Bpol'][self.x_grid.index(self.x_loc)][:-1],bounds_error=False,fill_value='extrapolate')(self.theta_ref[:-1]))

    # local equilibrium Miller parameterizations
    def miller(self,shape,theta,norm=False):
        """Compute Miller flux-surface parameterization given a set of shape parameters and a theta-grid.

        Args:
            shape (array): 1D array or list containing the Miller shape parameters [R0,Z0,r,kappa,delta].
            theta (array): 1D array containing the theta-grid.
            norm (bool, optional): Normalize the parameterized flux-surface coordinates by the center coordinates. Defaults to False.

        Returns:
            - R_param (array): 1D array containing the radial flux-surface parameterization coordinate sorted by theta_ref [0,2*pi].
            - Z_param (array): 1D array containing the vertical flux-surface parameterization coordinate sorted by theta_ref [0,2*pi].
            - theta_ref (array): 1D array containing the poloidal angle between radial and vertical flux-surface parameterization coordinates sorted ascending.
        """
        # flux-surface coordinate parameterization from [Miller PoP 5 (1998)] with Z0 added
        [self.R0,self.Z0,r,kappa,delta] = shape
        with np.errstate(invalid='ignore'):
            x = np.arcsin(delta)
        theta_R = theta + x * np.sin(theta)

        R_param = self.R0 + r * np.cos(theta_R)
        Z_param = self.Z0 + kappa * r * np.sin(theta)

        theta_ref = arctan2pi(Z_param-self.Z0,R_param-self.R0)
        if norm:
            R_param-=self.R0
            Z_param-=self.Z0
        
        return R_param, Z_param, theta_ref

    def miller_jr(self,shape,shape_deriv,theta,R,return_deriv=True):
        """Compute Miller flux-surface parameterization Jacobian given sets of shape and shape derivative parameters and a theta-grid.

        Args:
            shape (array): 1D array or list containing the Miller shape parameters [R0,Z0,r,kappa,delta].
            shape_deriv (array): 1D array or list containing the Miller shape derivative parameters [dR0dr,dZ0dr,s_kappa,s_delta].
            theta (array): 1D array containing the theta-grid.
            R (array): 1D array containing the radial flux-surface coordinate as output by miller().
            return_deriv (bool, optional): Switch to return the radial and poloidal derivatives in addition to the Jacobian or not. Defaults to True.

        Returns:
            dRdtheta (array, optional): 1D array containing the poloidal derivative of the radial flux-surface coordinate.
            dZdtheta (array, optional): 1D array containing the poloidal derivative of the vertical flux-surface coordinate.
            dRdr (array, optional): 1D array containing the radial derivative of the radial flux-surface coordinate.
            dZdr (array, optional): 1D array containing the radial derivative of the vertical flux-surface coordinate.
            J_r (array): 1D array containing the Jacobian for the Miller parameterization.
        """

        # define the parameters
        [R0,Z0,r,kappa,delta] = shape
        [dR0dr,dZ0dr,s_kappa,s_delta] = shape_deriv
        with np.errstate(invalid='ignore'):
            x = np.arcsin(delta)
        theta_R = theta + x * np.sin(theta)

        # compute the derivatives for the Jacobian
        dRdtheta = - r * np.sin(theta_R)*(1 + x * np.cos(theta))
        dZdtheta = kappa * r * np.cos(theta)
        dRdr = dR0dr + np.cos(theta_R) - s_delta * np.sin(theta) * np.sin(theta_R)
        dZdr = dZ0dr + kappa * (s_kappa + 1) * np.sin(theta)

        # compute the Jacobian
        J_r = R * (dRdr * dZdtheta - dRdtheta * dZdr)

        if return_deriv:
            return dRdtheta, dZdtheta, dRdr, dZdr, J_r
        else:
            return J_r

    def turnbull(self,shape,theta,norm=False):
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
        [self.R0,self.Z0,r,kappa,delta,zeta] = shape
        with np.errstate(invalid='ignore'):
            x = np.arcsin(delta)
        theta_R = theta + x * np.sin(theta)
        theta_Z = theta + zeta * np.sin(2 * theta)

        R_param = self.R0 + r * np.cos(theta_R)
        Z_param = self.Z0 + kappa * r * np.sin(theta_Z)
        theta_ref = arctan2pi(Z_param-self.Z0,R_param-self.R0)
        
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

    def turnbull_tilt(self,shape,theta,norm=False):
        """Compute Turnbull-tilt flux-surface parameterization given a set of shape parameters and a theta-grid.

        Args:
            shape (array): 1D array or list containing the Turnbull-tilt shape parameters [R0,Z0,r,kappa,delta,zeta,tilt].
            theta (array): 1D array containing the theta-grid.
            norm (bool, optional): Normalize the parameterized flux-surface coordinates by the center coordinates. Defaults to False.

        Returns:
            - R_param (array): 1D array containing the radial flux-surface parameterization coordinate sorted by theta_ref [0,2*pi].
            - Z_param (array): 1D array containing the vertical flux-surface parameterization coordinate sorted by theta_ref [0,2*pi].
            - theta_ref (array): 1D array containing the poloidal angle between radial and vertical flux-surface parameterization coordinates sorted ascending.
        """
        # flux-surface coordinate parameterization based on [Turnbull PoP 6 (1999)] with addition of tilt
        [self.R0,self.Z0,r,kappa,delta,zeta,tilt] = shape
        with np.errstate(invalid='ignore'):
            x = np.arcsin(delta)
        theta_R = theta + x * np.sin(theta) + tilt
        theta_Z = theta + zeta * np.sin(2 * theta)

        R_param = self.R0 + r * np.cos(theta_R)
        Z_param = self.Z0 + kappa * r * np.sin(theta_Z)

        theta_ref = arctan2pi(Z_param-self.Z0,R_param-self.R0)
        
        if norm:
            R_param-=self.R0
            Z_param-=self.Z0

        return R_param, Z_param, theta_ref

    def turnbull_tilt_jr(self,shape,shape_deriv,theta,R,return_deriv=True):
        """Compute Turnbull-tilt flux-surface parameterization Jacobian given sets of shape and shape derivative parameters and a theta-grid.

        Args:
            shape (array): 1D array or list containing the Turnbull-tilt shape parameters [R0,Z0,r,kappa,delta,zeta,tilt].
            shape_deriv (array): 1D array or list containing the Turnbull-tilt shape derivative parameters [dR0dr,dZ0dr,s_kappa,s_delta,s_zeta,s_tilt].
            theta (array): 1D array containing the theta-grid.
            R (array): 1D array containing the radial flux-surface coordinate as output by turnbull_tilt().
            return_deriv (bool, optional): Switch to return the radial and poloidal derivatives in addition to the Jacobian or not. Defaults to True.

        Returns:
            dRdtheta (array, optional): 1D array containing the poloidal derivative of the radial flux-surface coordinate.
            dZdtheta (array, optional): 1D array containing the poloidal derivative of the vertical flux-surface coordinate.
            dRdr (array, optional): 1D array containing the radial derivative of the radial flux-surface coordinate.
            dZdr (array, optional): 1D array containing the radial derivative of the vertical flux-surface coordinate.
            J_r (array): 1D array containing the Jacobian for the Miller parameterization.
        """
        # define the parameters
        [R0,Z0,r,kappa,delta,zeta,tilt] = shape
        [dR0dr,dZ0dr,s_kappa,s_delta,s_zeta,s_tilt] = shape_deriv
        with np.errstate(invalid='ignore'):
            x = np.arcsin(delta)
        theta_R = theta + x * np.sin(theta) + tilt
        dtheta_Rdtheta = 1 + x * np.cos(theta)
        theta_Z = theta + zeta * np.sin(2 * theta)
        dtheta_Zdtheta = 1 + 2 * zeta * np.cos(2 * theta)
        
        # compute the derivatives for the Jacobian
        dRdtheta = - r * np.sin(theta_R) * dtheta_Rdtheta
        dZdtheta = kappa * r * np.cos(theta_Z) * dtheta_Zdtheta
        dRdr = dR0dr + np.cos(theta_R) - (s_delta * np.sin(theta) + s_tilt) * np.sin(theta_R)
        dZdr = dZ0dr + kappa * ((s_kappa + 1) * np.sin(theta_Z) + s_zeta * np.sin(2 * theta) * np.cos(theta_Z))
        
        # compute Mercier-Luc arclength derivative and |grad r|
        J_r = R * (dRdr * dZdtheta - dRdtheta * dZdr)

        if return_deriv:
            return dRdtheta, dZdtheta, dRdr, dZdr, J_r
        else:
            return J_r

    # Harmonic expansion method Miller-like parameterizations
    def fourier(self,shape,theta,norm=None):
        """Compute general Fourier expansion flux-surface parameterization given a set of shape parameters and a theta-grid.

        Args:
            shape (array): 1D array or list containing the Fourier expansion shape parameters [R0,Z0,aR_1,bR_1,aZ_1,bZ_1,...,aR_n,bR_n,aZ_n,bZ_n].
            theta (array): 1D array containing the theta-grid.
            norm (bool, optional): Normalize the parameterized flux-surface coordinates by the center coordinates. Defaults to False.

        Returns:
            - R_param (array): 1D array containing the radial flux-surface parameterization coordinate sorted by theta_ref [0,2*pi].
            - Z_param (array): 1D array containing the vertical flux-surface parameterization coordinate sorted by theta_ref [0,2*pi].
            - theta_ref (array): 1D array containing the poloidal angle between radial and vertical flux-surface parameterization coordinates sorted ascending.
        """
        # flux-surface coordinate parameterization from [Candy PPCF 51 (2009)]
        [self.R0, self.Z0] = shape[:2] # only difference from [Candy PPCF 51 (2009)], R0 = 0.5*aR_0, Z0 = 0.5*aZ_0
        R_fourier, Z_fourier = 0,0
        N = int((len(shape)-2)/4)
        for n in range(1,N+1):
            aR_n = shape[2 + (n-1)*4]
            bR_n = shape[3 + (n-1)*4]
            aZ_n = shape[4 + (n-1)*4]
            bZ_n = shape[5 + (n-1)*4]
            R_fourier += aR_n * np.cos(n * theta) + bR_n * np.sin(n * theta)
            Z_fourier += aZ_n * np.cos(n * theta) + bZ_n * np.sin(n * theta)

        R_param = self.R0 + R_fourier
        Z_param = self.Z0 + Z_fourier

        theta_ref = arctan2pi(Z_param-self.Z0,R_param-self.R0)

        if norm:
            R_param-=self.R0
            Z_param-=self.Z0

        return R_param, Z_param, theta_ref

    def fourier_jr(self,shape,shape_deriv,theta,R,return_deriv=True):
        """Compute general Fourier expansion flux-surface parameterization Jacobian given sets of shape and shape derivative parameters and a theta-grid.

        Args:
            shape (array): 1D array or list containing the Fourier expansion shape parameters [R0,Z0,aR_1,bR_1,aZ_1,bZ_1,...,aR_n,bR_n,aZ_n,bZ_n].
            shape_deriv (array): 1D array or list containing the Fourier expansion shape derivative parameters [dR0dr,dZ0dr,daR_1dr,dbR_1dr,daZ_1dr,dbZ_1dr,...,daR_ndr,dbR_ndr,daZ_ndr,dbZ_ndr].
            theta (array): 1D array containing the theta-grid.
            R (array): 1D array containing the radial flux-surface coordinate as output by fourier().
            return_deriv (bool, optional): Switch to return the radial and poloidal derivatives in addition to the Jacobian or not. Defaults to True.

        Returns:
            dRdtheta (array, optional): 1D array containing the poloidal derivative of the radial flux-surface coordinate.
            dZdtheta (array, optional): 1D array containing the poloidal derivative of the vertical flux-surface coordinate.
            dRdr (array, optional): 1D array containing the radial derivative of the radial flux-surface coordinate.
            dZdr (array, optional): 1D array containing the radial derivative of the vertical flux-surface coordinate.
            J_r (array): 1D array containing the Jacobian for the Miller parameterization.
        """
        # flux-surface coordinate parameterization from [Candy PPCF 51 (2009)]
        [dRdr, dZdr] = shape_deriv[:2]
        dRdtheta, dZdtheta = 0,0
        N = int((len(shape)-2)/4)
        for n in range(1,N+1):
            aR_n = shape[2 + (n-1)*4]
            bR_n = shape[3 + (n-1)*4]
            aZ_n = shape[4 + (n-1)*4]
            bZ_n = shape[5 + (n-1)*4]

            daR_ndr = shape_deriv[2 + (n-1)*4]
            dbR_ndr = shape_deriv[3 + (n-1)*4]
            daZ_ndr = shape_deriv[4 + (n-1)*4]
            dbZ_ndr = shape_deriv[5 + (n-1)*4]

            dRdtheta += - aR_n * n * np.sin(n * theta) + bR_n * n * np.cos(n * theta)
            dZdtheta += - aZ_n * n * np.sin(n * theta) + bZ_n * n * np.cos(n * theta)

            dRdr += daR_ndr * np.cos(n * theta) + dbR_ndr * np.sin(n * theta)
            dZdr += daZ_ndr * np.cos(n * theta) + dbZ_ndr * np.sin(n * theta)

        # compute Jacobian
        J_r = R * (dRdr * dZdtheta - dRdtheta * dZdr)

        if return_deriv:
            return dRdtheta, dZdtheta, dRdr, dZdr, J_r
        else:
            return J_r

    def miller_general(self,shape,theta,norm=False):
        """Compute GENE miller_general flux-surface parameterization given a set of shape parameters and a theta-grid.

        Args:
            shape (array): 1D array or list containing the GENE miller_general shape parameters [cN_0,...cN_n,sN_0,...,sN_n].
            theta (array): 1D array containing the theta-grid.
            norm (bool, optional): Normalize the parameterized flux-surface coordinates by the center coordinates of the flux-surface of interest. Defaults to False.

        Returns:
            - R_param (array): 1D array containing the radial flux-surface parameterization coordinate sorted by theta_ref [0,2*pi].
            - Z_param (array): 1D array containing the vertical flux-surface parameterization coordinate sorted by theta_ref [0,2*pi].
            - theta_ref (array): 1D array containing the poloidal angle between radial and vertical flux-surface parameterization coordinates sorted ascending.
        """
        # Fourier expansion flux-surface coordinate parameterization from GENE
        n_harmonics = int(len(shape)/2)
        # set a fixed R0,Z0 center for the local equilibrium
        self.R0 = self.eq.fluxsurfaces['R0'][self.x_grid.index(self.x_loc)]
        self.Z0 = self.eq.fluxsurfaces['Z0'][self.x_grid.index(self.x_loc)]

        # enfore sN_0 = 0.0
        shape[n_harmonics] = 0.0
        cN = shape[:n_harmonics]
        sN = shape[n_harmonics:]
        rshape = np.zeros_like(theta)
        for n,c in enumerate(cN):
            rshape += cN[n] * np.cos(n*theta) + sN[n] * np.sin(n*theta)
        R_param = self.R0 + rshape * np.cos(theta)
        Z_param = self.Z0 + rshape * np.sin(theta)

        theta_ref = arctan2pi(Z_param-self.Z0,R_param-self.R0)

        if norm:
            R_param-=self.R0
            Z_param-=self.Z0

        return R_param, Z_param, theta_ref

    def miller_general_jr(self,shape,shape_deriv,theta,R,return_deriv=True):
        """Compute GENE miller_general flux-surface parameterization Jacobian given sets of shape and shape derivative parameters and a theta-grid.

        Args:
            shape (array): 1D array or list containing the Fourier expansion shape parameters [cN_0,...cN_n,sN_0,...,sN_n].
            shape_deriv (array): 1D array or list containing the Fourier expansion shape derivative parameters [dcN_0dr,...dcN_ndr,dsN_0dr,...,dsN_ndr].
            theta (array): 1D array containing the theta-grid.
            R (array): 1D array containing the radial flux-surface coordinate as output by miller_general().
            return_deriv (bool, optional): Switch to return the radial and poloidal derivatives in addition to the Jacobian or not. Defaults to True.

        Returns:
            dRdtheta (array, optional): 1D array containing the poloidal derivative of the radial flux-surface coordinate.
            dZdtheta (array, optional): 1D array containing the poloidal derivative of the vertical flux-surface coordinate.
            dRdr (array, optional): 1D array containing the radial derivative of the radial flux-surface coordinate.
            dZdr (array, optional): 1D array containing the radial derivative of the vertical flux-surface coordinate.
            J_r (array): 1D array containing the Jacobian for the Miller parameterization.
        """
        # Fourier expansion flux-surface coordinate parameterization from GENE
        n_harmonics = int(len(shape)/2)
        cN = shape[:n_harmonics]
        sN = [0.0]+list(shape)[n_harmonics+1:]
        cN_dr = shape_deriv[:n_harmonics]
        sN_dr = [0.0]+list(shape_deriv)[n_harmonics+1:]
        rShape = np.zeros_like(theta)
        drShapedr = np.zeros_like(theta)
        drShapedtheta = np.zeros_like(theta)
        for n,c in enumerate(cN):
            rShape += cN[n] * np.cos(n*theta) + sN[n] * np.sin(n*theta)
            drShapedr += cN_dr[n] * np.cos(n*theta) + sN_dr[n] * np.sin(n*theta)
            drShapedtheta += -cN[n] * n * np.sin(n*theta) + sN[n] * n * np.cos(n*theta)

        dRdtheta = drShapedtheta * np.cos(theta) - rShape * np.sin(theta)
        dZdtheta = drShapedtheta * np.sin(theta) + rShape * np.cos(theta)

        dRdr = drShapedr * np.cos(theta)
        dZdr = drShapedr * np.sin(theta)

        # compute Jacobian
        J_r = R * (dRdr * dZdtheta - dRdtheta * dZdr)

        if return_deriv:
            return dRdtheta, dZdtheta, dRdr, dZdr, J_r
        else:
            return J_r

    def mxh(self,shape,theta,norm=None):
        """Compute MXH flux-surface parameterization given a set of shape parameters and a theta-grid.

        Args:
            shape (array): 1D array or list containing the MXH shape parameters [R0,Z0,r,kappa,c_0,c_1,s_1,...,c_n,s_n].
            theta (array): 1D array containing the theta-grid.
            norm (bool, optional): Normalize the parameterized flux-surface coordinates by the center coordinates. Defaults to False.

        Returns:
            - R_param (array): 1D array containing the radial flux-surface parameterization coordinate sorted by theta_ref [0,2*pi].
            - Z_param (array): 1D array containing the vertical flux-surface parameterization coordinate sorted by theta_ref [0,2*pi].
            - theta_ref (array): 1D array containing the poloidal angle between radial and vertical flux-surface parameterization coordinates sorted ascending.
        """
        # flux-surface coordinate parameterization from [Arbon PPCF 63 (2020)]
        # Compute major radius, elevation, minor radius and elongation from bounding box
        self.R0 = (np.max(self.fs['R'][:-1])+np.min(self.fs['R'][:-1]))/2
        self.Z0 = (np.max(self.fs['Z'][:-1])+np.min(self.fs['Z'][:-1]))/2
        r = (np.max(self.fs['R'][:-1])-np.min(self.fs['R'][:-1]))/2
        kappa = ((np.max(self.fs['Z'][:-1])-np.min(self.fs['Z'][:-1]))/2)/r
        shape[:4] = [self.R0,self.Z0,r,kappa]
        c_0 = shape[4]
        theta_R = theta + c_0
        N = int((len(shape)-5)/2)
        for n in range(1,N+1):
            c_n = shape[5 + (n-1)*2]
            s_n = shape[6 + (n-1)*2]
            theta_R += c_n * np.cos(n * theta) + s_n * np.sin(n * theta)

        R_param = self.R0 + r * np.cos(theta_R)
        Z_param = self.Z0 + kappa * r * np.sin(theta)

        theta_ref = arctan2pi(Z_param-self.Z0,R_param-self.R0)

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
        # Compute major radius, elevation, minor radius and elongation from bounding box
        R0 = (np.max(self.fs['R'][:-1])+np.min(self.fs['R'][:-1]))/2
        Z0 = (np.max(self.fs['Z'][:-1])+np.min(self.fs['Z'][:-1]))/2
        r = (np.max(self.fs['R'][:-1])-np.min(self.fs['R'][:-1]))/2
        kappa = ((np.max(self.fs['Z'][:-1])-np.min(self.fs['Z'][:-1]))/2)/r
        c_0 = shape[4]
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

        dRdtheta = - r * np.sin(theta_R) * dtheta_Rdtheta
        dZdtheta = kappa * r * np.cos(theta)
        dRdr = dR0dr + np.cos(theta_R) - np.sin(theta_R) * dtheta_Rdr
        dZdr = dZ0dr + kappa * (s_kappa + 1) * np.sin(theta)

        # compute Mercier-Luc arclength derivative and |grad r|
        J_r = R * (dRdr * dZdtheta - dRdtheta * dZdr)

        if return_deriv:
            return dRdtheta, dZdtheta, dRdr, dZdr, J_r
        else:
            return J_r

    # bpol parameterization
    def param_bpol(self,param_jr,shape,shape_deriv,theta,R,dpsidr,method='jacobian',verbose=True):
        """Compute Bpol parameterization for a given Jacobian, sets of shape and shape derivative parameters and dpsi/dr on a given theta-grid

        Args:
            param_jr (array): 1D array containing the param Jacobian on the theta-grid basis.
            shape (array): 1D array or list containing the param shape parameters.
            shape_deriv (array): 1D array or list containing the param shape derivative parameters.
            theta (array): 1D array containing the theta-grid.
            R (array): 1D array containing the radial flux-surface coordinate as output by param().
            dpsidr (floar): Float value of dpsi/dr at the flux-surface of interest.
            method (str, optional): Token setting the Bpol calculation method. For the Miller, Turnbull-Miller and Turnbull-tilt parameterizations an analytic expression for Bpol is available, otherwise default to Jacobian calculation. Defaults to 'jacobian'.
            verbose (bool, optional): Switch to turn on/off print statements. Defaults to True.

        Returns:
            array: 1D array containing the parameterized Bpol values on the theta-grid basis. 
        """
        if method=='analytic':
            if param_jr in [self.miller_jr,self.turnbull_jr,self.turnbull_tilt_jr]:
                if param_jr==self.miller_jr:
                    # define the parameters
                    [R0,Z0,r,kappa,delta] = shape
                    [dR0dr,dZ0dr,s_kappa,s_delta] = shape_deriv
                    with np.errstate(invalid='ignore'):
                        x = np.arcsin(delta)
                    theta_R = theta + x * np.sin(theta)
                    dtheta_Rdtheta = 1 + x * np.cos(theta)
                    
                    Bp_nom = np.sqrt(np.sin(theta_R)**2 * dtheta_Rdtheta**2 + kappa**2 * np.cos(theta)**2)
                    Bp_denom = kappa * (np.cos(x * np.sin(theta)) + dR0dr * np.cos(theta) + (dZ0dr/kappa)*(np.sin(theta_R)*dtheta_Rdtheta) + (s_kappa - s_delta * np.cos(theta) + (1+ s_kappa) * x * np.cos(theta)) * np.sin(theta) * np.sin(theta_R))

                elif param_jr == self.turnbull_jr:
                    # define the parameters
                    [R0,Z0,r,kappa,delta,zeta] = shape
                    [dR0dr,dZ0dr,s_kappa,s_delta,s_zeta] = shape_deriv
                    with np.errstate(invalid='ignore'):
                        x = np.arcsin(delta)
                    theta_R = theta + x * np.sin(theta)
                    dtheta_Rdtheta = 1 + x * np.cos(theta)
                    theta_Z = theta + zeta * np.sin(2 * theta)
                    dtheta_Zdtheta = 1 + 2 * zeta * np.cos(2 * theta)

                    Bp_nom = np.sqrt(np.sin(theta_R)**2 * dtheta_Rdtheta**2 + kappa**2 * np.cos(theta_Z)**2 * dtheta_Zdtheta**2)
                    Bp_denom = kappa * np.cos(theta_Z) * dtheta_Zdtheta * (dR0dr + np.cos(theta_R) - s_delta * np.sin(theta) * np.sin(theta_R)) + np.sin(theta_R) * dtheta_Rdtheta * (dZ0dr + kappa * ((s_kappa + 1) * np.sin(theta_Z) + s_zeta * np.sin(2 * theta) * np.cos(theta_Z)))

                elif param_jr == self.turnbull_tilt_jr:
                    # define the parameters
                    [R0,Z0,r,kappa,delta,zeta,tilt] = shape
                    [dR0dr,dZ0dr,s_kappa,s_delta,s_zeta,s_tilt] = shape_deriv
                    with np.errstate(invalid='ignore'):
                        x = np.arcsin(delta)
                    theta_R = theta + x * np.sin(theta) + tilt
                    dtheta_Rdtheta = 1 + x * np.cos(theta)
                    theta_Z = theta + zeta * np.sin(2 * theta)
                    dtheta_Zdtheta = 1 + 2 * zeta * np.cos(2 * theta)

                    Bp_nom = np.sqrt(np.sin(theta_R)**2 * dtheta_Rdtheta**2 + kappa**2 * np.cos(theta_Z)**2 * dtheta_Zdtheta**2)
                    Bp_denom = kappa * np.cos(theta_Z) * dtheta_Zdtheta * (dR0dr + np.cos(theta_R) - s_delta * np.sin(theta) * np.sin(theta_R) + s_tilt) + np.sin(theta_R) * dtheta_Rdtheta * (dZ0dr + kappa * ((s_kappa + 1) * np.sin(theta_Z) + s_zeta * np.sin(2 * theta) * np.cos(theta_Z)))

                grad_r_norm =  Bp_nom / Bp_denom
            else:
                method = 'jacobian'
                if verbose:
                    print("Bpol method `analytic' is not available for the chosen parameterization, switching to `jacobian'...")
        
        elif method=='jacobian':
            dRdtheta, dZdtheta, dRdr, dZdr, J_r = param_jr(shape,shape_deriv,theta,R,return_deriv=True)

            # compute the Mercier-Luc arclength derivative and |grad r|
            dl_dtheta = np.sqrt(dRdtheta**2 + dZdtheta**2)
            grad_r_norm = (R/J_r)*dl_dtheta

        # Poloidal magnetic flux density
        Bp_param = (dpsidr / R) * grad_r_norm
        
        return Bp_param

    # cost functions
    def cost_param(self,params):
        """Compute the cost function f_i(x) for flux-surface coordinates.

        Args:
            params (array): 1D array or list containing the param shape parameters.

        Returns:
            array: 1D array containing the weighted sum of squared error cost function with a length of 3*n_theta
        """
        # compute the flux-surface parameterization for a given shape set `params`
        self.R_param, self.Z_param, self.theta_ref = self.param(params, self.theta, norm=True)
        
        # compute the reference flux-surface coordinates
        if self._param in ['mxh']:
            self.R_ref = self.fs['R'][:-1] - self.R0
            self.Z_ref = self.fs['Z'][:-1] - self.Z0
        else:
            # interpolate the actual flux-surface contour to the theta basis 
            self.R_ref = np.array(interpolate_periodic(self.fs['theta_RZ'][:-1], self.fs['R'][:-1],self.theta_ref)) - self.R0
            self.Z_ref = np.array(interpolate_periodic(self.fs['theta_RZ'][:-1], self.fs['Z'][:-1],self.theta_ref)) - self.Z0

        # define the cost function
        L1_norm = np.abs(np.array([self.R_param,self.Z_param])-np.array([self.R_ref,self.Z_ref])).flatten()
        L2_norm = np.sqrt((self.R_param-self.R_ref)**2+(self.Z_param-self.Z_ref)**2)

        cost = self.n_theta*np.hstack((L1_norm,L2_norm))

        return cost

    def cost_bpol(self,params):
        """Compute the cost function f_i(x) for flux-surface Bpol.

        Args:
            params (array): 1D array or list containing the param shape derivative parameters.

        Returns:
            array: 1D array containing the weighted sum of squared error cost function with a length of n_theta
        """
        self.Bp_param_opt = self.param_bpol(self.param_jr, self.shape, params, self.theta, self.R_param[:-1], self.dpsidr)
        
        theta_RZ = self.eq.fluxsurfaces['theta_RZ'][self.x_grid.index(self.x_loc)][:-1]
        Bp_RZ = self.eq.fluxsurfaces['Bpol'][self.x_grid.index(self.x_loc)][:-1]
        self.Bp_ref_opt = np.array(interpolate.interp1d(theta_RZ,Bp_RZ,bounds_error=False,fill_value='extrapolate')(self.theta_ref[:-1]))

        cost = self.n_theta*np.abs(self.Bp_param_opt-self.Bp_ref_opt)

        return cost

    def cost_deriv(self,params):
        """Compute the cost function f_i(x) for flux-surface coordinate derivatives and Bpol.

        Args:
            params (array): 1D array or list containing the param shape derivative parameters.

        Returns:
            array: 1D array containing the weighted sum of squared error cost function with a length of 3*n_theta
        """
        # compute the flux-surface parameterization derivatives for given shape sets `self.shape` and `params`
        self.dRdtheta, self.dZdtheta, self.dRdr, self.dZdr, self.J_r = self.param_jr(self.shape,params,self.theta,self.R_param[:-1],return_deriv=True)

        # Bpol stuff
        self.Bp_param_opt = self.param_bpol(self.param_jr, self.shape, params, self.theta, self.R_param[:-1], self.dpsidr)
        
        theta_RZ = self.eq.fluxsurfaces['theta_RZ'][self.x_grid.index(self.x_loc)][:-1]
        Bp_RZ = self.eq.fluxsurfaces['Bpol'][self.x_grid.index(self.x_loc)][:-1]
        self.Bp_ref_opt = np.array(interpolate.interp1d(theta_RZ,Bp_RZ,bounds_error=False,fill_value='extrapolate')(self.theta_ref[:-1]))

        # define the cost function
        L1_norm_ddr = np.abs(np.array([self.dRdr,self.dZdr])-np.array([self.dR_refdr,self.dZ_refdr])).flatten()
        L1_norm_ddtheta = np.abs(np.array([self.dRdtheta,self.dZdtheta])-np.array([self.dR_refdtheta,self.dZ_refdtheta])).flatten()
        L1_norm_bpol = np.abs(self.Bp_param_opt-self.Bp_ref_opt)

        cost = self.n_theta*np.hstack((L1_norm_ddr,L1_norm_ddtheta,L1_norm_bpol))

        return cost

    # extraction tools
    @classmethod
    def extract_analytic_shape(cls, fluxsurface):
        """Extract Turnbull-Miller geometry parameters [Turnbull PoP 6 1113 (1999)] from a flux-surface contour. Adapted from 'extract_miller_from_eqdsk.py' by D. Told.

        Args:
            `fluxsurface` (dict): flux-surface data containing R, Z, R0, Z0, r, theta_RZ (poloidal angle between R and Z) and the R_Zmax,Z_max and R_Zmin,Z_min coordinates.

        Returns:
            (dict): the Miller parameters and R_miller,Z_miller on the same poloidal basis as the input flux-surface.
        """
        miller_geo = {}

        # compute triangularity (delta) and elongation (kappa) of flux-surface
        miller_geo['delta_u'] = (fluxsurface['R0'] - fluxsurface['R_Zmax'])/fluxsurface['r']
        miller_geo['delta_l'] = (fluxsurface['R0'] - fluxsurface['R_Zmin'])/fluxsurface['r']
        miller_geo['delta'] = (miller_geo['delta_u']+miller_geo['delta_l'])/2
        x = np.arcsin(miller_geo['delta'])
        
        miller_geo['kappa_u'] = (fluxsurface['Z_max'] - fluxsurface['Z_Rmax'])/fluxsurface['r']
        miller_geo['kappa_l'] = (fluxsurface['Z_Rmax'] - fluxsurface['Z_min'])/fluxsurface['r']
        miller_geo['kappa'] = (fluxsurface['Z_max'] - fluxsurface['Z_min'])/(2*fluxsurface['r'])

        # generate theta grid and interpolate the flux-surface trace to the Miller parameterization
        theta_zeta = np.array([0.25*np.pi,0.75*np.pi,1.25*np.pi,1.75*np.pi])
        theta_miller = fluxsurface['theta_RZ'] if len(fluxsurface['theta_RZ']) > 1 else theta_zeta
        R_miller = fluxsurface['R0'] + fluxsurface['r']*np.cos(theta_miller+x*np.sin(theta_miller))
        Z_miller = fluxsurface['Z0'] + miller_geo['kappa']*fluxsurface['r']*np.sin(theta_miller)
        if len(fluxsurface['R']) > 1 and len(fluxsurface['Z']) > 1:
            Z_miller = np.hstack((
                interpolate.interp1d(fluxsurface['R'][:np.argmin(fluxsurface['R'])],fluxsurface['Z'][:np.argmin(fluxsurface['R'])],bounds_error=False)(R_miller[:find(np.min(fluxsurface['R']),R_miller)]),
                interpolate.interp1d(fluxsurface['R'][np.argmin(fluxsurface['R']):],fluxsurface['Z'][np.argmin(fluxsurface['R']):],bounds_error=False)(R_miller[find(np.min(fluxsurface['R']),R_miller):])
            ))

        # derive the squareness (zeta) from the Miller parameterization
        Z_zeta = interpolate.interp1d(theta_miller,Z_miller,bounds_error=False,fill_value='extrapolate')(theta_zeta)

        # invert the Miller parameterization of Z, holding off on subtracting theta/sin(2*theta)
        zeta_4q = np.arcsin((Z_zeta-fluxsurface['Z0'])/(miller_geo['kappa']*fluxsurface['r']))/np.sin(2*theta_zeta)

        # apply a periodic correction for the arcsin of the flux-surface quadrants
        zeta_4q = np.array([1,-1,-1,1])*zeta_4q+np.array([0,-np.pi,-np.pi,0])

        miller_geo['zeta_uo'] = zeta_4q[0] - (theta_zeta[0]/np.sin(2*theta_zeta[0]))
        miller_geo['zeta_ui'] = zeta_4q[1] - (theta_zeta[1]/np.sin(2*theta_zeta[1]))
        miller_geo['zeta_li'] = zeta_4q[2] - (theta_zeta[1]/np.sin(2*theta_zeta[1]))
        miller_geo['zeta_lo'] = zeta_4q[3] - (theta_zeta[0]/np.sin(2*theta_zeta[0]))

        # compute the average squareness of the flux-surface
        miller_geo['zeta'] = 0.25*(miller_geo['zeta_uo']+miller_geo['zeta_ui']+miller_geo['zeta_li']+miller_geo['zeta_lo'])

        miller_geo['R_miller'] = np.array([fluxsurface['R0']])
        miller_geo['Z_miller'] = np.array([fluxsurface['Z0']])
        if len(fluxsurface['theta_RZ']) > 1:
            miller_geo['R_miller'] = R_miller
            miller_geo['Z_miller'] = fluxsurface['Z0']+miller_geo['kappa']*fluxsurface['r']*np.sin(fluxsurface['theta_RZ']+miller_geo['zeta']*np.sin(2*fluxsurface['theta_RZ']))

        return miller_geo

    def extract_fft_shape(self,fluxsurface,R0,Z0,theta,n_harmonics):
        """Extract Fourier coefficients from a flux-surface contour from a truncated FFT.

        Args:
            fluxsurface (dict): Dict containing containing R, Z, R0, Z0, r, theta_RZ (poloidal angle between R and Z) and the R_Zmax,Z_max and R_Zmin,Z_min coordinates of a flux-surface.
            R0 (float): Float value of the flux-surface center radial coordinate.
            Z0 (float): Float value of the flux-surface center vertical coordinate.
            theta (array): 1D array containing the theta-grid.
            n_harmonics (int): Integer value of the number of harmonics at which to truncate the FFT.

        Returns:
            - cN (array): 1D array containing the cos() Fourier coefficients.
            - sN (array): 1D array containing the sin() Fourier coefficients.
        """
        R_r = fluxsurface['R']
        Z_r = fluxsurface['Z']
        theta_RZ = fluxsurface['theta_RZ']

        # sort the flux-surface coordinates between 0 and 2*pi
        theta_RZ, R_r, Z_r = zipsort(theta_RZ,R_r,Z_r)
        aN_r = np.sqrt((R_r-R0)**2+(Z_r-Z0)**2)
        aN_equidistant = interpolate_periodic(theta_RZ,aN_r,theta)

        # take FFT of the flux-surface distance aN
        fft_aN = np.fft.fft(aN_equidistant)
        
        # truncate at n_harmonics, NOTE: for `miller_general` in GENE n_harmonics <=32
        cN = 2.0*np.real(fft_aN)[0:n_harmonics]/len(theta)
        cN[0] *= 0.5
        sN = -2.0*np.imag(fft_aN)[0:n_harmonics]/len(theta)
        sN[0] *= 0.5

        return cN, sN

    # auxiliary functions
    def printer(self,printer,shape,labels,shape_deriv,labels_deriv,lref='a'):
        """Print magnetic geometry related input values for gyrokinetic codes.

        Args:
            printer (str): String token specifying which code format values are to be printed in.
            shape (array): 1D array or list containing the param shape parameters.
            labels (list): list of str labels of shape labels for parameterized flux-surface.
            shape_deriv (array): 1D array or list containing the param shape derivative parameters.
            labels_deriv (list): list of str labels of shape derivative labels for parameterized Bpol.
            lref (str, optional): String token specifying the reference length used to normalize the output. Defaults to 'a'.
        """
        print('Printing input values for {} code...'.format(printer))
        i_x_loc = self.x_grid.index(self.x_loc)

        fs = {}
        for i_key,key in enumerate(labels):
            fs.update({key:shape[i_key]})
        for i_key,key in enumerate(labels_deriv):
            fs.update({key:shape_deriv[i_key]})
        # get the other derived quantities required for print
        for key in ['q','s','fpol']:
            if key in self.eq.fluxsurfaces:
                fs.update({key:self.eq.fluxsurfaces[key][i_x_loc]})
            elif key in self.eq.derived:
                fs.update({key:self.eq.derived[key][i_x_loc]})

        if printer.lower() == 'gene':
            if lref == 'a':
                Lref = self.eq.derived['a']
            elif lref =='R':
                Lref = fs['R0']
            
            if self._param in ['miller','turnbull','turnbull_tilt']:
                print('&geometry')
                print('trpeps  = {}\t! {} = {}'.format((fs['r']/fs['R0']),self.x_label,self.x_loc))
                print('q0      = {}'.format(fs['q']))
                print('shat    = {}'.format(fs['s']))
                print('amhd    = {}'.format(-1))
                print('drR     = {}'.format(fs['dR0dr']))
                print('drZ     = {}'.format(fs['dZ0dr']))
                print('kappa   = {}'.format(fs['kappa']))
                print('s_kappa = {}'.format(fs['s_kappa']))
                print('delta   = {}'.format(fs['delta']))
                print('s_delta = {}'.format(fs['s_delta']))
                if self._param in ['turnbull','turnbull_tilt']:
                    print('zeta    = {}'.format(fs['zeta']))
                    print('s_zeta  = {}'.format(fs['s_zeta']))
                if self._param == 'turnbull_tilt':
                    print('tilt    = {}'.format(fs['tilt']))
                    print('s_tilt  = {}'.format(fs['s_tilt']))
                print('minor_r = {}'.format(self.eq.derived['a']/Lref))#1.0,
                print('major_R = {}'.format(fs['R0']/Lref))#R0/a,
                print('/')
            
                print('\nAdditional information:')
                print('&units')
                if lref == 'a':
                    print('Lref    = {} !for Lref=a convention'.format(Lref))
                elif lref == 'R':
                    print('Lref    = {} !for Lref=R0 convention'.format(fs['R0']))
                print('Bref    = {}'.format(np.abs(fs['fpol']/fs['R0'])))
                print('Bunit    = {}'.format(np.abs(self.Bunit)))
                print('Bunit/Bref    = {}'.format(np.abs(self.Bunit/(fs['fpol']/fs['R0']))))
                print('a*drho_tor/dr    = {} !for gradient coversion omn(rho_tor) -> a/Ln'.format(self.eq.derived['a']*self.dxdr[i_x_loc]))
                print('R0*drho_tor/dr    = {} !for gradient coversion omn(rho_tor) -> R0/Ln'.format(fs['R0']*self.dxdr[i_x_loc]))
                print('/')
            
            elif self._param in ['mxh']:
                cN_m = []
                cNdr_m = []
                sN_m = [0.0]
                sNdr_m = [0.0]
                for label in self.param_labels:
                    if 'c_' in label:
                        cN_m.append(fs[label])
                    elif 's_' in label:
                        sN_m.append(fs[label])
                for label in self.deriv_labels:
                    if 'rdc_' in label:
                        cNdr_m.append(fs[label])
                    elif 'rds_' in label:
                        sNdr_m.append(fs[label])

                print('&geometry')
                print('trpeps  = {}\t! {} = {}'.format((fs['r']/fs['R0']),self.x_label,self.x_loc))
                print('q0      = {}'.format(fs['q']))
                print('shat    = {}'.format(fs['s']))
                print('amhd    = {}'.format(-1))
                print('drR     = {}'.format(fs['dR0dr']))
                print('drZ     = {}'.format(fs['dZ0dr']))
                print('kappa   = {}'.format(fs['kappa']))
                print('s_kappa = {}'.format(fs['s_kappa']))
                print('cN_m  = {}'.format(' '.join('{:15.8e}'.format(value) for value in cN_m)))
                print('cNdr_m  = {}'.format(' '.join('{:15.8e}'.format(value) for value in cNdr_m)))
                print('sN_m  = {}'.format(' '.join('{:15.8e}'.format(value) for value in sN_m)))
                print('sNdr_m  = {}'.format(' '.join('{:15.8e}'.format(value) for value in sNdr_m)))
                print('minor_r = {}'.format(self.eq.derived['a']/Lref))#1.0,
                print('major_R = {}'.format(fs['R0']/Lref))#R0/a,
                print('/')
            
                print('\nAdditional information:')
                print('delta   = {:15.8e}'.format(np.sin(sN_m[1])))
                print('s_delta = {:15.8e}'.format(sNdr_m[1]))
                print('zeta    = {:15.8e}'.format(-sN_m[2]))
                print('s_zeta  = {:15.8e}'.format(-sNdr_m[2]))
                print('')
                print('&units')
                if lref == 'a':
                    print('Lref    = {} !for Lref=a convention'.format(Lref))
                elif lref == 'R':
                    print('Lref    = {} !for Lref=R0 convention'.format(fs['R0']))
                print('Bref    = {}'.format(np.abs(fs['fpol']/fs['R0'])))
                print('Bunit    = {}'.format(np.abs(self.Bunit)))
                print('Bunit/Bref    = {}'.format(np.abs(self.Bunit/(fs['fpol']/fs['R0']))))
                print('a*drho_tor/dr    = {} !for gradient coversion omn(rho_tor) -> a/Ln'.format(self.eq.derived['a']*self.dxdr[i_x_loc]))
                print('R0*drho_tor/dr    = {} !for gradient coversion omn(rho_tor) -> R0/Ln'.format(fs['R0']*self.dxdr[i_x_loc]))
                print('/')

            else:
                print('The selected geometry parameterization is currently not supported by GENE!')
        
        elif printer.lower() == 'tglf':
            Lref = self.eq.derived['a']
            if self._param in ['miller','turnbull']:
                print('RMIN_LOC = {}'.format(fs['r']/Lref))
                print('RMAJ_LOC = {}'.format(fs['R0']/Lref))
                print('ZMAJ_LOC = {}'.format(fs['Z0']))
                print('DRMAJDX_LOC = {}'.format(fs['dR0dr']))
                print('DZMAJDX_LOC = {}'.format(fs['dZ0dr']))
                print('KAPPA_LOC = {}'.format(fs['kappa']))
                print('S_KAPPA_LOC = {}'.format(fs['s_kappa']))
                print('DELTA_LOC = {}'.format(fs['delta']))
                print('S_DELTA_LOC = {}'.format(fs['s_delta']*(1.0-fs['delta']**2.0)**0.5))
                if self._param == 'turnbull':
                    print('ZETA_LOC = {}'.format(fs['zeta']))
                    print('S_ZETA_LOC = {}'.format(fs['s_zeta']))
                print('Q_LOC = {}'.format(fs['q']))
                print('Q_PRIME_LOC = {}'.format(fs['s']*(fs['q']/(fs['r']/Lref))**2))
                #print('P_PRIME_LOC = {}'.format(-fs['amhd']/(8.0*np.pi*(fs['q']/Lref)*(fs['R0']/Lref)*fs['r'])))
            else:
                print('The selected geometry parameterization is currently not supported by TGLF!')
        else:
            print('Selected option {} is not supported for printing iput format! For fitted shape parameters see below:'.format(printer))
            for i_key,key in enumerate(self.param_labels):
                print('{} = {}'.format(key,self.shape[i_key]))
            for i_key,key in enumerate(self.deriv_labels):
                print('{} = {}'.format(key,self.shape_deriv[i_key]))

        return

    def plot_all(self,analytic_shape=None,opt_bpol=None):
        """Plot a summury of a magnetic equilibrium and local parameterization.

        Args:
            analytic_shape (bool,optional): Bool to include contour-averaged analytical shape parameters in the plots.
            opt_bpol (bool,optional): Bool to include optimized shape derivative parameters in the plots.
        """
        i_x_loc = self.x_grid.index(self.x_loc)

        fs = {}
        # get all the flux-surface shape parameters
        for key in self.param_labels:
            if key in self.eq.fluxsurfaces['fit_geo']:
                fs.update({key:self.eq.fluxsurfaces['fit_geo'][key]})
        for key in self.deriv_labels:
            if key+'_opt' in self.eq.fluxsurfaces['fit_geo']:
                fs.update({key:self.eq.fluxsurfaces['fit_geo'][key+'_opt']})
            if key in self.eq.fluxsurfaces['fit_geo']:
                fs.update({key:self.eq.fluxsurfaces['fit_geo'][key]})
        # get the other derived quantities required for print
        for key in ['q','s','fpol']:
            if key in self.eq.fluxsurfaces:
                fs.update({key:self.eq.fluxsurfaces[key]})
            elif key in self.eq.derived:
                fs.update({key:self.eq.derived[key]})

        for i_key,key in enumerate(fs.keys()):
            if '_opt' not in key:
                plt.figure(i_key)
                plt.title(key)
                plt.plot(self.x_grid,fs[key])
            else:
                plt.figure(find(key.replace('_opt',''),list(fs.keys())))
                plt.title(key)
                plt.plot(self.x_loc,fs[key],'*')
        
        fig = plt.figure(constrained_layout=True,figsize=(15,10))
        fig.suptitle('MEGPy diagnostic'.format(self.x_label,self.x_loc))
        axes = fig.subplot_mosaic(
            """
            AACDEF
            AACDEF
            AAGHIJ
            BBGHIJ
            """
        )
        
        plt.show()

        return

