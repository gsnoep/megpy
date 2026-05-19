import  scipy
import  math
import  numpy                           as      np
import  matplotlib.pyplot               as      plt
import  scipy.io                        as      io
from    scipy.optimize                  import  minimize
from    mpi4py                          import  MPI
from    scipy.interpolate               import  UnivariateSpline    as sp 


# make a function to read the file
def read_file(filename):
    file2read = io.netcdf_file(filename,'r',mmap=False)
    # retrieve rmnc, zmns, xn, ns, and nfp
    rmnc = file2read.variables['rmnc']
    zmns = file2read.variables['zmns']
    xn = (file2read.variables['xn']).data
    xm = (file2read.variables['xm']).data
    ns = (file2read.variables['ns']).data
    nfp = (file2read.variables['nfp']).data
    # retrieve pres and iotas
    pres = file2read.variables['presf'].data
    iota = file2read.variables['iotaf'].data
    pres = pres[1::]
    iota = iota[1::]
    phi_last = file2read.variables['phi'].data[-1]
    edge_toroidal_flux_over_2pi = phi_last / (2*np.pi)
    L_reference = file2read.variables['Aminor_p'].data
    # close the file
    file2read.close()
    return rmnc, zmns, xn, xm, ns, nfp, pres, iota, edge_toroidal_flux_over_2pi, L_reference


# return (R,Z) coordinates of the vmec cross section
def vmec_RZ(filename,fit_modes=2**8):
    rmnc, zmns, xn, xm, ns, nfp, pres, iota, edge_toroidal_flux_over_2pi, L_reference = read_file(filename)
    # make grid of theta and zeta
    nzeta= 1
    theta = np.linspace(0,2*np.pi,num=fit_modes)
    zeta = np.linspace(0,2*np.pi/nfp,num=nzeta,endpoint=False)
    # number of modes
    nmodes = len(xn)
    iradius = ns-1
    R = np.zeros((ns-1,fit_modes,nzeta))
    Z = np.zeros((ns-1,fit_modes,nzeta))
    for iradius in range(ns-1):
        for itheta in range(fit_modes):
            for izeta in range(nzeta):
                for imode in range(nmodes):
                    angle = xm[imode]*theta[itheta] - xn[imode]*zeta[izeta]
                    R[iradius,itheta,izeta] = R[iradius,itheta,izeta] + rmnc[iradius,imode]*math.cos(angle)
                    Z[iradius,itheta,izeta] = Z[iradius,itheta,izeta] + zmns[iradius,imode]*math.sin(angle)
    return R, Z


# return (R,Z) coordinates of Miller cross section
def miller_RZ(R0,Z0,r,kappa,delta,zeta,fit_modes=2**8):
    theta = np.linspace(0,2*np.pi,num=fit_modes)
    Z = Z0 + r * kappa * np.sin(theta + zeta * np.sin(2*theta))
    R = R0 + r * np.cos(theta + np.arcsin(delta) * np.sin(theta))
    return R,Z


# return the fourier coefficients of the cross section in (R,Z) coordinates
def RZ_fourier(R,Z):
    # create complex (R,Z)
    C = R + 1j*Z
    # now do fft
    C_hat = np.fft.fft(C,norm='forward')
    return C_hat


# define fitting function which returns the difference between the Fourier coefficients
def fit_func(params,R,Z,fit_modes):
    # first, get the Fourier coefficients of the VMEC cross section
    C_hat = RZ_fourier(R,Z)
    # now, get the Fourier coefficients of the Miller cross section
    R0 = params[0]
    Z0 = params[1]
    r = params[2]
    kappa = params[3]
    delta = params[4]
    zeta = params[5]
    Rm,Zm = miller_RZ(R0,Z0,r,kappa,delta,zeta,fit_modes=fit_modes)
    Cm_hat = RZ_fourier(Rm,Zm)
    # return the difference between the two
    diff = np.abs(C_hat - Cm_hat)
    return np.sum(diff**2)


# finally, do the fitting for all the cross sections. To this end, loop over all radii
def fit_all_cross_sections(R,Z,fit_modes):
    # get number of radii
    nr      = R.shape[0]
    # initialize arrays
    R0      = np.zeros(nr)
    Z0      = np.zeros(nr)
    r       = np.zeros(nr)
    kappa   = np.zeros(nr)
    delta   = np.zeros(nr)
    zeta    = np.zeros(nr)
    # define initial guess
    params0 = np.array([1.0,0.0,0.1,1.0,0.0,0.0])
    # loop over all radii
    for ir in range(nr):
        # get the (R,Z) coordinates
        Rr  = (R[ir,:,:]).flatten()
        Zr  = (Z[ir,:,:]).flatten()
        # do the minimization, bounded.
        # R0 should be positive
        # Z0 can be anything
        # r should be positive
        # kappa should be positive
        # delta should be between -1 and 1 
        # zeta can be anything
        bnds = ((0.0,None),(None,None),(0.0,None),(0.0,None),(-1.0,1.0),(None,None))
        res = minimize(fit_func,params0,args=(Rr,Zr,fit_modes),bounds=bnds,tol=1e-16)
        # print the minimization result
        # get the result
        R0[ir] = res.x[0]
        Z0[ir] = res.x[1]
        r[ir] = res.x[2]
        kappa[ir] = res.x[3]
        delta[ir] = res.x[4]
        zeta[ir] = res.x[5]
        # update the initial guess
        # params0 = res.x

    return R0, Z0, r, kappa, delta, zeta


# now make a wrapper function which takes as input the filename and outputs the Miller parameters
def vmec2miller(filename,fit_modes=2**6):
    # first, get the (R,Z) coordinates of the VMEC cross section
    R, Z = vmec_RZ(filename,fit_modes=fit_modes)
    # now, do the fitting
    R0, Z0, r, kappa, delta, zeta = fit_all_cross_sections(R,Z,fit_modes)
    # we construct the r derivatives of the Miller parameters
    # this is done with splines
    # let us first define the spline parameters
    k_int = 3
    ext_int = 0
    s_val = 0.0
    # also retrieve pres, iota, Bref
    rmnc, zmns, xn, xm, ns, nfp, pres, iota, edge_toroidal_flux_over_2pi, L_reference = read_file(filename)
    Bref    = 2 * np.abs(edge_toroidal_flux_over_2pi) / L_reference**2
    mu0     = 4*np.pi*1e-7
    mag_pres= Bref**2 / 2 / mu0
    # now construct the spline
    pres_spline = sp(r,pres,k=k_int,ext=ext_int,s=s_val)
    iota_spline = sp(r,iota,k=k_int,ext=ext_int,s=s_val)
    R0_spline = sp(r,R0,k=k_int,ext=ext_int,s=s_val)
    Z0_spline = sp(r,Z0,k=k_int,ext=ext_int,s=s_val)
    kappa_spline = sp(r,kappa,k=k_int,ext=ext_int,s=s_val)
    delta_spline = sp(r,delta,k=k_int,ext=ext_int,s=s_val)
    zeta_spline = sp(r,zeta,k=k_int,ext=ext_int,s=s_val)
    # now, define the derivatives
    presp = pres_spline.derivative()
    iotap = iota_spline.derivative()
    R0p = R0_spline.derivative()
    Z0p = Z0_spline.derivative()
    kappap = kappa_spline.derivative()
    deltap = delta_spline.derivative()
    zetap = zeta_spline.derivative()
    # finally, define the Miller parameters
    alpha = - (1/iota_spline(r))**2 * R0_spline(r) * presp(r) / mag_pres
    shear = - r * iotap(r)/iota_spline(r)
    dR0dr = R0p(r)
    dZ0dr = Z0p(r)
    s_kappa = r * kappap(r)/kappa_spline(r)
    s_delta = r * deltap(r)/np.sqrt(1 - delta_spline(r)**2)
    s_zeta  = r * zetap(r)
    # return the Miller parameters (R0,Z0,r,kappa,delta,zeta,dR0dr,dZ0dr,s_kappa,s_delta,s_zeta,1/iota,shear,alpha,Bref)
    return R0,Z0,r,kappa,delta,zeta,dR0dr,dZ0dr,s_kappa,s_delta,s_zeta,1/iota,shear,alpha,Bref


test = True
# now, let us test the code
if test==True:
    # run test on wout_Lmode.nc
    name = 'wout_output.nc'
    R0,Z0,r,kappa,delta,zeta,dR0dr,dZ0dr,s_kappa,s_delta,s_zeta,q,shear,alpha,Bref = vmec2miller(name)

    # plot the results as a function of radius in several subplots. print Bref
    fig, ax = plt.subplots(4,4,figsize=(8,8),tight_layout=True,sharex=True)
    ax[0,0].plot(r,R0)
    ax[0,0].set_title('R0')
    ax[0,1].plot(r,Z0)
    ax[0,1].set_title('Z0')
    ax[0,2].plot(r,dR0dr)
    ax[0,2].set_title('dR0dr')
    ax[0,3].plot(r,dZ0dr)
    ax[0,3].set_title('dZ0dr')
    ax[1,0].plot(r,r)
    ax[1,0].set_title('r')
    ax[1,1].plot(r,kappa)
    ax[1,1].set_title('kappa')
    ax[1,2].plot(r,delta)
    ax[1,2].set_title('delta')
    ax[1,3].plot(r,zeta)
    ax[1,3].set_title('zeta')
    ax[2,0].plot(r,s_kappa)
    ax[2,0].set_title('s_kappa')
    ax[2,1].plot(r,s_delta)
    ax[2,1].set_title('s_delta')
    ax[2,2].plot(r,s_zeta)
    ax[2,2].set_title('s_zeta')
    ax[2,3].plot(r,q)
    ax[2,3].set_title('q')
    ax[3,0].plot(r,shear)
    ax[3,0].set_title('shear')
    ax[3,1].plot(r,alpha)
    ax[3,1].set_title('alpha')

    # print Bref
    print('Bref = ',Bref)

    # show the plot
    plt.show()

    # also plot 3 cross sections to compare fit with original
    # do for 3 radii: inner, middle, outer
    # import R,Z from file
    R, Z = vmec_RZ(name,fit_modes=2**8)
    fig, ax = plt.subplots(1,3,figsize=(12,4),tight_layout=True)
    # inner
    ax[0].plot(R[1,:,0],Z[1,:,0],label='VMEC')
    Rm,Zm = miller_RZ(R0[1],Z0[1],r[1],kappa[1],delta[1],zeta[1],fit_modes=2**8)
    ax[0].plot(Rm,Zm,label='Miller')
    ax[0].set_title('inner')
    ax[0].legend()
    # middle, R.shape[0]//2
    ax[1].plot(R[R.shape[0]//2,:,0],Z[R.shape[0]//2,:,0],label='VMEC')
    Rm,Zm = miller_RZ(R0[R.shape[0]//2],Z0[R.shape[0]//2],r[R.shape[0]//2],kappa[R.shape[0]//2],delta[R.shape[0]//2],zeta[R.shape[0]//2],fit_modes=2**8)
    ax[1].plot(Rm,Zm,label='Miller')
    ax[1].set_title('middle')
    ax[1].legend()
    # outer
    ax[2].plot(R[-1,:,0],Z[-1,:,0],label='VMEC')
    Rm,Zm = miller_RZ(R0[-1],Z0[-1],r[-1],kappa[-1],delta[-1],zeta[-1],fit_modes=2**8)
    ax[2].plot(Rm,Zm,label='Miller')
    ax[2].set_title('outer')
    ax[2].legend()
    # show the plot
    plt.show()  
