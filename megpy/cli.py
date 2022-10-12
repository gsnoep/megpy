import os
import argparse

from .equilibrium import Equilibrium
from .localequilibrium import LocalEquilibrium
from .utils import *

def is_valid_file(file):
    if not os.path.exists(file):
        raise argparse.ArgumentTypeError('The provided path {} does not lead to an existing EQDSK file, check input!'.format(file))
    else:
        return file

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse():
    parser = argparse.ArgumentParser(description='MEGPy: Extract flux surface shaping parameters (Miller, Turnbull-Miller, etc) from EQDSK files.')
    parser.add_argument('-m', '--method', dest='method', required=False, type=str, default='turnbull', help='parameterization to be used for fitting the flux surfaces and Bpol', metavar='STR')
    parser.add_argument('-f', '--file_path', dest='file', required=True, type=is_valid_file, help='valid path to an EQDSK file', metavar='FILE')
    parser.add_argument('-r', '--refine', dest='refine', required=False, type=int, default=0, help='refine the poloidal flux map psi(R,Z) <r> times', metavar='INT')
    parser.add_argument('-x', '--x_fluxsurface', dest='x_fs', required=True, type=float, help='radial location to extract flux surface and Bpol shaping parameters', metavar='POSITION')
    parser.add_argument('-l', '--x_label', dest='x_label', required=False, type=str, default='rho_tor', help='label of the radial location (e.g. rho_tor, rho_pol, r/a)', metavar='STR')
    parser.add_argument('-p', '--printer', dest='printer', required=False, type=str, default='GENE', help='radial location to extract flux surface shaping parameters', metavar='STR')
    parser.add_argument('--printer_lref', dest='lref', required=False, type=str, default='a', help='radial location to extract flux surface shaping parameters', metavar='STR')
    parser.add_argument('-c', '--cost_f', dest='cost_f', required=False, type=str, default='l1l2', help='radial location to extract flux surface shaping parameters', metavar='STR')
    parser.add_argument('--n_x', dest='n_x', required=False, type=int, default=10, help='number of radial points used for radial derivatives', metavar='INT')
    parser.add_argument('--n_theta', dest='n_theta', required=False, type=int, default=7200, help='number of radial points used for radial derivatives', metavar='INT')
    parser.add_argument('--n_harmonics', dest='n_harmonics', required=False, type=int, default=2, help='number of radial points used for radial derivatives', metavar='INT')
    parser.add_argument('-a', '--analytic-geo', dest='analytic_geo', required=False, nargs='?', const=True, default=True, type=str2bool, help='include the analytical Miller parameters for comparison y/n', metavar='BOOL')
    parser.add_argument('--opt-bpol', dest='opt_bpol', required=False, nargs='?', const=False, default=False, type=str2bool, help='refine the poloidal flux map psi(R,Z) <i> times', metavar='BOOL')
    args = parser.parse_args()

    eq = Equilibrium()
    eq.read_geqdsk(f_path=args.file)
    if args.refine > 1:
        eq.add_derived(refine=args.refine)
    else:
        eq.add_derived()
    if args.x_label == 'r/a':
        print('Processing Equilibrium to determine r/a(psi)...')
        eq.add_fluxsurfaces()

    locgeo = LocalEquilibrium(args.method,
                           eq,
                           args.x_fs,
                           x_label=args.x_label,
                           cost_f=args.cost_f,
                           n_x=args.n_x,
                           n_theta=args.n_theta,
                           n_harmonics=args.n_harmonics,
                           incl_analytic_geo=args.analytic_geo,
                           opt_bpol=args.opt_bpol,
                           verbose=0)
    
    if args.printer:
        locgeo.printer(args.printer,locgeo.shape,locgeo.param_labels,locgeo.shape_bpol,locgeo.bpol_labels)
        #if incl_analytic_geo:
        #    self.printer(printer,self.shape_analytic,self.param_labels,self.shape_bpol_analytic,self.bpol_labels)
        
    if not args.no_plots:
        locgeo.plot_all(incl_analytic_geo=args.analytic_geo,opt_bpol=args.opt_bpol)



str_megpy=["\n                    .#&%%%-.\n"
          ,"                <===#%%%%%%%%%%.\n"
          ,"                   ?==%%( )%%%%\\\n"
          ,"                    )%%%%%%%%%%%%\\\n"
          ,"                    )%%%%%%%%%%%..%%%\n"
          ,"                    )%%%%%%%%&..    .\%.\n"
          ,"                     %%%%%%%% &\,.  ...\\%.\n"
          ,"                     M%%%%%%%...&&%\%%%%%%%%%-\n"
          ,"                       %%%%%.       .\%%%%%%%%%%.\n"
          ,"                        %%%..             .\%%%%%%%.\n"
          ,"                         E...               .\%%%%%%%-.\n"
          ,"                            &...              )%%%%(%%%.\n"
          ,"                               G..   .&).     )GS\   \%%%%%%-\n"
          ,"                                  \&))  \&(%^^          \&%%%%%&.\n"
          ,"                                   )%%    %%                 \%%%%%%%&.\n"
          ,"                                .&&     .%                        \&\\%%%%\\&..\n"
          ,"                             .%<       (\                               %   \\\n"
          ,"                      .::-P-..-&  &&-Y&--.\n\n"]