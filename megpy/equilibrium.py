"""
created by gsnoep on 11 August 2022, with contributions from aho

Module to handle any and all methods related to magnetic equilibrium data

The Equilibrium class can:
- read and write magnetic equilibria files (only eqdsk g-files for now),
- add derived quantities (e.g. phi, rho_tor, rho_pol, etc.) to the Equilibrium, 
- add flux surfaces traces (shaping parameters included), 
- map 1D profiles with a magnetics derived coordinate basis on the Equilibrium,
- refine the complete Equilibrium by interpolating the R,Z basis .
"""

# imports
import os
import re
import copy
import numpy as np
import time
import json
import codecs

from scipy import interpolate, integrate
from sys import stdout
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from . import tracer
from .localequilibrium import LocalEquilibrium
from .utils import *


class Equilibrium():
    """
    Class to handle any and all data related to the magnetic equilibrium in a magnetic confinement fusion device.
    """
    def __init__(self,verbose=True):
        self.raw = {} # storage for all raw eqdsk data
        self.derived = {} # storage for all data derived from eqdsk data
        self.fluxsurfaces = {} # storage for all data related to flux surfaces
        # specify the eqdsk file formate, based on 'G EQDSK FORMAT - L Lao 2/7/97'
        self._eqdsk_format = {
            0:{'vars':['case','idum','nw','nh'],'size':[4]},
            1:{'vars':['rdim', 'zdim', 'rcentr', 'rleft', 'zmid'],'size':[5]},
            2:{'vars':['rmaxis', 'zmaxis', 'simag', 'sibry', 'bcentr'],'size':[5]},
            3:{'vars':['current', 'simag2', 'xdum', 'rmaxis2', 'xdum'],'size':[5]},
            4:{'vars':['zmaxis2', 'xdum', 'sibry2', 'xdum', 'xdum'],'size':[5]},
            5:{'vars':['fpol'],'size':['nw']},
            6:{'vars':['pres'],'size':['nw']},
            7:{'vars':['ffprim'],'size':['nw']},
            8:{'vars':['pprime'],'size':['nw']},
            9:{'vars':['psirz'],'size':['nh','nw']},
            10:{'vars':['qpsi'],'size':['nw']},
            11:{'vars':['nbbbs','limitr'],'size':[2]},
            12:{'vars':['rbbbs','zbbbs'],'size':['nbbbs']},
            13:{'vars':['rlim','zlim'],'size':['limitr']},
        }
        self._sanity_values = ['rmaxis','zmaxis','simag','sibry'] # specify the sanity values used for consistency check of eqdsk file
        self._max_values = 5 # maximum number of values per line
        self.verbose = verbose

    ## I/O functions
    def read_geqdsk(self,f_path=None,add_derived=False):
        """Read an eqdsk g-file from file into `Equilibrium` object

        Args:
            `f_path` (str): the path to the eqdsk g-file, including the file name (!).
            `add_derived` (bool): [True] also add derived quantities (e.g. phi, rho_tor) to the `Equilibrium` object upon reading the g-file, or [False, default] not.

        Returns:
            [default] self
        
        Raises:
            ValueError: Raise an exception when no `f_path` is provided
        """
        if self.verbose:
            print('Reading eqdsk g-file to Equilibrium...')

        # check if eqdsk file path is provided and if it exists
        if f_path is None or not os.path.isfile(f_path):
            raise ValueError('Invalid file or path provided!')
        
        # read the g-file
        with open(f_path,'r') as file:
            lines = file.readlines()
        
        if lines:
            # start at the top of the file
            current_row = 0
            # go through the eqdsk format key by key and collect all the values for the vars in each format row
            for key in self._eqdsk_format:
                if current_row < len(lines):
                    # check if the var size is a string refering to a value to be read from the eqdsk file and backfill it, for loop for multidimensional vars
                    for i,size in enumerate(self._eqdsk_format[key]['size']):
                        if isinstance(size,str):
                            self._eqdsk_format[key]['size'][i] = self.raw[size]

                    # compute the row the current eqdsk format key ends
                    if len(self._eqdsk_format[key]['vars']) != np.prod(self._eqdsk_format[key]['size']):
                        end_row = current_row + int(np.ceil(len(self._eqdsk_format[key]['vars'])*np.prod(self._eqdsk_format[key]['size'])/self._max_values))
                    else:
                        end_row = current_row + int(np.ceil(np.prod(self._eqdsk_format[key]['size'])/self._max_values))

                    # check if there are values to be collected
                    if end_row > current_row:
                        _lines = lines[current_row:end_row]
                        for i_row, row in enumerate(_lines):
                            try:
                                # split the row string into separate values by ' ' as delimiter, adding a space before a minus sign if it is the delimiter
                                values = list(filter(None,re.sub(r'(?<![Ee])-',' -',row).rstrip('\n').split(' ')))
                                # select all the numerical values in the list of sub-strings of the current row, but keep them as strings so the fortran formatting remains
                                numbers = [j for i in [num for num in (re.findall(r'^(?![A-Z]).*-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?', value) for value in values)] for j in i]
                                # select all the remaining sub-strings and store them in a separate list
                                strings = [value for value in values if value not in numbers]
                                # handle the exception of the first line where in the case description numbers and strings can be mixed
                                if current_row == 0:
                                    numbers = numbers[-3:]
                                    strings = [string for string in values if string not in numbers] 
                                # convert the list of numerical sub-strings to their actual int or float value and collate the strings in a single string
                                numbers = [number(value) for value in numbers]
                                strings = ' '.join(strings)
                                _values = numbers
                                if strings:
                                    _values.insert(0,strings)
                            except:
                                _values = row.strip()
                            _lines[i_row] = _values
                        # unpack all the values between current_row and end_row in the eqdsk file and flatten the resulting list of lists to a list
                        values = [value for row in _lines for value in row]

                        # handle the exception of len(eqdsk_format[key]['vars']) > 1 and the data being stored in value pairs 
                        if len(self._eqdsk_format[key]['vars']) > 1 and len(self._eqdsk_format[key]['vars']) != self._eqdsk_format[key]['size'][0]:
                            # make a shadow copy of values
                            _values = copy.deepcopy(values)
                            # empty the values list
                            values = []
                            # collect all the values belonging to the n-th variable in the format list and remove them from the shadow value list until empty
                            for j in range(len(self._eqdsk_format[key]['vars']),0,-1):
                                values.append(np.array(_values[0::j]))
                                _values = [value for value in _values if value not in values[-1]]
                        # store and reshape the values in a np.array() in case eqdsk_format[key]['size'] > max_values
                        elif self._eqdsk_format[key]['size'][0] > self._max_values:
                            values = [np.array(values).reshape(self._eqdsk_format[key]['size'])]
                        # store the var value pairs in the eqdsk dict
                        self.raw.update({var:values[k] for k,var in enumerate(self._eqdsk_format[key]['vars'])})
                    # update the current position in the 
                    current_row = end_row
            
            # store any remaining lines as a comment, in case of CHEASE/LIUQE
            if current_row < len(lines):
                comment_lines = []
                for line in lines[current_row+1:]:
                    if isinstance(line,list):
                        comment_lines.append(' '.join([str(text) for text in line]))
                    else:
                        if line.strip():
                            comment_lines.append(str(line))
                self.raw['comment'] = '\n'.join(comment_lines)

            # sanity check the eqdsk values
            for key in self._sanity_values:
                # find the matching sanity key in eqdsk
                sanity_pair = [keypair for keypair in self.raw.keys() if keypair.startswith(key)][1]
                #print(sanity_pair)
                if self.raw[key]!=self.raw[sanity_pair]:
                    raise ValueError('Inconsistent '+key+': %7.4g, %7.4g'%(self.raw[key], self.raw[sanity_pair])+'. CHECK YOUR EQDSK FILE!')

            if add_derived:
                self.add_derived()
            
            return self
    
    def write_geqdsk(self,f_path=None):
        """ Write an `Equilibrium` object to an eqdsk g-file 

        Args:
            f_path (str): the target path of generated eqdsk g-file, including the file name (!).
        
        Returns:
            
        """
        if self.verbose:
            print('Writing Equilibrium to eqdsk g-file...')

        if self.raw:
            if not isinstance(f_path, str):
                raise TypeError("filepath field must be a string. EQDSK file write aborted.")

            maxv = int(self._max_values)

            if os.path.isfile(f_path):
                print("{} exists, overwriting file with EQDSK file!".format(f_path))
            eq = {"xdum": 0.0}
            for linenum in self._eqdsk_format:
                if "vars" in self._eqdsk_format[linenum]:
                    for key in self._eqdsk_format[linenum]["vars"]:
                        if key in self.raw:
                            eq[key] = copy.deepcopy(self.derived[key])
                        elif key in ["nbbbs","limitr","rbbbs","zbbbs","rlim","zlim"]:
                            eq[key] = None
                            if key in self.derived:
                                eq[key] = copy.deepcopy(self.derived[key])
                        else:
                            raise TypeError("%s field must be specified. EQDSK file write aborted." % (key))
            if eq["nbbbs"] is None or eq["rbbbs"] is None or eq["zbbbs"] is None:
                eq["nbbbs"] = 0
                eq["rbbbs"] = []
                eq["zbbbs"] = []
            if eq["limitr"] is None or eq["rlim"] is None or eq["zlim"] is None:
                eq["limitr"] = 0
                eq["rlim"] = []
                eq["zlim"] = []

            eq["xdum"] = 0.0
            with open(f_path, 'w') as ff:
                ff.write("%-48s%4d%4d%4d\n" % (eq["case"], eq["idum"], eq["nw"], eq["nh"]))
                ff.write("%16.9E%16.9E%16.9E%16.9E%16.9E\n" % (eq["rdim"], eq["zdim"], eq["rcentr"], eq["rleft"], eq["zmid"]))
                ff.write("%16.9E%16.9E%16.9E%16.9E%16.9E\n" % (eq["rmaxis"], eq["zmaxis"], eq["simag"], eq["sibry"], eq["bcentr"]))
                ff.write("%16.9E%16.9E%16.9E%16.9E%16.9E\n" % (eq["current"], eq["simag"], eq["xdum"], eq["rmaxis"], eq["xdum"]))
                ff.write("%16.9E%16.9E%16.9E%16.9E%16.9E\n" % (eq["zmaxis"], eq["xdum"], eq["sibry"], eq["xdum"], eq["xdum"]))
                for ii in range(0, len(eq["fpol"])):
                    ff.write("%16.9E" % (eq["fpol"][ii]))
                    if (ii + 1) % maxv == 0 and (ii + 1) != len(eq["fpol"]):
                        ff.write("\n")
                ff.write("\n")
                for ii in range(0, len(eq["pres"])):
                    ff.write("%16.9E" % (eq["pres"][ii]))
                    if (ii + 1) % maxv == 0 and (ii + 1) != len(eq["pres"]):
                        ff.write("\n")
                ff.write("\n")
                for ii in range(0, len(eq["ffprim"])):
                    ff.write("%16.9E" % (eq["ffprim"][ii]))
                    if (ii + 1) % maxv == 0 and (ii + 1) != len(eq["ffprim"]):
                        ff.write("\n")
                ff.write("\n")
                for ii in range(0, len(eq["pprime"])):
                    ff.write("%16.9E" % (eq["pprime"][ii]))
                    if (ii + 1) % maxv == 0 and (ii + 1) != len(eq["pprime"]):
                        ff.write("\n")
                ff.write("\n")
                kk = 0
                for ii in range(0, eq["nh"]):
                    for jj in range(0, eq["nw"]):
                        ff.write("%16.9E" % (eq["psirz"][ii, jj]))
                        if (kk + 1) % maxv == 0 and (kk + 1) != (eq["nh"] * eq["nw"]):
                            ff.write("\n")
                        kk = kk + 1
                ff.write("\n")
                for ii in range(0, len(eq["qpsi"])):
                    ff.write("%16.9E" % (eq["qpsi"][ii]))
                    if (ii + 1) % maxv == 0 and (ii + 1) != len(eq["qpsi"]):
                        ff.write("\n")
                ff.write("\n")
                ff.write("%5d%5d\n" % (eq["nbbbs"], eq["limitr"]))
                kk = 0
                for ii in range(0, eq["nbbbs"]):
                    ff.write("%16.9E" % (eq["rbbbs"][ii]))
                    if (kk + 1) % maxv == 0 and (ii + 1) != eq["nbbbs"]:
                        ff.write("\n")
                    kk = kk + 1
                    ff.write("%16.9E" % (eq["zbbbs"][ii]))
                    if (kk + 1) % maxv == 0 and (ii + 1) != eq["nbbbs"]:
                        ff.write("\n")
                    kk = kk + 1
                ff.write("\n")
                kk = 0
                for ii in range(0, eq["limitr"]):
                    ff.write("%16.9E" % (eq["rlim"][ii]))
                    if (kk + 1) % maxv == 0 and (kk + 1) != eq["limitr"]:
                        ff.write("\n")
                    kk = kk + 1
                    ff.write("%16.9E" % (eq["zlim"][ii]))
                    if (kk + 1) % maxv == 0 and (kk + 1) != eq["limitr"]:
                        ff.write("\n")
                    kk = kk + 1
                ff.write("\n")
            print('Output EQDSK file saved as {}.'.format(f_path))

        else:
            print("g-eqdsk could not be written")

        return

    def read_json(self,path='./',fname='Equilibrium.json'):
        if self.verbose:
            print("Reading Equilibrium {}".format(path+fname))
        with open(path+fname,'r') as file:
            equilibrium_json = json.load(file)
        
        list_to_array(equilibrium_json)

        self.raw = copy.deepcopy(equilibrium_json['raw'])
        self.derived = copy.deepcopy(equilibrium_json['derived'])
        self.fluxsurfaces = copy.deepcopy(equilibrium_json['fluxsurfaces'])

        return self

    def write_json(self,path='./',fname='Equilibrium.json',metadata=None):
        equilibrium = {'raw':copy.deepcopy(self.raw),'derived':copy.deepcopy(self.derived),'fluxsurfaces':copy.deepcopy(self.fluxsurfaces)}
        if metadata:
            equilibrium.update({'metadata':metadata})
        array_to_list(equilibrium)
        json.dump(equilibrium, codecs.open(path+fname, 'w', encoding='utf-8'), separators=(',', ':'), indent=4)

        if self.verbose:
            print('Generated megpy.Equilibrium file at: {}'.format(path+fname))

        return

    def read_ex2gk_pkl(self,f_path,use_fitted=False,add_derived=False):
        """Read an EX2GK pickle file containing eqdsk g-file data into `Equilibrium` object

        Args:
            f_path (str): the path to the EX2GK pickle file, including the file name (!).
            use_fitted (bool, optional): use GPR fitted pressure, pprime and q profiles. Defaults to False.
            add_derived (bool, optional): add derived quantities. Defaults to False.

        Raises:
            ValueError: invalid file path provided
        """
        if self.verbose:
            print('Reading EX2GK equilibrium information to Equilibrium...')
        # check if eqdsk file path is provided and if it exists
        if f_path is None or (isinstance(f_path,str) and not os.path.isfile(f_path)):
            raise ValueError('Invalid file path provided!')
        
        # get the eqdsk keys from the g-file format
        eqdsk_keys = []
        for key in self._eqdsk_format.keys():
            if isinstance(self._eqdsk_format[key]['vars'],list):
                eqdsk_keys += self._eqdsk_format[key]['vars']
        
        # read the EX2GK data pickle
        ex2gk_data = read_pickle(f_path)

        if 'META_SHOT' not in ex2gk_data:
            ex2gk_data.update({'META_SHOT':'unknown'})

        # TODO: verify that the required variables are present in ex2gk data

        # collect and collate all the required eqdsk data from ex2gk_data
        ex2gk_eqdsk = {'case':'EX2GK #'+str(ex2gk_data['META_SHOT'])+', '+ex2gk_data['ED_PPROV'],
                       'idum':42,
                       'nw':ex2gk_data['ED_PSI'].shape[1],
                       'nh':ex2gk_data['ED_PSI'].shape[0],
                       'rdim':np.nanmax(ex2gk_data['ED_PSIX']) - np.nanmin(ex2gk_data['ED_PSIX']),
                       'zdim':np.nanmax(ex2gk_data['ED_PSIY']) - np.nanmin(ex2gk_data['ED_PSIY']),
                       'rcentr':ex2gk_data['ZD_RVAC'],
                       'rleft':np.nanmin(ex2gk_data['ED_PSIX']),
                       'zmid':(np.nanmax(ex2gk_data['ED_PSIY']) + np.nanmin(ex2gk_data['ED_PSIY'])) / 2.0,
                       'rmaxis':ex2gk_data['ZD_RMAG'],
                       'zmaxis':ex2gk_data['ZD_ZMAG'],
                       'simag':ex2gk_data['ZD_PSIAXS'],
                       'sibry':ex2gk_data['ZD_PSIBND'],
                       'bcentr':ex2gk_data['ZD_BVAC'],
                       'current':ex2gk_data['ZD_IPLA'],
                       'fpol':ex2gk_data['ED_F'],
                       'pres':ex2gk_data['ED_P'],
                       'ffprim':ex2gk_data['ED_FFP'],
                       'pprime':ex2gk_data['ED_PP'],
                       'psirz':ex2gk_data['ED_PSI'],
                       'qpsi':ex2gk_data['ED_Q'],}

        # check if boundary definition is present
        if 'ED_BND' in ex2gk_data:
            ex2gk_eqdsk.update({'nbbbs':len(ex2gk_data['ED_BND']),
                                'rbbbs':ex2gk_data['ED_BND'],
                                'zbbbs':ex2gk_data['ED_BNDX']})
        if 'ED_LIM' in ex2gk_data:
            ex2gk_eqdsk.update({'limitr':len(ex2gk_data['ED_LIM']),
                                'rlim':ex2gk_data['ED_LIM'],
                                'zlim':ex2gk_data['ED_LIMX'],})

        # optional: use the GPR fitted pressure, pprime and q profiles instead of the eqdsk values
        if use_fitted:
            ex2gk_eqdsk.update({'pres':ex2gk_data['PD_PEI'],
                                'pprime':ex2gk_data['PD_DPEI'],
                                'qpsi':ex2gk_data['PD_Q'],})
        
        # insert the eqdsk data from EX2GK into raw
        for key in eqdsk_keys:
            if key in ex2gk_eqdsk:
                self.raw.update({key:ex2gk_eqdsk[key]})

        # optional: add derived quantities
        if add_derived:
            self.add_derived()

        return

    def read_ids_equilibrium(self,f_path=None,add_derived=False):

        if self.verbose:
            print('Reading equilibrium IDS (HDF5) information to Equilibrium...')
        # check if ids file path is provided and if it exists
        if f_path is None or (isinstance(f_path,str) and not os.path.isfile(f_path)):
            raise ValueError('Invalid file path provided!')

        # get the eqdsk keys from the g-file format
        eqdsk_keys = []
        for key in self._eqdsk_format.keys():
            if isinstance(self._eqdsk_format[key]['vars'],list):
                eqdsk_keys += self._eqdsk_format[key]['vars']

        ids_data = read_hdf5(f_path)

        ids_eqdsk = {}
        if 'equilibrium' in ids_data:

            cocos_factor = 2.0 * np.pi   # IDS uses COCOS=11 convention, G-EQDSK has no convention (?) but EX2GK uses COCOS=1
            ids_eq_data = ids_data['equilibrium']

            rgrid = ids_eq_data['time_slice[]&profiles_2d[]&grid&dim1'][-1,0]
            zgrid = ids_eq_data['time_slice[]&profiles_2d[]&grid&dim2'][-1,0]
            ids_eqdsk.update({
                'nw': len(rgrid),
                'nh': len(zgrid),
                'rdim': np.nanmax(rgrid) - np.nanmin(rgrid),
                'zdim': np.nanmax(zgrid) - np.nanmin(zgrid),
                'rcentr': ids_eq_data['vacuum_toroidal_field&r0'][()],
                'rleft': np.nanmin(rgrid),
                'zmid': (np.nanmax(zgrid) + np.nanmin(zgrid)) / 2.0,
                'rmaxis': ids_eq_data['time_slice[]&global_quantities&magnetic_axis&r'][-1],
                'zmaxis': ids_eq_data['time_slice[]&global_quantities&magnetic_axis&z'][-1],
                'simag': ids_eq_data['time_slice[]&global_quantities&psi_axis'][-1] / cocos_factor,
                'sibry': ids_eq_data['time_slice[]&global_quantities&psi_boundary'][-1] / cocos_factor,
                'bcentr': ids_eq_data['vacuum_toroidal_field&b0'][()],
                'current': ids_eq_data['time_slice[]&global_quantities&ip'][-1],
                'fpol': ids_eq_data['time_slice[]&profiles_1d&f'][-1],
                'pres': ids_eq_data['time_slice[]&profiles_1d&pressure'][-1],
                'ffprim': ids_eq_data['time_slice[]&profiles_1d&f_df_dpsi'][-1] * cocos_factor,
                'pprime': ids_eq_data['time_slice[]&profiles_1d&dpressure_dpsi'][-1] * cocos_factor,
                'psirz': ids_eq_data['time_slice[]&profiles_2d[]&psi'][-1,0] / cocos_factor,
                'qpsi': ids_eq_data['time_slice[]&profiles_1d&q'][-1],
            })

            # check if boundary definition is present
            if 'time_slice[]&boundary&x_point[]&AOS_SHAPE' in ids_eq_data and ids_eq_data['time_slice[]&boundary&x_point[]&AOS_SHAPE'][-1,0] > 0:
                nbnd = len(ids_eq_data['time_slice[]&boundary&outline&r'][-1])
                ids_eqdsk.update({
                    'rbbbs': ids_eq_data['time_slice[]&boundary&outline&r'][-1],
                    'zbbbs': ids_eq_data['time_slice[]&boundary&outline&z'][-1],
                    'nbbbs': nbnd,
                })
            else:
                nlim = len(ids_eq_data['time_slice[]&boundary&outline&r'][-1])
                ids_eqdsk.update({
                    'rlim': ids_eq_data['time_slice[]&boundary&outline&r'][-1],
                    'zlim': ids_eq_data['time_slice[]&boundary&outline&z'][-1],
                    'limitr': nlim,
                })
 
        # insert the eqdsk data from IDS into raw
        for key in eqdsk_keys:
            if key in ids_eqdsk:
                self.raw.update({key:ids_eqdsk[key]})

        # optional: add derived quantities
        if add_derived:
            self.add_derived()

        return

    @classmethod
    def from_ids_equilibrium(cls,f_path=None,add_derived=False,verbose=False):
        out = cls(verbose=verbose)
        out.read_ids_equilibrium(f_path=f_path,add_derived=add_derived)
        return out

    ## physics functions
    def add_derived(self,f_path=None,refine=None,just_derived=False,incl_fluxsurfaces=False,analytic_shape=False,incl_B=False,tracer_diag=None,verbose=False):
        """Add quantities derived from the raw `Equilibrium.read_geqdsk()` output, such as phi, rho_pol, rho_tor to the `Equilibrium` object.
        Can also be called directly if `f_path` is defined.

        Args:
            `f_path` (str): path to the eqdsk g-file, including the file name (!)
            `refine` (int): the number of desired `Equilibrium` grid points, if the native refine is lower than this value, it is refined using the `refine()` method
            `just_derived` (bool): [True] return only the derived quantities dictionary, or [False, default] return the `Equilibrium` object
            `incl_fluxsurfaces` (bool): include fluxsurface tracing output in the added derived quantities
            `analytic_shape` (bool): include the analytical flux surface Miller shaping parameters as defined in literature. Defaults to False.

        Returns:
            self or dict if just_derived

        Raises:
            ValueError: Raises an exception when `Equilibrium.raw` is empty and no `f_path` is provided
        """
        if self.verbose or verbose:
            print('Adding derived quantities to Equilibrium...')

        if self.raw == {}:
            try:
                self.read_geqdsk(f_path=f_path)
            except:
                raise ValueError('Unable to read provided EQDSK file, check file and/or path')

        # introduce shorthands for data and derived locations for increased readability
        raw = self.raw
        derived = self.derived

        # check refine requirements
        if refine:
            self.refine = refine
            self.refine_equilibrium(nw=int(refine*raw['nw']),nh=int(refine*raw['nh']))
        else:
            derived.update(self.raw)

        # compute R and Z grid vectors
        derived['R'] = np.array([derived['rleft'] + i*(derived['rdim']/(derived['nw']-1)) for i in range(derived['nw'])])
        derived['Z'] = np.array([derived['zmid'] - 0.5*derived['zdim'] + i*(derived['zdim']/(derived['nh']-1)) for i in range(derived['nh'])])

        # equidistant psi grid
        derived['psi'] = np.linspace(derived['simag'],derived['sibry'],derived['nw'])

        # corresponding rho_pol grid
        psi_norm = np.abs((derived['psi'] - derived['simag'])/(derived['sibry'] - derived['simag']))
        derived['rho_pol'] = np.sqrt(psi_norm)

        if 'rbbbs' in derived and 'zbbbs' in derived:
            # ensure the boundary coordinates are stored from midplane lfs to midplane hfs
            i_split = find(np.max(derived['rbbbs']),self.derived['rbbbs'])
            derived['rbbbs'] = np.hstack((derived['rbbbs'][i_split:],derived['rbbbs'][:i_split],derived['rbbbs'][i_split]))
            derived['zbbbs'] = np.hstack((derived['zbbbs'][i_split:],derived['zbbbs'][:i_split],derived['zbbbs'][i_split]))
            
            # find the indexes of 'zmaxis' on the high field side (hfs) and low field side (lfs) of the separatrix
            i_zmaxis_hfs = int(len(derived['zbbbs'])/3)+find(derived['zmaxis'],derived['zbbbs'][int(len(derived['zbbbs'])/3):int(2*len(derived['zbbbs'])/3)])
            i_zmaxis_lfs = int(2*len(derived['zbbbs'])/3)+find(derived['zmaxis'],derived['zbbbs'][int(2*len(derived['zbbbs'])/3):])
            
            # find the index of 'zmaxis' in the R,Z grid
            i_zmaxis = find(derived['zmaxis'],derived['Z'])

            # find indexes of separatrix on HFS, magnetic axis, separatrix on LFS in R
            i_R_hfs = find(derived['rbbbs'][i_zmaxis_hfs],derived['R'][:int(len(derived['R'])/2)])
            i_rmaxis = find(derived['rmaxis'],derived['R'])
            i_R_lfs = int(len(derived['R'])/2)+find(derived['rbbbs'][i_zmaxis_lfs],derived['R'][int(len(derived['R'])/2):])

            # HFS and LFS R and psirz
            R_hfs = derived['R'][i_R_hfs:i_rmaxis]
            R_lfs = derived['R'][i_rmaxis:i_R_lfs]
            psirzmaxis_hfs = derived['psirz'][i_zmaxis,i_R_hfs:i_rmaxis]
            psirzmaxis_lfs = derived['psirz'][i_zmaxis,i_rmaxis:i_R_lfs]

            # nonlinear R grid at 'zmaxis' based on equidistant psi grid for 'fpol', 'pres', 'ffprim', 'pprime' and 'qpsi'
            derived['R_psi_hfs'] = interpolate.interp1d(psirzmaxis_hfs,R_hfs,fill_value='extrapolate')(derived['psi'][::-1])
            derived['R_psi_lfs'] = interpolate.interp1d(psirzmaxis_lfs,R_lfs,fill_value='extrapolate')(derived['psi'])
        
            # find the R,Z values of the x-point, !TODO: should add check for second x-point in case of double-null equilibrium
            i_xpoint_Z = find(np.min(derived['zbbbs']),derived['zbbbs']) # assuming lower null, JET-ILW shape for now
            derived['R_x'] = derived['rbbbs'][i_xpoint_Z]
            derived['Z_x'] = derived['zbbbs'][i_xpoint_Z]

            bbbs_center = tracer.contour_center({'X':derived['rbbbs'],'Y':derived['zbbbs'],'level':derived['sibry'],'label':1.0})

            derived['R0'] = bbbs_center['X0']
            derived['Z0'] = bbbs_center['Y0']
            derived['a'] = bbbs_center['r']

        # compute LFS phi (toroidal flux in W/rad) grid from integrating q = d psi/d phi
        derived['phi'] = integrate.cumulative_trapezoid(derived['qpsi'],derived['psi'],initial=0)

        # construct the corresponding rho_tor grid
        if derived['phi'][-1] !=0:
            phi_norm = derived['phi']/derived['phi'][-1] # as phi[0] = 0 this term is dropped
        else:
            phi_norm = np.ones_like(derived['phi'])*np.NaN
            print('Could not construct valid rho_tor')
        derived['rho_tor']  = np.sqrt(phi_norm)

        # compute the rho_pol and rho_tor grids corresponding to the R,Z grid
        psirz_norm = np.abs((derived['psirz'] - derived['simag'])/(derived['sibry'] - derived['simag']))
        derived['rhorz_pol'] = np.sqrt(psirz_norm)

        derived['phirz'] = interpolate.interp1d(derived['psi'],derived['phi'],kind=5,bounds_error=False)(derived['psirz'])
        phirz_norm = abs(derived['phirz']/(derived['phi'][-1]))
        derived['rhorz_tor'] = np.sqrt(phirz_norm)

        # compute the toroidal current density
        derived['j_tor'] = derived['R_psi_lfs']*derived['pprime']+derived['ffprim']/derived['R_psi_lfs']

        # compute the poloidal magnetic flux density
        R,Z = np.meshgrid(derived['R'],derived['Z'])
        [derived['dpsirzdz'],derived['dpsirzdr']] = np.gradient(derived['psirz'],derived['Z'],derived['R'],edge_order=2)
        derived['B_r'] = derived['dpsirzdz']/R
        derived['B_z'] = -derived['dpsirzdr']/R
        derived['B_pol_rz'] = np.sqrt(derived['B_r']**2 + derived['B_z']**2)

        # compute the toroidal magnetic flux density
        derived['B_tor_rz'] = interpolate.interp1d(derived['psi'],derived['fpol'],bounds_error=False,fill_value='extrapolate')(derived['psirz'])/R

        if incl_fluxsurfaces:
            self.add_fluxsurfaces(refine=refine,analytic_shape=analytic_shape,incl_B=incl_B,tracer_diag=tracer_diag)
              
        if just_derived:
            return self.derived 
        else:
            return self

    def add_fluxsurfaces(self,x=None,x_label='rho_tor',refine=None,analytic_shape=False,incl_B=False,tracer_diag=None,verbose=False):
        """Add flux surfaces to an `Equilibrium`.
        
        Args:
            `raw` (dict, optional):  the raw `Equilibrium` data, [default] self.raw if None is set.
            `derived` (dict, optional): the derived `Equilibrium` quantities, [default] self.derived if None is set.
            `fluxsurfaces` (dict, optional): the `Equilibrium` flux surface data, each key a variable containing an array, [default] self.fluxsurfaces if None is set.
            `analytic_shape` (bool, optional): [True] include the flux surface Miller shaping parameters delta, kappa and zeta, or [False, default] not.

        Returns:
            self.
        """
        if self.verbose or verbose:
            print('Adding fluxsurfaces to Equilibrium...')

        # check if self.fluxsurfaces contains all the flux surfaces specified by derived['rho_tor'] already
        if self.fluxsurfaces and self.derived and len(self.fluxsurfaces['rho_tor']) == len(self.derived['rho_tor']):
            # skip
            print('Skipped adding flux surfaces to Equilibrium as it already contains fluxsurfaces')
        else:
            with np.errstate(divide='ignore'):
                # set the default locations if None is specified
                raw = self.raw
                derived = self.derived
                if not self.derived:
                    self.add_derived()
                fluxsurfaces = self.fluxsurfaces

                if refine and derived['nw'] != refine*derived['nw']:
                    self.refine(nw=refine*derived['nw'],nh=refine*derived['nh'],self_consistent=False)
                    self.add_derived()
                
                R = copy.deepcopy(derived['R'])
                Z = copy.deepcopy(derived['Z'])
                psirz = copy.deepcopy(derived['psirz'])

                if tracer_diag == 'mesh':
                    fig = plt.figure()
                    ax = fig.add_subplot(projection='3d')
                    R_,Z_ = np.meshgrid(R,Z)
                    ax.plot_wireframe(R_,Z_,psirz, rstride=10, cstride=10)
                    ax.set_xlabel('R [m]')
                    ax.set_ylabel('Z [m]')
                    ax.set_zlabel('$\\Psi$')
                    plt.show()

                # find the approximate location of the magnetic axis on the psirz map
                i_rmaxis = find(self.derived['rmaxis'],R)
                i_zmaxis = find(self.derived['zmaxis'],Z)

                if tracer_diag:
                    plt.figure()
                    if tracer_diag == 'fs':
                        plt.plot(derived['rmaxis'],derived['zmaxis'],'bx')

                # add the flux surface data for rho_tor > 0
                if not x:
                    if x_label in self.derived:
                        x_list = self.derived[x_label][1:]
                    else:
                        x_list = self.derived['psi'][1:]
                else:
                    x_list = list(x)
                
                threshold = derived['sibry']
                interp_method = 'normal'
                if np.max(psirz[np.where(psirz!=0.0)]) <= threshold:
                    threshold = np.max(psirz[np.where(psirz!=0.0)])
                #    interp_method = 'bounded_extrapolation'
                elif np.min(psirz[np.where(psirz!=0.0)]) >= threshold:
                    threshold = np.min(psirz[np.where(psirz!=0.0)])
                #    interp_method = 'bounded_extrapolation'

                tracer_timing = 0.
                analytic_timing = 0.
                for i_x_fs,x_fs in enumerate(x_list):
                    # print a progress %
                    if self.verbose or verbose:
                        stdout.write('\r {}% completed'.format(round(100*(find(x_fs,x_list)+1)/len(x_list))))
                        stdout.flush()
                    # check that rho stays inside the lcfs
                    if x_fs > 0 and x_fs < 0.999:
                        # compute the psi level of the flux surface
                        psi_fs = float(interpolate.interp1d(derived[x_label],derived['psi'])(x_fs))
                        q_fs = float(interpolate.interp1d(derived[x_label],derived['qpsi'])(x_fs))
                        fpol_fs = float(interpolate.interp1d(derived[x_label],derived['fpol'])(x_fs))

                        # trace the flux surface contour and relabel the tracer output
                        time0 = time.time()
                        fs = tracer.contour(R,Z,psirz,psi_fs,threshold,i_center=[i_rmaxis,i_zmaxis],tracer_diag=tracer_diag,interp_method=interp_method)
                        tracer_timing += time.time()-time0
                        fs.update({x_label:x_fs, 'psi':psi_fs, 'q':q_fs, 'fpol':fpol_fs})
                        if x_label != 'rho_tor' and 'rho_tor' in derived:
                            fs.update({'rho_tor':interpolate.interp1d(derived[x_label],derived['rho_tor'])(x_fs)})
                        keys = copy.deepcopy(list(fs.keys()))
                        for key in keys:
                            if 'X' in key or 'Y' in key:
                                _key = (key.replace('X','R')).replace('Y','Z')
                                fs[_key] = fs.pop(key)
                        del fs['label']
                        del fs['level']
                        if not np.isfinite(fs['r']):
                            r_res = np.sqrt((R[i_rmaxis] - derived['rmaxis']) ** 2 + (Z[i_zmaxis] - derived['zmaxis']) ** 2)
                            fs['r'] = r_res * (psi_fs - derived['simag']) / (psirz[i_zmaxis,i_rmaxis] - derived['simag'])
                            if fs['r'] == 0.0:
                                fs['r'] = 1.0e-4
                        if not np.isfinite(fs['R0']):
                            fs['R0'] = derived['rmaxis']
                        if not np.isfinite(fs['Z0']):
                            fs['Z0'] = derived['zmaxis']
                        if not np.isfinite(fs['R_Zmax']):
                            fs['R_Zmax'] = derived['rmaxis']
                        if not np.isfinite(fs['R_Zmin']):
                            fs['R_Zmin'] = derived['rmaxis']
                        if not np.isfinite(fs['Z_max']):
                            fs['Z_max'] = derived['zmaxis'] + fs['r']
                        if not np.isfinite(fs['Z_min']):
                            fs['Z_min'] = derived['zmaxis'] - fs['r']

                        if analytic_shape:
                            time1 = time.time()
                            fs['miller_geo'] = LocalEquilibrium.extract_analytic_shape(fs)
                            analytic_timing += time.time()-time1

                        _incl_B = False
                        if incl_B:
                            if isinstance(incl_B,list):
                                _incl_B = incl_B[i_x_fs]
                            else:
                                _incl_B = incl_B

                        if _incl_B:

                            B_pol_fs = 0.0
                            B_tor_fs = fs['fpol'] / fs['R0']
                            if len(fs['R']) > 5:
                                # to speed up the Bpol interpolation generate a reduced Z,R mesh
                                i_R_in = find(fs['R_in'],self.derived['R'])-2
                                i_R_out = find(fs['R_out'],self.derived['R'])+2
                                i_Z_min = find(fs['Z_min'],self.derived['Z'])-2
                                i_Z_max = find(fs['Z_max'],self.derived['Z'])+2
                                R_mesh,Z_mesh = np.meshgrid(self.derived['R'][i_R_in:i_R_out],self.derived['Z'][i_Z_min:i_Z_max])
                                RZ_mesh = np.column_stack((Z_mesh.flatten(),R_mesh.flatten()))

                                # interpolate Bpol and Btor
                                B_pol_fs = interpolate.griddata(RZ_mesh,self.derived['B_pol_rz'][i_Z_min:i_Z_max,i_R_in:i_R_out].flatten(),(fs['Z'],fs['R']),method='cubic')
                                #B_pol_fs = np.array([])
                                #for i_R,RR in enumerate(fs['R']):
                                #    B_pol_fs = np.append(B_pol_fs,interpolate.interp2d(self.derived['R'][i_R_in:i_R_out],self.derived['Z'][i_Z_min:i_Z_max],self.derived['B_pol_rz'][i_Z_min:i_Z_max,i_R_in:i_R_out],bounds_error=False,fill_value='extrapolate')(RR,fs['Z'][i_R]))
                                B_tor_fs = interpolate.interp1d(self.derived['psi'],self.derived['fpol'],bounds_error=False)(np.array([psi_fs]))[0]/fs['R']
                            fs.update({'Bpol':B_pol_fs, 'Btor':B_tor_fs, 'B':np.sqrt(B_pol_fs**2+B_tor_fs**2)})

                            flux_integrand = np.sqrt(np.diff(fs['R']) ** 2.0 + np.diff(fs['Z']) ** 2.0) / np.abs(fs['Bpol'][:-1])
                            fs['Vprime'] = np.sum(flux_integrand)
                            fs['1/R'] = np.sum(flux_integrand / fs['R'][:-1]) / np.sum(flux_integrand)

                        else:
                            fs.update({'Bpol':np.array([]), 'Btor':np.array([]), 'B':np.array([]), 'Vprime':np.array([]), '1/R':np.array([])})

                        fs = list_to_array(fs)

                        # merge the flux surface data into the Equilibrium()
                        merge_trees(fs,fluxsurfaces)
                analytic_timing /= len(x_list)
                tracer_timing /= len(x_list)

                if verbose:
                    stdout.write('\n')
                    print('tracer time:{:.3f}s / flux-surface'.format(tracer_timing))
                    print('analytic extraction time:{:.3f}s / flux-surface'.format(analytic_timing))
                
                if tracer_diag == 'fs':
                    plt.show()

                if not x:
                    if 'rbbbs' in raw and 'zbbbs' in raw:
                        # find the geometric center, minor radius and extrema of the lcfs manually
                        lcfs = tracer.contour_center({'X':derived['rbbbs'],'Y':derived['zbbbs'],'level':derived['sibry'],'label':1.0})
                        keys = copy.deepcopy(list(lcfs.keys()))
                        for key in keys:
                            if 'X' in key or 'Y' in key:
                                _key = (key.replace('X','R')).replace('Y','Z')
                                lcfs[_key] = lcfs.pop(key)
                        lcfs.update({'theta_RZ':arctan2pi(lcfs['Z']-lcfs['Z0'],lcfs['R']-lcfs['R0'])})
                    else:
                        lcfs = tracer.contour(R,Z,psirz,derived['sibry'],derived['sibry'],i_center=[i_rmaxis,i_zmaxis],interp_method='bounded_extrapolation',return_self=False)
                        keys = copy.deepcopy(list(lcfs.keys()))
                        for key in keys:
                            if 'X' in key or 'Y' in key:
                                _key = (key.replace('X','R')).replace('Y','Z')
                                lcfs[_key] = lcfs.pop(key)
                        derived.update({'rbbbs':lcfs['R'],'zbbbs':lcfs['Z'],'nbbbs':len(lcfs['R'])})
                    if analytic_shape:
                        lcfs.update({'miller_geo':LocalEquilibrium.extract_analytic_shape(lcfs)})
                
                    lcfs.update({x_label:x_fs, 'psi':psi_fs, 'q':derived['qpsi'][-1], 'fpol':derived['fpol'][-1]})
                    if x_label != 'rho_tor' and 'rho_tor' in derived:
                        lcfs.update({'rho_tor':interpolate.interp1d(derived[x_label],derived['rho_tor'])(x_fs)})
                    keys = copy.deepcopy(list(lcfs.keys()))
                    for key in keys:
                        if 'X' in key or 'Y' in key:
                            _key = (key.replace('X','R')).replace('Y','Z')
                            lcfs[_key] = lcfs.pop(key)
                    del lcfs['label']
                    del lcfs['level']

                    _incl_B = False
                    if incl_B:
                        if isinstance(incl_B,list):
                            _incl_B = incl_B[i_x_fs]
                        else:
                            _incl_B = incl_B

                    if _incl_B:

                        B_pol_fs = 0.0
                        B_tor_fs = lcfs['fpol'] / lcfs['R0']
                        if len(lcfs['R']) > 5:
                            # to speed up the Bpol interpolation generate a reduced Z,R mesh
                            i_R_in = find(lcfs['R_in'],self.derived['R'])-2
                            i_R_out = find(lcfs['R_out'],self.derived['R'])+2
                            i_Z_min = find(lcfs['Z_min'],self.derived['Z'])-2
                            i_Z_max = find(lcfs['Z_max'],self.derived['Z'])+2
                            R_mesh,Z_mesh = np.meshgrid(self.derived['R'][i_R_in:i_R_out],self.derived['Z'][i_Z_min:i_Z_max])
                            RZ_mesh = np.column_stack((Z_mesh.flatten(),R_mesh.flatten()))

                            # interpolate Bpol and Btor
                            B_pol_fs = interpolate.griddata(RZ_mesh,self.derived['B_pol_rz'][i_Z_min:i_Z_max,i_R_in:i_R_out].flatten(),(lcfs['Z'],lcfs['R']),method='cubic')
                            B_tor_fs = interpolate.interp1d(self.derived['psi'],self.derived['fpol'],bounds_error=False)(np.array([psi_fs]))[0]/lcfs['R']
                        lcfs.update({'Bpol':B_pol_fs, 'Btor':B_tor_fs, 'B':np.sqrt(B_pol_fs**2+B_tor_fs**2)})

                        flux_integrand = np.sqrt(np.diff(lcfs['R']) ** 2.0 + np.diff(lcfs['Z']) ** 2.0) / np.abs(lcfs['Bpol'][:-1])
                        lcfs['Vprime'] = np.sum(flux_integrand)
                        lcfs['1/R'] = np.sum(flux_integrand / lcfs['R'][:-1]) / np.sum(flux_integrand)

                    else:
                        lcfs.update({'Bpol':np.array([]), 'Btor':np.array([]), 'B':np.array([]), 'Vprime':np.array([]), '1/R':np.array([])})

                    lcfs = list_to_array(lcfs)

                    # append the lcfs values to the end of the flux surface data
                    merge_trees(lcfs,fluxsurfaces)

                    # add a zero at the start of all flux surface quantities
                    for key in fluxsurfaces:
                        if key in ['R']:
                            fluxsurfaces[key].insert(0,np.array([derived['rmaxis']]))
                        elif key in ['Z']:
                            fluxsurfaces[key].insert(0,np.array([derived['zmaxis']]))
                        elif key in ['R0','R_Zmax','R_Zmin','R_in','R_out']:
                            fluxsurfaces[key].insert(0,derived['rmaxis'])
                        elif key in ['Z0','Z_max','Z_min']:
                            fluxsurfaces[key].insert(0,derived['zmaxis'])
                        elif key in ['q','kappa','delta','zeta','s_kappa','s_delta','s_zeta']:
                            fluxsurfaces[key].insert(0,fluxsurfaces[key][0])
                        else:
                            if isinstance(fluxsurfaces[key],dict):
                                for _key in fluxsurfaces[key]:
                                    if _key in ['delta_u','delta_l','delta','kappa','zeta_uo','zeta_ui','zeta_li','zeta_lo','zeta']:
                                        fluxsurfaces[key][_key].insert(0,fluxsurfaces[key][_key][0])
                                    else:
                                        fluxsurfaces[key][_key].insert(0,0.*fluxsurfaces[key][_key][-1])
                            elif isinstance(fluxsurfaces[key],list):
                                fluxsurfaces[key].insert(0,0.*fluxsurfaces[key][-1])

                # add the midplane average geometric flux surface quantities to derived
                derived['Ro'] = np.array(fluxsurfaces['R0'])
                if derived['Ro'][0] == 0.0:
                    derived['Ro'][0] = derived['Ro'][1] # clear the starting zero
                derived['R0'] = derived['Ro'][-1] # midplane average major radius of the lcfs
                derived['Zo'] = np.array(fluxsurfaces['Z0'])
                derived['Z0'] = derived['Zo'][-1] # average elevation of the lcfs
                derived['r'] = np.array(fluxsurfaces['r'])
                if x and 'rbbbs' and 'zbbbs' in raw:
                    derived['a'] = tracer.contour_center({'X':derived['rbbbs'],'Y':derived['zbbbs'],'level':derived['sibry'],'label':1.0})['r']
                else:
                    derived['a'] = derived['r'][-1] # midplane average minor radius of the lcfs
                derived['epsilon'] = derived['r']/derived['Ro']
                derived['r/a'] = derived['r']/derived['a']

                # add the midplane average major radius and elevation derivatives to derived
                derived['dRodr'] = np.gradient(derived['Ro'],derived['r'])
                derived['dZodr'] = np.gradient(derived['Zo'],derived['r'])

                # add the magnetic shear to derived
                derived['s'] = derived['r']*np.gradient(np.log(fluxsurfaces['q']),derived['r'],edge_order=2)

                # add several magnetic field quantities to derived
                derived['Bref_eqdsk'] = derived['fpol'][0]/derived['rmaxis']
                derived['Bref_miller'] = fluxsurfaces['fpol']/derived['Ro']
                #derived['B_unit'] = interpolate.interp1d(derived['r'],(1/derived['r'])*np.gradient(derived['phi'],derived['r'],edge_order=2))(derived['r'])
                derived['B_unit'] = interpolate.interp1d(derived['r'],(fluxsurfaces['q']/derived['r'])*np.gradient(fluxsurfaces['psi'],derived['r'],edge_order=2))(derived['r'])
                
                # add beta and alpha, assuming the pressure profile included in the equilibrium and Bref=derived['Bref_eqdsk]
                #derived['beta'] = 8*np.pi*1E-7*derived['pres']/(derived['Bref_eqdsk']**2)
                #derived['alpha'] = -1*derived['qpsi']**2*derived['Ro']*np.gradient(derived['beta'],derived['r'])

                if analytic_shape:
                    derived['miller_geo'] = list_to_array(copy.deepcopy(fluxsurfaces['miller_geo']))

                    # compute the shear of the Turnbull-Miller shaping parameters
                    derived['miller_geo']['s_kappa'] = derived['r']*np.gradient(np.log(derived['miller_geo']['kappa']),derived['r'],edge_order=2)
                    derived['miller_geo']['s_delta'] = (derived['r']/np.sqrt(1-derived['miller_geo']['delta']**2))*np.gradient(derived['miller_geo']['delta'],derived['r'],edge_order=2)
                    derived['miller_geo']['s_delta_ga'] = derived['r']*np.gradient(derived['miller_geo']['delta'],derived['r'],edge_order=2)
                    derived['miller_geo']['s_zeta'] = derived['r']*np.gradient(derived['miller_geo']['zeta'],derived['r'],edge_order=2)
                
                return self

    def map_on_equilibrium(self,x=None,y=None,x_label=None,interp_order=9,extrapolate=False):
        """Map a 1D plasma profile on to the `x_label` radial coordinate basis of this `Equilibrium`.

        Args:
            `x` (array): vector of the radial basis of the existing profile in units of `x_label`.
            `y` (array): vector of the 1D profile of the plasma quantity that is mapped on this `Equilibrium`.
            `x_label` (str): label of the radial coordinate specification of `x`.
            `interp_order` (float): the interpolation order used in the remapping of the 1D profile, [default] 9 based on experience.
            `extrapolate` (bool): [True] use `fill_value` = 'extrapolate' in the interpolation, or [False, default] not.

        Returns:
            Two vectors, the x vector of `Equilibrium.derived` [`x_label`] and the y vector of the remapped 1D profile.
        """

        # remap the provided y profile, onto the x basis of x_label in this equilibrium
        if extrapolate:
            y_interpolated = interpolate.interp1d(x,y,kind=interp_order,bounds_error=False,fill_value='extrapolate')(self.derived[x_label])
        else:
            y_interpolated = interpolate.interp1d(x,y,kind=interp_order,bounds_error=False)(self.derived[x_label])
        
        return self.derived[x_label],y_interpolated
    
    def refine_equilibrium(self,nw=None,nh=None,nbbbs=None,interp_order=9,verbose=False):
        """Refine the R,Z refine of the `Equilibrium` through interpolation, assuming a g-EQDSK file as origin.

        Args:
            `nw` (int): desired grid refine of the 1D profiles and 2D psi(R,Z) map radial coordinate.
            `nh` (int): desired grid refine of the 2D psi(R,Z) map vertical coordinate.
            `nbbbs` (int, optional): desired grid refine of the last closed flux surface plasma boundary trace.
            `interp_order` (int, optional): the interpolation order used in the remapping of the 1D profiles, [default] 9 based on experience.
            `retain_original` (bool, optional): [True] store the original raw g-EQDSK equilibrium data in `Equilibrium.original`, or [False, default] not.
            `self_consistent` (bool, optional): [True, default] re-derive and re-trace all the existing additional data when applied to an existing `Equilibrium`, or [False] not.

        Returns:
            self

        Raises:
            ValueError: if the provided `nw` is smaller than the native refine of the `Equilibrium`.
        """
        if verbose:
            print('Refining Equilibrium to {}x{}...'.format(nw,nw))
        
        old_w = np.linspace(0,1,self.raw['nw'])
        old_h = np.linspace(0,1,self.raw['nh'])
        if nw:
            new_w = np.linspace(0,1,nw)
            refine_w = True
        else:
            new_w = old_w
            refine_w = False
            #raise ValueError('Provided nw does not refine the equilibrium, provided:{} < exisiting:{}'.format(nw,self.derived['nw']))
        if nh:
            new_h = np.linspace(0,1,nh)
            refine_h = True
        else:
            new_h = old_h
            refine_h = False
            #raise ValueError('Provided nh does not refine the equilibrium, provided:{} < exisiting:{}'.format(nh,self.derived['nh']))
        if nbbbs and nbbbs > self.raw['nbbbs']:
            pass
        
        for quantity in self.raw.keys():
            if isinstance(self.raw[quantity],np.ndarray):
                if refine_w and self.raw[quantity].size == old_w.size:
                    self.derived[quantity] = interpolate.interp1d(old_w,self.raw[quantity],kind=interp_order)(new_w)
                elif (refine_h or refine_w) and self.raw[quantity].size == old_w.size*old_h.size:
                    _old_w,_old_h = np.meshgrid(old_w,old_h)
                    old_hw = np.column_stack((_old_w.flatten(),_old_h.flatten()))
                    _new_w,_new_h = np.meshgrid(new_w,new_h)
                    self.derived[quantity] = interpolate.griddata(old_hw,self.raw[quantity].flatten(),(_new_w,_new_h),method='cubic')
                    #self.derived[quantity] = interpolate.interp2d(old_w,old_h,self.raw[quantity],kind='quintic')(new_w,new_h)
                else:
                    self.derived[quantity] = copy.deepcopy(self.raw[quantity])
            else:
                self.derived[quantity] = copy.deepcopy(self.raw[quantity])
        
        self.derived['nw'] = nw
        self.derived['nh'] = nh

        return self

    def plot_derived(self,):

        return self
    
    def plot_fluxsurfaces(self,):

        return self
