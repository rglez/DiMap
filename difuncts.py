#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on : Sun Mar 15 13:18:50 2020
@author    : Roy Gonzalez-Aleman
@mail      : rglez.developer@gmail.com
"""
import glob
import itertools as it
import math
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import os
import pandas as pd
import pickle
import prody as prd
import tempfile
import time
from collections import OrderedDict, namedtuple as nt
from multiprocessing import Pool
from scipy.interpolate import griddata
from subprocess import run


def parse_pdb(pdb_file):
    """
    DESCRIPTION
    Parse .pdb files for finding system and atomic information.

    Arguments:
        pdb_file (srt): path to a formatted .pdb file.
    Returns:
        parsed_pdb (pandas.DataFrame): dataframe with parsed information in
        columns: serial, chain, resnum, resname, atomname, atomtype,
        atomcharge, atomicnumber, neighbors, valence, element and index.
    """

    # ---- open file ----------------------------------------------------------
    pdb_file = '/home/tom/TomHub/DiMap/example/1-2PG.pdb'
    with open(pdb_file, 'rt') as psf:
        lines = psf.readlines()
    # ---- parse atomic info --------------------------------------------------
    info = []
    info_tup = nt('Atom', ['Record', 'AtomNum', 'AtomName', 'AltLoc',
                           'ResName', 'ChainID', 'ResNum', 'Insert', 'x', 'y',
                           'z', 'Occup', 'BFactor', 'SegID', 'Symbol',
                           'Charge'])
    for line in lines:
        if ('ATOM' in line) or ('HETATM' in line):
            line = line.strip()
            splitted = [line[0:6], line[6:11], line[11:16], line[16:17],
                        line[17:21], line[21:22], line[22:26], line[26:30],
                        line[30:38], line[38:46], line[46:54], line[54:60],
                        line[60:66], line[66:76], line[76:78], line[78:80]]
            splitted = [x.strip() for i, x in enumerate(splitted)]
            info.append(info_tup(*splitted))
    # ---- parse bonded info --------------------------------------------------
    parsed_pdb = pd.DataFrame(info)
    parsed_pdb = parsed_pdb.astype(
     {'Record': 'str', 'AtomNum': 'int32', 'AtomName': 'str', 'ResName': 'str',
      'x': 'float32', 'y': 'float32', 'z': 'float32', 'Occup': 'float32',
      'BFactor': 'float32', 'SegID': 'str', 'Symbol': 'str'})

    return parsed_pdb


def write(pdb_df, output_name):
    """
    Writes a well formated pdb file from a DataFrame. Created because
    'to_string()' pandas.DataFrame's method has a bug that prints an extra
    white space between every two columns.
    RETURNS output_name.
    """
    # ---- Formatters in agreement with PDB standards ----------------------- #
    formatters = ('{:6}{:>5}{:>5}{:1}{:4}{:1}{:>4}{:4}{:>8.3f}{:>8.3f}' +
                  '{:>8.3f}{:>6}{:>6}{:>10}{:>2}{:2}')
    # ---- Formatting each line of a pandas.DataFrame ----------------------- #
    with open(output_name, 'wt') as output:
        for line in range(len(pdb_df)):
            row = list(pdb_df.iloc[line])
            row = formatters.format(row[0], row[1], row[2], row[3], row[4],
                                    row[5], row[6], row[7], row[8], row[9],
                                    row[10], row[11], row[12], row[13],
                                    row[14], row[15])
            output.write(row + '\n')
        output.write('END \n')
    return output_name


def parse_psf(psf_file):
    """
    DESCRIPTION
    Parse .psf_14 files for finding system and atomic information as well as
    bonded information and each atom neighbors.

    Arguments:
        psf_file (srt): path to a formatted .psf_14 file.
    Returns:
        parsed_psf (pandas.DataFrame): dataframe with parsed information in
        columns: serial, chain, resnum, resname, atomname, atomtype,
        atomcharge, atomicnumber, neighbors, valence, element and index.
    """

    # ---- open file ----------------------------------------------------------
    with open(psf_file, 'rt') as psf:
        lines = psf.readlines()
    # ---- parse atomic info --------------------------------------------------
    atomic_str = '!NATOM'
    for i, line in enumerate(lines):
        if atomic_str in line:
            n, total = i, int(line.split()[0])
    info = []
    info_tup = nt('Atom', ['serial', 'chain', 'resnum', 'resname', 'atomname',
                  'atomtype', 'atomcharge', 'atomicnumber'])
    for line in lines[n + 1: n + 1 + total]:
        splitted = line.split()[:8]
        splitted[0] = int(splitted[0])
        splitted[2] = int(splitted[2])
        splitted[6] = float(splitted[6])
        splitted[7] = float(splitted[7])
        info.append(info_tup(*splitted))
    parsed_psf = pd.DataFrame(info)
    # ---- parse bonded info --------------------------------------------------
    bond_str = '!NBOND'
    for i, line in enumerate(lines):
        if bond_str in line:
            n, total = i, int(line.split()[0])
    pairs = []
    for line in lines[n+1:]:
        splitted = line.split()
        if not splitted:
            break
        pairs.extend([int(x) - 1 for x in splitted])
    neighbors = OrderedDict()
    for i, x in enumerate(info):
        neighbors[i] = []
    for i, x in enumerate(range(0, len(pairs), 2)):
        a = pairs[x]
        b = pairs[x+1]
        neighbors[a].append(b)
        neighbors[b].append(a)
    # ---- dataframe of info --------------------------------------------------
    parsed_psf['neighbors'] = neighbors.values()
    parsed_psf['valence'] = [len(x) for x in neighbors.values()]
    parsed_psf['element'] = [x[0] for x in parsed_psf.atomname]
    parsed_psf['index'] = parsed_psf.serial - 1
    return parsed_psf


def find_glycosidics(parsed_psf):
    """
    DESCRIPTION
    Find the index of atoms involved in the glycosidic linkage of a (di/poly)-
    saccharide.

    Arguments:
        parsed_psf (pandas.DataFrame): dataframe with parsed information in
        columns.
    Returns:
        glycosidic (list): nested list containing a list of list for every
        glycosidic linkage detected.
    """

    oxygens = parsed_psf[(parsed_psf['element'] == 'O') &
                         (parsed_psf['valence'] == 2)]
    glycosidic = []
    names = []
    for i in range(oxygens.shape[0]):
        a, b = oxygens.iloc[i].neighbors
        if parsed_psf.iloc[a].resnum != parsed_psf.iloc[b].resnum:
            o = oxygens.iloc[i]['index']
            if parsed_psf.iloc[a].atomname == 'H1':
                c1 = a
                cx = b
            else:
                c1 = b
                cx = a
            n_c1 = parsed_psf.iloc[c1].neighbors
            h1 = (parsed_psf.iloc[n_c1].element == 'H').idxmax()
            n_cx = parsed_psf.iloc[cx].neighbors
            hx = (parsed_psf.iloc[n_cx].element == 'H').idxmax()
            glycosidic.append([[h1, c1, o, cx], [c1, o, cx, hx]])
            names.append('{}_{}_{}_{}'.format(parsed_psf.iloc[c1].resname,
                                              parsed_psf.iloc[c1].atomname,
                                              parsed_psf.iloc[cx].atomname,
                                              parsed_psf.iloc[cx].resname))
    assert len(names) == 1
    return names[0], glycosidic


class log:
    """
    Defines a namd.log object. Only the **name** of the object is needed to
    initialize it.
    """

    def __init__(self, name):
        self.name = name
        self.lines = self.get_raw_lines()
        self.all_energies = self.get_all_energies()

    def get_raw_lines(self):
        """ Gets all lines in a **namd.log** file using **readlines**.

        Returns:
            (list): **self.readlines()**
        """
        with open(self.name, 'rt') as inp:
            return inp.readlines()

    def get_all_energies(self):
        """ Gets all energy terms printed on a **namd.log** as a DataFrame.

        Energies are retrieved for every line that contains **ENERGY:** and
        whose lenght are exactly the same as the titles of the energy terms.

        Returns:
            (pd.DataFrame): a DataFrame containing all energy terms found in
            **namd.log** object.
        """
        lines = self.lines
        # getting titles
        for line in lines:
            if 'ETITLE:' in line:
                titles = line.split()[1:]
                break
        # getting all energies
        all_energies = []
        for line in lines:
            if 'ENERGY:' in line and len(line.split()) - 1 == len(titles):
                all_energies.append(line.split()[1:])
        # getting a DataFrame of floats
        return pd.DataFrame(data=all_energies, columns=titles).astype(float)

    def get_last_energy(self):
        """ Gets **only** the last energy of a **namd.log** file.

        Returns:
            (float): last energy term on **namd.log** file corresponding to the
            **TOTAL** energy of the system.
        """
        return self.get_all_energies()['TOTAL'].iloc[-1]

    def get_dcd_energies(self):
        """ Gets energy **TOTAL** terms of the system that were printed on a
        **.dcd** file.

        .. warning ::

          Remember that len(dcd_steps) **might not be ==** len(all_energies).

        Returns:
            (pd.DataFrame): a DataFrame containing timesteps (TS) and energy
            of the step (TOTAL).

        """
        lines = self.lines
        dcd_written_steps = []
        for line in lines:
            if 'WRITING COORDINATES TO DCD FILE' in line:
                dcd_written_steps.append(int(line.split(sep='STEP')[1]))
        dcd_energies = pd.DataFrame()
        dcd_energies['TS'] = dcd_written_steps
        dcd_energies['TOTAL'] = self.all_energies['TOTAL'][dcd_written_steps]
        return dcd_energies


# =============================================================================
# central.gnr
# =============================================================================
def generic_matplotlib():
    """
    Some customizations of matplotlib.
    """
    mpl.rc('figure', figsize=[12, 8], dpi=300)
    mpl.rc('xtick', direction='in', top=True)
    mpl.rc('xtick.major', top=False, )
    mpl.rc('xtick.minor', top=True, visible=True)
    mpl.rc('ytick', direction='in', right=True)
    mpl.rc('ytick.major', right=True, )
    mpl.rc('ytick.minor', right=True, visible=True)

    mpl.rc('axes', labelsize=20)
    mpl.rc('lines', linewidth=8, color='k')
    mpl.rc('font', family='monospace', size=20)
    mpl.rc('grid', alpha=0.5, color='gray', linewidth=1, linestyle='--')


def overwrite_dir(dir_name):
    """
    Overwrites an existing directory silently.
    """
    try:
        os.mkdir(dir_name)
        return dir_name
    except FileExistsError:
        return False


# =============================================================================
# central.mm
# =============================================================================
def conf_creator(PDB, PSF, PRM, steps=1000, per_cycle=20):
    """
    Creates a configuration file for NAMD minimization job.
    RETURNS name of created file.
    """
    file = PDB
    output_name = "OPT_" + PDB
    with open(file[:-4]+".CONF", "wt") as output:
        output.write(
         'structure           ' + PSF + '\n'
         'coordinates         ' + PDB + '\n'
         'set outputname      ' + output_name + '\n'
         'temperature         ' + '0\n'
         'paraTypeCharmm      ' + 'on\n'
         'parameters          ' + PRM + '\n'
         'exclude             ' + 'scaled1-4\n'
         '1-4scaling          ' + '1.0\n'
         # 'cutoff              ' + '12.0\n'

         # GBIS
         'cutoff             ' + '16.0\n'
         'GBIS               ' + 'on\n'
         'solventDielectric  ' + '80\n'
         'switchdist          ' + '15.0\n'
         'pairlistdist        ' + '18\n'


         'switching           ' + 'on\n'
         # 'switchdist          ' + '10.0\n'
         # 'pairlistdist        ' + '13.5\n'
         'timestep            ' + '1.0\n'
         'nonbondedFreq       ' + '1\n'
         'fullElectFrequency  ' + '2\n'
         'stepspercycle       ' + '{}\n'.format(per_cycle) +
         'outputName          ' + '$outputname\n'
         'dcdfreq             ' + str(1) + '\n'  # OJO str(steps)
         'xstFreq             ' + str(1) + '\n'
         'outputEnergies      ' + str(1) + '\n'
         'outputTiming        ' + str(1) + '\n'
         'binaryoutput        ' + 'no\n'
         'if {1} {\n'
         'fixedAtoms          ' + 'on\n'
         'fixedAtomsFile      ' + PDB + '\n'
         'fixedAtomsCol       ' + 'B\n'
         'fixedAtomsForces    ' + 'on\n'
         '}\n'
         'minimization        ' + 'on\n'
         'minimize            ' + str(steps) + '\n'
         )
        return file[:-4]+'.CONF'


def mk_mesh(x=5, y=5, xmin=-180, xmax=180, ymin=-180, ymax=180):
    """
    Creates a list of X,Y pairs of values in a grid format.
    RETURN created meshgrid.
    """
    x_values = list(np.arange(xmin, xmax, x))
    y_values = list(np.arange(ymin, ymax, y))
    meshgrid = []
    for y in range(len(y_values)):
        for x in range(len(x_values)):
            xy_pair = [y_values[y], x_values[x]]
            meshgrid.append(xy_pair)
    return meshgrid

def mk_mesh2(xy_min=-180, xy_max=180, xy_step=60):
    """
    Creates a list of X,Y pairs of values in a grid format.
    RETURN created meshgrid.
    """

    xy_values = range(xy_min, xy_max + xy_step, xy_step)
    xy_values_up1_down = []
    xy_values_up_down_second_fix = []
    xy_values_down_down = []
    xy_values_up_down = []

    for i in xy_values:
        for j in xy_values:
            if i != j:
                down_down = (i, j)
                counter_flow = (i, -1 * j)
                xy_values_down_down.append(down_down)
                xy_values_up_down.append(counter_flow)
                fix_one_flow = (i, j)
                fix_second_flow = (j, -1*i)
                xy_values_up1_down.append(fix_one_flow)
                xy_values_up_down_second_fix.append(fix_second_flow)

    xy_values_up_down1 = xy_values_up1_down.copy()
    xy_values_up1_down.reverse()
    xy_values_up_up = xy_values_down_down.copy()
    xy_values_down_down.reverse()
    xy_values_down_up = xy_values_up_down.copy()
    xy_values_up_down.reverse()
    xy_values_up_down_second_fix1 = xy_values_up_down_second_fix.copy()
    xy_values_up_down_second_fix.reverse()

    meshes = {'down-up': xy_values_up_down, 'up-down': xy_values_down_up,
              'up-up': xy_values_up_up, 'down-down': xy_values_down_down,
              'up*-down': xy_values_up1_down, 'down*-up': xy_values_up_down1,
              'up-down*': xy_values_up_down_second_fix1,
              'down-up*': xy_values_up_down_second_fix}

    return meshes


def extract_values(itertools_product, reverse=False):
    if reverse:
        return [(x[1], x[0]) for x in itertools_product]
    return [(x[0], x[1]) for x in itertools_product]


def mk_mesh1(xy_min=-180, xy_max=180, xy_step=60):
    # create combinations of flows
    flows = ['up', 'down']
    flows_combinations = set(it.combinations([*flows, *flows], 2))
    # define up & down values once
    values = {'up': range(xy_min, xy_max, xy_step),
              'down': range(xy_max, xy_min, -xy_step)}
    # always fix one position
    fixed = [0, 1]
    fixed_combinations = set([x for x in it.combinations([*fixed, *fixed], 2)
                              if x[0] != x[1]])

    # explore all possible combinations
    meshes = {}
    for combination in flows_combinations:
        for fix in fixed_combinations:
            key = "{}-{}-{}-{}".format(*combination, *fix)
            flow1, flow2 = combination
            if fix[0] == 0:
                iter_product = it.product(values[flow2], values[flow1])
                meshes.update({key: extract_values(iter_product, reverse=True)}
                              )
            else:
                iter_product = it.product(values[flow1], values[flow2])
                meshes.update({key: extract_values(iter_product, reverse=False)
                               })
    return meshes

# =============================================================================
# central.rotations
# =============================================================================
def get_both_sides(PSF_file, dihedral):
    """
    "Cuts" a non-cyclic molecule in two parts. One of them (side_rot) contains
    atoms that will rotate using Rodrigues rotation approach.
    RETURN side_rot
    """
    # ======================================================================= #
    connect = get_psf_info(PSF_file)[3]            # dict of connectivities
    b = dihedral[1]                                # smaller will be fixed
    c = dihedral[2]                                # greater will be rotated
    # ======================================================================= #
    # ---- First search for closest neighbors ------------------------------- #
    side_rot = []                                 # getting neighbors of c
    side_rot.extend(connect[c])                   # (...)
    side_rot.remove(b)                            # (...) that are != b
    # ---- Subsequent searchs ----------------------------------------------- #
    while True:
        m = len(side_rot)
        for atom in side_rot:
            if atom != b:
                side_rot.extend(connect[atom])
                side_rot = list(set(side_rot))
                s = len(side_rot)
        # ---- Breaking search if no new neighbor is found ------------------ #
        if s - m == 0:
            break
    # ---- Excluding b and c from those that will rotate -------------------- #
    side_rot = set(side_rot)
    side_rot.remove(b)
    return side_rot


def get_psf_info(PSF_file):
    """
    Get information from CHARMM topology file (PSF)
    RETURN[0] total number of atoms
    RETURN[1] dataframe of pdb_like section in PSF_file
    RETURN[2] list of tuples with connected pairs
    RETURN[3] dictionary of connectivities {int(atom):list(connectivities)}
    RETURN[4] dictionary of connectivities {int(atom):list(elements)}
    RETURN[5] strings containing applied patches
    """
    # ==== Finding string patterns positions ================================ #
    with open(PSF_file, 'rt') as psf:
        lines = psf.readlines()
        patches = [line.rstrip() for line in lines if 'REMARKS patch' in line]
    str_0 = '!NBOND:'
    str_1 = '!NTHETA:'
    str_2 = '!NATOM'
    for index, line in enumerate(lines):
        if str_0 in line.split():
            start = index + 1
        if str_1 in line.split():
            end = index - 1
        if str_2 in line.split():
            NATOMS = int(line.split()[0])  # RETURN[0]
            start2 = index + 1
    # ======================================================================= #

    # ---- Getting pdb-like info section. RETURN[1] ------------------------- #
    pdb_like = lines[start2:(start - 2)]
    pdb_row = [line.split() for line in pdb_like]
    pdb_DF = pd.DataFrame(pdb_row, columns=['num', 'seg', 'resnum', 'resname',
                                            'atomname', 'atomtype', 'charge',
                                            'mass', 'field'])
    # ---- Getting pairs of connected atoms. RETURN[2] ---------------------- #
    extract = lines[start:end]
    row_bonds = [line.split() for line in extract]
    bonds = []
    for x in row_bonds:
        bonds.extend(x)
    connected_pairs = []
    while len(bonds) != 0:
        connected_pairs.append((int(bonds[0]) - 1, int(bonds[1]) - 1))
        bonds.pop(0)
        bonds.pop(0)
    # ---- Getting connectivities as dict of index of atoms. RETURN[3] ------ #
    connectivities = {}
    for atom in range(0, NATOMS):
        neighbors = []
        for pair in connected_pairs:
            if atom in pair:
                neighbors.extend(pair)
        neighbors = set(neighbors)
        neighbors.remove(atom)
        neighbors = list(neighbors)
        connectivities.setdefault(atom, neighbors)
    # ---- Getting connectivities as dict of name of atoms. RETURN[4] ------- #
    elements = {}
    for element in range(0, NATOMS):
        voisins = []
        for pair in connected_pairs:
            if element in pair:
                voisins.extend(pair)
        voisins = set(voisins)
        voisins.remove(element)
        voisins = list(voisins)
        voisins = [pdb_DF.iloc[x].atomtype[0] for x in voisins]
        elements.setdefault(element, voisins)

    return NATOMS, pdb_DF, connected_pairs, connectivities, elements, patches


def measure_dihedral(cartesians, dihedral):
    """ Measures a dihedral angle from given cartesian coordinates.

    Args:
        cartesians (np.array): array of cartesian coordinates x, y and z.
        dihedral (list): a 4-member list containing **indices** of a
                         dihedral angle. **indices** must be in cartesian.
    Returns:
        (float): dihedral angle in range [-180 : +180].
    """
    a, b, c, d = dihedral
    # ---- Vectors ---------------------------------------------------------- #
    vec1 = cartesians[b] - cartesians[a]
    vec2 = cartesians[c] - cartesians[b]
    vec3 = cartesians[d] - cartesians[c]
    # ---- Products --------------------------------------------------------- #
    cross_1 = np.cross(vec1, vec2)
    cross_2 = np.cross(vec2, vec3)
    sign = np.cross(cross_1, cross_2)
    sign_check = np.dot(vec2, sign)
    # ---- Getting angle ---------------------------------------------------- #
    cos_angle = np.dot(cross_1, cross_2) / (np.linalg.norm(cross_1) *
                                            np.linalg.norm(cross_2))
    angle = round(math.degrees(math.acos(round(cos_angle, 2))), 2)
    # ---- Checking sign ---------------------------------------------------- #
    if sign_check < 0:
        angle_sign = -angle
    else:
        angle_sign = angle
    return angle_sign


def set_dihedral_value(cartesians, dihedral, angle, side_rot):
    """ Sets **dihedral** to a specific **angle** value using
    "Rodrigues Rotation Method".

    Args:
        cartesians (np.array): array of cartesian coordinates x, y and z.
        dihedral (list)      : a 4-member list containing **indices** of a
                               dihedral angle.
        angle (float, int)   : **delta** of the rotation that will be made.
        side_rot (set)       : set of atoms' index that defines the side of the
                               molecule that will be rotated.
    Returns:
        (np.array): an array of rotated cartesians.
    """
    x0 = measure_dihedral(cartesians, dihedral)
    xf = angle
    if {
        (x0 >= 0 and xf >= 0) or
        (x0 < 0 <= xf) or
        ((x0 < 0 and xf < 0) and xf > x0)
       }:
        incremental_angle = xf - x0
    else:
        incremental_angle = 360 + (xf - x0)
    rotated_cartesians = rotate_dihedral(cartesians, dihedral,
                                         incremental_angle, side_rot)
    return rotated_cartesians


def module(vector):
    """Determines the norm (lenght) of a vector.

    Args:
        vector (np.array): an array containing x, y, and z cartesian
                           coordinates.
    Returns:
        (float): module of **vector**.
    """
    module = np.sqrt((vector**2).sum())
    return module


def rotate_dihedral(cartesians, dihedral, angle, side_rot):
    """ Rotates by **angle** degrees a specified dihedral.

    Args:
        cartesians (np.array): array of cartesian coordinates x, y and z.
        dihedral (list)      : a 4-member list containing **indices** of a
                               dihedral angle.
        angle (float, int)   : **delta** of the rotation that will be made.
        side_rot (set)       : set of atoms' index that defines the side of the
                               molecule that will be rotated.
    Returns:
        (pd.DataFrame): a DataFrame with rotated cartesians.
    """
    # ---- Importing global coordinates and reseting origin ----------------- #
    A, B, C, D = dihedral
    cartesians = cartesians - cartesians[B]
    # ---- Defyning axis around to which rotate ----------------------------- #
    B = cartesians[B]
    C = cartesians[C]
    axis = C - B
    axis = axis / module(axis)
    # ---- Rotate "angle" degrees (radians) each atom "upper" axis ---------- #
    angle = np.radians(angle)
    for atom in side_rot:
        d_vec = (np.dot(axis, cartesians[atom])) * axis
        r_vec = cartesians[atom] - d_vec
        r_prime = (r_vec*np.cos(angle)) + (np.cross(axis, r_vec))*np.sin(angle)
        rotated_vec = d_vec + r_prime
        cartesians[atom] = rotated_vec
    return cartesians


# =============================================================================
#
# =============================================================================

def plot_dimap(N, log_file, out_name):
    """
    """
    table = pd.read_table(log_file, skiprows=1, delimiter='\s+',
                          names=['phi', 'psi', 'e'])
    table['e'][table.e > N] = N

    x_to_plot = np.asarray(table.phi)
    y_to_plot = np.asarray(table.psi)
    z_to_plot = np.asarray(table.e)

    xi = np.linspace(-180, 180, 720)
    yi = np.linspace(-180, 180, 720)

    plt.figure(1)
    xmax, xmin = max(xi), min(xi)
    ymax, ymin = max(yi), min(yi)
    grid_xi = np.empty((yi.shape[0], xi.shape[0]))
    grid_yi = np.empty((yi.shape[0], xi.shape[0]))
    for c in range(yi.shape[0]):
        grid_xi[c, :] = xi
    for c in range(xi.shape[0]):
        grid_yi[:, c] = yi

    zi = griddata((x_to_plot, y_to_plot), z_to_plot, (grid_xi, grid_yi),
                  method='linear')
    plt.contour(xi, yi, zi, range(0, int(N), 2), linewidths=0.5, colors='k',
                linestyles='dashed')
    plt.pcolormesh(xi, yi, zi, cmap=plt.get_cmap('rainbow'))
    plt.colorbar()

    incr = (xmax-xmin)/100
    plt.xlim(xmin-incr, xmax+incr)
    incr = (ymax-ymin)/100
    plt.ylim(ymin-incr, ymax+incr)
    plt.xlabel('PHI')
    plt.ylabel('PSI')
    plt.savefig(out_name)
    plt.close()


def naive_clustering(traj, energ, rms_lim):
    """
    """
    idxs = np.where(energ <= 5)[0]
    idxs_energies = energ[idxs]

    leaders = []
    clusters = np.ndarray(idxs.size, dtype=int)
    clusters.fill(-1)
    counter = -1
    while idxs_energies.min() != np.inf:
        counter += 1
        minim_idx = idxs_energies.idxmin()
        leaders.append(minim_idx)
        rmsds = md.rmsd(traj, traj, minim_idx, precentered=True)[idxs]*10
        neighb = np.where(rmsds <= rms_lim)[0].tolist()
        neighb_idx = idxs[neighb]
        clusters[neighb] = counter
        idxs_energies[neighb_idx] = np.inf

    CLUSTERS = []
    for x in np.unique(clusters):
        CLUSTERS.append(idxs[np.where(clusters == x)[0]])
    return CLUSTERS, leaders


def dimap_exploration(pdb_file, psf_file, dihedrals, grid_space,
                      xmin, xmax, ymin, ymax, prm_file, minim_steps,
                      namd2, nproc):
    """
    """
    parsed_pdb = parse_pdb(pdb_file)
    a, b, c, d = dihedrals[0][0]
    e = dihedrals[0][1][-1]
    # initial DataFrame
    PDB_DF = parse_pdb(pdb_file)
    trajectory_pdb = md.load(pdb_file)
    # getting atoms to rotate
    side_rot_PHI = get_both_sides(psf_file, dihedrals[0][0])
    side_rot_PSI = get_both_sides(psf_file, dihedrals[0][1])
    # mesh of angle pairs (to be consumed)
    mesh = mk_mesh(grid_space, grid_space, xmin, xmax, ymin, ymax)
    # Containers
    phis = []
    psis = []
    energies = []
    while len(mesh) != 0:
        # =====================================================================
        # Rotations
        # =====================================================================
        cartesians = np.asarray(parsed_pdb[['x', 'y', 'z']])
        # setting phi value
        pdb_rot_first = set_dihedral_value(
         cartesians, dihedrals[0][0], mesh[0][0], side_rot_PHI)
        # setting psi value
        pdb_rotated = set_dihedral_value(
         pdb_rot_first, dihedrals[0][1], mesh[0][1], side_rot_PSI)
        # updating initial DataFrame
        PDB_DF['x'] = pdb_rotated.T[0]
        PDB_DF['y'] = pdb_rotated.T[1]
        PDB_DF['z'] = pdb_rotated.T[2]
        # fixing values of phi & psi in occupancy column of the DataFrame
        for atom in [a, b, c, d, e]:
            PDB_DF.iloc[:, 12][atom] = '1.00'

        # =====================================================================
        # Minimization
        # =====================================================================
        # writing the fixed pdb
        str_name = '{}-{}-{}-{}-{}.PHI.pdb'.format(len(mesh), a, b, c, d)
        written_pdb = write(PDB_DF, 'rot--'+str_name)
        # make minimizations for written file
        config_file = conf_creator(written_pdb, psf_file,
                                   prm_file, steps=minim_steps)
        run('{0} +p{1} {2} > {2}.log'.format(namd2, nproc, config_file),
            shell=True)

        # =====================================================================
        # Information
        # =====================================================================
        # torsion angles
        phis.append(mesh[0][0])
        psis.append(mesh[0][1])
        # absolute energies
        loginfo = log(config_file+'.log')
        energies.append(loginfo.get_last_energy())
        # frame coordinates
        new_pdb = md.load_pdb('OPT_'+written_pdb+'.coor')
        trajectory_pdb = trajectory_pdb.join(new_pdb)
        # cleaning operations
        trash = ['*.coor', '*.dcd', '*.vel', '*.xsc', '*.xst', '*.CONF',
                 '*.BAK', 'OPT_*', 'rot--*']
        [os.remove(y) for x in trash for y in glob.glob(x)]
        mesh.pop(0)
    return phis, psis, energies, trajectory_pdb


def dimap_parallel(pdb_file, parsed_pdb, psf_file, dihedrals, grid_space,
                   xmin, xmax, ymin, ymax, prm_file, minim_steps,
                   namd2, nproc):
    """
    """
    # =================================================================
    # Load data to make inputs
    # =================================================================
    a, b, c, d = dihedrals[0][0]
    e = dihedrals[0][1][-1]
    # initial DataFrame
    PDB_DF = parsed_pdb.copy()
    # getting atoms to rotate
    side_rot_PHI = get_both_sides(psf_file, dihedrals[0][0])
    side_rot_PSI = get_both_sides(psf_file, dihedrals[0][1])
    # mesh of angle pairs (to be consumed)
    mesh = mk_mesh(grid_space, grid_space, xmin, xmax, ymin, ymax)
    meshes = mk_mesh1()

    for mesh in meshes:
        print('Temp dir created. Writing files.')
        for i, x in enumerate(meshes[mesh]):
            # =============================================================
            # Rotations
            # =============================================================
            cartesians = np.asarray(parsed_pdb[['x', 'y', 'z']])
            # setting phi value
            pdb_rot_first = set_dihedral_value(
             cartesians, dihedrals[0][0], x[0], side_rot_PHI)
            # setting psi value
            pdb_rotated = set_dihedral_value(
             pdb_rot_first, dihedrals[0][1], x[1], side_rot_PSI)
            # updating initial DataFrame
            PDB_DF['x'] = pdb_rotated.T[0]
            PDB_DF['y'] = pdb_rotated.T[1]
            PDB_DF['z'] = pdb_rotated.T[2]
            # tuple_info = {}
            tuple_meshes = {}
            # fixing values of phi & psi in occupancy column of the DataFrame
            for atom in [a, b, c, d, e]:
                PDB_DF.loc[atom, 'BFactor'] = '1.00'
            # =============================================================
            # Writing
            # =============================================================
                str_name = '{}-{}-{}-{}-{}.PHI.pdb'.format(i, a, b, c, d)
                written_pdb = write(PDB_DF, 'rot--' + str_name)
                config_file = conf_creator(written_pdb, psf_file, prm_file,
                                           steps=minim_steps)
                # =============================================================
                # config_files per meshes
                # =============================================================
            # tuple_meshes.update({config_file: dict(phi=x[0], psi=x[1],
            #                                        energy=0.0,
            #                                        frame=written_pdb)
            #                             for k in meshes.keys()})
                for config_file in meshes:
                    tuple_meshes.update({config_file: {'phi': x[0],
                                                        'psi': x[1],
                                                        'energy': 0.0,
                                                        'frame': written_pdb}
                                        for keys in meshes})
            config_file = conf_creator(written_pdb, psf_file, prm_file,
                                       steps=minim_steps)
            # """ estas son las ideas que tengo hasta ahora el error esta en
            # la minimizacion ahora"""

            #==============================================================
            # Minimization
            # =============================================================
            command = '{0} +p{2} {1} > {1}.log'.format(
                            namd2, config_file, nproc)
            os.system(command)
            coor_file = 'OPT_rot--' + str_name + '.coor'
            pdb_file = 'OPT_rot--' + str_name
            shutil.copy(coor_file, '.'.join(coor_file.split('.')[:-1]))
            parsed_pdb = parse_pdb(pdb_file)
            # =================================================================
            # Information
            # =================================================================
            # absolute energies
            loginfo = log(config_file + '.log')
            tuple_meshes[config_file]['energy'] = loginfo.get_last_energy()

        # frame coordinates
            keys = sorted(list(tuple_meshes.keys()),
                      key=lambda f: int(f.split('rot--')[1].split('-')[0]))
            to_join = []
            for key in keys:
                frame = 'OPT_' + tuple_meshes[key]['frame'] + '.coor'
                to_join.append(md.load_pdb(frame))
            os.chdir(os.path.expanduser('~'))
        print('temp dir removed')
    return tuple_meshes, md.join(to_join)


def plot_exploration(xmin, xmax, ymin, ymax, restricted, cl, dataf0, leads):
    """
    """
    generic_matplotlib()
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel('PHI')
    plt.ylabel('PSI')
    plt.scatter(restricted.phi, restricted.psi, c=restricted.energy, s=350,
                marker='.', alpha=0.8, cmap='rainbow')
    plt.colorbar()
    for i, c in enumerate(cl):
        p = dataf0.loc[c].phi
        s = dataf0.loc[c].psi
        plt.scatter(p, s, s=350, marker='.', alpha=0.8, edgecolors='k',
                    label='Cluster-{}'.format(i+1))
    plt.legend()
    for lead in leads:
        p, s, e = dataf0.iloc[lead]
        plt.scatter(p, s, s=50, marker='.', alpha=0.8, c='k')

    plt.savefig('exploration')
    plt.close()


def pickle_to_file(data, file_name):
    """ Serialize data using **pickle**.

    Args:
        data (object)  : any serializable object.
        file_name (str): name of the **pickle** file to be created.
    Returns:
        (str): file_name
    """
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
    return file_name


def unpickle_from_file(file_name):
    """ Unserialize a **pickle** file.

    Args:
        file_name (str): file to unserialize.
    Returns:
        (object): an unserialized object.
    """
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    return data


def combinatory(top_iterable):
    """
    """
    # ---- Constructors parameters ------------------------------------------ #
    LEN = [len(x) for x in top_iterable]
    N = np.product(LEN)
    DIV = [np.product(LEN[0:x+1]) for x in range(len(top_iterable))]
    REP = [int(N/D) for D in DIV]
    repetitions = [int(N/(REP[x]*LEN[x])) for x in range(len(top_iterable))]
    # ---- Combinatory ------------------------------------------------------ #
    columns = []
    for index, iterable in enumerate(top_iterable):
        col = []
        for idx, element in enumerate(iterable):
            r = REP[index]
            while r != 0:
                col.append(element)
                r -= 1
        columns.append(col)
    # ---- Final product ---------------------------------------------------- #
    COMB = [iterable*repetitions[index]
            for index, iterable in enumerate(columns)]
    # ---- List of angles creation ------------------------------------------ #
    ANG = []
    for index in range(len(COMB[0])):
        conformer = []
        for idx, iterable in enumerate(COMB):
            conformer.append(COMB[idx][index])
        ANG.append(conformer)
    return ANG


def get_minima_from_database(database_dir, names):
    database = os.listdir(database_dir)
    minima = []
    for i, name in enumerate(names):
        if name not in database:
            # abort if a detected dihedral is not in database
            print('\n{} is not in database. Aborting!'.format(name))
            break
        else:
            # find minima for the corresponding angles
            min_file = os.path.join(database_dir, name, '{}.min'.format(name))
            min_data = unpickle_from_file(min_file)
            minimum = []
            for key in min_data:
                idx = min_data[key]['index']
                phi = min_data[key]['phis'].iloc[idx]
                psi = min_data[key]['psis'].iloc[idx]
                minimum.append((phi, psi))
            minima.append(minimum)
    return minima


def generate_conformers(psf_file, pdb_file, indices, combinatory):
    parsed = prd.parsePDB(pdb_file)
    coords_stack = [parsed.getCoords()]
    conformers = []
    for conformer in combinatory:
        for i, phi_psi in enumerate(conformer):
            phi_val, psi_val = phi_psi
            phi_idx, psi_idx = indices[i]
            phi_side_rot = get_both_sides(psf_file, phi_idx)
            psi_side_rot = get_both_sides(psf_file, psi_idx)
            coords_stack.append(
                set_dihedral_value(coords_stack[-1], phi_idx, phi_val,
                                   phi_side_rot))
            coords_stack.append(
                set_dihedral_value(coords_stack[-1], psi_idx, psi_val,
                                   psi_side_rot))
        conformers.append(coords_stack[-1])
    return conformers


def minimimze_conformers(pdb_parsed, psf_file, prm_file, conformers, namd2,
                         nproc):
    tuple_info = {}
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)
        # ---- generation of config files
        config_files = []
        for i, conformer in enumerate(conformers):
            pdb_parsed.setCoords(conformer)
            pdb_name = 'premin_{}.pdb'.format(i)
            written_pdb = prd.writePDB(pdb_name, pdb_parsed)
            config_file = conf_creator(pdb_name, psf_file, prm_file,
                                           steps=1000, per_cycle=20)
            tuple_info.update(
                {config_file: {'energy': 0.0, 'frame': written_pdb}})
            config_files.append(config_file)

        commands = ['{0} +p1 {1} > {1}.log'.format(namd2, config_file) for
                    config_file in config_files]
        # ---- minimizations
        pool = Pool(nproc)
        start2 = time.time()
        print('Starting minimizations with {} cores.'.format(nproc))
        pool.map(os.system, commands)
        print(
            'Minimizations of rot-- files in {}'.format(time.time() - start2))
        # ---- recopiling information
        for config_file in config_files:
            loginfo = log(config_file + '.log')
            tuple_info[config_file]['energy'] = loginfo.get_last_energy()
        keys = sorted(list(tuple_info.keys()),
                      key=lambda f: int(f.split('_')[1].split('.')[0]))
        to_join = []
        for key in keys:
            frame = 'OPT_' + tuple_info[key]['frame'] + '.coor'
            to_join.append(md.load_pdb(frame))
        os.chdir(os.path.expanduser('~'))
        return tuple_info, md.join(to_join)
