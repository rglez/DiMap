#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on : Mon Mar 16 12:44:51 2020
@author    : Roy Gonzalez-Aleman
@mail      : rglez.developer@gmail.com
"""

import os
import sys
import time
import shutil
import pandas as pd
import configparser as conf

import functions as dimap

starting_time = time.time()

# =============================================================================
# Parsing configuration file arguments
# =============================================================================
argv = sys.argv[1]
config = conf.ConfigParser()
config.read(argv)
# [files] section
psf_file = config.get('files', 'psf_path')
pdb_file = config.get('files', 'pdb_path')
prm_file = config.get('files', 'prm_path')
# [params] section
grid_space = config.getint('params', 'angle_step')
e_cutoff = config.getfloat('params', 'energy_cut')
rms_lim = config.getfloat('params', 'rmsd_cut')
xmin = config.getfloat('params', 'xmin')
xmax = config.getfloat('params', 'xmax')
ymin = config.getfloat('params', 'ymin')
ymax = config.getfloat('params', 'ymax')
# [namd] section
namd2 = config.get('namd', 'namd_path')
nproc = config.getint('namd', 'nprocs')
minim_steps = config.getint('namd', 'minim_steps')
# [output] section
base_name = config.get('outputs', 'base_name')
dcd_out_name = '{}.dcd'.format(base_name)


# =============================================================================
# Parsing specified files
# =============================================================================
parsed_psf = dimap.parse_psf(psf_file)
psf_name = os.path.basename(psf_file)
pdb_name = os.path.basename(pdb_file)

if config.getboolean('params', 'auto_dihedrals'):
    dihedrals = dimap.find_glycosidics(parsed_psf)
else:
    phi_angle = [int(x) for x in config.get('params', 'phi').split()]
    psi_angle = [int(x) for x in config.get('params', 'psi').split()]
    dihedrals = [[phi_angle, psi_angle]]


# =============================================================================
# Rotations (Rodrigues Method)
# =============================================================================
# phi & psi definition --------------------------------------------------------
a, b, c, d = dihedrals[0][0]
e = dihedrals[0][1][-1]
# creation of production folder -----------------------------------------------
opt_folder = 'out_{}-{}_{}_{}_{}_{}'.format(base_name, a, b, c, d, e)
err1 = 'ERROR: Directory {} exists. Please back it up !!!'.format(opt_folder)
assert dimap.overwrite_dir(opt_folder), err1
# rotations -------------------------------------------------------------------
phis, psis, energies, trajectory_pdb = dimap.dimap_exploration(
        pdb_file, psf_file, dihedrals, grid_space, xmin, xmax, ymin, ymax,
        prm_file, minim_steps, namd2, nproc)


# =============================================================================
# Inside the final folder
# =============================================================================
os.chdir(opt_folder)
# Create total Exploration.log and extract restricted exploration -------------
dataf0 = pd.DataFrame(data=list(zip(phis, psis, energies)),
                      columns=['phi', 'psi', 'energy'])
dataf0['energy'] = dataf0['energy'] - dataf0['energy'].min()
with open('exploration.log', 'wt') as expl:
    dataf0.to_string(expl, index=False)
restricted = dataf0[dataf0.energy <= e_cutoff]
# Trajectory saving -----------------------------------------------------------
PSF = shutil.copy(psf_file, '.')
trajectory_pdb = trajectory_pdb[1:]
trajectory_pdb.save_dcd(dcd_out_name)
trajectory_pdb.center_coordinates()
# Clustering ------------------------------------------------------------------
cl, leads = dimap.naive_clustering(trajectory_pdb, dataf0.energy,
                                   e_cutoff, rms_lim)
# Plots -----------------------------------------------------------------------
out_name = base_name
dimap.plot_exploration(xmin, xmax, ymin, ymax, restricted, cl, dataf0, leads)
dimap.plot_dimap(e_cutoff, 'exploration.log', out_name)


# =============================================================================
# Timing
# =============================================================================
final_time = time.time() - starting_time
print('\n\nDiMap execution took: {:18.2f} sec'.format(final_time))
