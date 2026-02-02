#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on : Mon Mar 16 12:44:51 2020
@author    : Roy Gonzalez-Aleman
@mail      : roy.gonzalez.aleman@gmail.com
"""

import os
import shutil
import time

import pandas as pd

import difuncts as dimap
import diparse

starting_time = time.time()

# =============================================================================
# 1. Parsing configuration file arguments
# =============================================================================
# argv = sys.argv[1]
argv = '/media/tomas/e9ad14c8-6bd8-4a0c-b5c1-02d8a2c93c7c/2026/PROJ_ROY/example/dissac.conf'
cfg = diparse.parser(argv)

# =============================================================================
# 2. Parsing dihedrals (either automatic or read from cfg)
# =============================================================================
parsed_psf = dimap.parse_psf(cfg.psf_file)
if cfg.auto:
    basename, dihedrals = dimap.find_glycosidics(parsed_psf)
else:
    phi_angle = [int(x) for x in cfg.phi]
    psi_angle = [int(x) for x in cfg.psi]
    dihedrals = [[phi_angle, psi_angle]]
    c1 = phi_angle[1]
    cx = phi_angle[3]
    basename = '{}_{}_{}_{}'.format(parsed_psf.iloc[c1].resname,
                                    parsed_psf.iloc[c1].atomname,
                                    parsed_psf.iloc[cx].atomname,
                                    parsed_psf.iloc[cx].resname)

print('Selected dihedrals are: {}'.format(dihedrals))
print('Basename for outputs is: {}'.format(basename))

# =============================================================================
# 3. Rotations (Rodrigues Method)
# =============================================================================
# phi & psi definition
a, b, c, d = dihedrals[0][0]
e = dihedrals[0][1][-1]

# creation of production folder
opt_folder = os.path.abspath(os.path.join(cfg.root_dir, '{}'.format(basename)))
err1 = 'ERROR: Directory {} exists. Please back it up !!!'.format(opt_folder)
assert dimap.overwrite_dir(opt_folder), err1

# rotations
parsed_pdb = dimap.parse_pdb(cfg.pdb_file)
mapdict, traj = dimap.dimap_parallel(cfg.pdb_file, parsed_pdb,cfg.psf_file,
                                     dihedrals, cfg.grid_space, cfg.xmin,
                                     cfg.xmax, cfg.ymin, cfg.ymax,
                                     cfg.prm_file, cfg.minim_steps, cfg.namd2,
                                     cfg.nproc)
# =============================================================================
# 4. Inside the final folder
# =============================================================================
os.chdir(opt_folder)

# Create total Exploration.log and extract restricted exploration
dataf0 = pd.DataFrame(mapdict).T[['phi', 'psi', 'energy']]
dataf0 = dataf0.astype({'phi': float, 'psi': float, 'energy': float})
dataf0 = dataf0.reset_index(drop=True)
dataf0['energy'] = dataf0['energy'] - dataf0['energy'].min()

with open('{}.log'.format(basename), 'wt') as expl:
    dataf0.to_string(expl, index=False)
restricted = dataf0[dataf0.energy <= cfg.e_cutoff]

# Trajectory saving -----------------------------------------------------------
PSF = shutil.copy(cfg.psf_file, basename + '.psf_14')
traj.save_dcd(basename + '.dcd')
traj.center_coordinates()

# Clustering ------------------------------------------------------------------
cl, leads = dimap.naive_clustering(traj, dataf0.energy, cfg.rms_lim)

# Minima database -------------------------------------------------------------
minima = {
    x: {'alternatives': cl[i],
        'energies': dataf0.energy[cl[i]],
        'phis': dataf0.phi[cl[i]],
        'psis': dataf0.psi[cl[i]],
        'index': cl[i].tolist().index(x)} for i, x in enumerate(leads)}
dimap.pickle_to_file(minima, '{}.min'.format(basename))

# Plots -----------------------------------------------------------------------
dimap.plot_exploration(cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax,
                       restricted, cl, dataf0, leads)
dimap.plot_dimap(cfg.e_cutoff, '{}.log'.format(basename), basename)

# =============================================================================
# Timing
# =============================================================================
final_time = time.time() - starting_time
print('\n\nDiMap execution took: {:3.2f} sec'.format(final_time))
