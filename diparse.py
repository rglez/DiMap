# Created by roy.gonzalez-aleman at 16/10/2022

import configparser
import os


def assert_existence(path, ext=None):
    existence = os.path.exists(path)
    assert existence, '\nNo such file or directory: %s' % path
    if ext:
        sms = '\n {} must have .{} extension.'.format(path, ext)
        assert os.path.basename(path).split('.')[-1] == ext, sms
    return path


def read_cfg(cfg_path):
    assert_existence(cfg_path)
    config = configparser.ConfigParser(allow_no_value=True,
                                       inline_comment_prefixes='#')
    config.optionxform = str
    config.read(cfg_path)
    return config


class parser:
    def __init__(self, config_path):
        # read the configuration file & define class attributes
        self.cfg_path = config_path
        self.cfg = read_cfg(self.cfg_path)
        self.sections = self.cfg.sections()
        self.allowed_sections = ['files', 'namd', 'params']
        self.root_dir = os.path.split(config_path)[0]

        # checks
        sms1 = '\n\nMismatch in the number of allowed sections. Verify that' \
               ' only these sections were specified in the configuration ' \
               'file: {}.'.format(self.allowed_sections)
        sms2 = '\n\nError in the nomenclature of allowed sections. Verify' \
               ' that only these sections were specified in the' \
               ' configuration file: {}.'.format(self.allowed_sections)

        assert len(self.sections) == len(self.allowed_sections), sms1
        assert set(self.sections) == set(self.allowed_sections), sms2

        # [files] section
        self.psf_file = self.cfg.get('files', 'psf_path')
        self.pdb_file = self.cfg.get('files', 'pdb_path')
        self.prm_file = self.cfg.get('files', 'prm_path')

        # [params] section
        self.grid_space = self.cfg.getfloat('params', 'angle_step')
        self.e_cutoff = self.cfg.getfloat('params', 'energy_cut')
        self.rms_lim = self.cfg.getfloat('params', 'rmsd_cut')
        self.auto = self.cfg.getboolean('params', 'auto_dihedrals')
        self.xmin = self.cfg.getfloat('params', 'xmin')
        self.xmax = self.cfg.getfloat('params', 'xmax')
        self.ymin = self.cfg.getfloat('params', 'ymin')
        self.ymax = self.cfg.getfloat('params', 'ymax')
        self.phi = self.cfg.get('params', 'phi').split()
        self.psi = self.cfg.get('params', 'psi').split()

        # [namd] section
        self.namd2 = self.cfg.get('namd', 'namd_path')
        self.nproc = self.cfg.getint('namd', 'nprocs')
        self.minim_steps = self.cfg.getint('namd', 'minim_steps')


# self = parser('/home/roy.gonzalez-aleman/RoyHub/DiMap/inputs/dissac.conf')
