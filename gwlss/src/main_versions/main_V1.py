#!/usr/bin/env python3

"""
The code is organized as the coding etiquette required: 
first all the needed functions and then the main.

To go to the preferred part, just search for "FUNCTIONS_SECTION" or "MAIN_SECTION",
using the words-finder, depending on your need.

"""

##############################		 ##############################

import os

import argparse
import shutil

import numpy as np
import json
import sys
import time
import importlib
import matplotlib.pyplot as plt

import functions_cross_correlation as fcc

from astropy.cosmology import Planck18
from astropy.cosmology import FlatLambdaCDM
import astropy.cosmology.units as cu
from astropy import units as u
import astropy.constants as const
import numdifftools as nd
import scipy.special as ss
import scipy.integrate as sint
from scipy.integrate import quad
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import scipy.linalg
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import RectBivariateSpline
import scipy.interpolate as si
import scipy.optimize as optimize
import scipy.integrate as integrate

plt.rc('font',size=20,family='serif')

configspath = 'configs/'

#############################
"""
Reads user inputs at runtime:
    --config = name of the config file (without .py).
    --fout = output folder path (where to save results).
"""
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='', type=str, required=False) # path to config file, in.json format
parser.add_argument("--fout", default='', type=str, required=True) # path to output folder
FLAGS = parser.parse_args()
##############################		 ##############################
##############################		 ##############################




##############################		 ##############################
"""
   __                  _   _                 
  / _|                | | (_)                
 | |_ _   _ _ __   ___| |_ _  ___  _ __  ___ 
 |  _| | | | '_ \ / __| __| |/ _ \| '_ \/ __|
 | | | |_| | | | | (__| |_| | (_) | | | \__ \
 |_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|___/

FUNCTIONS_SECTION: Here are defined all the used functions, divide  in terms of the section addressed                                                                      
"""

#############################
############################# Loading GW detector parameter
#############################
def load_detector_params(GW_det: str, yr: float):
	"""
	Load and return gravitational wave detector parameters based on the detector name.

	Parameters:
	- GW_det (str): Name of the gravitational wave detector.
	- yr (float): Time unit in years, used to scale A.

	Returns:
	- dict: Dictionary containing A, Z_0, Alpha, Beta, log_delta_dl, log_loc, log_dl,
			s_a to s_d, be_a to be_d.
	"""
	# Default parameters (common in many cases)
	s_params = {
		's_a': -5.59e-3,
		's_b': 2.92e-2,
		's_c': 3.44e-3,
		's_d': 2.58e-3,
	}
	be_params = {
		'be_a': -1.45,
		'be_b': -1.39,
		'be_c': 1.98,
		'be_d': -0.363,
	}

	# Detector-specific settings
	detectors = {
		'ET_Delta_2CE': {
			'A': 40.143 * yr, 'Z_0': 1.364, 'Alpha': 2.693, 'Beta': 0.625,
			'files': ['log_delta_dl_ET_Delta_2CE.npy', 'log_loc_ET_Delta_2CE.npy', 'log_dl_ET_Delta_2CE.npy']
		},
		'ET_2L_2CE': {
			'A': 32.795 * yr, 'Z_0': 1.244, 'Alpha': 2.729, 'Beta': 0.614,
			'files': ['log_delta_dl_ET_2L_2CE.npy', 'log_loc_ET_2L_2CE.npy', 'log_dl_ET_2L_2CE.npy']
		},
		'ET_Delta_2CE_cut': {
			'A': 437.98 * yr, 'Z_0': 6.84, 'Alpha': 1.687, 'Beta': 1.07,
			'files': ['log_delta_dl_ET_Delta_2CE_hardcut.npy', 'log_loc_ET_Delta_2CE_hardcut.npy',
					  'log_dl_ET_Delta_2CE_hardcut.npy']
		},
		'ET_2L_2CE_cut': {
			'A': 465.1 * yr, 'Z_0': 7.09, 'Alpha': 1.72, 'Beta': 1.06,
			'files': ['log_delta_dl_ET_2L_2CE_hardcut.npy', 'log_loc_ET_2L_2CE_hardcut.npy',
					  'log_dl_ET_2L_2CE_hardcut.npy']
		},
		'ET_Delta_1CE': {
			'A': 69.695 * yr, 'Z_0': 1.79, 'Alpha': 2.539, 'Beta': 0.658,
			'files': ['log_delta_dl_ET_Delta_1CE.npy', 'log_loc_ET_Delta_1CE.npy', 'log_dl_ET_Delta_1CE.npy']
		},
		'ET_2L_1CE': {
			'A': 49.835 * yr, 'Z_0': 1.533, 'Alpha': 2.619, 'Beta': 0.638,
			'files': ['log_delta_dl_ET_2L_1CE.npy', 'log_loc_ET_2L_1CE.npy', 'log_dl_ET_2L_1CE.npy']
		},
		'ET_Delta': {
			'A': 99 * yr, 'Z_0': 6.89, 'Alpha': 1.25, 'Beta': 0.97,
			'files': ['log_delta_dl_ET_Delta_cut.npy', 'log_loc_ET_Delta_cut.npy', 'log_dl_ET_Delta_cut.npy'],
			's_params': {'s_a': -8.39e-3, 's_b': 4.54e-2, 's_c': 1.36e-2, 's_d': -2.04e-3}
		},
		'ET_2L': {
			'A': 61.34 * yr, 'Z_0': 1.97, 'Alpha': 1.93, 'Beta': 0.7,
			'files': ['log_delta_dl_ET_2L_cut.npy', 'log_loc_ET_2L_cut.npy', 'log_dl_ET_2L_cut.npy'],
			's_params': {'s_a': -8.39e-3, 's_b': 4.54e-2, 's_c': 1.36e-2, 's_d': -2.04e-3}
		},
		'LVK': {
			'A': 60.585 * yr, 'Z_0': 2.149, 'Alpha': 1.445, 'Beta': 0.910,
			'files': ['log_delta_dl_LVK.npy', 'log_loc_LVK.npy', 'log_dl_LVK.npy'],
			's_params': {'s_a': -0.122, 's_b': 3.15, 's_c': -7.61, 's_d': 7.33},
			'be_params': {'be_a': -1.04, 'be_b': -0.176, 'be_c': 105.0, 'be_d': -436.0}
		}
	}

	if GW_det not in detectors:
		raise ValueError(f"Unknown detector: {GW_det}")

	det = detectors[GW_det]
	log_delta_dl, log_loc, log_dl = [np.load(f'det_param/{f}') for f in det['files']]

	# Merge detector-specific overrides
	s_p = det.get('s_params', s_params)
	be_p = det.get('be_params', be_params)

	return {
		'A': det['A'], 'Z_0': det['Z_0'], 'Alpha': det['Alpha'], 'Beta': det['Beta'],
		'log_delta_dl': log_delta_dl, 'log_loc': log_loc, 'log_dl': log_dl,
		**s_p, **be_p
	}



#############################
############################# Loading Galaxy detector parameters
#############################
def load_galaxy_detector_params(gal_det: str):
	"""
    Load and return galaxy survey detector parameters based on the detector name.

    Parameters:
    - gal_det (str): Name of the galaxy detector (e.g., 'euclid_photo', 'euclid_spectro', 'ska').

    Returns:
    - dict: Dictionary containing bg0–bg3, sg0–sg3, bin_centers_fit, values_fit, spline,
            and optionally sig_gal and f_sky.
    """
	detectors = {
		'euclid_photo': {
			'bg': [0.5125, 1.377, 0.222, -0.249],
			'sg': [0.0842, 0.0532, 0.298, -0.0113],
			'bin_centers': [0.001, 0.14, 0.26, 0.39, 0.53, 0.69, 0.84, 1.00, 1.14, 1.30, 1.44, 1.62, 1.78, 1.91, 2.1,
							2.25],
			'values': [0, 0.758, 2.607, 4.117, 3.837, 3.861, 3.730, 3.000, 2.827, 1.800, 1.078, 0.522, 0.360, 0.251,
					   0.1, 0],
			'spline_s': 0.1
		},
		'euclid_spectro': {
			'bg': [0.853, 0.04, 0.713, -0.164],
			'sg': [1.231, -1.746, 1.810, -0.505],
			'bin_centers': [0.8, 1, 1.07, 1.14, 1.2, 1.35, 1.45, 1.56, 1.67, 1.9],
			'values': [0., 0.2802, 0.2802, 0.2571, 0.2571, 0.2184, 0.2184, 0.2443, 0.2443, 0.],
			'spline_s': 0,
			'sig_gal': 0.001
		},
		'ska': {
			'bg': [0.853, 0.04, 0.713, -0.164],
			'sg': [1.36, 1.76, -1.18, 0.28],
			'bin_centers': [0.01, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95,
							1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95],
			'values': [0, 1.21872309, 1.74931326, 1.81914498, 1.6263191, 1.33347361, 1.05034008, 0.79713276,
					   0.58895358, 0.42322164, 0.29564803, 0.20296989, 0.1366185, 0.09011826, 0.0586648,
					   0.03724468, 0.02323761, 0.01423011, 0.00848182, 0.00492732],
			'spline_s': 0.001,
			'sig_gal': 0.001,
			'f_sky': 0.7
		}
	}

	if gal_det not in detectors:
		raise ValueError(f"Unknown galaxy detector: {gal_det}")

	det = detectors[gal_det]
	spline = UnivariateSpline(det['bin_centers'], det['values'], s=det['spline_s'])

	# Build result dictionary
	result = {
		'bg0': det['bg'][0], 'bg1': det['bg'][1], 'bg2': det['bg'][2], 'bg3': det['bg'][3],
		'sg0': det['sg'][0], 'sg1': det['sg'][1], 'sg2': det['sg'][2], 'sg3': det['sg'][3],
		'bin_centers_fit': np.array(det['bin_centers']),
		'values_fit': np.array(det['values']),
		'spline': spline
	}

	# Optionally include extra fields like sig_gal or f_sky
	for key in ['sig_gal', 'f_sky']:
		if key in det:
			result[key] = det[key]

	return result

#############################
############################# Compute the bin edges for the 2 datasets
#############################
def compute_bin_edges(
		bin_strategy, n_bins_dl, n_bins_z,
		bin_int, zM_bin, dlM_bin, zm_bin,
		fiducial_universe, A, Z_0, Alpha, Beta,
		spline, fcc
):
	"""
    Compute bin edges in redshift and luminosity distance according to the specified strategy.

    Parameters:
        bin_strategy (str): Strategy name ('right_cosmo', 'equal_space right_cosmo', etc.)
        n_bins_dl (int): Number of bins for luminosity distance.
        n_bins_z (int): Number of bins for redshift.
        bin_int (array): z sampling for galaxy distribution.
        zM_bin, dlM_bin, zm_bin: z and dL bin boundaries.
        fiducial_universe: astropy cosmology instance.
        A, Z_0, Alpha, Beta: GW rate model parameters.
        spline: spline fitted to galaxy number density.
        fcc: a utility module that provides equal_interval function.

    Returns:
        bin_edges (array): Redshift bin edges.
        bin_edges_dl (array): Luminosity distance bin edges [Gpc].
    """

	if n_bins_dl <= n_bins_z:
		print('number of bins in distance must be greater than bins in z, set automatically to n_bins_z+1')
		n_bins_dl = n_bins_z + 1

	if bin_strategy == 'right_cosmo':
		gal_bin = spline(bin_int)
		gal_bin[gal_bin < 0] = 0
		interval_gal = fcc.equal_interval(gal_bin, bin_int, n_bins_z)
		bin_edges = bin_int[interval_gal]

		bin_edges_dl = np.zeros(n_bins_dl + 1)
		for i, z in enumerate(bin_edges):
			bin_edges_dl[i] = fiducial_universe.luminosity_distance(z).value / 1000

		dlm_bin = fiducial_universe.luminosity_distance(zM_bin).value
		bin_int_GW = np.linspace(dlm_bin / 1000, dlM_bin / 1000, n_bins_dl * 100)

		GW_bin = A * (bin_int_GW / Z_0) ** Alpha * np.exp(-(bin_int_GW / Z_0) ** Beta)
		interval_GW = fcc.equal_interval(GW_bin, bin_int_GW, n_bins_dl - n_bins_z)

		bin_edges_dl[n_bins_z:] = bin_int_GW[interval_GW]

	elif bin_strategy == 'equal_space right_cosmo':
		bin_edges = np.linspace(zm_bin, zM_bin, n_bins_z + 1)

		bin_edges_dl = np.zeros(n_bins_dl + 1)
		for i, z in enumerate(bin_edges):
			bin_edges_dl[i] = fiducial_universe.luminosity_distance(z).value / 1000

		dlm_bin = fiducial_universe.luminosity_distance(zM_bin).value
		bin_int_GW = np.linspace(dlm_bin / 1000, dlM_bin / 1000, n_bins_dl - n_bins_z + 1)

		bin_edges_dl[n_bins_z:] = bin_int_GW

	elif bin_strategy == 'wrong_cosmo':
		from astropy.cosmology import FlatLambdaCDM
		wrong_universe = FlatLambdaCDM(H0=65, Om0=0.32)

		gal_bin = spline(bin_int)
		gal_bin[gal_bin < 0] = 0
		interval_gal = fcc.equal_interval(gal_bin, bin_int, n_bins_z)
		bin_edges = bin_int[interval_gal]

		bin_edges_dl = np.zeros(n_bins_dl + 1)
		for i, z in enumerate(bin_edges):
			bin_edges_dl[i] = wrong_universe.luminosity_distance(z).value / 1000

		dlm_bin = wrong_universe.luminosity_distance(zM_bin).value
		bin_int_GW = np.linspace(dlm_bin / 1000, dlM_bin / 1000, n_bins_dl * 100)

		GW_bin = A * (bin_int_GW / Z_0) ** Alpha * np.exp(-(bin_int_GW / Z_0) ** Beta)
		interval_GW = fcc.equal_interval(GW_bin, bin_int_GW, n_bins_dl - n_bins_z)

		bin_edges_dl[n_bins_z:] = bin_int_GW[interval_GW]

	elif bin_strategy == 'equal_pop':
		gal_bin = spline(bin_int)
		gal_bin[gal_bin < 0] = 0
		interval_gal = fcc.equal_interval(gal_bin, bin_int, n_bins_z)
		bin_edges = bin_int[interval_gal]

		bin_int_GW = np.linspace(dlm_bin / 1000, dlM_bin / 1000, n_bins_dl * 100)
		GW_bin = A * (bin_int_GW / Z_0) ** Alpha * np.exp(-(bin_int_GW / Z_0) ** Beta)
		interval_GW = fcc.equal_interval(GW_bin, bin_int_GW, n_bins_dl)
		bin_edges_dl = bin_int_GW[interval_GW]

	elif bin_strategy == 'equal_space':
		bin_edges = np.linspace(zm_bin, zM_bin, n_bins_z + 1)
		bin_edges_dl = np.linspace(dlm_bin / 1000, dlM_bin / 1000, n_bins_dl + 1)

	else:
		raise ValueError(f"Unknown binning strategy: {bin_strategy}")

	return bin_edges, bin_edges_dl


#############################
############################# HERE
#############################
def compute_nz_gal_and_total(gal_det, z_gal, bin_edges, sig_gal, spline, fcc):
    """
    Compute the galaxy redshift distribution per bin and the total galaxy number density.

    Parameters:
        gal_det (str): Galaxy detector name ('euclid_photo', 'euclid_spectro', 'ska').
        z_gal (array): Redshift values to evaluate.
        bin_edges (array): Bin edges in redshift.
        sig_gal (float): Galaxy redshift uncertainty.
        spline (UnivariateSpline): Spline for galaxy number density.
        fcc (module): Module with detector-specific nz functions.

    Returns:
        nz_gal (array): Galaxy distribution per redshift bin.
        gal_tot (array): Total galaxy number density over z_gal.
    """
    gal_scale_factors = {
        'euclid_photo': 8.35e7,
        'euclid_spectro': 1.25e7,
        'ska': 9.6e7
    }

    nz_func_map = {
        'euclid_photo': fcc.euclid_photo,
        'euclid_spectro': fcc.euclid_spec,
        'ska': fcc.ska
    }

    if gal_det not in nz_func_map:
        raise ValueError(f"Unknown galaxy detector: {gal_det}")

    nz_gal = nz_func_map[gal_det](z_gal, bin_edges, sig_gal)
    gal_tot = spline(z_gal) * gal_scale_factors[gal_det]

    return nz_gal, gal_tot

#############################
############################# HERE
#############################
def compute_kmax(z_values, P_interp, k_range, sigma_target=0.25):
	"""
	Compute k_max for a set of redshifts using the nonlinear scale criterion.

	Parameters:
		z_values (array): Array of redshifts.
		P_interp (RectBivariateSpline): Interpolated nonlinear power spectrum P(k, z).
		k_range (array): Array of k values [h/Mpc] for integration.
		sigma_target (float): Target sigma^2 for defining R (default 0.25).

	Returns:
		np.ndarray: Array of k_max values for each redshift.
	"""

	def j1(x):
		return 3 / x ** 2 * (np.sin(x) / x - np.cos(x))

	def sigma_squared(R, z_):
		def pk(k): return P_interp(z_, k)[0]

		integrand = lambda x: (1 / (2 * np.pi ** 2)) * x ** 2 * (j1(x * R) ** 2) * pk(x)
		return integrate.quad(integrand, k_range[0], k_range[-1], limit=10000)[0]

	kmax_list = []
	for z_ in z_values:
		sol = optimize.root_scalar(
			lambda R: sigma_squared(R, z_) - sigma_target,
			bracket=[0.01, 20],
			method='bisect'
		)
		R_nl = sol.root
		kmax = np.pi / R_nl / 2
		kmax_list.append(kmax)

	return np.array(kmax_list)

#############################
############################# Evolution bias (b) and Magnification bias (s) from arXiv:2309.04391v1
#############################
def compute_beta(H0, Omega_m, Omega_b, s_a, s_b, s_c, s_d, be_a, be_b, be_c, be_d):

	cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m, Ob0=Omega_b)

	conf_H = cosmo.H(z_gal).value/(1+z_gal)
	H_dot = -(3/2)*(cosmo.H(z_gal).value)**2*Omega_m*(1+z_gal)**3
	der_conf_H = 1/((1+z_gal)**2)*(H_dot + (cosmo.H(z_gal).value)**2)
	r_conf_H = cosmo.comoving_distance(z_gal).value*conf_H
	s = s_a + s_b*z_gal + s_c*z_gal**2 + s_d*z_gal**3
	gamma = r_conf_H/(1+r_conf_H)
	b = be_a + be_b*z_gal + be_c*z_gal**2 + be_d*z_gal**3
	beta = 5*s-1+gamma*(2/r_conf_H+gamma*((der_conf_H/(conf_H**2))-1/r_conf_H)-1-b)

	return beta

#############################
############################# HERE
#############################
def compute_s_gal(z_gal, gal_det, sg0, sg1, sg2, sg3):
	if gal_det == 'ska':
		return (sg0 + sg1 * z_gal + sg2 * z_gal ** 2 + sg3 * z_gal ** 3) * z_gal
	else:
		return sg0 + sg1 * z_gal + sg2 * z_gal ** 2 + sg3 * z_gal ** 3


#############################
############################# HERE
#############################
def compute_parameter_derivatives(parameters, FLAGS, n_bins_z, n_bins_dl, covariance_matrices):
	for param in parameters:
		print(f"\nComputing the derivative with respect to the {param['name']}...\n")

		step = param["step"]
		method = param.get("method", "central")

		# Define numerical derivative objects
		partial_der_GG = nd.Derivative(lambda x: param["derivative_args"](x)[0], step=step, method=method)
		partial_der_GWGW = nd.Derivative(lambda x: param["derivative_args"](x)[1], step=step, method=method)
		partial_der_GGW = nd.Derivative(lambda x: param["derivative_args"](x)[2], step=step, method=method)

		# Evaluate derivatives at the true parameter value
		der_GG = partial_der_GG(param["true_value"])
		der_GWGW = partial_der_GWGW(param["true_value"])
		der_GGW = partial_der_GGW(param["true_value"])

		# Construct derivative vector and covariance matrix
		der_vec = fcc.vector_cl(cl_cross=der_GGW, cl_auto1=der_GG, cl_auto2=der_GWGW)
		der_cov_mat = fcc.covariance_matrix(der_vec, n_bins_z, n_bins_dl)

		# Store the covariance matrix
		covariance_matrices[param["key"]] = der_cov_mat

		# Save to file
		np.save(os.path.join(FLAGS.fout, f"{param['key']}.npy"), der_cov_mat)




#############################
############################# HERE
#############################
def compute_partial_derivatives_gal(bgal, der_bgal):

	def func_GG(b):
		bgal_temp = np.copy(bgal)
		bgal_temp[i] = b
		return Cl_func(H0_true, Omega_m_true, Omega_b_true, 2.12605, 0.96, bgal_temp, bias_GW)[0]

	def func_GWGW(b):
		bgal_temp = np.copy(bgal)
		bgal_temp[i] = b
		return Cl_func(H0_true, Omega_m_true, Omega_b_true, 2.12605, 0.96, bgal_temp, bias_GW)[1]

	def func_GGW(b):
		bgal_temp = np.copy(bgal)
		bgal_temp[i] = b
		return Cl_func(H0_true, Omega_m_true, Omega_b_true, 2.12605, 0.96, bgal_temp, bias_GW)[2]

	for i in range(len(bgal)):
		print('\nComputing the derivative with respect to the galaxy bias in bin %i...\n' % i)

		der_bgal_GG = nd.Derivative(func_GG, step=step)(bgal[i])
		der_bgal_GWGW = nd.Derivative(func_GWGW, step=step)(bgal[i])
		der_bgal_GGW = nd.Derivative(func_GGW, step=step)(bgal[i])

		der_bgal_vec = fcc.vector_cl(cl_cross=der_bgal_GGW, cl_auto1=der_bgal_GG, cl_auto2=der_bgal_GWGW)
		der_bgal_cov_mat = fcc.covariance_matrix(der_bgal_vec, n_bins_z, n_bins_dl)

		der_bgal[i] = der_bgal_cov_mat

		np.save(os.path.join(FLAGS.fout, 'der_bgal_cov_mat_bin_%i.npy' % i), der_bgal_cov_mat)

	return der_bgal

#############################
############################# HERE
#############################
def compute_partial_derivatives_GW(bGW, der_bGW):

	def func_GG(b):
		bGW_temp = np.copy(bGW)
		bGW_temp[i] = b
		return Cl_func(H0_true, Omega_m_true, Omega_b_true, 2.12605, 0.96, bias_gal, bGW_temp)[0]

	def func_GWGW(b):
		bGW_temp = np.copy(bGW)
		bGW_temp[i] = b
		return Cl_func(H0_true, Omega_m_true, Omega_b_true, 2.12605, 0.96, bias_gal, bGW_temp)[1]

	def func_GGW(b):
		bGW_temp = np.copy(bGW)
		bGW_temp[i] = b
		return Cl_func(H0_true, Omega_m_true, Omega_b_true, 2.12605, 0.96, bias_gal, bGW_temp)[2]

	for i in range(len(bGW)):
		print('\nComputing the derivative with respect to the GW bias in bin %i...\n' % i)

		der_bGW_GG = nd.Derivative(func_GG, step=step)(bGW[i])
		der_bGW_GWGW = nd.Derivative(func_GWGW, step=step)(bGW[i])
		der_bGW_GGW = nd.Derivative(func_GGW, step=step)(bGW[i])

		der_bGW_vec = fcc.vector_cl(cl_cross=der_bGW_GGW, cl_auto1=der_bGW_GG, cl_auto2=der_bGW_GWGW)
		der_bGW_cov_mat = fcc.covariance_matrix(der_bGW_vec, n_bins_z, n_bins_dl)

		der_bGW[i] = der_bGW_cov_mat

		np.save(os.path.join(FLAGS.fout, 'der_bGW_cov_mat_bin_%i.npy' % i), der_bGW_cov_mat)

	return der_bGW


#############################
############################# HERE
#############################
def compute_lmin(z):
	conditions = [z < 0.5, (z >= 0.5) & (z < 0.75), (z >= 0.75) & (z < 1.25), z >= 1.25]
	values = [0, 5, 10, 15]
	return np.select(conditions, values, default=np.nan)

#############################
############################# HERE
#############################
def generate_matrix(arr):
	n = len(arr)
	matrix = np.zeros((n, n))
	for i in range(n):
		matrix[i, :i + 1] = arr[:i + 1]
		matrix[i, i + 1:] = arr[i]
	return matrix


#############################
############################# HERE
#############################
def symm(matrix):
	return np.triu(matrix) + np.triu(matrix, k=1).T


#############################
############################# HERE
#############################
def apply_lmin_lmax_mask(all_der_lmin, n_param, n_bins_z, n_bins_dl, bin_centers, ell_matrix):
	for i in range(n_param):
		for ii in range(n_bins_z + n_bins_dl):
			for iii in range(n_bins_z + n_bins_dl):
				# Compute min redshift and corresponding lmin
				z_temp = min(bin_centers[ii], bin_centers[iii])
				lmin_temp = compute_lmin(z_temp).astype(int)

				# Mask all multipoles below lmin
				if lmin_temp != 0:
					all_der_lmin[i, ii, iii, :lmin_temp] = 0

				# Mask all multipoles beyond lmax (with a buffer of -5)
				lmax_temp = (ell_matrix[ii, iii] - 5).astype(int)
				all_der_lmin[i, ii, iii, lmax_temp:] = 0

	return all_der_lmin

#############################
############################# Define a function to rotate Fisher matrix from (H0, Ob) to (H0, ob)
#############################
def rotate_fisher_Ob_to_ob(or_matrix, Ob=0.048, H0=67.7, pos={'H0': 0, 'Ob': 2}):
	nparams = or_matrix.shape[0]
	rotMatrix = np.identity(nparams)  # Identity rotation for non-rotated params

	# Jacobian matrix for variable transformation from (H0, Ob) to (H0, ob)
	J_H0Ob_to_H0ob = np.array([
		[1, 0],
		[-2 * Ob / H0, 1e04 / H0 ** 2]
	])

	# Replace corresponding submatrix in rotation matrix
	rotMatrix[np.ix_([pos['H0'], pos['Ob']], [pos['H0'], pos['Ob']])] = J_H0Ob_to_H0ob

	# Apply rotation: F' = J^T F J
	matrix = rotMatrix.T @ or_matrix @ rotMatrix
	return matrix

#############################
############################# HERE
#############################

def compute_sigma_params(fisher_inv):
    """
    Compute 1σ uncertainties for H0, Omega_m, Omega_b, A_s, n_s.
    """
    diag = np.diag(fisher_inv)
    return np.sqrt(diag[0]), np.sqrt(diag[1]), np.sqrt(diag[2]), np.sqrt(diag[3]), np.sqrt(diag[4])


#############################
############################# HERE
#############################
def compute_relative_errors(sigma_H0, sigma_omega, sigma_omega_b, sigma_As, sigma_ns, H0_true, Omega_m_true, Omega_b_true):
    """
    Compute 2σ relative percentage errors for H0, Omega_m, Omega_b, A_s, n_s.
    """
    return (
        2 * sigma_H0 / H0_true*100,
        2 * sigma_omega / Omega_m_true*100,
        2 * sigma_omega_b / Omega_b_true*100,
        2 * sigma_As / 2.12605*100,
        2 * sigma_ns / 0.96*100
    )

#############################
############################# HERE
#############################

##############################		 ##############################
"""
  __  __          _____ _   _ 
 |  \/  |   /\   |_   _| \ | |
 | \  / |  /  \    | | |  \| |
 | |\/| | / /\ \   | | | . ` |
 | |  | |/ ____ \ _| |_| |\  |
 |_|  |_/_/    \_\_____|_| \_|
                              
                              
MAIN_SECTION:
"""
##############################		 ##############################

if __name__=='__main__':

	# Copy settings in output folder
	shutil.copy(os.path.join( configspath, FLAGS.config+'.py'), os.path.join(FLAGS.fout, 'config_original.py'))

	# import variables in config file
	sys.path.append(configspath)
	config = importlib.import_module( FLAGS.config, package=None)

	# import colibri
	sys.path.insert(0, config.colibri_path)	
	import colibri.cosmology as cc
	import colibri.limber_GW as LLG

	# print input args
	from inspect import getmembers, ismodule
	config_items = {item[0]: item[1] for item in getmembers(config) if '__' not in item[0]}
	print('Config items:')
	print(config_items)

	##############################		 ##############################
	"""
    					INITIAL SETTINGS FROM CONFIG
    """
	##############################		 ##############################
	# GW detector (ET_Delta_2CE, ET_2L_2CE, ET_Delta_1CE, ET_2L_1CE, ET_Delta, ET_2L, LVK)
	GW_det = config.GW_det

	# Years of observation
	yr = config.yr

	# Define the number of bins
	n_bins_z = config.n_bins_z
	n_bins_dl = config.n_bins_dl

	# Define the galaxy bin range
	zm_bin = config.zm_bin
	zM_bin = config.zM_bin

	# Define the GW bin range in redshift (will be converted in dl using the fiducial model)
	zm_bin_GW = config.zm_bin_GW
	zM_bin_GW = config.zM_bin_GW

	# Set the binning strategy (right_cosmo, wrong_cosmo(H0=65, Om0=0.32), equal_pop, equal_space)
	bin_strategy = config.bin_strategy

	# Include the lensing
	Lensing = config.Lensing

	# "True" values of the cosmological parameters
	H0_true = config.H0_true
	Omega_m_true = config.Omega_m_true

	# Fraction of the sky covered from the survey
	f_sky = config.f_sky
	f_sky_GW = config.f_sky_GW

	# Errors on the galaxy distribution
	sig_gal = config.sig_gal

	# galaxy survey (euclid_photo, euclid_spectro, ska)
	gal_det = config.gal_det

	l_min = 5

	# Number of parameters Fisher
	n_param = 5 + n_bins_dl + n_bins_z
	Omega_b_true = 0.048

	# Compute power spectra (True)
	fourier = True

	# Define the redshift total range
	zm = 0.001
	zM = 7

	# Define the luminosity distance total range
	dlm = 1
	dlM = 100000


	##############################		 ##############################
	"""
                        LOADING GW AND GALAXY PARAMETERS
    """
	##############################		 ##############################
	# Load gravitational wave detector parameters
	gw_params = load_detector_params(GW_det, yr)

	# Load galaxy detector parameters
	gal_params = load_galaxy_detector_params(gal_det)

	# Call of the single parameters: first GW and second Galaxies
	A = gw_params['A']
	Alpha = gw_params['Alpha']
	log_dl = gw_params['log_dl']
	s_a = gw_params['s_a']
	Z_0=gw_params['Z_0']
	Beta=gw_params['Beta']

	spline = gal_params['spline']
	bg0 = gal_params['bg0']
	sig_gal = gal_params.get('sig_gal', None)  # may not exist for all detectors

	##############################         ##############################
	"""
     DEFINE FIDUCIAL COSMOLOGICAL MODEL AND COMPUTE CORRESPONDING LUMINOSITY DISTANCES
    """
	##############################         ##############################

	# Luminosity distance interval, equal to the redshift one assuming fiducial cosmology
	fiducial_universe = FlatLambdaCDM(H0=H0_true, Om0=Omega_m_true, Ob0=Omega_b_true)

	dlm_bin = fiducial_universe.luminosity_distance(zm_bin_GW).value  # Minimum luminosity distance from fiducial model
	dlM_bin = fiducial_universe.luminosity_distance(zM_bin_GW).value  # Maximum luminosity distance from fiducial model

	z_gal = np.linspace(zm, zM, 1200)  # Redshift grid for galaxy distribution
	dl_GW = np.linspace(dlm, dlM, 1200)  # Luminosity distance grid for gravitational wave sources

	##############################		 ##############################
	"""
								BIN STRATEGY
	"""
	##############################		 ##############################

	bin_int = np.linspace(zm_bin, zM_bin, n_bins_z * 1000)  # Fine redshift grid for binning
	bin_int_GW = np.linspace(dlm_bin / 1000, dlM_bin / 1000, n_bins_dl * 1000)  # Fine luminosity distance grid for GW binning (in Gpc)

	# Compute bin edges using the specified strategy and cosmology
	bin_edges, bin_edges_dl = compute_bin_edges(bin_strategy, n_bins_dl, n_bins_z, bin_int, zM_bin, dlM_bin, zm_bin, fiducial_universe, A, Z_0, Alpha, Beta, spline, fcc)

	# Convert luminosity distance bin edges to redshift using the fiducial cosmology
	bin_z_fiducial = (bin_edges_dl * u.Gpc).to(cu.redshift,cu.redshift_distance(fiducial_universe, kind="luminosity")).value

	# Save bin edges for later use
	np.save(os.path.join(FLAGS.fout, 'bin_edges_GW_fiducial.npy'), bin_z_fiducial)
	np.save(os.path.join(FLAGS.fout, 'bin_edges_GW.npy'), bin_edges_dl)
	np.save(os.path.join(FLAGS.fout, 'bin_edges_gal.npy'), bin_edges)

	# Compute redshift distribution and total number of galaxies
	nz_gal, gal_tot = compute_nz_gal_and_total(gal_det, z_gal, bin_edges, sig_gal, gal_params['spline'], fcc)

	gal_tot[gal_tot < 0] = 0  # Remove negative values (if any)
	n_tot_gal = np.trapz(gal_tot, z_gal)  # Integrate total galaxy distribution
	print('\nthe total number of galaxies across all redshift: ', n_tot_gal * 4 * np.pi * f_sky)

	# Compute fraction of galaxies in each redshift bin
	bin_frac_gal = np.zeros(shape=(n_bins_z))
	for i in range(n_bins_z):
		bin_frac_gal[i] = sint.simps(nz_gal[i], z_gal)

	n_gal_bins = np.sum(bin_frac_gal)  # Sum of galaxy fractions across bins
	print('the total number of galaxies in our bins: ', n_gal_bins * 4 * np.pi * f_sky)


	##############################         ##############################
	"""
                    PLOTTING THE GALAXY BIN DISTRIBUTION
    """
	##############################         ##############################

	# Plot each galaxy bin's redshift distribution
	for i in range(n_bins_z):
		plt.plot(z_gal, nz_gal[i])

	# Plot vertical lines indicating bin edges
	for i in range(n_bins_z):
		plt.axvline(bin_edges[i], c='black', alpha=0.5)
	plt.axvline(bin_edges[-1], c='black', alpha=0.5, label='bin edges')

	# Label axes and configure plot appearance
	plt.xlabel(r'$z$')  # Redshift axis label
	plt.ylabel(r'$w_i$')  # Weight or distribution label
	plt.xlim(zm_bin - 0.3, zM_bin + 0.3)  # X-axis limits

	plt.title('Galaxy bin distribution')  # Plot title
	plt.plot(z_gal, gal_tot, ls='--', alpha=0.8, color='red', label='total\ndistribution')  # Plot total galaxy distribution
	plt.savefig(os.path.join(FLAGS.fout, 'gal_distr.pdf'), bbox_inches='tight')  # Save the plot

	plt.close()  # Close the figure

	# Print statistics about galaxy bins
	print('mean number of galaxies in each bin: ', np.mean(bin_frac_gal))
	print('mean shot noise in each bin: ', np.mean(shot_noise_gal))

	##############################         ##############################
	"""
    # Determine representative redshifts and compute nonlinear power spectrum
    """
	##############################         ##############################

	# Initialize array to store the peak redshift of each galaxy bin
	redshift = np.zeros(shape=n_bins_z)
	for i in range(n_bins_z):
		a = np.argmax(nz_gal[i])  # Index of maximum value in the redshift distribution
		redshift[i] = z_gal[a]  # Assign corresponding redshift

	# Define k and z arrays for evaluating the nonlinear power spectrum
	kk_nl = np.geomspace(1e-4, 1e2, 200)  # Logarithmically spaced k values
	zz_nl = np.linspace(zm_bin_GW, zM_bin_GW, 100)  # Linearly spaced redshift values

	# Initialize cosmology for power spectrum calculation
	C = cc.cosmo(Omega_m=Omega_m_true, h=H0_true / 100)

	# Compute nonlinear matter power spectrum using CAMB with Halofit
	_, P_vals = C.camb_Pk(z=zz_nl, k=kk_nl, nonlinear=True, halofit='mead2020')

	# Interpolate power spectrum over redshift and k
	P_interp = RectBivariateSpline(zz_nl, kk_nl, P_vals)

	# Use peak redshifts of bins as centers for computing kmax
	zcenters_use = redshift

	# Compute maximum usable wavenumber at each redshift bin center
	kmax = compute_kmax(zcenters_use, P_interp, kk_nl)

	##############################         ##############################
	"""
    COMPUTING MULTIPOLE LIMITS AND GW BIN DISTRIBUTION STATISTICS
    """
	##############################         ##############################

	# Compute maximum multipole l for each bin using comoving distance and kmax
	l_max_nl = np.asarray(
		[fiducial_universe.comoving_distance(zcenters_use[i]).value * k_ for i, k_ in enumerate(kmax)]).astype(int)

	# Compute localization error parameters for GW bins
	sigma_sn_GW, l_max_loc = fcc.loc_error_param(bin_edges_dl, log_loc, log_dl, l_min, 10000)

	# Determine lengths of arrays
	n = len(l_max_nl)
	m = len(l_max_loc)

	# Extend l_max_nl to match length of l_max_loc
	l_max_nl_ = np.concatenate((l_max_nl, l_max_loc[-(m - n):]))

	# Compute final l_max per bin as the minimum between localization and nonlinear limits
	l_max_bin = np.minimum(l_max_loc, l_max_nl_)

	# Determine whether the limiting factor is localization (0) or nonlinear scale (1)
	loc_or_nl = np.where(l_max_loc <= l_max_nl_, 0, 1)

	# Compute overall maximum multipole
	l_max = np.max(l_max_nl_)

	# Define multipole vector with increasing step sizes at higher l
	ll = np.sort(np.unique(np.concatenate([
		np.arange(l_min, 20, step=2),
		np.arange(20, 50, step=5),
		np.arange(50, 100, step=10),
		np.arange(100, l_max + 1, step=25)
	])))
	ll[-1] = l_max  # Ensure maximum l is included
	ll_total = np.arange(l_min, l_max + 1)

	# Compute normalization factor for Cl's
	c = ll * (ll + 1.) / (2. * np.pi)

	# Print diagnostics
	print('l vector: ', ll)
	print('l max bin: ', l_max_bin)

	# Save computed arrays
	np.save(os.path.join(FLAGS.fout, 'ell_max.npy'), l_max_bin)
	np.save(os.path.join(FLAGS.fout, 'loc_nl.npy'), loc_or_nl)

	##############################         ##############################
	"""
                COMPUTING AND PLOTTING GW BIN DISTRIBUTION STATISTICS
    """
	##############################         ##############################

	# Compute the merger rate distribution and related quantities from luminosity distance bins
	z_GW, bin_convert, ndl_GW, nGW, merger_rate_tot = fcc.merger_rate_dl(
		dl=dl_GW,
		bin_dl=bin_edges_dl,
		log_dl=log_dl,
		log_delta_dl=log_delta_dl,
		H0=H0_true,
		omega_m=Omega_m_true,
		omega_b=Omega_b_true,
		A=A,
		Z_0=Z_0,
		Alpha=Alpha,
		Beta=Beta,
		normalize=False
	)

	# Integrate the total merger rate over the full luminosity distance range (in Gpc)
	n_tot_GW = np.trapz(merger_rate_tot, dl_GW / 1000) * 4 * np.pi
	print('\nthe total number of GW across all distance: ', n_tot_GW)

	# Calculate the fraction of GW sources in each luminosity distance bin
	bin_frac_GW = np.zeros(shape=(n_bins_dl))
	for i in range(n_bins_dl):
		bin_frac_GW[i] = np.trapz(ndl_GW[i], dl_GW / 1000)

	# Sum all bin fractions to get the total number in bins (should match total GW if complete)
	n_GW_bins = np.sum(bin_frac_GW)
	print('the total number of GW in our bins: ', n_GW_bins * 4 * np.pi)

	# Plot the GW bin distributions over distance
	for i in range(n_bins_dl):
		plt.plot(dl_GW / 1000, ndl_GW[i])  # Plot each bin distribution
	for i in range(n_bins_dl):
		plt.axvline(bin_edges_dl[i], c='black', alpha=0.5)  # Add bin edges
	plt.axvline(bin_edges_dl[-1], c='black', alpha=0.5, label='bin edges')

	# Configuration of the plot
	plt.xlabel(r'$d_L[Gpc]$')
	plt.ylabel(r'$w_i$')
	plt.title('GW bin distribution')
	plt.plot(dl_GW / 1000, merger_rate_tot, ls='--', alpha=0.8, color='red', label='total\ndistribution')
	plt.savefig(os.path.join(FLAGS.fout, 'GW_distr.pdf'), bbox_inches='tight')
	plt.close()

	# Print per-bin and mean statistics for GW shot noise
	print('fraction of GW per sterad in each bin', bin_frac_GW)
	shot_noise_GW = 1 / bin_frac_GW
	print('shot noise per bin', shot_noise_GW)
	print('mean number of GW in each bin: ', np.mean(bin_frac_GW))
	print('mean shot noise in each bin: ', np.mean(shot_noise_GW))

	##############################         ##############################
	"""
            FIGURES FOR COMPARING GALAXY AND GW DISTRIBUTIONS
    """
	##############################         ##############################
	fig = plt.figure(figsize=(18, 7), tight_layout=True)

	# First subplot: galaxy redshift distributions per bin
	ax = fig.add_subplot(121)
	for i in range(n_bins_z):
		plt.plot(z_gal, nz_gal[i])
	for i in range(n_bins_z + 1):
		plt.axvline(bin_edges[i], c='black', alpha=0.5)  # Bin edges
	plt.xlabel(r'$z$')
	plt.ylabel(r'$\frac{dN}{dzd\Omega}$')
	plt.title('Galaxy distribution')
	plt.xlim(zm_bin - 0.3, zM_bin + 0.3)
	plt.plot(z_gal, gal_tot, ls='--', alpha=0.5, color='red')  # Total galaxy distribution

	# Second subplot: GW merger rate distribution per bin in redshift
	ax = fig.add_subplot(122)
	for i in range(n_bins_dl):
		plt.plot(z_GW, ndl_GW[i])
	for i in range(n_bins_dl + 1):
		plt.axvline(bin_convert[i], c='black', alpha=0.5)  # Bin edges converted to redshift
	plt.xlabel(r'$z$')
	plt.ylabel(r'$\frac{dN}{dzd\Omega}$')
	plt.title('Merger rate, fiducial model')
	plt.xlim(zm_bin - 0.3, bin_convert[-1] + 0.5)
	plt.plot(z_GW, merger_rate_tot, ls='--', alpha=0.5, color='red')  # Total merger rate

	# Save the comparison plot
	plt.savefig(os.path.join(FLAGS.fout, 'distr_compare.pdf'), bbox_inches='tight')
	plt.close()

	##############################         ##############################
	"""
	        DEFINITION OF Cl_func DEPENDING ON THE PRESENCE OF THE LENSING
	"""
	##############################         ##############################
	# If lensing is included in the analysis
	if Lensing:

		# Define function to compute Cl including lensing, clustering, and RSD contributions
		def Cl_func(H_0, Omega_m, Omega_b, A_s, n_s, b_gal, b_GW, npoints=13, npoints_x=20, grid_x='lin', zmin=1e-05,
					nlow=5, nhigh=5):

			# Define cosmology and initialize Limber integrator
			C = cc.cosmo(Omega_m=Omega_m, Omega_b=Omega_b, h=H_0 / 100, As=1e-9 * A_s, ns=n_s)
			S = LLG.limber(cosmology=C, z_limits=[zm, zM])

			# Define k and z grids for power spectrum
			kk = np.geomspace(1e-4, 1e2, 301)
			zz = np.linspace(0, zM, 101)

			# Compute nonlinear matter power spectrum with CAMB
			_, pkz = C.camb_Pk(z=zz, k=kk, nonlinear=True, halofit='mead2020')
			S.load_power_spectra(z=zz, k=kk, power_spectra=pkz)

			# Generate GW distribution from fiducial parameters
			z_GW, bin_GW_converted, ndl_GW, nGW, total = fcc.merger_rate_dl(
				dl=dl_GW, bin_dl=bin_edges_dl, log_dl=log_dl, log_delta_dl=log_delta_dl,
				H0=H_0, omega_m=Omega_m, omega_b=Omega_b, A=A, Z_0=Z_0, Alpha=Alpha, Beta=Beta, normalize=False
			)

			# Load bin edges for all observables
			S.load_bin_edges(bin_edges, bin_z_fiducial, name_1='lensing_gal', name_2='lensing_GW')
			S.load_bin_edges(bin_edges, bin_z_fiducial, name_1='galaxy', name_2='GW')
			S.load_bin_edges(bin_edges, bin_z_fiducial, name_1='rsd', name_2='lsd')

			# Compute galaxy magnification slope parameter beta
			beta = compute_beta(H_0, Omega_m, Omega_b, s_a, s_b, s_c, s_d, be_a, be_b, be_c, be_d)

			# Load window functions for each observable
			S.load_galaxy_clustering_window_functions(z=z_gal, nz=nz_gal, ll=ll, bias=b_gal, name='galaxy')
			S.load_rsd_window_functions(z=z_gal, nz=nz_gal, H_0=H_0, omega_m=Omega_m, omega_b=Omega_b, ll=ll,
										name='rsd')
			S.load_gravitational_wave_window_functions(z=z_GW, ndl=ndl_GW, H_0=H_0, omega_m=Omega_m, omega_b=Omega_b,
													   ll=ll, bias=b_GW, name='GW')
			S.load_lsd_window_functions(z=z_GW, ndl=ndl_GW, H_0=H_0, omega_m=Omega_m, omega_b=Omega_b, ll=ll,
										name='lsd')
			S.load_galaxy_lensing_window_functions(z=z_gal, nz=nz_gal, H_0=H_0, omega_m=Omega_m, ll=ll,
												   name='lensing_gal')
			S.load_gw_lensing_window_functions(z=z_GW, ndl=ndl_GW, H_0=H_0, omega_m=Omega_m, omega_b=Omega_b, ll=ll,
											   name='lensing_GW')

			# Compute all angular power spectra using Limber integrals
			Cl = S.limber_angular_power_spectra(l=ll, windows=['galaxy', 'GW', 'rsd', 'lsd'])
			Cl_lens = S.limber_angular_power_spectra_lensing_auto(
				l=ll, s_gal=s_gal, beta=beta, H_0=H_0, omega_m=Omega_m, omega_b=Omega_b,
				windows=['lensing_gal', 'lensing_GW'], npoints=npoints, npoints_x=npoints_x,
				zmin=zmin, grid_x=grid_x, nlow=nlow, nhigh=nhigh
			)
			Cl_lens_cross = S.limber_angular_power_spectra_lensing_cross(
				l=ll, s_gal=s_gal, beta=beta, H_0=H_0, omega_m=Omega_m, omega_b=Omega_b,
				windows=None, npoints=npoints, npoints_x=npoints_x,
				zmin=zmin, grid_x=grid_x, nlow=nlow, nhigh=nhigh
			)

			# Extract auto and cross angular power spectra
			Cl_delta_GG = Cl['galaxy-galaxy']
			Cl_delta_GWGW = Cl['GW-GW']
			Cl_delta_GGW = Cl['galaxy-GW']

			Cl_len_GG = Cl_lens['lensing_gal-lensing_gal']
			Cl_len_GWGW = Cl_lens['lensing_GW-lensing_GW']
			Cl_len_GGW = Cl_lens['lensing_gal-lensing_GW']

			Cl_RSD_GG = Cl['rsd-rsd']
			Cl_RSD_GWGW = Cl['lsd-lsd']
			Cl_RSD_GGW = Cl['rsd-lsd']

			Cl_delta_len_GG = Cl_lens_cross['galaxy-lensing_gal']
			Cl_delta_len_GWGW = Cl_lens_cross['GW-lensing_GW']
			Cl_delta_len_GGW = Cl_lens_cross['galaxy-lensing_GW']
			Cl_delta_len_GWG = Cl_lens_cross['GW-lensing_gal']

			Cl_delta_RSD_GG = Cl['galaxy-rsd']
			Cl_delta_RSD_GWGW = Cl['GW-lsd']
			Cl_delta_RSD_GGW = Cl['galaxy-lsd']
			Cl_delta_RSD_GWG = Cl['GW-rsd']

			Cl_RSD_len_GG = Cl_lens_cross['rsd-lensing_gal']
			Cl_RSD_len_GWGW = Cl_lens_cross['lsd-lensing_GW']
			Cl_RSD_len_GGW = Cl_lens_cross['rsd-lensing_GW']
			Cl_RSD_len_GWG = Cl_lens_cross['lsd-lensing_gal']

			# Ensure matrix symmetry where needed
			Cl_delta_len_GWG = np.swapaxes(Cl_delta_len_GWG, 0, 1)
			Cl_delta_RSD_GWG = np.swapaxes(Cl_delta_RSD_GWG, 0, 1)
			Cl_RSD_len_GWG = np.swapaxes(Cl_RSD_len_GWG, 0, 1)

			# Combine all contributions to total angular power spectra
			Cl_GG = Cl_delta_GG + Cl_len_GG + Cl_RSD_GG + 2 * Cl_delta_len_GG + 2 * Cl_delta_RSD_GG + 2 * Cl_RSD_len_GG
			Cl_GWGW = Cl_delta_GWGW + Cl_len_GWGW + Cl_RSD_GWGW + 2 * Cl_delta_len_GWGW + 2 * Cl_delta_RSD_GWGW + 2 * Cl_RSD_len_GWGW
			Cl_GGW = (Cl_delta_GGW + Cl_len_GGW + Cl_RSD_GGW +
					  Cl_delta_len_GGW + Cl_delta_len_GWG +
					  Cl_delta_RSD_GGW + Cl_delta_RSD_GWG +
					  Cl_RSD_len_GGW + Cl_RSD_len_GWG)

			return Cl_GG, Cl_GWGW, Cl_GGW

	# If lensing is not included, compute only density clustering spectra
	else:

		# Define function to compute Cl from galaxy and GW clustering only
		def Cl_func(H_0, Omega_m, Omega_b, A_s, n_s, b_gal, b_GW, npoints=13, npoints_x=20, grid_x='mix', zmin=1e-05,
					nlow=5, nhigh=5):

			# Define cosmology and Limber integrator
			C = cc.cosmo(Omega_m=Omega_m, Omega_b=Omega_b, h=H_0 / 100, As=1e-9 * A_s, ns=n_s)
			S = LLG.limber(cosmology=C, z_limits=[zm, zM])

			# Define power spectrum grids
			kk = np.geomspace(1e-4, 1e2, 500)
			zz = np.linspace(0, zM, 100)

			# Compute nonlinear matter power spectrum
			_, pkz = C.camb_Pk(z=zz, k=kk, nonlinear=True, halofit='mead2020')
			S.load_power_spectra(z=zz, k=kk, power_spectra=pkz)

			# Generate GW source distribution
			z_GW, bin_GW_converted, ndl_GW, nGW, total = fcc.merger_rate_dl(
				dl=dl_GW, bin_dl=bin_edges_dl, log_dl=log_dl, log_delta_dl=log_delta_dl,
				H0=H_0, omega_m=Omega_m, omega_b=Omega_b, A=A, Z_0=Z_0, Alpha=Alpha, Beta=Beta, normalize=False
			)

			# Load binning and window functions
			S.load_bin_edges(bin_edges, bin_GW_converted)
			S.load_galaxy_clustering_window_functions(z=z_gal, nz=nz_gal, ll=ll, bias=b_gal, name='galaxy')
			S.load_gravitational_wave_window_functions(z=z_GW, ndl=ndl_GW, H_0=H_0, omega_m=Omega_m, omega_b=Omega_b,
													   ll=ll, bias=b_GW, name='GW')

			# Compute angular power spectra (density terms only)
			Cl = S.limber_angular_power_spectra(l=ll, windows=None)

			# Extract auto and cross terms
			Cl_delta_GG = Cl['galaxy-galaxy']
			Cl_delta_GWGW = Cl['GW-GW']
			Cl_delta_GGW = Cl['galaxy-GW']

			return Cl_delta_GG, Cl_delta_GWGW, Cl_delta_GGW

	##############################         ##############################
	"""
    					COMPUTING FIDUCIAL BIASES
    """
	##############################         ##############################

	# Parameters for GW bias model
	A_GW = 1.2
	gamma = 0.59

	# Compute mean redshift for each GW bin and corresponding GW bias
	z_mean_GW = (bin_z_fiducial[:-1] + bin_z_fiducial[1:]) * 0.5
	bias_GW = A_GW * (1. + z_mean_GW) ** gamma
	np.save(os.path.join(FLAGS.fout, 'bias_fiducial_GW'), bias_GW)

	# Compute mean redshift for each galaxy bin and galaxy bias using polynomial model
	z_mean_gal = (bin_edges[:-1] + bin_edges[1:]) * 0.5
	bias_gal = bg0 + bg1 * z_mean_gal + bg2 * z_mean_gal ** 2 + bg3 * z_mean_gal ** 3
	np.save(os.path.join(FLAGS.fout, 'bias_fiducial_gal'), bias_gal)

	# Compute magnification slope s(z) depending on galaxy detector
	s_gal= compute_s_gal(z_gal, gal_det, sg0, sg1, sg2, sg3)


	##############################         ##############################
	"""
					COMPUTING LOCALIZATION NOISE MATRICES
	"""
	##############################         ##############################
	# Compute shot noise matrices for galaxies and GW
	noise_gal = fcc.shot_noise_mat_auto(shot_noise_gal, ll_total)
	noise_GW = fcc.shot_noise_mat_auto(shot_noise_GW, ll_total)

	# Initialize localization noise attenuation matrices
	noise_loc = np.zeros(shape=(n_bins_dl, len(ll_total)))
	noise_loc_auto = np.zeros(shape=(n_bins_dl, len(ll_total)))

	# Compute exponential localization damping for each bin and multipole
	for i in range(n_bins_dl):
		for l in range(len(ll_total)):
			ell_term = ll_total[l] * (ll_total[l] + 1) * (sigma_sn_GW[i] / (2 * np.pi) ** (3 / 2))
			if ell_term < 30:
				noise_loc[i, l] = np.exp(-ell_term)
				noise_loc_auto[i, l] = np.exp(-2 * ell_term)
			else:
				noise_loc[i, l] = np.exp(-30)
				noise_loc_auto[i, l] = np.exp(-30)

	# Create full 3D localization noise matrix for galaxy-GW cross-spectra
	noise_loc_mat = np.zeros(shape=(n_bins_z, n_bins_dl, len(ll_total)))
	for i in range(n_bins_z):
		for ii in range(n_bins_dl):
			noise_loc_mat[i, ii, :] = noise_loc[ii, :]

	# Create full 3D localization noise matrix for GW-GW auto-spectra
	noise_loc_mat_auto = np.zeros(shape=(n_bins_dl, n_bins_dl, len(ll_total)))
	for i in range(n_bins_dl):
		for ii in range(i, n_bins_dl):
			noise_loc_mat_auto[i, ii, :] = noise_loc_auto[ii, :]
	for i in range(n_bins_dl):
		for ii in range(i + 1, n_bins_dl):
			noise_loc_mat_auto[ii, i] = noise_loc_mat_auto[i, ii]

	##############################         ##############################
	"""
						COMPUTING THE POWER SPECTRUM
	"""
	##############################         ##############################
	# Print status message for power spectrum computation
	print('\nComputing the Power Spectrum...\n')

	# Compute angular power spectra from Cl_func with fiducial cosmological and bias parameters
	Cl_GG, Cl_GWGW, Cl_GGW = Cl_func(
		H_0=H0_true,
		Omega_m=Omega_m_true,
		Omega_b=Omega_b_true,
		A_s=2.12605,
		n_s=0.96,
		b_gal=bias_gal,
		b_GW=bias_GW
	)

	# Initialize arrays to store interpolated Cl on the full ell grid
	Cl_GG_total = np.zeros(shape=(n_bins_z, n_bins_z, len(ll_total)))
	Cl_GWGW_total = np.zeros(shape=(n_bins_dl, n_bins_dl, len(ll_total)))
	Cl_GGW_total = np.zeros(shape=(n_bins_z, n_bins_dl, len(ll_total)))

	# Interpolate Cl_GG to full ell grid
	for i in range(n_bins_z):
		for ii in range(n_bins_z):
			Cl_GG_interp = si.interp1d(ll, Cl_GG[i, ii])
			Cl_GG_total[i, ii] = Cl_GG_interp(ll_total)

	# Interpolate Cl_GWGW to full ell grid
	for i in range(n_bins_dl):
		for ii in range(n_bins_dl):
			Cl_GWGW_interp = si.interp1d(ll, Cl_GWGW[i, ii])
			Cl_GWGW_total[i, ii] = Cl_GWGW_interp(ll_total)

	# Interpolate Cl_GGW to full ell grid
	for i in range(n_bins_z):
		for ii in range(n_bins_dl):
			Cl_GGW_interp = si.interp1d(ll, Cl_GGW[i, ii])
			Cl_GGW_total[i, ii] = Cl_GGW_interp(ll_total)

	# Save all computed Cls and noise terms to disk
	np.save(os.path.join(FLAGS.fout, 'Cl_GG'), Cl_GG_total)
	np.save(os.path.join(FLAGS.fout, 'Cl_GWGW'), Cl_GWGW_total)
	np.save(os.path.join(FLAGS.fout, 'Cl_GGW'), Cl_GGW_total)
	np.save(os.path.join(FLAGS.fout, 'noise_GW'), noise_GW)
	np.save(os.path.join(FLAGS.fout, 'noise_gal'), noise_gal)
	np.save(os.path.join(FLAGS.fout, 'noise_loc_auto'), noise_loc_mat_auto)
	np.save(os.path.join(FLAGS.fout, 'noise_loc_cross'), noise_loc_mat)

	# Apply localization damping matrices to GW-GW and GW-Galaxy spectra
	Cl_GWGW_total *= noise_loc_mat_auto
	Cl_GGW_total *= noise_loc_mat

	# Add shot noise to the auto-correlations
	Cl_GWGW_total += noise_GW
	Cl_GG_total += noise_gal


	##############################         ##############################
	"""
    					COMPUTING FIDUCIAL COVARIANCE MATRIX
    """
	##############################         ##############################

	# Construct the full data vector from Cl auto- and cross-spectra
	vec = fcc.vector_cl(cl_cross=Cl_GGW_total, cl_auto1=Cl_GG_total, cl_auto2=Cl_GWGW_total)

	# Compute the covariance matrix of the fiducial spectra
	cov_mat = fcc.covariance_matrix(vec, n_bins_z, n_bins_dl)

	# Save the fiducial covariance matrix to file
	np.save(os.path.join(FLAGS.fout, 'cov_mat'), cov_mat)


	##############################         ##############################
	"""
					COMPUTING PARAMETER DERIVATIVE MATRICE
	"""
	##############################         ##############################
	# Initialize dictionary to store parameter-specific covariance derivatives
	covariance_matrices = {}

	# Define list of cosmological parameters for which to compute derivatives
	parameters = [
		{
			"name": "Hubble constant",
			"symbol": "H0",
			"true_value": H0_true,
			"step": 1e-2,
			"derivative_args": lambda x: Cl_func(x, Omega_m_true, Omega_b_true, 2.12605, 0.96, bias_gal, bias_GW),
			"key": "der_H0_cov_mat"
		},
		{
			"name": "Matter density parameter",
			"symbol": "Omega_m",
			"true_value": Omega_m_true,
			"step": 1e-6,
			"method": "central",
			"derivative_args": lambda x: Cl_func(H0_true, x, Omega_b_true, 2.12605, 0.96, bias_gal, bias_GW),
			"key": "der_omega_cov_mat"
		},
		{
			"name": "Baryon density parameter",
			"symbol": "Omega_b",
			"true_value": Omega_b_true,
			"step": 1e-5,
			"method": "central",
			"derivative_args": lambda x: Cl_func(H0_true, Omega_m_true, x, 2.12605, 0.96, bias_gal, bias_GW),
			"key": "der_omega_b_cov_mat"
		},
		{
			"name": "Amplitude of the primordial power spectrum",
			"symbol": "A_s",
			"true_value": 2.12605,
			"step": 1e-3,
			"derivative_args": lambda x: Cl_func(H0_true, Omega_m_true, Omega_b_true, x, 0.96, bias_gal, bias_GW),
			"key": "der_As_cov_mat"
		},
		{
			"name": "Spectral index of the primordial power spectrum",
			"symbol": "n_s",
			"true_value": 0.96,
			"step": 1e-3,
			"derivative_args": lambda x: Cl_func(H0_true, Omega_m_true, Omega_b_true, 2.12605, x, bias_gal, bias_GW),
			"key": "der_ns_cov_mat"
		}
	]

	# Compute and save the derivative covariance matrices for each parameter
	compute_parameter_derivatives(parameters, FLAGS, n_bins_z, n_bins_dl, covariance_matrices)


	##############################         ##############################
	"""
    			COMPUTING DERIVATIVES WITH RESPECT TO BIASES
    """
	##############################         ##############################

	# Define step size for numerical differentiation
	step = 1e-3

	# Initialize array to store galaxy bias derivatives for each bin pair
	der_bgal = np.zeros(shape=(n_bins_z, n_bins_dl + n_bins_z, n_bins_dl + n_bins_z, len(ll)))

	# Compute covariance matrix derivatives with respect to galaxy bias parameters
	der_bgal_cov_mat = compute_partial_derivatives_gal(bias_gal, der_bgal)

	# Initialize array to store GW bias derivatives for each bin pair
	der_bGW = np.zeros(shape=(n_bins_dl, n_bins_dl + n_bins_z, n_bins_dl + n_bins_z, len(ll)))

	# Compute covariance matrix derivatives with respect to GW bias parameters
	der_bGW_cov_mat = compute_partial_derivatives_GW(bias_GW, der_bGW)

	# Notify that all bias-related derivatives have been computed
	print('\nAll derivative computed\n')

	# Initialize full derivative array including all cosmological and bias parameters
	all_der = np.zeros((n_param, n_bins_z + n_bins_dl, n_bins_z + n_bins_dl, len(ll)))

	# Assign precomputed cosmological parameter derivatives
	all_der[0] = covariance_matrices['der_H0_cov_mat']
	all_der[1] = covariance_matrices['der_omega_cov_mat']
	all_der[2] = covariance_matrices['der_omega_b_cov_mat']
	all_der[3] = covariance_matrices['der_As_cov_mat']
	all_der[4] = covariance_matrices['der_ns_cov_mat']

	# Insert galaxy bias derivatives at appropriate indices
	for i in range(n_bins_z):
		index = n_param - n_bins_dl - n_bins_z + i
		all_der[index] = der_bgal_cov_mat[i]

	# Insert GW bias derivatives at appropriate indices
	for i in range(n_bins_dl):
		index = n_param - n_bins_dl + i
		all_der[index] = der_bGW_cov_mat[i]

	# Print shape for verification
	print(all_der.shape)

	##############################         ##############################
	"""
    INTERPOLATING DERIVATIVES TO FULL MULTIPOLE RANGE AND APPLYING LMIN & LMAX MASKING
    """
	##############################         ##############################

	# Initialize array to store all derivatives interpolated over full multipole range
	all_der_total = np.zeros((n_param, n_bins_z + n_bins_dl, n_bins_z + n_bins_dl, len(ll_total)))

	# Interpolate derivatives from original ell grid (ll) to full ell_total grid
	for i in range(n_param):
		for ii in range(n_bins_z + n_bins_dl):
			for iii in range(n_bins_z + n_bins_dl):
				all_der_interp = si.interp1d(ll, all_der[i, ii, iii])
				all_der_total[i, ii, iii] = all_der_interp(ll_total)

	# Initialize masking array to zero out invalid ell values (e.g., below lmin or above lmax)
	all_der_lmin = np.ones_like(all_der_total)

	# Combine galaxy and GW bin centers for redshift-based lmin calculation
	bin_centers = np.concatenate((z_mean_gal, z_mean_GW), axis=0)

	# Combine nonlinear and localization-based ell max values for each bin
	ell_max_total = np.concatenate((l_max_nl, l_max_bin), axis=0)

	# Generate ell cutoff matrix (cross-bin max-allowed multipole)
	ell_matrix = generate_matrix(ell_max_total)

	# Ensure symmetry of ell matrix (important for covariance structure)
	for i in range(len(ell_max_total)):
		for ii in range(len(ell_max_total)):
			ell_matrix[i, ii] = min(ell_matrix[i, ii], ell_max_total[ii])
	ell_matrix = symm(ell_matrix)

	# Apply lmin and lmax masks to the derivative array
	all_der_lmin = apply_lmin_lmax_mask(all_der_lmin,n_param,n_bins_z,n_bins_dl,bin_centers,ell_matrix)

	# Apply the lmin/lmax mask to the full derivative tensor
	all_der_total = all_der_total * all_der_lmin

	# Save the masked and interpolated derivative array
	np.save(os.path.join(FLAGS.fout, 'all_der_total.npy'), all_der_total)

	##############################         ##############################
	"""
    COMPUTING FISHER MATRIX AND ROTATING TO NEW PARAMETER BASIS
    """
	##############################         ##############################

	# Compute Fisher information matrix using the full derivative set and covariance
	fisher = fcc.fisher_matrix(cov_mat, all_der_total, ll_total, f_sky)

	# Rotate the Fisher matrix to new basis
	fisher = rotate_fisher_Ob_to_ob(fisher)

	# Save the rotated Fisher matrix
	np.save(os.path.join(FLAGS.fout, 'fisher_mat.npy'), fisher)

	##############################         ##############################
	"""
    EXTRACTING 1σ ERRORS AND RELATIVE ERRORS FROM FISHER MATRIX
    """
	##############################         ##############################

	# Invert the Fisher matrix and extract the 2×2 marginalized submatrix (H0, Omega_m)
	fisher_inv = scipy.linalg.inv(fisher)
	fisher_marg = fisher_inv[:2, :2]

	# Compute 1σ uncertainties for cosmological parameters (sqrt of diagonal elements)
	sigma_H0, sigma_omega, sigma_omega_b, sigma_As, sigma_ns = compute_sigma_params(fisher_inv)

	# Compute 1σ uncertainties for galaxy bias in each redshift bin
	sigma_bias_gal = np.zeros(shape=(n_bins_z))
	for i in range(n_bins_z):
		index = n_param - n_bins_dl - n_bins_z + i
		sigma_bias_gal[i] = np.sqrt(fisher_inv[index, index])

	# Compute 1σ uncertainties for GW bias in each luminosity distance bin
	sigma_bias_GW = np.zeros(shape=(n_bins_dl))
	for i in range(n_bins_dl):
		index = n_param - n_bins_dl + i
		sigma_bias_GW[i] = np.sqrt(fisher_inv[index, index])

	# Print absolute 1σ errors
	print('\nH_0 = ', sigma_H0)
	print('Omega_m = ', sigma_omega)
	print('Omega_b = ', sigma_omega_b)
	print('A_s = ', sigma_As)
	print('n_s = ', sigma_ns)

	for i in range(n_bins_z):
		print('bias galaxy bin %i = ' % (i + 1), sigma_bias_gal[i])
	for i in range(n_bins_dl):
		print('bias GW bin %i = ' % (i + 1), sigma_bias_GW[i])

	# Compute relative 2σ percentage errors for cosmological parameters
	rel_err_H0, rel_err_omega, rel_err_omega_b, rel_err_As, rel_err_ns = compute_relative_errors(
		sigma_H0, sigma_omega, sigma_omega_b, sigma_As, sigma_ns,
		H0_true, Omega_m_true, Omega_b_true)

	# Compute relative 2σ percentage errors for biases
	rel_err_bias_gal = np.zeros(shape=(n_bins_z))
	for i in range(n_bins_z):
		rel_err_bias_gal[i] = 2 * sigma_bias_gal[i] / bias_gal[i] * 100

	rel_err_bias_GW = np.zeros(shape=(n_bins_dl))
	for i in range(n_bins_dl):
		rel_err_bias_GW[i] = 2 * sigma_bias_GW[i] / bias_GW[i] * 100

	# Print relative errors
	print('\nrelative errors:\n')
	print('H_0 = ', rel_err_H0)
	print('Omega_m = ', rel_err_omega)
	print('Omega_b = ', rel_err_omega_b)
	print('A_s = ', rel_err_As)
	print('n_s = ', rel_err_ns)

	##############################         ##############################
	"""
    		SAVING RELATIVE ERRORS 
    """
	##############################         ##############################

	# Write relative errors of cosmological parameters to results_error.txt
	with open(os.path.join(FLAGS.fout, 'results_error.txt'), 'a') as file:
		file.write(
			'\ndetector: %s, year: %i, lensing:%s, bin strategy: %s, n_bins_z: %i, n_bins_dl: %i\nH_0 = %.2f, Omega_m = %.2f, Omega_b = %.2f, A_s = %.2f, n_s = %.2f\n' %
			(GW_det, yr, Lensing, bin_strategy, n_bins_z, n_bins_dl, rel_err_H0, rel_err_omega, rel_err_omega_b, rel_err_As,
			 rel_err_ns))

	# Write and print relative errors for galaxy bias
	for i in range(n_bins_z):
		print('bias galaxy bin %i = ' % (i + 1), rel_err_bias_gal[i])
		with open(os.path.join(FLAGS.fout, 'results_error.txt'), 'a') as file:
			file.write('bias galaxy bin %i = %.2f\n' % (i + 1, rel_err_bias_gal[i]))

	# Write and print relative errors for GW bias
	for i in range(n_bins_dl):
		print('bias GW bin %i = ' % (i + 1), rel_err_bias_GW[i])
		with open(os.path.join(FLAGS.fout, 'results_error.txt'), 'a') as file:
			file.write('bias GW bin %i = %.2f\n' % (i + 1, rel_err_bias_GW[i]))

	##############################         ##############################
	"""
				PLOTTING 2D GAUSSIAN CONTOUR FOR H0 AND OMEGA_M
	"""
	##############################         ##############################

	# Define the mean and covariance for 2D Gaussian on (H0, Omega_m)
	mean = np.array([H0_true, Omega_m_true])
	cov_matrix = fisher_marg

	# Generate a grid around the mean for contour evaluation
	scale = 0.05
	x, y = np.meshgrid(
		np.linspace(H0_true - scale * H0_true, H0_true + scale * H0_true, 200),
		np.linspace(Omega_m_true - scale * Omega_m_true, Omega_m_true + scale * Omega_m_true, 200)
	)
	pos = np.dstack((x, y))

	# Evaluate multivariate Gaussian PDF on the grid
	pdf = multivariate_normal(mean, cov_matrix).pdf(pos)
	pdf /= np.max(pdf)

	# Plot 68% confidence contour
	confidence_level = 0.68
	contour = plt.contour(x, y, pdf, levels=[confidence_level], colors='blue')
	plt.clabel(contour, fontsize=10, fmt='%0.2f')
	plt.contourf(x, y, pdf, levels=[confidence_level, 1000], cmap='Blues', alpha=0.3)

	# Compute and show percentage errors in legend
	perc_err_H0 = 2 * sigma_H0 / H0_true * 100
	perc_err_Om = 2 * sigma_omega / Omega_m_true * 100
	plt.scatter(H0_true, Omega_m_true, c='blue', s=15,
				label='$\\sigma_{H_0}/H_0=%.1f\\%%$\n$\\sigma_{\\Omega_m}/\\Omega_m=%.1f\\%%$' % (perc_err_H0, perc_err_Om))

	# Finalize and save plot
	plt.grid(True, linestyle='--', alpha=0.5)
	plt.legend(fontsize=15)
	plt.xlabel('$H_0$')
	plt.ylabel('$\\Omega_m$')
	plt.title('%s' % GW_det)
	plt.savefig(os.path.join(FLAGS.fout, 'contour_plot.pdf'), bbox_inches='tight')
	plt.close()

	# Save summary results to results.txt
	with open(os.path.join(FLAGS.fout, 'results.txt'), 'a') as file:
		file.write(
			'\ndetector: %s, year: %i, lensing:%s, bin strategy: %s, n_bins_z: %i, n_bins_dl: %i, z_min: %f, z_max: %f, err_gal: %f, l_max: %i, n_gal: %i, n_GW: %f, sigma_H0_perc: %.2f, sigma_omega_m_perc: %.2f\n' %
			(GW_det, yr, Lensing, bin_strategy, n_bins_z, n_bins_dl, zm_bin, zM_bin, sig_gal, l_max, n_gal_bins, n_GW_bins,
			 perc_err_H0, perc_err_Om))
