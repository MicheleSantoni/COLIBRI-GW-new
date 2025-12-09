#!/usr/bin/env python3
#-----------------------------------------------------------------------------------------

import os

import argparse
import shutil
import json
import sys
import importlib
from inspect import getmembers, ismodule, isfunction
from tabulate import tabulate
from scipy.interpolate import interp1d


import functions_cross_correlation as fcc
import functions_extra_main as fem
import colibri.cosmology as cc
import colibri.limber_GW as LLG

from astropy.cosmology import FlatLambdaCDM
import astropy.cosmology.units as cu
from astropy import units as u

from functools import partial
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd
from scipy.integrate import trapezoid, simpson
from scipy.stats import multivariate_normal
import scipy.linalg
from scipy.interpolate import RectBivariateSpline
import scipy.interpolate as si

import time

plt.rc('font',size=20,family='serif')

configspath = 'configs/'

#-----------------------------------------------------------------------------------------

"""
Reads user inputs at runtime:
    --config = name of the config file (without .py).
    --fout = output folder path (where to save results).
"""
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='', type=str, required=False) # path to config file, in.json format
parser.add_argument("--fout", default='', type=str, required=True) # path to output folder
FLAGS = parser.parse_args()
#-----------------------------------------------------------------------------------------


if __name__=='__main__':
	os.makedirs(FLAGS.fout, exist_ok=True)

	sys.path.append(configspath)

	# Dictionary to hold imported config modules
	configs = {}
	config_names = FLAGS.config.split(',')  # if FLAGS.config is a comma-separated string

	for cfg_name in config_names:
		# Copy the config file into the output directory
		src = os.path.join(configspath, f"{cfg_name}.py")
		dst = os.path.join(FLAGS.fout, f"{cfg_name}_original.py")
		shutil.copy(src, dst)

		# Dynamically import the config module
		configs[cfg_name] = importlib.import_module(cfg_name)

	# import colibri
	sys.path.insert(0, configs['config_template_detectors'].colibri_path)

	# Initialize cosmology for power spectrum calculation
	cosmo_params = configs['config_template_LCDM_alphas'].COSMO_PARAMS

	print("\n=== Config Items ===")
	for name, module in configs.items():
		config_items = {
			key: val for key, val in getmembers(module)
			if not key.startswith('__') and not ismodule(val) and not isfunction(val)
		}
		print(f"\n{name}:")
		print(tabulate(config_items.items(), headers=["Parameter", "Value"], tablefmt="github"))
		break


	print('\n')

	def truncate(val, length=20):
		s = str(val)
		return s if len(s) <= length else s[:length] + "..."

	rows = [(k, truncate(v)) for k, v in cosmo_params.items()]

	table = []
	for i in range(0, len(rows), 2):
		row = rows[i:i + 2]
		if len(row) == 1:
			row.append(("", ""))
		table.append(row[0] + row[1])

	print(tabulate(table, headers=["Param", "Value", "Param", "Value"], tablefmt="github"))
	#print('\n')

	#-----------------------------------------------------------------------------------------
    #					INITIAL SETTINGS FROM CONFIG(s)
	#-----------------------------------------------------------------------------------------
	# GW detector (ET_Delta_2CE, ET_2L_2CE, ET_Delta_1CE, ET_2L_1CE, ET_Delta, ET_2L, LVK)
	GW_det = configs['config_template_detectors'].GW_det

	# Years of observation
	yr = configs['config_template_detectors'].yr

	# Parameters for GW bias model
	A_GW = configs['config_template_detectors'].A_GW
	gamma = configs['config_template_detectors'].gamma

	# Define the number of bins
	n_bins_z = configs['config_template_detectors'].n_bins_z
	n_bins_dl = configs['config_template_detectors'].n_bins_dl

	# Define the galaxy bin range
	z_m_bin = configs['config_template_detectors'].z_m_bin
	z_M_bin = configs['config_template_detectors'].z_M_bin

	# Define the GW bin range in redshift (will be converted in dl using the fiducial model)
	z_m_bin_GW = configs['config_template_detectors'].z_m_bin_GW
	z_M_bin_GW = configs['config_template_detectors'].z_M_bin_GW

	# Set the binning strategy (right_cosmo, wrong_cosmo(H0=65, Om0=0.32), equal_pop, equal_space)
	bin_strategy = configs['config_template_detectors'].bin_strategy

	# Include the lensing
	Lensing = configs['config_template_detectors'].Lensing

	# Fraction of the sky covered from the survey
	f_sky = configs['config_template_detectors'].f_sky
	f_sky_GW = configs['config_template_detectors'].f_sky_GW

	# Errors on the galaxy distribution
	sig_gal = configs['config_template_detectors'].sig_gal

	# galaxy survey (euclid_photo, euclid_spectro, ska)
	gal_det = configs['config_template_detectors'].gal_det

	l_min = configs['config_template_detectors'].l_min

	# Compute power spectra (True)
	fourier = configs['config_template_detectors'].fourier

	# Define the redshift total range
	z_m = configs['config_template_detectors'].z_m
	z_M = configs['config_template_detectors'].z_M

	# Define the luminosity distance total range
	dlm = configs['config_template_detectors'].dlm
	dlM = configs['config_template_detectors'].dlM

	# "True" values of the cosmological parameters
	H0_true = cosmo_params['h'] * 100  # if you want H0 in km/s/Mpc
	Omega_m_true = cosmo_params['Omega_m']
	Omega_b_true = cosmo_params['Omega_b']
	A_s = cosmo_params['A_s']*10**(9)
	n_s = cosmo_params['n_s']

	parameters_smg = cosmo_params['parameters_smg']
	#print(type(parameters_smg), parameters_smg)  # This should show <class 'str'>
	parameters_smg = list(map(float, parameters_smg.split(','))) 	# Convert to list of floats
	alpha_B = parameters_smg[1]
	alpha_M = parameters_smg[2]
	#print(alpha_M,alpha_B)

	w_0 = -1.0
	w_a = 0.0

	# Parameters in the Fisher
	parameters = [
		{
			"name": "Hubble constant",
			"symbol": "H0",
			"true_value": H0_true,
			"step": 1e-2,
			"derivative_args": lambda x: Cl_func_wrapped(cosmo_params, x, Omega_m_true, Omega_b_true, A_s, n_s, alpha_M,
														 alpha_B, w_0, w_a),
			"key": "der_H0_cov_mat"
		},
		{
			"name": "Matter density parameter",
			"symbol": "Omega_m",
			"true_value": Omega_m_true,
			"step": 1e-4,
			"method": "central",
			"derivative_args": lambda x: Cl_func_wrapped(cosmo_params, H0_true, x, Omega_b_true, A_s, n_s, alpha_M,
														 alpha_B,
														 w_0, w_a),
			"key": "der_omega_cov_mat"
		},
		{
			"name": "Baryon density parameter",
			"symbol": "Omega_b",
			"true_value": Omega_b_true,
			"step": 1e-3,
			"method": "central",
			"derivative_args": lambda x: Cl_func_wrapped(cosmo_params, H0_true, Omega_m_true, x, A_s, n_s, alpha_M,
														 alpha_B,
														 w_0, w_a),
			"key": "der_omega_b_cov_mat"
		},
		{
			"name": "Amplitude of the primordial power spectrum",
			"symbol": "A_s",
			"true_value": 2.100549,
			"step": 1e-3,
			"derivative_args": lambda x: Cl_func_wrapped(cosmo_params, H0_true, Omega_m_true, Omega_b_true, x, n_s,
														 alpha_M,
														 alpha_B, w_0, w_a),
			"key": "der_As_cov_mat"
		},
		{
			"name": "Spectral index of the primordial power spectrum",
			"symbol": "n_s",
			"true_value": n_s,
			"step": 1e-3,
			"derivative_args": lambda x: Cl_func_wrapped(cosmo_params, H0_true, Omega_m_true, Omega_b_true, A_s, x,
														 alpha_M,
														 alpha_B, w_0, w_a),
			"key": "der_ns_cov_mat"
		},
		{
			"name": "Alpha M",
			"symbol": "aM",
			"true_value": alpha_M,
			"step": 1e-2,
			"derivative_args": lambda x: Cl_func_wrapped(cosmo_params, H0_true, Omega_m_true, Omega_b_true, A_s, n_s, x,
														 alpha_B, w_0, w_a),
			"key": "der_aM_cov_mat"
		},
		{
			"name": "Alpha B",
			"symbol": "aB",
			"true_value": alpha_B,
			"step": 1e-3,
			"derivative_args": lambda x: Cl_func_wrapped(cosmo_params, H0_true, Omega_m_true, Omega_b_true, A_s, n_s,
														 alpha_M, x, w_0, w_a),
			"key": "der_aB_cov_mat"
		},
		{
			"name": "w_0",
			"symbol": "w_0",
			"true_value": w_0,
			"step": 1e-5,
			"derivative_args": lambda x: Cl_func_wrapped(cosmo_params, H0_true, Omega_m_true, Omega_b_true, A_s, n_s,
														 alpha_M, alpha_B, x, w_a),
			"key": "der_w_0_cov_mat"
		},
		{
			"name": "w_a",
			"symbol": "w_a",
			"true_value": w_a,
			"step": 1e-5,
			"derivative_args": lambda x: Cl_func_wrapped(cosmo_params, H0_true, Omega_m_true, Omega_b_true, A_s, n_s,
														 alpha_M, alpha_B, w_0, x),
			"key": "der_w_a_cov_mat"
		}
	]

	# Number of parameters Fisher
	n_cosmo_param = len(parameters)
	n_param = n_cosmo_param + n_bins_z + n_bins_dl

	#-----------------------------------------------------------------------------------------
    #                   LOADING GW AND GALAXY PARAMETERS
	#-----------------------------------------------------------------------------------------
	# Load gravitational wave detector parameters
	gw_params = fem.load_detector_params(GW_det, yr)

	# Load galaxy detector parameters
	gal_params = fem.load_galaxy_detector_params(gal_det)

	# Call of the single parameters: first GW and second Galaxies
	A = gw_params['A']
	Alpha = gw_params['Alpha']
	log_loc = gw_params['log_loc']
	log_delta_dl = gw_params['log_delta_dl']
	log_dl = gw_params['log_dl']
	Z_0=gw_params['Z_0']
	Beta=gw_params['Beta']
	s_a, s_b, s_c, s_d = [gw_params[k] for k in ['s_a', 's_b', 's_c', 's_d']]
	be_a, be_b, be_c, be_d = [gw_params[k] for k in ['be_a', 'be_b', 'be_c', 'be_d']]

	spline = gal_params['spline']
	bg0 = gal_params['bg0']
	bg1 = gal_params['bg1']
	bg2 = gal_params['bg2']
	bg3 = gal_params['bg3']
	sg0 = gal_params['sg0']
	sg1= gal_params['sg1']
	sg2= gal_params['sg2']
	sg3= gal_params['sg3']
	sig_gal = gal_params.get('sig_gal', None)  # may not exist for all detectors




	#-----------------------------------------------------------------------------------------
	# 	DEFINE FIDUCIAL COSMOLOGICAL MODEL AND COMPUTE CORRESPONDING LUMINOSITY DISTANCES
	#-----------------------------------------------------------------------------------------
	# Luminosity distance interval, equal to the redshift one assuming fiducial cosmology
	Hi_Cosmo = cc.cosmo(**cosmo_params)

	def dL_from_C(Hi_Cosmo, z):
		"""
		Luminosity distance d_L(z) in Mpc from your colibri cosmology `Hi_Cosmo`.
		Assumes Hi_Cosmo.comoving_distance(z) returns comoving distance in Mpc/h.
		"""
		z = np.asarray(z, dtype=float)
		chi_Mpc = np.asarray(Hi_Cosmo.comoving_distance(z)) / Hi_Cosmo.h  # -> Mpc
		return (1.0 + z) * chi_Mpc

	dlm_bin = dL_from_C(Hi_Cosmo, z_m_bin_GW)  # min d_L from C
	dlM_bin = dL_from_C(Hi_Cosmo, z_M_bin_GW)  # max d_L from C

	z_gal = np.linspace(z_m, z_M, 1200)  # Redshift grid for galaxy distribution
	dl_GW = np.linspace(dlm, dlM, 1200)  # Luminosity distance grid for gravitational wave sources

	#-----------------------------------------------------------------------------------------
	#							BIN STRATEGY
	#-----------------------------------------------------------------------------------------
	def z_from_dL(Hi_Cosmo, dL_Mpc, z_max=10.0, ngrid=20001):
		"""
		Invert d_L(z) -> z using a precomputed grid and linear interpolation.
		dL_Mpc: array-like in Mpc.
		Returns z with same shape as dL_Mpc.
		"""
		# Build a monotonic (z, dL) grid
		z_grid = np.linspace(0.0, float(z_max), int(ngrid))
		dL_grid = dL_from_C(Hi_Cosmo, z_grid)  # [Mpc]

		# Ensure strict monotonicity for interp1d by uniquifying dL_grid
		# (d_L is monotonic in standard cosmologies; numerical noise can cause ties)
		order = np.argsort(dL_grid)
		dL_sorted = dL_grid[order]
		z_sorted = z_grid[order]
		# Drop any duplicates in dL_sorted
		mask = np.concatenate(([True], np.diff(dL_sorted) > 0))
		dL_unique = dL_sorted[mask]
		z_unique = z_sorted[mask]

		inv = interp1d(dL_unique, z_unique, kind='linear', bounds_error=False, fill_value='extrapolate',assume_sorted=True)

		return inv(np.asarray(dL_Mpc, dtype=float))


	bin_int = np.linspace(z_m_bin, z_M_bin, n_bins_z * 1000)  # fine z-grid for galaxies

	# --- call with GW limits too ---
	bin_edges, bin_edges_dl = fem.compute_bin_edges_new(
		bin_strategy, n_bins_dl, n_bins_z,
		bin_int, z_M_bin, dlM_bin, z_m_bin,
		Hi_Cosmo, A, Z_0, Alpha, Beta, spline
	)

	# convert luminosity-distance bin edges (Gpc) to redshift using C
	dL_edges_Mpc = 1000.0 * np.asarray(bin_edges_dl,dtype=float)  #  Gpc -> Mpc
	bin_z_fiducial = z_from_dL(Hi_Cosmo, dL_edges_Mpc)  # array of z edges

	# Compute redshift distribution and total number of galaxies
	nz_gal, gal_tot = fem.compute_nz_gal_and_total(gal_det, z_gal, bin_edges, sig_gal, gal_params['spline'])

	gal_tot[gal_tot < 0] = 0  # Remove negative values (if any)
	n_tot_gal = trapezoid(gal_tot, z_gal)  # Integrate total galaxy distribution

	# Compute fraction of galaxies in each redshift bin
	bin_frac_gal = np.zeros(shape=(n_bins_z))
	for i in range(n_bins_z):
		bin_frac_gal[i] = simpson(nz_gal[i], z_gal)

	shot_noise_gal= 1/bin_frac_gal
	n_gal_bins = np.sum(bin_frac_gal)  # Sum of galaxy fractions across bins

	# Save bin edges for later use
	np.save(os.path.join(FLAGS.fout, 'bin_edges_GW_fiducial.npy'), bin_z_fiducial)
	np.save(os.path.join(FLAGS.fout, 'bin_edges_GW.npy'), bin_edges_dl)
	np.save(os.path.join(FLAGS.fout, 'bin_edges_gal.npy'), bin_edges)
	np.save(os.path.join(FLAGS.fout,'nz_gal.npy'),nz_gal)

	#-----------------------------------------------------------------------------------------
	#                PLOTTING THE GALAXY BIN DISTRIBUTION
	#-----------------------------------------------------------------------------------------
	fem.plot_galaxy_bin_distributions(z_gal, nz_gal, gal_tot, bin_edges, n_bins_z, z_m_bin, z_M_bin, FLAGS.fout)

	# Print statistics about galaxy bins
	print('\nthe total number of galaxies across all redshift: ', n_tot_gal * 4 * np.pi * f_sky)
	print('\nthe total number of galaxies in our bins: ', n_gal_bins * 4 * np.pi * f_sky)
	print('\nmean number of galaxies in each bin: ', np.mean(bin_frac_gal))
	print('\nmean shot noise in each bin: ', np.mean(shot_noise_gal))

	with open(os.path.join(FLAGS.fout, "galaxy_bin_distributions.txt"), "w") as f:
		f.write("Diagnostics for this run\n\n")
		f.write("z_gal ="+str(z_gal.tolist())+"\n")
		f.write("nz_gal ="+str(nz_gal.tolist())+"\n")
		f.write("gal_tot = " + str(gal_tot.tolist()) + "\n")
		f.write("bin_frac_gal = " + str(bin_frac_gal.tolist()) + "\n")
		f.write("shot_noise_gal = " + str(shot_noise_gal.tolist()) + "\n")
		f.write("n_bins_z         = " + str(n_bins_z) + "\n")
		f.write("z_m_bin            = " + str(z_m_bin) + "\n")
		f.write("z_M_bin     = " + str(z_M_bin) + "\n")
		f.write("the total number of galaxies across all redshift  = " + str(n_tot_gal * 4 * np.pi * f_sky) + "\n")
		f.write("the total number of galaxies in our bins     = " + str(n_gal_bins * 4 * np.pi * f_sky) + "\n")
		f.write("mean number of galaxies in each bin     = " + str(np.mean(bin_frac_gal)) + "\n")
		f.write("mean shot noise in each bin    = " + str(np.mean(shot_noise_gal)) + "\n")

	print("\nDiagnostics saved!")

	#-----------------------------------------------------------------------------------------
	# 		DETERMINE REPRESENTATIVE REDSHIFTS AND COMPUTE NONLINEAR POWER SPECTRUM
	#-----------------------------------------------------------------------------------------
	# Initialize array to store the peak redshift of each galaxy bin
	redshift = np.zeros(shape=n_bins_z)
	for i in range(n_bins_z):
		a = np.argmax(nz_gal[i])  # Index of maximum value in the redshift distribution
		redshift[i] = z_gal[a]  # Assign corresponding redshift

	# Define k and z arrays for evaluating the nonlinear power spectrum
	kk_nl = np.geomspace(1e-4, 1e2, 200)  # Logarithmically spaced k values [h/Mpc]
	zz_nl = np.linspace(z_m_bin_GW, z_M_bin_GW, 100)  # Linearly spaced redshift values

	# Compute nonlinear matter power spectrum using HI_CLASS
	_,kk_nl,zz_nl,P_vals = Hi_Cosmo.hi_class_pk(cosmo_params, kk_nl, zz_nl, True) # in (Mpc/h)^3

	# Interpolate power spectrum over redshift and k
	P_interp = RectBivariateSpline(kk_nl, zz_nl, P_vals)

	# Use peak redshifts of bins as centers for computing k_max
	z_centers_use = redshift

	# Compute maximum usable wavenumber at each redshift bin center
	k_max = fem.compute_k_max(z_centers_use, P_interp, kk_nl)

	z_centers_use = np.asarray(z_centers_use, float).ravel()
	k_max = np.asarray(k_max, float).ravel()

	#-----------------------------------------------------------------------------------------
	#			COMPUTING MULTIPOLE LIMITS AND GW BIN DISTRIBUTION STATISTICS
	#-----------------------------------------------------------------------------------------
	# Compute maximum multipole l for each bin using comoving distance and k_max
	chi_C_hMpc = np.asarray(Hi_Cosmo.comoving_distance(z_centers_use), float).ravel()
	ell_C = chi_C_hMpc * k_max
	l_max_nl = np.rint(ell_C).astype(int)

	#print('\nl_max_nl=',l_max_nl.tolist())

	# sanitize edges: 1-D, finite, strictly increasing, unique, and covering data range
	edges = np.asarray(bin_edges_dl, float).ravel()
	edges = edges[np.isfinite(edges)]
	edges = np.unique(edges)  # sorted + deduplicated

	# ensure edges cover the data range (recommended)
	dl_vals = np.power(10.0, np.asarray(log_dl, float).ravel())  # Gpc if log_dl=log10(Gpc)
	emin, emax = edges[0], edges[-1]
	dmin, dmax = dl_vals.min(), dl_vals.max()

	eps = 1e-12
	if dmin < emin:
		edges = np.insert(edges, 0, dmin * (1 - eps))
	if dmax > emax:
		edges = np.append(edges, dmax * (1 + eps))

	assert np.all(np.diff(edges) > 0), "bin edges must be strictly increasing"

	# Compute localization error parameters for GW bins
	# now call your function with the sanitized edges
	bin_edges_dl=edges
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
		np.arange(100, l_max + 1, step=25)])))
	ll[-1] = l_max  # Ensure maximum l is included
	ll_total = np.arange(l_min, l_max + 1)

	# Compute normalization factor for Cl's
	c = ll * (ll + 1.) / (2. * np.pi)

	# Save computed arrays
	np.save(os.path.join(FLAGS.fout, 'ell_max.npy'), l_max_bin)
	np.save(os.path.join(FLAGS.fout, 'loc_nl.npy'), loc_or_nl)

	#-----------------------------------------------------------------------------------------
	#            COMPUTING AND PLOTTING GW BIN DISTRIBUTION STATISTICS
	#-----------------------------------------------------------------------------------------
	# Compute the merger rate distribution and related quantities from luminosity distance bins
	z_GW, bin_convert, ndl_GW, n_GW, merger_rate_tot = fcc.merger_rate_dl_new(
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
				C=Hi_Cosmo,
				normalize=False
			)

	np.save(os.path.join(FLAGS.fout, 'z_GW'), z_GW)
	np.save(os.path.join(FLAGS.fout, 'ndl_GW'), ndl_GW)
	np.save(os.path.join(FLAGS.fout, 'n_GW'), n_GW)

	# Integrate the total merger rate over the full luminosity distance range (in Gpc)
	n_tot_GW = trapezoid(merger_rate_tot, dl_GW / 1000) * 4 * np.pi
	print('\nthe total number of GW across all distance: ', n_tot_GW)

	# Calculate the fraction of GW sources in each luminosity distance bin
	bin_frac_GW = np.zeros(shape=n_bins_dl)
	for i in range(n_bins_dl):
		bin_frac_GW[i] = trapezoid(ndl_GW[i], dl_GW / 1000)

	# Sum all bin fractions to get the total number in bins (should match total GW if complete)
	n_GW_bins = np.sum(bin_frac_GW)
	print('\nthe total number of GW in our bins: ', n_GW_bins * 4 * np.pi)

	fem.plot_gw_bin_distributions(
		dl_GW=dl_GW,
		ndl_GW=ndl_GW,
		merger_rate_tot=merger_rate_tot,
		bin_edges_dl=bin_edges_dl,
		n_bins_dl=n_bins_dl,
		output_path=FLAGS.fout
	)

	# Print per-bin and mean statistics for GW shot noise
	print('\nfraction of GW per sterad in each bin', bin_frac_GW)
	shot_noise_GW = 1 / bin_frac_GW
	print('\nshot noise per bin', shot_noise_GW)
	print('\nmean number of GW in each bin: ', np.mean(bin_frac_GW))
	print('\nmean shot noise in each bin: ', np.mean(shot_noise_GW))

	#-----------------------------------------------------------------------------------------
    #        FIGURES FOR COMPARING GALAXY AND GW DISTRIBUTIONS
	#-----------------------------------------------------------------------------------------
	fem.plot_distribution_comparison(
		z_gal, nz_gal, gal_tot, bin_edges, n_bins_z, z_m_bin, z_M_bin,
		z_GW, ndl_GW, merger_rate_tot, bin_convert, n_bins_dl, FLAGS.fout)

	#-----------------------------------------------------------------------------------------
	#       DEFINITION OF Cl_func DEPENDING ON THE PRESENCE OF THE LENSING
	#-----------------------------------------------------------------------------------------
	# If lensing is included in the analysis
	if Lensing:
		# Define function to compute Cl including lensing, clustering, and RSD contributions
		def Cl_func(Hi_Cosmo,params,gw_params,dl_GW,bin_edges_dl,z_gal,ll,b_gal, b_GW,save,n_points=13, n_points_x=20, grid_x='lin', z_min=1e-05,n_low=5, n_high=5):

			# Define cosmology and initialize Limber integrator
			#print('params contains:',params)
			S = LLG.limber(cosmology=Hi_Cosmo, z_limits=[z_m, z_M])

			# Define k and z grids for power spectrum
			kk = np.geomspace(1e-4, 1e2, 301)
			zz = np.linspace(0, z_M, 101)

			# Compute nonlinear matter power spectrum with HI_CLASS
			bg,_,_, pkz = Hi_Cosmo.hi_class_pk(params, kk, zz, True)

			#np.save(os.path.join(FLAGS.fout, 'background'),bg )

			S.load_power_spectra(k=kk,z=zz, power_spectra=pkz)

			# Generate GW distribution from fiducial parameters
			A = gw_params['A']
			Alpha = gw_params['Alpha']
			log_delta_dl = gw_params['log_delta_dl']
			log_dl = gw_params['log_dl']
			Z_0 = gw_params['Z_0']
			Beta = gw_params['Beta']

			h= params['h']
			H_0 = params['h'] * 100
			Omega_m = params['Omega_m']
			Omega_b = params['Omega_b']

			z_GW, bin_GW_converted, ndl_GW, n_GW, total = fcc.merger_rate_dl_new(
				dl=dl_GW,
				bin_dl=bin_edges_dl,
				log_dl=log_dl,
				log_delta_dl=log_delta_dl,
				H0=H_0,
				omega_m=Omega_m,
				omega_b=Omega_b,
				A=A,
				Z_0=Z_0,
				Alpha=Alpha,
				Beta=Beta,
				C=Hi_Cosmo,
				normalize=False
			)
			# Load bin edges for all observables
			S.load_bin_edges(bin_edges, bin_z_fiducial, name_1='lensing_gal', name_2='lensing_GW')
			S.load_bin_edges(bin_edges, bin_z_fiducial, name_1='galaxy', name_2='GW')
			S.load_bin_edges(bin_edges, bin_z_fiducial, name_1='rsd', name_2='lsd')

			# Compute galaxy magnification slope parameter beta
			s_a, s_b, s_c, s_d = [gw_params[k] for k in ['s_a', 's_b', 's_c', 's_d']]
			be_a, be_b, be_c, be_d = [gw_params[k] for k in ['be_a', 'be_b', 'be_c', 'be_d']]

			print('QUI BETA HA FlatLambdaCDM')
			beta = fem.compute_beta(H_0, Omega_m, Omega_b,z_gal, s_a, s_b, s_c, s_d, be_a, be_b, be_c, be_d)

			print('\nLoading the window functions...\n')
			# Load window functions for each observable
			S.load_galaxy_clustering_window_functions(bg,z=z_gal, n_z=nz_gal, ll=ll, bias=b_gal, name='galaxy') ###
			S.load_gravitational_wave_window_functions(bg,z=z_GW, n_dl=ndl_GW,ll=ll, bias=b_GW, name='GW') ###

			S.load_rsd_window_functions(bg, z=z_gal, n_z=nz_gal, ll=ll,name='rsd') ###
			S.load_lsd_window_functions(bg,z=z_GW, n_dl=ndl_GW, ll=ll,name='lsd') ###

			S.load_galaxy_lensing_window_functions(z=z_gal, n_z=nz_gal, H_0=H_0, omega_m=Omega_m, ll=ll,name='lensing_gal') ###
			S.load_gw_lensing_window_functions(bg,z=z_GW, n_dl=ndl_GW, H_0=H_0, omega_m=Omega_m, ll=ll,name='lensing_GW') ###

			print('\nComputing the angular power spectra...\n')
			# Compute all angular power spectra using Limber integrals
			start = time.time()
			Cl = S.limber_angular_power_spectra(bg,h=h,l=ll, windows=['galaxy', 'GW', 'rsd', 'lsd'])
			end = time.time()
			print(f"Time took {end - start:.4f} seconds \n")

			print('\nComputing the angular power spectra cross-correlation...\n')

			def rel_diff_Cl(Cl_a, Cl_b, eps=1e-12):
				"""
                Returns a dict of relative differences with same keys/shapes as inputs.
                For each key: (A - B) / denom, where denom is A unless |A|<eps, then B (and 1.0 if both ~0).
                """
				out = {}
				common = set(Cl_a.keys()) & set(Cl_b.keys())
				if not common:
					raise ValueError(f"No common keys: A={list(Cl_a.keys())}, B={list(Cl_b.keys())}")
				for k in sorted(common):
					A = np.asarray(Cl_a[k], dtype=float)
					B = np.asarray(Cl_b[k], dtype=float)
					if A.shape != B.shape:
						raise ValueError(f"Shape mismatch for {k}: {A.shape} vs {B.shape}")
					denom = np.where(np.abs(A) > eps, A,
									 np.where(np.abs(B) > eps, B, 1.0))
					out[k] = (A - B) / denom
				return out

			def summarize_rel_diff(diff_dict, ref_a, ref_b, tol=0.0, autocorr_only=False, only_nonzero=True):
				"""
                Prints per-bin summaries. If autocorr_only=True, only i==j bins are checked.
                If only_nonzero=True, prints bins where max|rel diff| > tol.
                """
				for k, D in diff_dict.items():
					A = np.asarray(ref_a[k], dtype=float)
					B = np.asarray(ref_b[k], dtype=float)
					I, J, L = D.shape
					printed_header = False
					for i in range(I):
						for j in range(J):
							if autocorr_only and i != j:
								continue
							rd = D[i, j, :]
							max_rel = np.nanmax(np.abs(rd))
							if only_nonzero and not (max_rel > tol):
								continue
							if not printed_header:
								print(f"\n[{k}]  bins={I}x{J}, L={L}")
								printed_header = True
							a = A[i, j, :]
							b = B[i, j, :]
							d = a - b
							max_abs = np.nanmax(np.abs(d))
							rms_rel = np.sqrt(np.nanmean(rd ** 2))
							# cosine similarity (guard against zero vectors)
							na = np.sqrt(np.dot(a, a))
							nb = np.sqrt(np.dot(b, b))
							cos = (np.dot(a, b) / (na * nb)) if (na > 0 and nb > 0) else np.nan
							print(
								f"  bin({i},{j}): max|Δ|={max_abs:.3e}  max|rel|={max_rel:.3e}  RMS_rel={rms_rel:.3e}  cos={cos:.6f}")


			print('\n prima')
			Cl_lens_cross_prima = S.limber_angular_power_spectra_lensing_cross_prima(l=ll, s_gal=s_gal, beta=beta,
																					 H_0=H_0,
																					 omega_m=Omega_m, omega_b=Omega_b,
																					 windows=None, n_points=n_points,
																					 n_points_x=n_points_x,
																					 z_min=z_min, grid_x=grid_x,
																					 n_low=n_low,
																					 n_high=n_high)
			print('\n dopo')
			start = time.time()
			Cl_lens_cross = S.limber_angular_power_spectra_lensing_cross(bg,
																		 l=ll, s_gal=s_gal, beta=beta,
																		 windows=None, n_points=n_points,
																		 n_points_x=n_points_x,
																		 z_min=z_min, grid_x=grid_x, n_low=n_low,
																		 n_high=n_high)
			end = time.time()
			print(f"Time took {end - start:.4f} seconds \n")

			# 1) Full comparison (all bins)
			diff_all = rel_diff_Cl(Cl_lens_cross, Cl_lens_cross_prima)
			summarize_rel_diff(diff_all, Cl_lens_cross, Cl_lens_cross_prima, tol=0.0, autocorr_only=False,only_nonzero=True)

			# 2) Only auto-correlations (i == j)
			diff_auto = rel_diff_Cl(Cl_lens_cross, Cl_lens_cross_prima)
			summarize_rel_diff(diff_auto, Cl_lens_cross, Cl_lens_cross_prima, tol=0.0, autocorr_only=True,	only_nonzero=True)

			def capture_summary(diff_dict, ref_a, ref_b, tol=0.0, autocorr_only=False, only_nonzero=True):
				lines = []
				for k, D in diff_dict.items():
					A = np.asarray(ref_a[k], dtype=float)
					B = np.asarray(ref_b[k], dtype=float)
					I, J, L = D.shape
					header_written = False
					for i in range(I):
						for j in range(J):
							if autocorr_only and i != j:
								continue
							rd = D[i, j, :]
							max_rel = np.nanmax(np.abs(rd))
							if only_nonzero and not (max_rel > tol):
								continue
							if not header_written:
								lines.append(f"\n[{k}]  bins={I}x{J}, L={L}")
								header_written = True
							a = A[i, j, :];
							b = B[i, j, :];
							d = a - b
							max_abs = np.nanmax(np.abs(d))
							rms_rel = np.sqrt(np.nanmean(rd ** 2))
							na = np.sqrt(np.dot(a, a));
							nb = np.sqrt(np.dot(b, b))
							cos = (np.dot(a, b) / (na * nb)) if (na > 0 and nb > 0) else np.nan
							lines.append(f"  bin({i},{j}): max|Δ|={max_abs:.3e}  max|rel|={max_rel:.3e}  "
										 f"RMS_rel={rms_rel:.3e}  cos={cos:.6f}")
				return "\n".join(lines)

			with open(os.path.join(FLAGS.fout, "summary_diff_all.txt"), "w") as f:
				f.write(capture_summary(diff_all, Cl_lens_cross, Cl_lens_cross_prima, tol=0.0, autocorr_only=False,
										only_nonzero=True))

			with open(os.path.join(FLAGS.fout, "summary_diff_auto.txt"), "w") as f:
				f.write(capture_summary(diff_auto, Cl_lens_cross, Cl_lens_cross_prima, tol=0.0, autocorr_only=True,
										only_nonzero=True))

			print('\nComputing the angular power spectra autocorrelation...\n')
			# start = time.time()

			Cl_lens = S.limber_angular_power_spectra_lensing_auto(bg,
																  l=ll, s_gal=s_gal, beta=beta,
																  windows=['lensing_gal', 'lensing_GW'],
																  n_points=n_points, n_points_x=n_points_x,
																  z_min=z_min, grid_x=grid_x, n_low=n_low,
																  n_high=n_high)



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

			if save:
				print('\nSaving all the Cl results...\n')
				# Galaxy-Galaxy
				np.save(os.path.join(FLAGS.fout, 'Cl_delta_GG'), Cl_delta_GG)
				np.save(os.path.join(FLAGS.fout, 'Cl_len_GG'), Cl_len_GG)
				np.save(os.path.join(FLAGS.fout, 'Cl_RSD_GG'), Cl_RSD_GG)
				np.save(os.path.join(FLAGS.fout, 'Cl_delta_len_GG'), Cl_delta_len_GG)
				np.save(os.path.join(FLAGS.fout, 'Cl_delta_RSD_GG'), Cl_delta_RSD_GG)
				np.save(os.path.join(FLAGS.fout, 'Cl_RSD_len_GG'), Cl_RSD_len_GG)

				# GW-GW
				np.save(os.path.join(FLAGS.fout, 'Cl_delta_GWGW'), Cl_delta_GWGW)
				np.save(os.path.join(FLAGS.fout, 'Cl_len_GWGW'), Cl_len_GWGW)
				np.save(os.path.join(FLAGS.fout, 'Cl_RSD_GWGW'), Cl_RSD_GWGW)
				np.save(os.path.join(FLAGS.fout, 'Cl_delta_len_GWGW'), Cl_delta_len_GWGW)
				np.save(os.path.join(FLAGS.fout, 'Cl_delta_RSD_GWGW'), Cl_delta_RSD_GWGW)
				np.save(os.path.join(FLAGS.fout, 'Cl_RSD_len_GWGW'), Cl_RSD_len_GWGW)

				# Galaxy-GW
				np.save(os.path.join(FLAGS.fout, 'Cl_delta_GGW'), Cl_delta_GGW)
				np.save(os.path.join(FLAGS.fout, 'Cl_len_GGW'), Cl_len_GGW)
				np.save(os.path.join(FLAGS.fout, 'Cl_RSD_GGW'), Cl_RSD_GGW)
				np.save(os.path.join(FLAGS.fout, 'Cl_delta_len_GGW'), Cl_delta_len_GGW)
				np.save(os.path.join(FLAGS.fout, 'Cl_delta_RSD_GGW'), Cl_delta_RSD_GGW)
				np.save(os.path.join(FLAGS.fout, 'Cl_RSD_len_GGW'), Cl_RSD_len_GGW)

			return Cl_GG, Cl_GWGW, Cl_GGW

	# If lensing is not included, compute only density clustering spectra
	else:

		# Define function to compute Cl from galaxy and GW clustering only
		def Cl_func(Hi_Cosmo,params,gw_params,dl_GW,bin_edges_dl,z_gal,ll,b_gal, b_GW,save,n_points=13, n_points_x=20, grid_x='lin', z_min=1e-05,n_low=5, n_high=5):

			S = LLG.limber(cosmology=Hi_Cosmo, z_limits=[z_m, z_M])

			# Define power spectrum grids
			kk = np.geomspace(1e-4, 1e2, 500)
			zz = np.linspace(0, z_M, 100)

			# Compute nonlinear matter power spectrum
			bg,_,_, pkz = Hi_Cosmo.hi_class_pk(params, kk, zz, True) # pkz (Mpc/h)^3

			S.load_power_spectra(z=zz, k=kk, power_spectra=pkz)

			# Generate GW distribution from fiducial parameters
			A = gw_params['A']
			Alpha = gw_params['Alpha']
			log_delta_dl = gw_params['log_delta_dl']
			log_dl = gw_params['log_dl']
			Z_0 = gw_params['Z_0']
			Beta = gw_params['Beta']

			h= params['h']
			H_0 = params['h'] * 100
			Omega_m = params['Omega_m']
			Omega_b = params['Omega_b']

			# Generate GW source distribution
			z_GW, bin_GW_converted, ndl_GW, n_GW, total = fcc.merger_rate_dl_new(
				dl=dl_GW,
				bin_dl=bin_edges_dl,
				log_dl=log_dl,
				log_delta_dl=log_delta_dl,
				H0=H_0,
				omega_m=Omega_m,
				omega_b=Omega_b,
				A=A,
				Z_0=Z_0,
				Alpha=Alpha,
				Beta=Beta,
				C=Hi_Cosmo,
				normalize=False
			)
			print('\nLoading the window functions...\n')

			# Load binning and window functions
			S.load_bin_edges(bin_edges, bin_GW_converted)
			S.load_galaxy_clustering_window_functions(bg,z=z_gal, n_z=nz_gal, ll=ll, bias=b_gal, name='galaxy')
			S.load_gravitational_wave_window_functions(bg,z=z_GW, n_dl=ndl_GW, ll=ll, bias=b_GW, name='GW')

			print('\nComputing the angular power spectra...\n')
			# Compute angular power spectra (density terms only)
			Cl = S.limber_angular_power_spectra(bg,h=h, l=ll, windows=None)

			# Galaxy-GW
			Cl_delta_GGW = Cl['galaxy-GW']

			# Galaxy-Galaxy
			Cl_delta_GG = Cl['galaxy-galaxy']

			# GW-GW
			Cl_delta_GWGW = Cl['GW-GW']

			if save:
				print('\nSaving all the Cl results...\n')
				np.save(os.path.join(FLAGS.fout, 'Cl_delta_GG'), Cl_delta_GG)
				np.save(os.path.join(FLAGS.fout, 'Cl_delta_GWGW'), Cl_delta_GWGW)
				np.save(os.path.join(FLAGS.fout, 'Cl_delta_GGW'), Cl_delta_GGW)

			return Cl_delta_GG, Cl_delta_GWGW, Cl_delta_GGW

	#-----------------------------------------------------------------------------------------
    #					COMPUTING FIDUCIAL BIASES
	#-----------------------------------------------------------------------------------------
	# Compute mean redshift for each GW bin and corresponding GW bias
	z_mean_GW = (bin_z_fiducial[:-1] + bin_z_fiducial[1:]) * 0.5
	bias_GW = A_GW * (1. + z_mean_GW) ** gamma

	# Compute mean redshift for each galaxy bin and galaxy bias using polynomial model
	z_mean_gal = (bin_edges[:-1] + bin_edges[1:]) * 0.5
	bias_gal = bg0 + bg1 * z_mean_gal + bg2 * z_mean_gal ** 2 + bg3 * z_mean_gal ** 3

	# Compute magnification slope s(z) depending on galaxy detector
	s_gal= fem.compute_s_gal(z_gal, gal_det, sg0, sg1, sg2, sg3)

	np.save(os.path.join(FLAGS.fout, 'bias_fiducial_gal'), bias_gal)
	np.save(os.path.join(FLAGS.fout, 'bias_fiducial_GW'), bias_GW)

	with open(os.path.join(FLAGS.fout, "fisher_info_diagnostics.txt"), "w") as f:
		f.write("Diagnostics for this run\n\n")
		f.write("z_gal ="+str(z_gal.tolist())+"\n")
		f.write("dl_GW ="+str(dl_GW.tolist())+"\n")
		f.write("z_centers_use = " + str(z_centers_use.tolist()) + "\n")
		f.write("k_max         = " + str(k_max.tolist()) + "\n")
		f.write("ll            = " + str(ll.tolist()) + "\n")
		f.write("l_max_bin     = " + str(l_max_bin.tolist()) + "\n")
		f.write("ll_total      = " + str(ll_total.tolist()) + "\n")
		f.write("l_max_nl      = " + str(l_max_nl.tolist()) + "\n")
		f.write("z_mean_GW     = " + str(z_mean_GW.tolist()) + "\n")
		f.write("z_mean_gal    = " + str(z_mean_gal.tolist()) + "\n")

	print("\nDiagnostics for FISHER saved!")

	#-----------------------------------------------------------------------------------------
	#				COMPUTING LOCALIZATION NOISE MATRICES
	#-----------------------------------------------------------------------------------------
	print('\nComputing localization noise matrices...\n')

	noise_gal = fcc.shot_noise_mat_auto(shot_noise_gal, ll_total)
	noise_GW = fcc.shot_noise_mat_auto(shot_noise_GW, ll_total)

	noise_loc = np.zeros(shape=(n_bins_dl, len(ll_total)))
	noise_loc_auto = np.zeros(shape=(n_bins_dl, len(ll_total)))

	for i in range(n_bins_dl):
		for l in range(len(ll_total)):
			if (ll_total[l] * (ll_total[l] + 1) * (sigma_sn_GW[i] / (2 * np.pi) ** (3 / 2))) < 30:
				noise_loc[i, l] = np.exp(-ll_total[l] * (ll_total[l] + 1) * (sigma_sn_GW[i] / (2 * np.pi) ** (3 / 2)))
				noise_loc_auto[i, l] = np.exp(
					-2 * ll_total[l] * (ll_total[l] + 1) * (sigma_sn_GW[i] / (2 * np.pi) ** (3 / 2)))
			else:
				noise_loc[i, l] = np.exp(-30)
				noise_loc_auto[i, l] = np.exp(-30)

	noise_loc_mat = np.zeros(shape=(n_bins_z, n_bins_dl, len(ll_total)))
	noise_loc_mat_auto = np.zeros(shape=(n_bins_dl, n_bins_dl, len(ll_total)))

	for i in range(n_bins_z):
		for ii in range(n_bins_dl):
			noise_loc_mat[i, ii, :] = noise_loc[ii, :]

	for i in range(n_bins_dl):
		for ii in range(i, n_bins_dl):
			noise_loc_mat_auto[i, ii, :] = noise_loc_auto[ii, :]

	for i in range(n_bins_dl):
		for ii in range(i + 1, n_bins_dl):
			noise_loc_mat_auto[ii, i] = noise_loc_mat_auto[i, ii]

	#-----------------------------------------------------------------------------------------
	#					COMPUTING THE POWER SPECTRUM
	#-----------------------------------------------------------------------------------------
	# Print status message for power spectrum computation
	print('\nComputing the Power Spectrum...\n')

	# Compute angular power spectra from Cl_func with fiducial cosmological and bias parameters
	Cl_GG, Cl_GWGW, Cl_GGW = Cl_func(Hi_Cosmo,cosmo_params,gw_params,dl_GW,bin_edges_dl,z_gal,ll,b_gal= bias_gal, b_GW = bias_GW, save=True)

	Cl_GG_total = np.zeros(shape=(n_bins_z, n_bins_z, len(ll_total)))
	Cl_GWGW_total = np.zeros(shape=(n_bins_dl, n_bins_dl, len(ll_total)))
	Cl_GGW_total = np.zeros(shape=(n_bins_z, n_bins_dl, len(ll_total)))

	for i in range(n_bins_z):
		for ii in range(n_bins_z):
			Cl_GG_interp = si.interp1d(ll, Cl_GG[i,ii])
			Cl_GG_total[i,ii] = Cl_GG_interp(ll_total)

	for i in range(n_bins_dl):
		for ii in range(n_bins_dl):
			Cl_GWGW_interp = si.interp1d(ll, Cl_GWGW[i,ii])
			Cl_GWGW_total[i,ii] = Cl_GWGW_interp(ll_total)

	for i in range(n_bins_z):
		for ii in range(n_bins_dl):
			Cl_GGW_interp = si.interp1d(ll, Cl_GGW[i,ii])
			Cl_GGW_total[i,ii] = Cl_GGW_interp(ll_total)

	np.save(os.path.join(FLAGS.fout, 'Cl_GG'), Cl_GG_total)
	np.save(os.path.join(FLAGS.fout, 'Cl_GWGW'), Cl_GWGW_total)
	np.save(os.path.join(FLAGS.fout, 'Cl_GGW'), Cl_GGW_total)
	np.save(os.path.join(FLAGS.fout, 'noise_GW'), noise_GW)
	np.save(os.path.join(FLAGS.fout, 'noise_gal'), noise_gal)
	np.save(os.path.join(FLAGS.fout, 'noise_loc_auto'), noise_loc_mat_auto)
	np.save(os.path.join(FLAGS.fout, 'noise_loc_cross'), noise_loc_mat)

	Cl_GWGW_total = Cl_GWGW_total * noise_loc_mat_auto
	Cl_GGW_total = Cl_GGW_total * noise_loc_mat

	Cl_GWGW_total += noise_GW
	Cl_GG_total += noise_gal

	#-----------------------------------------------------------------------------------------
    #					COMPUTING FIDUCIAL COVARIANCE MATRIX
	#-----------------------------------------------------------------------------------------
	print('\nComputing fiducial covariance matrix...\n')
	# Construct the full data vector from Cl auto- and cross-spectra
	vec = fcc.vector_cl(cl_cross=Cl_GGW_total, cl_auto1=Cl_GG_total, cl_auto2=Cl_GWGW_total)

	# Compute the covariance matrix of the fiducial spectra
	cov_mat = fcc.covariance_matrix(vec, n_bins_z, n_bins_dl)

	# Save the fiducial covariance matrix to file
	np.save(os.path.join(FLAGS.fout, 'cov_mat'), cov_mat)

	#-----------------------------------------------------------------------------------------
	#				COMPUTING PARAMETER DERIVATIVE MATRIx
	#-----------------------------------------------------------------------------------------
	print('\nComputing parameter derivative matrix...\n')
	# Initialize dictionary to store parameter-specific covariance dell_totalrivatives
	covariance_matrices = {}


	def Cl_func_wrapped(cosmo_params, H0, Omega_m, Omega_b, A_s, n_s, alpha_M, alpha_B, w_0, w_a):
		"""
        Unified wrapper for computing Cl given any set of cosmological parameters.
        Includes both ΛCDM and scalar modified gravity (SMG) parameters.
        """
		params = deepcopy(cosmo_params)

		# ΛCDM-like parameters
		params['h'] = H0 / 100.0
		params['Omega_m'] = Omega_m
		params['Omega_b'] = Omega_b
		params['A_s'] = A_s*10**(-9)
		params['n_s'] = n_s

		# Modified gravity (SMG) parameters
		# x_k, x_b, x_m, x_t, (M_*)^ 2_ini
		# '1.0, 0.0, 0.0, 0.0, 1.0' -> LCDM case
		params['parameters_smg'] = f"1.0,{alpha_B},{alpha_M},0.0,1.0"
		#print(params['parameters_smg'])
		#print(params)

		# Cosmology object from CLASS / hi_class
		Hi_Cosmo = cc.cosmo(**params)

		return Cl_func(Hi_Cosmo, params, gw_params, dl_GW, bin_edges_dl, z_gal, ll, bias_gal, bias_GW,save=False)


	# Compute and save the derivative covariance matrices for each parameter
	fem.compute_parameter_derivatives(parameters, FLAGS, n_bins_z, n_bins_dl, covariance_matrices)

	#-----------------------------------------------------------------------------------------
    #			COMPUTING DERIVATIVES WITH RESPECT TO BIASES
	#-----------------------------------------------------------------------------------------
	# Initialize array to store galaxy bias derivatives for each bin pair
	der_b_gal = np.zeros(shape=(n_bins_z, n_bins_dl + n_bins_z, n_bins_dl + n_bins_z, len(ll)))
	# Define step size for numerical differentiation
	step = configs['config_template_detectors'].step

	def compute_partial_derivatives_gal(b_gal,bias_GW, der_b_gal, step):
		"""
        Compute numerical derivatives of power spectra with respect to galaxy bias in each bin.

        Parameters:
        - b_gal: Array of galaxy bias values per bin
        - der_b_gal: Output array to store derivative covariance matrices

        Returns:
        - der_b_gal: Updated array with derivative covariance matrices per bias bin
        """

		for i in range(len(b_gal)):
			print('\nComputing the derivative with respect to the galaxy bias in bin %i...\n' % i)

			def func_GG(b):
				b_gal_temp = np.copy(b_gal)
				b_gal_temp[i] = b
				return Cl_func(Hi_Cosmo, cosmo_params, gw_params, dl_GW, bin_edges_dl, z_gal, ll, b_gal_temp,
							bias_GW,save=False)[0]

			def func_GWGW(b):
				b_gal_temp = np.copy(b_gal)
				b_gal_temp[i] = b
				return Cl_func(Hi_Cosmo, cosmo_params, gw_params, dl_GW, bin_edges_dl, z_gal, ll, b_gal_temp,
							bias_GW,save=False)[1]

			def func_GGW(b):
				b_gal_temp = np.copy(b_gal)
				b_gal_temp[i] = b
				return Cl_func(Hi_Cosmo, cosmo_params, gw_params, dl_GW, bin_edges_dl, z_gal, ll, b_gal_temp,
							bias_GW,save=False)[2]

			# Compute finite difference derivatives
			print('\n-------> Computing the derivative with respect to the galaxy bias in bin %i: G-G...' % i)
			der_b_gal_GG = nd.Derivative(func_GG, step=step)(b_gal[i])
			print('\n-------> Computing the derivative with respect to the galaxy bias in bin %i: GW-GW...' % i)
			der_b_gal_GWGW = nd.Derivative(func_GWGW, step=step)(b_gal[i])
			print('\n-------> Computing the derivative with respect to the galaxy bias in bin %i: G-GW...' % i)
			der_b_gal_GGW = nd.Derivative(func_GGW, step=step)(b_gal[i])

			# Construct and store covariance matrix
			der_b_gal_vec = fcc.vector_cl(cl_cross=der_b_gal_GGW, cl_auto1=der_b_gal_GG, cl_auto2=der_b_gal_GWGW)
			der_b_gal_cov_mat = fcc.covariance_matrix(der_b_gal_vec, n_bins_z, n_bins_dl)

			der_b_gal[i] = der_b_gal_cov_mat
			np.save(os.path.join(FLAGS.fout, 'der_b_gal_cov_mat_bin_%i' % i), der_b_gal_cov_mat)
			print("\nThe derivative with respect to the galaxy bias in bin %i has been computed.\n" % i)

		return der_b_gal

	# Compute covariance matrix derivatives with respect to galaxy bias parameters
	der_b_gal_cov_mat = compute_partial_derivatives_gal(bias_gal,bias_GW, der_b_gal,step)

	# Initialize array to store GW bias derivatives for each bin pair
	der_bGW = np.zeros(shape=(n_bins_dl, n_bins_dl + n_bins_z, n_bins_dl + n_bins_z, len(ll)))


	def compute_partial_derivatives_GW(b_GW,bias_gal, der_b_GW, step):
		"""
        Compute numerical derivatives of power spectra with respect to GW (gravitational wave) bias parameters in each bin.

        Parameters:
        - b_GW: Array of GW bias values per bin
        - der_b_GW: Output array to store derivative covariance matrices

        Returns:
        - der_b_GW: Updated array with derivative covariance matrices per bias bin
        """

		for i in range(len(b_GW)):
			print('\nComputing the derivative with respect to the GW bias in bin %i...\n' % i)

			# Define internal functions to return power spectra for modified bias
			def func_GG(b):
				b_GW_temp = np.copy(b_GW)
				b_GW_temp[i] = b
				return Cl_func(Hi_Cosmo, cosmo_params, gw_params, dl_GW, bin_edges_dl, z_gal, ll, bias_gal, b_GW_temp,save=False)[0]

			def func_GWGW(b):
				b_GW_temp = np.copy(b_GW)
				b_GW_temp[i] = b
				return Cl_func(Hi_Cosmo, cosmo_params, gw_params, dl_GW, bin_edges_dl, z_gal, ll, bias_gal, b_GW_temp,save=False)[1]

			def func_GGW(b):
				b_GW_temp = np.copy(b_GW)
				b_GW_temp[i] = b
				return Cl_func(Hi_Cosmo, cosmo_params, gw_params, dl_GW, bin_edges_dl, z_gal, ll, bias_gal, b_GW_temp,save=False)[2]

			# Compute numerical derivatives for each power spectrum
			print('\n-------> Computing the derivative with respect to the GW bias in bin %i: G-G...' % i)
			der_b_GW_GG = nd.Derivative(func_GG, step=step)(b_GW[i])
			print('\n-------> Computing the derivative with respect to the GW bias in bin %i: GW-GW...' % i)
			der_b_GW_GWGW = nd.Derivative(func_GWGW, step=step)(b_GW[i])
			print('\n-------> Computing the derivative with respect to the GW bias in bin %i: G-GW...' % i)
			der_b_GW_GGW = nd.Derivative(func_GGW, step=step)(b_GW[i])

			# Assemble derivative vector and covariance matrix
			der_b_GW_vec = fcc.vector_cl(cl_cross=der_b_GW_GGW, cl_auto1=der_b_GW_GG, cl_auto2=der_b_GW_GWGW)
			der_b_GW_cov_mat = fcc.covariance_matrix(der_b_GW_vec, n_bins_z, n_bins_dl)

			# Store result and save to file
			der_b_GW[i] = der_b_GW_cov_mat
			np.save(os.path.join(FLAGS.fout, 'der_b_GW_cov_mat_bin_%i.npy' % i), der_b_GW_cov_mat)
			print("\nThe derivative with respect to the GW bias in bin %i has been computed.\n" % i)

		return der_b_GW

	# Compute covariance matrix derivatives with respect to GW bias parameters
	der_bGW_cov_mat = compute_partial_derivatives_GW(bias_GW,bias_gal, der_bGW,step)


	# Notify that all bias-related derivatives have been computed
	print('\n ---------------- All derivative computed ---------------- \n')

	# Initialize full derivative array including all cosmological and bias parameters
	all_der = np.zeros((n_param, n_bins_z + n_bins_dl, n_bins_z + n_bins_dl, len(ll)))

	# Assign precomputed cosmological parameter derivatives
	all_der[0] = covariance_matrices['der_H0_cov_mat']
	all_der[1] = covariance_matrices['der_omega_cov_mat']
	all_der[2] = covariance_matrices['der_omega_b_cov_mat']
	all_der[3] = covariance_matrices['der_As_cov_mat']
	all_der[4] = covariance_matrices['der_ns_cov_mat']
	all_der[5] = covariance_matrices['der_aM_cov_mat']
	all_der[6] = covariance_matrices['der_aB_cov_mat']
	all_der[7] = covariance_matrices['der_w_0_cov_mat']
	all_der[8] = covariance_matrices['der_w_a_cov_mat']

	# Insert galaxy bias derivatives at appropriate indices
	for i in range(n_bins_z):
		index = n_param - n_bins_dl - n_bins_z + i
		all_der[index] = der_b_gal_cov_mat[i]

	# Insert GW bias derivatives at appropriate indices
	for i in range(n_bins_dl):
		index = n_param - n_bins_dl + i
		all_der[index] = der_bGW_cov_mat[i]

	# Print shape for verification
	#print(all_der.shape)

	#-----------------------------------------------------------------------------------------
    # INTERPOLATING DERIVATIVES TO FULL MULTIPOLE RANGE AND APPLYING LMIN & LMAX MASKING
	#-----------------------------------------------------------------------------------------
	all_der_total=fem.process_derivatives(
    all_der=all_der,
    ll=ll,
    ll_total=ll_total,
    z_mean_gal=z_mean_gal,
    z_mean_GW=z_mean_GW,
    l_max_nl=l_max_nl,
    l_max_bin=l_max_bin,
    FLAGS=FLAGS)

	#-----------------------------------------------------------------------------------------
    # 			COMPUTING FISHER MATRIX AND ROTATING TO NEW PARAMETER BASIS
	#-----------------------------------------------------------------------------------------
	print('\nComputing the Fisher Matrix...\n')
	# Compute Fisher information matrix using the full derivative set and covariance
	fisher = fcc.fisher_matrix(cov_mat, all_der_total, ll_total, f_sky)

	# Rotate the Fisher matrix to new basis
	fisher = fem.rotate_fisher_Ob_to_ob(fisher)

	# Save the rotated Fisher matrix
	np.save(os.path.join(FLAGS.fout, 'fisher_mat.npy'), fisher)

	#-----------------------------------------------------------------------------------------
    #			EXTRACTING 1σ ERRORS AND RELATIVE ERRORS FROM FISHER MATRIX
	#-----------------------------------------------------------------------------------------
	print('\nExtracting the errors...\n')

	fisher_inv = scipy.linalg.inv(fisher)
	fisher_marg = fisher_inv[:2, :2]

	# Compute 1σ uncertainties for cosmological parameters (sqrt of diagonal elements)
	sigma_H0,sigma_omega,sigma_omega_b,sigma_As,sigma_ns,sigma_alpha_M, sigma_alpha_B, sigma_w_0, sigma_w_a = fem.compute_and_print_sigma_params(fisher_inv)

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


	for i in range(n_bins_z):
		print('bias galaxy bin %i = ' % (i + 1), sigma_bias_gal[i])
	for i in range(n_bins_dl):
		print('bias GW bin %i = ' % (i + 1), sigma_bias_GW[i])

	# Compute relative 2σ percentage errors for cosmological parameters
	rel_err_H0, rel_err_omega, rel_err_omega_b, rel_err_As, rel_err_ns, rel_err_alpha_M, rel_err_alpha_B, rel_err_w_0, rel_err_w_a = fem.compute_and_print_relative_errors(
		sigma_H0, sigma_omega, sigma_omega_b, sigma_As, sigma_ns,
		sigma_alpha_M, sigma_alpha_B, sigma_w_0, sigma_w_a,
		H0_true, Omega_m_true, Omega_b_true, A_s, n_s, alpha_M, alpha_B, w_0, w_a)

	# Compute relative 2σ percentage errors for biases
	rel_err_bias_gal = np.zeros(shape=(n_bins_z))
	for i in range(n_bins_z):
		rel_err_bias_gal[i] = 2 * sigma_bias_gal[i] / bias_gal[i] * 100

	rel_err_bias_GW = np.zeros(shape=(n_bins_dl))
	for i in range(n_bins_dl):
		rel_err_bias_GW[i] = 2 * sigma_bias_GW[i] / bias_GW[i] * 100


	#-----------------------------------------------------------------------------------------
    #							SAVING RELATIVE ERRORS
	#-----------------------------------------------------------------------------------------
	print('\nSaving errors...\n')
	# Write relative errors of cosmological parameters to results_error.txt
	with open(os.path.join(FLAGS.fout, 'results_error.txt'), 'a') as file:
		file.write(
			'\ndetector: %s, year: %i, lensing:%s, bin strategy: %s, n_bins_z: %i, n_bins_dl: %i\nH_0 = %.2f, Omega_m = %.2f, Omega_b = %.2f, A_s = %.2f, n_s = %.2f, alpha_M = %.2f,alpha_B = %.2f,w_0 = %.2f,w_a = %.2f\n' %
			(GW_det, yr, Lensing, bin_strategy, n_bins_z, n_bins_dl, rel_err_H0, rel_err_omega, rel_err_omega_b,
			 rel_err_As, rel_err_ns, rel_err_alpha_M, rel_err_alpha_B, rel_err_w_0, rel_err_w_a))

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

	#-----------------------------------------------------------------------------------------
	#			PLOTTING 2D GAUSSIAN CONTOUR FOR H0_true AND Omega_m_true
	#-----------------------------------------------------------------------------------------
	fem.plot_gaussian_contour(H0_true, Omega_m_true, sigma_H0, sigma_omega, fisher_marg, GW_det, FLAGS.fout)

	# -----------------------------------------------------------------------------------------
	#							SAVING THE FINAL RESULTS
	# -----------------------------------------------------------------------------------------

	with open(os.path.join(FLAGS.fout, 'results.txt'), 'a') as file:
		file.write('\ndetector: %s, year: %i, lensing:%s, bin strategy: %s, n_bins_z: %i, n_bins_dl: %i, z_min: %f, z_max: %f, '
				   'err_gal: %f, l_max: %i, n_gal: %i, n_GW: %f,sigma_H0_perc: %.2f, sigma_omega_m_perc: %.2f, sigma_alpha_M_perc: %.2f, sigma_alpha_B_perc: %.2f\n'
				   % (GW_det, yr, Lensing, bin_strategy, n_bins_z, n_bins_dl, z_m_bin, z_M_bin, sig_gal, l_max, n_gal_bins,n_GW_bins, rel_err_H0, rel_err_omega, rel_err_alpha_M, rel_err_alpha_B))

	print('''
	  ________  ________   _______   ______ 
	 /_  __/ / / / ____/  / ____/ | / / __ \\
	  / / / /_/ / __/    / __/ /  |/ / / / /
	 / / / __  / /___   / /___/ /|  / /_/ / 
	/_/ /_/ /_/_____/  /_____/_/ |_/_____/  
	''')

