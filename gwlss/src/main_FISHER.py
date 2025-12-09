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
	'''
	print("\n====================================================== CONFIGURATION ITEMS ======================================================")
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
	'''

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

	expansion_smg = cosmo_params['expansion_smg']
	#print(type(expansion_smg_smg), expansion_smg_smg)
	expansion_smg_list = list(map(float, expansion_smg.split(',')))  # Convert to list of floats
	if len(expansion_smg_list)==1:
		Omega_DE = expansion_smg_list[0]
		w_0 = -1
		w_a = 0
	else:
		Omega_DE = expansion_smg_list[0]
		w_0 = expansion_smg_list[1]
		w_a = expansion_smg_list[2]

	#print(Omega_DE,w_0,w_a)

	# Parameters in the Fisher
	parameters = [
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
			"step": 1e-2,
			"derivative_args": lambda x: Cl_func_wrapped(cosmo_params, H0_true, Omega_m_true, Omega_b_true, A_s, n_s,
														 alpha_M, x, w_0, w_a),
			"key": "der_aB_cov_mat"
		},
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
	]
	'''
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
	'''

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

	# -----------------------------------------------------------------------------------------
	# 	DEFINE FIDUCIAL COSMOLOGICAL MODEL AND COMPUTE CORRESPONDING LUMINOSITY DISTANCES
	# -----------------------------------------------------------------------------------------
	# Luminosity distance interval, equal to the redshift one assuming fiducial cosmology
	print("\nComputing the cosmology ...")
	fiducial_universe = FlatLambdaCDM(H0=H0_true, Om0=Omega_m_true, Ob0=Omega_b_true)

	dlm_bin = fiducial_universe.luminosity_distance(z_m_bin_GW).value  # Minimum luminosity distance from fiducial model
	dlM_bin = fiducial_universe.luminosity_distance(z_M_bin_GW).value  # Maximum luminosity distance from fiducial model

	z_gal = np.linspace(z_m, z_M, 2000)  # Redshift grid for galaxy distribution
	dl_GW = np.linspace(dlm, dlM, 2000)  # Luminosity distance grid for gravitational wave sources in Mpc

	# -----------------------------------------------------------------------------------------
	#							BIN STRATEGY
	# -----------------------------------------------------------------------------------------
	bin_int = np.linspace(z_m_bin, z_M_bin, n_bins_z * 1000)  # Fine redshift grid for binning
	bin_int_GW = np.linspace(dlm_bin / 1000, dlM_bin / 1000, n_bins_dl * 1000)  # Fine luminosity distance grid for GW binning (in Gpc)

	# Compute bin edges using the specified strategy and cosmology
	bin_edges, bin_edges_dl = fem.compute_bin_edges(bin_strategy, n_bins_dl, n_bins_z, bin_int, z_M_bin, dlM_bin, z_m_bin, fiducial_universe, A, Z_0, Alpha, Beta, spline)

	# Convert luminosity distance bin edges to redshift using the fiducial cosmology
	bin_z_fiducial = (bin_edges_dl * u.Gpc).to(cu.redshift,
											   cu.redshift_distance(fiducial_universe, kind="luminosity")).value

	# Compute redshift distribution and total number of galaxies
	nz_gal, gal_tot = fem.compute_nz_gal_and_total(gal_det, z_gal, bin_edges, sig_gal, gal_params['spline'])

	gal_tot[gal_tot < 0] = 0  # Remove negative values (if any)
	n_tot_gal = trapezoid(gal_tot, z_gal)  # Integrate total galaxy distribution
	print('\nthe total number of galaxies across all redshift: ', n_tot_gal * 4 * np.pi * f_sky)

	# Compute fraction of galaxies in each redshift bin
	bin_frac_gal = np.zeros(shape=(n_bins_z))
	for i in range(n_bins_z):
		bin_frac_gal[i] = simpson(nz_gal[i], z_gal)

	shot_noise_gal = 1 / bin_frac_gal
	n_gal_bins = np.sum(bin_frac_gal)  # Sum of galaxy fractions across bins
	print('the total number of galaxies in our bins: ', n_gal_bins * 4 * np.pi * f_sky)

	# Save bin edges for later use
	np.save(os.path.join(FLAGS.fout, 'bin_edges_GW_fiducial.npy'), bin_z_fiducial)
	np.save(os.path.join(FLAGS.fout, 'bin_edges_GW.npy'), bin_edges_dl)
	np.save(os.path.join(FLAGS.fout, 'bin_edges_gal.npy'), bin_edges)
	np.save(os.path.join(FLAGS.fout, 'nz_gal.npy'), nz_gal)

	# -----------------------------------------------------------------------------------------
	#           PLOTTING AND SAVING THE GALAXY BIN DISTRIBUTION AND INFORMATION
	# -----------------------------------------------------------------------------------------
	fem.plot_galaxy_bin_distributions(z_gal, nz_gal, gal_tot, bin_edges, n_bins_z, z_m_bin, z_M_bin, FLAGS.fout)

	# Print statistics about galaxy bins
	print('\nthe total number of galaxies across all redshift: ', n_tot_gal * 4 * np.pi * f_sky)
	print('the total number of galaxies in our bins: ', n_gal_bins * 4 * np.pi * f_sky)
	print('mean number of galaxies in each bin: ', np.mean(bin_frac_gal))
	print('mean shot noise in each bin: ', np.mean(shot_noise_gal))

	with open(os.path.join(FLAGS.fout, "galaxy_bin_distributions.txt"), "w") as f:
		f.write("Diagnostics for this run: z_gal,nz_gal, gal_tot, bin_frac_gal, shot_noise_gal, n_bins_z, z_m_bin, z_M_bin and info galaxies and shot noise \n\n")
		f.write("z_gal =" + str(z_gal.tolist()) + "\n\n")
		f.write("nz_gal =" + str(nz_gal.tolist()) + "\n\n")
		f.write("gal_tot = " + str(gal_tot.tolist()) + "\n\n")
		f.write("bin_frac_gal = " + str(bin_frac_gal.tolist()) + "\n\n")
		f.write("shot_noise_gal = " + str(shot_noise_gal.tolist()) + "\n\n")
		f.write("n_bins_z         = " + str(n_bins_z) + "\n\n")
		f.write("z_m_bin            = " + str(z_m_bin) + "\n\n")
		f.write("z_M_bin     = " + str(z_M_bin) + "\n\n")
		f.write("the total number of galaxies across all redshift  = " + str(n_tot_gal * 4 * np.pi * f_sky) + "\n\n")
		f.write("the total number of galaxies in our bins     = " + str(n_gal_bins * 4 * np.pi * f_sky) + "\n\n")
		f.write("mean number of galaxies in each bin     = " + str(np.mean(bin_frac_gal)) + "\n\n")
		f.write("mean shot noise in each bin    = " + str(np.mean(shot_noise_gal)) + "\n\n")

	print("\nDiagnostics GALAXY BIN DISTRIBUTION saved!\n")

	# -----------------------------------------------------------------------------------------
	# 		DETERMINE REPRESENTATIVE REDSHIFTS AND COMPUTE NONLINEAR POWER SPECTRUM
	# -----------------------------------------------------------------------------------------
	print('\nComputing power spectrum...\n')
	# Initialize array to store the peak redshift of each galaxy bin
	redshift = np.zeros(shape=n_bins_z)
	for i in range(n_bins_z):
		a = np.argmax(nz_gal[i])  # Index of maximum value in the redshift distribution
		redshift[i] = z_gal[a]  # Assign corresponding redshift


	# Define k and z arrays for evaluating the nonlinear power spectrum
	kk_nl_input = np.geomspace(1e-4, 1e2, 200)  # Logarithmically spaced k values
	zz_nl_input = np.linspace(z_m_bin_GW, z_M_bin_GW, 100)  # Linearly spaced redshift values

	# Compute nonlinear matter power spectrum using HI_CLASS
	Hi_Cosmo = cc.cosmo(**cosmo_params)

	# Compute nonlinear matter power spectrum using HI_CLASS
	bg, kk_nl, zz_nl, P_vals = Hi_Cosmo.hi_class_pk(cosmo_params, kk_nl_input, zz_nl_input,True) #Halofit

	print('\nSaving the background...')
	np.save(os.path.join(FLAGS.fout, 'background.npy'), bg)

	# Interpolate power spectrum over redshift and k
	P_interp = RectBivariateSpline(kk_nl, zz_nl, P_vals)

	# Use peak redshifts of bins as centers for computing k_max
	z_centers_use = redshift

	# Compute maximum usable wavenumber at each redshift bin center
	k_max = fem.compute_k_max(z_centers_use, P_interp, kk_nl)

	# -----------------------------------------------------------------------------------------
	#			COMPUTING MULTIPOLE LIMITS AND GW BIN DISTRIBUTION STATISTICS
	# -----------------------------------------------------------------------------------------
	# Compute maximum multipole l for each bin using comoving distance and k_max
	l_max_nl = np.asarray([fiducial_universe.comoving_distance(z_centers_use[i]).value * k_ for i, k_ in enumerate(k_max)]).astype(int)

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
		np.arange(100, l_max + 1, step=25)])))
	ll[-1] = l_max  # Ensure maximum l is included
	ll_total = np.arange(l_min, l_max + 1)

	# Compute normalization factor for Cl's
	c = ll * (ll + 1.) / (2. * np.pi)

	# Save computed arrays
	np.save(os.path.join(FLAGS.fout, 'ell_max.npy'), l_max_bin)
	np.save(os.path.join(FLAGS.fout, 'loc_nl.npy'), loc_or_nl)

	# -----------------------------------------------------------------------------------------
	#            COMPUTING AND PLOTTING GW BIN DISTRIBUTION STATISTICS
	# -----------------------------------------------------------------------------------------
	# Compute the merger rate distribution and related quantities from luminosity distance bins
	z_GW, bin_convert, ndl_GW, n_GW, merger_rate_tot = fcc.merger_rate_dl(  #_new
		dl=dl_GW, #Mpc
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

	np.save(os.path.join(FLAGS.fout, 'z_GW'), z_GW)
	np.save(os.path.join(FLAGS.fout, 'ndl_GW'), ndl_GW)
	np.save(os.path.join(FLAGS.fout, 'n_GW'), n_GW)

	# Integrate the total merger rate over the full luminosity distance range (in Gpc)
	n_tot_GW = trapezoid(merger_rate_tot, dl_GW / 1000) * 4 * np.pi   # Mpc/1000 = Gpc
	print('\nthe total number of GW across all distance: ', n_tot_GW)

	# Calculate the fraction of GW sources in each luminosity distance bin
	bin_frac_GW = np.zeros(shape=n_bins_dl)
	for i in range(n_bins_dl):
		bin_frac_GW[i] = trapezoid(ndl_GW[i], dl_GW / 1000)

	# Sum all bin fractions to get the total number in bins (should match total GW if complete)
	n_GW_bins = np.sum(bin_frac_GW)
	print('the total number of GW in our bins: ', n_GW_bins * 4 * np.pi)

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
	print('shot noise per bin', shot_noise_GW)
	print('mean number of GW in each bin: ', np.mean(bin_frac_GW))
	print('mean shot noise in each bin: ', np.mean(shot_noise_GW))

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

			# z_GW no dim; bin_GW_converted no dim; ndl_GW 1/Gpc; n_GW [Gpc]; total no dim
			z_GW, bin_GW_converted, ndl_GW, n_GW, total = fcc.merger_rate_dl_new(
				bg=bg,
				dl=dl_GW,
				bin_dl=bin_edges_dl,
				log_dl=log_dl,
				log_delta_dl=log_delta_dl,
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

			#beta_new =fem.compute_beta_new(Hi_Cosmo,z_gal,Omega_m, s_a, s_b, s_c, s_d, be_a, be_b, be_c, be_d)
			#print('Relative dif % beta', ((beta_new/beta)-1 )*100)

			print('\nLoading the window functions...\n')
			# Load window functions for each observable
			S.load_galaxy_clustering_window_functions(bg, h, z=z_gal, n_z=nz_gal, ll=ll, bias=b_gal, name='galaxy') ### OK
			S.load_gravitational_wave_window_functions(bg, C=Hi_Cosmo, z=z_GW, n_dl=ndl_GW, ll=ll, bias=b_GW, name='GW')  ### OK

			S.load_rsd_window_functions(bg,h, z=z_gal, n_z=nz_gal, ll=ll,name='rsd') ### OK
			S.load_lsd_window_functions(bg, h,C=Hi_Cosmo, z=z_GW, n_dl=ndl_GW, ll=ll, name='lsd') ### OK

			S.load_galaxy_lensing_window_functions(z=z_gal, n_z=nz_gal, H_0=H_0, omega_m=Omega_m, ll=ll,name='lensing_gal') ### in units Mpc   OK
			S.load_gw_lensing_window_functions(bg=bg,C=Hi_Cosmo, z=z_GW,h=h, n_dl=ndl_GW, H_0=H_0, omega_m=Omega_m, ll=ll,name='lensing_GW') ### OK


			print('\nComputing the angular power spectra...\n')
			# Compute all angular power spectra using Limber integrals
			#start = time.time()

			Cl = S.limber_angular_power_spectra(bg,h=h,l=ll, windows=['galaxy', 'GW', 'rsd', 'lsd'])

			#end = time.time()
			#print(f"Time took {end - start:.4f} seconds \n")

			print('\nComputing the angular power spectra cross-correlation...\n')


			Cl_lens_cross = S.limber_angular_power_spectra_lensing_cross(bg, l=ll, s_gal=s_gal, beta=beta,
																		 windows=None, n_points=n_points,
																		 n_points_x=n_points_x,
																		 z_min=z_min, grid_x=grid_x,
																		 n_low=n_low,
																		 n_high=n_high)


			print('\nComputing the angular power spectra autocorrelation...\n')
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

			h = params['h']
			H_0 = params['h'] * 100
			Omega_m = params['Omega_m']
			Omega_b = params['Omega_b']

			# Generate GW source distribution
			# z_GW no dim; bin_GW_converted ; ndl_GW 1/Gpc; n_GW [Gpc]; total no dim
			z_GW, bin_GW_converted, ndl_GW, n_GW, total = fcc.merger_rate_dl_new(
				bg=bg,
				dl=dl_GW, #Mpc
				bin_dl=bin_edges_dl,
				log_dl=log_dl,
				log_delta_dl=log_delta_dl,
				A=A,
				Z_0=Z_0,
				Alpha=Alpha,
				Beta=Beta,
				C=Hi_Cosmo,
				normalize=False
			)
			'''
			with open(os.path.join(FLAGS.fout, "analysis_Cl_func.txt"), "w") as f:
				f.write(
					"Diagnostics for Cl_func and windows \n\n")
				f.write("z_gal =" + str(z_gal.tolist()) + "\n\n")
				f.write("nz_gal =" + str(nz_gal.tolist()) + "\n\n")
				f.write("z_GW = " + str(z_GW.tolist()) + "\n\n")
				f.write("ndl_GW = " + str(ndl_GW.tolist()) + "\n\n")
				f.write("ll = " + str(ll.tolist()) + "\n\n")
				f.write("b_GW         = " + str(b_GW.tolist()) + "\n\n")

			np.save(os.path.join(FLAGS.fout, 'background'), bg)
			'''

			print('\nLoading the window functions...')

			# Load binning and window functions
			S.load_bin_edges(bin_edges, bin_GW_converted)
			S.load_galaxy_clustering_window_functions(bg,h, z=z_gal, n_z=nz_gal, ll=ll, bias=b_gal,name='galaxy')


			S.load_gravitational_wave_window_functions(bg, C=Hi_Cosmo, z=z_GW, n_dl=ndl_GW, ll=ll,bias=b_GW, name='GW')

			'''
			S.load_galaxy_clustering_window_functions_old(z=z_gal, nz=nz_gal, ll=ll, bias=b_gal, name='gal_old')
			S.compare_galaxy_windows(
				z=z_GW, ll=ll,
				outdir='./cmp_gal',
				prefix='cmp_gal',
				name_bg='galaxy',
				name_legacy='gal_old',
				save_overlays=False,
				save_rel=False,
				title='GAL CLUSTERING'
			)
			'''

			#S.load_gravitational_wave_window_functions_old(z=z_GW, ndl=ndl_GW, ll=ll, bias=b_GW, H0=H_0,omega_m=Omega_m, omega_b=Omega_b, name='gw_old')
			#S.compare_galaxy_windows(
			#	z=z_GW, ll=ll,
			#	outdir='./cmp_gw',
			#	prefix='cmp_gw',
			#	name_bg='GW',
			#	name_legacy='gw_old',
			#	save_overlays=False,
			#	save_rel=False,
			#	title='GW CLUSTERING'
			#)


			print('\nComputing the angular power spectra...')
			# Compute angular power spectra (density terms only)

			#S.limber_angular_power_spectra_old(l=ll, windows=None)
			#S.limber_angular_power_spectra(bg, h=h, l=ll, windows=None)

			# Call after windows are loaded and power_spectra_interpolator is set
			#S.compare_angular_power_spectra(
			#	bg=bg, h=h, ell=ll,
			#	outdir=FLAGS.fout, prefix="Cl", spectra=["galaxy", "GW"]
			#)

			Cl = S.limber_angular_power_spectra(bg,h, l=ll, windows=None,outdir=FLAGS.fout)

			# Galaxy-GW
			Cl_delta_GGW = Cl['galaxy-GW']

			# Galaxy-Galaxy
			Cl_delta_GG = Cl['galaxy-galaxy']

			# GW-GW
			Cl_delta_GWGW = Cl['GW-GW']

			if save:
				print('\nSaving all the Cl results...')
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
		f.write("Diagnostics for this run: dl_GW, z_centers_use, k_max, ll, l_max_bin, ll_total, l_max_nl, z_mean_GW, z_mean_gal \n\n")
		f.write("dl_GW ="+str(dl_GW.tolist())+"\n\n")
		f.write("z_centers_use = " + str(z_centers_use.tolist()) + "\n\n")
		f.write("k_max         = " + str(k_max.tolist()) + "\n\n")
		f.write("ll            = " + str(ll.tolist()) + "\n\n")
		f.write("l_max_bin     = " + str(l_max_bin.tolist()) + "\n\n")
		f.write("ll_total      = " + str(ll_total.tolist()) + "\n\n")
		f.write("l_max_nl      = " + str(l_max_nl.tolist()) + "\n\n")
		f.write("z_mean_GW     = " + str(z_mean_GW.tolist()) + "\n\n")
		f.write("z_mean_gal    = " + str(z_mean_gal.tolist()) + "\n\n")

	print("\nDiagnostics for FISHER saved!")

	#-----------------------------------------------------------------------------------------
	#				COMPUTING LOCALIZATION NOISE MATRICES
	#-----------------------------------------------------------------------------------------
	print('\nComputing localization noise matrices...')

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
	print('\nComputing the Power Spectrum...')

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

	Cl_GG_total_with_noise= Cl_GG_total
	Cl_GWGW_total_with_noise = Cl_GWGW_total
	Cl_GGW_total_with_noise = Cl_GGW_total

	np.save(os.path.join(FLAGS.fout, 'Cl_GG_with_noise'), Cl_GG_total_with_noise)
	np.save(os.path.join(FLAGS.fout, 'Cl_GWGW_total_with_noise'), Cl_GWGW_total_with_noise)
	np.save(os.path.join(FLAGS.fout, 'Cl_GGW_total_with_noise'), Cl_GGW_total_with_noise)

	#-----------------------------------------------------------------------------------------
    #					COMPUTING FIDUCIAL COVARIANCE MATRIX
	#-----------------------------------------------------------------------------------------
	print('\nComputing fiducial covariance matrix...')
	# Construct the full data vector from Cl auto- and cross-spectra
	vec = fcc.vector_cl(cl_cross=Cl_GGW_total, cl_auto1=Cl_GG_total, cl_auto2=Cl_GWGW_total)

	# Compute the covariance matrix of the fiducial spectra
	cov_mat = fcc.covariance_matrix(vec, n_bins_z, n_bins_dl)

	# Save the fiducial covariance matrix to file
	np.save(os.path.join(FLAGS.fout, 'cov_mat'), cov_mat)

	#-----------------------------------------------------------------------------------------
	#				COMPUTING PARAMETER DERIVATIVE MATRIx
	#-----------------------------------------------------------------------------------------
	print('\nComputing parameter derivative matrix...')
	# Initialize dictionary to store parameter-specific covariance derivatives
	covariance_matrices = {}


	def Cl_func_wrapped(cosmo_params, H0, Omega_m, Omega_b, A_s, n_s, alpha_M, alpha_B, w_0, w_a):
		"""
        Unified wrapper for computing Cl given any set of cosmological parameters.
        Includes both ΛCDM and scalar modified gravity (SMG) parameters.
        """
		print('\nUpdating parameters...')
		params = deepcopy(cosmo_params)

		# ΛCDM-like parameters
		params['h'] = H0 / 100.0
		params['Omega_m'] = Omega_m
		params['Omega_b'] = Omega_b
		params['A_s'] = A_s*10**(-9)
		params['n_s'] = n_s

		# Modified gravity (SMG) parameters
		# x_k, x_b, x_m, x_t, (M_*)^ 2_ini -> '10.0, 0.0, 0.0, 0.0, 1.0' LCDM case
		params['parameters_smg'] = f"10.0,{alpha_B},{alpha_M},0.0,1.0"

		expansion_smg = cosmo_params['expansion_smg']
		expansion_smg_list = list(map(float, expansion_smg.split(',')))
		if len(expansion_smg_list) == 3:
			params['expansion_smg']=f"0.7,{w_0},{w_a}"

		print('\nThese are the values of the new cosmology:')
		print("COSMO: \t $\H_0$:", params['h']*100,"$\Omega_m$:", params['Omega_m'],'$\Omega_b$', params['Omega_b'], '$\A_s$',params['A_s'], "$n_s$",params['n_s'])
		print('MG: \t parameters_smg', params['parameters_smg'],'\t expansion model',params['expansion_smg'],'\n')

		# Cosmology object from CLASS / hi_class
		Hi_Cosmo_update = cc.cosmo(**params)

		return Cl_func(Hi_Cosmo_update, params, gw_params, dl_GW, bin_edges_dl, z_gal, ll, bias_gal, bias_GW,save=False)


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
	#all_der[7] = covariance_matrices['der_w_0_cov_mat']
	#all_der[8] = covariance_matrices['der_w_a_cov_mat']

	# Insert galaxy bias derivatives at appropriate indices
	for i in range(n_bins_z):
		index = n_param - n_bins_dl - n_bins_z + i
		all_der[index] = der_b_gal_cov_mat[i]

	# Insert GW bias derivatives at appropriate indices
	for i in range(n_bins_dl):
		index = n_param - n_bins_dl + i
		all_der[index] = der_bGW_cov_mat[i]

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
	print('\nComputing the Fisher Matrix...')
	# Compute Fisher information matrix using the full derivative set and covariance
	fisher = fcc.fisher_matrix(cov_mat, all_der_total, ll_total, f_sky)

	# Rotate the Fisher matrix to new basis
	fisher = fem.rotate_fisher_Ob_to_ob(fisher)

	# Save the rotated Fisher matrix
	np.save(os.path.join(FLAGS.fout, 'fisher_mat.npy'), fisher)

	#-----------------------------------------------------------------------------------------
    #			EXTRACTING 1σ ERRORS AND RELATIVE ERRORS FROM FISHER MATRIX
	#-----------------------------------------------------------------------------------------
	print('\nExtracting the errors...')

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
	print('\nSaving errors...')
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

