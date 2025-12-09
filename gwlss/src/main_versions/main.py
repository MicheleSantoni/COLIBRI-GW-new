#!/usr/bin/env python3

##############################

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


parser = argparse.ArgumentParser()
parser.add_argument("--config", default='', type=str, required=False) # path to config file, in.json format
parser.add_argument("--fout", default='', type=str, required=True) # path to output folder


FLAGS = parser.parse_args()


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




	##############
	
	# GW detector (ET_Delta_2CE, ET_2L_2CE, ET_Delta_1CE, ET_2L_1CE, ET_Delta, ET_2L, LVK)
	GW_det = config.GW_det

	# Years of observation
	yr = config.yr

	# Define the number of bins
	nbins_z = config.nbins_z
	nbins_dl = config.nbins_dl

	# Define the galaxy bin range
	zm_bin = config.zm_bin
	zM_bin = config.zM_bin

	# Define the GW bin range in redshift (will be converted in dl using the fiducual model)
	zm_bin_GW = config.zm_bin_GW
	zM_bin_GW = config.zM_bin_GW

	# Set the binning strategy (right_cosmo, wrong_cosmo(H0=65, Om0=0.32), equal_pop, equal_space)
	bin_strategy = config.bin_strategy

	# Include the lenisng
	Lensing = config.Lensing

	# "True" values of the cosmologcal parameters
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
	    
	if GW_det=='ET_Delta_2CE':
		A, Z_0, Alpha, Beta = 40.143*yr, 1.364, 2.693, 0.625
		log_delta_dl = np.load('det_param/log_delta_dl_ET_Delta_2CE.npy')
		log_loc = np.load('det_param/log_loc_ET_Delta_2CE.npy')
		log_dl = np.load('det_param/log_dl_ET_Delta_2CE.npy')
		s_a, s_b, s_c, s_d = -5.59*10**(-3), 2.92*10**(-2), 3.44*10**(-3), 2.58*10**(-3)
		be_a, be_b, be_c, be_d = -1.45, -1.39, 1.98, -0.363

	elif GW_det=='ET_2L_2CE':
		A, Z_0, Alpha, Beta = 32.795*yr, 1.244, 2.729, 0.614
		log_delta_dl = np.load('det_param/log_delta_dl_ET_2L_2CE.npy')
		log_loc = np.load('det_param/log_loc_ET_2L_2CE.npy')
		log_dl = np.load('det_param/log_dl_ET_2L_2CE.npy')
		s_a, s_b, s_c, s_d = -5.59*10**(-3), 2.92*10**(-2), 3.44*10**(-3), 2.58*10**(-3)
		be_a, be_b, be_c, be_d = -1.45, -1.39, 1.98, -0.363

	if GW_det=='ET_Delta_2CE_cut':
		A, Z_0, Alpha, Beta = 437.98*yr, 6.84, 1.687, 1.07
		log_delta_dl = np.load('det_param/log_delta_dl_ET_Delta_2CE_hardcut.npy')
		log_loc = np.load('det_param/log_loc_ET_Delta_2CE_hardcut.npy')
		log_dl = np.load('det_param/log_dl_ET_Delta_2CE_hardcut.npy')
		s_a, s_b, s_c, s_d = -5.59*10**(-3), 2.92*10**(-2), 3.44*10**(-3), 2.58*10**(-3)
		be_a, be_b, be_c, be_d = -1.45, -1.39, 1.98, -0.363

	elif GW_det=='ET_2L_2CE_cut':
		A, Z_0, Alpha, Beta = 465.1*yr, 7.09, 1.72, 1.06
		log_delta_dl = np.load('det_param/log_delta_dl_ET_2L_2CE_hardcut.npy')
		log_loc = np.load('det_param/log_loc_ET_2L_2CE_hardcut.npy')
		log_dl = np.load('det_param/log_dl_ET_2L_2CE_hardcut.npy')
		s_a, s_b, s_c, s_d = -5.59*10**(-3), 2.92*10**(-2), 3.44*10**(-3), 2.58*10**(-3)
		be_a, be_b, be_c, be_d = -1.45, -1.39, 1.98, -0.363

	elif GW_det=='ET_Delta_1CE':
		A, Z_0, Alpha, Beta = 69.695*yr, 1.79, 2.539, 0.658
		log_delta_dl = np.load('det_param/log_delta_dl_ET_Delta_1CE.npy')
		log_loc = np.load('det_param/log_loc_ET_Delta_1CE.npy')
		log_dl = np.load('det_param/log_dl_ET_Delta_1CE.npy')
		s_a, s_b, s_c, s_d = -5.59*10**(-3), 2.92*10**(-2), 3.44*10**(-3), 2.58*10**(-3)
		be_a, be_b, be_c, be_d = -1.45, -1.39, 1.98, -0.363

	elif GW_det=='ET_2L_1CE':
		A, Z_0, Alpha, Beta = 49.835*yr, 1.533, 2.619, 0.638
		log_delta_dl = np.load('det_param/log_delta_dl_ET_2L_1CE.npy')
		log_loc = np.load('det_param/log_loc_ET_2L_1CE.npy')
		log_dl = np.load('det_param/log_dl_ET_2L_1CE.npy')
		s_a, s_b, s_c, s_d = -5.59*10**(-3), 2.92*10**(-2), 3.44*10**(-3), 2.58*10**(-3)
		be_a, be_b, be_c, be_d = -1.45, -1.39, 1.98, -0.363
	    
	elif GW_det=='ET_Delta':
		#A, Z_0, Alpha, Beta = 330.897*yr, 6.352, 1.697, 0.922
		A, Z_0, Alpha, Beta = 99*yr, 6.89, 1.25, 0.97
		log_delta_dl = np.load('det_param/log_delta_dl_ET_Delta_cut.npy')
		log_loc = np.load('det_param/log_loc_ET_Delta_cut.npy')
		log_dl = np.load('det_param/log_dl_ET_Delta_cut.npy')
		s_a, s_b, s_c, s_d = -8.39*10**(-3), 4.54*10**(-2), 1.36*10**(-2), -2.04*10**(-3)
		be_a, be_b, be_c, be_d = -1.45, -1.39, 1.98, -0.363
	    
	elif GW_det=='ET_2L':
		A, Z_0, Alpha, Beta = 61.34*yr, 1.97, 1.93, 0.7
		log_delta_dl = np.load('det_param/log_delta_dl_ET_2L_cut.npy')
		log_loc = np.load('det_param/log_loc_ET_2L_cut.npy')
		log_dl = np.load('det_param/log_dl_ET_2L_cut.npy')
		s_a, s_b, s_c, s_d = -8.39*10**(-3), 4.54*10**(-2), 1.36*10**(-2), -2.04*10**(-3)
		be_a, be_b, be_c, be_d = -1.45, -1.39, 1.98, -0.363
	    
	elif GW_det=='LVK':
		A, Z_0, Alpha, Beta = 60.585*yr, 2.149, 1.445, 0.910
		log_delta_dl = np.load('det_param/log_delta_dl_LVK.npy')
		log_loc = np.load('det_param/log_loc_LVK.npy')
		log_dl = np.load('det_param/log_dl_LVK.npy')
		s_a, s_b, s_c, s_d = -1.22*10**(-1), 3.15, -7.61, 7.33
		be_a, be_b, be_c, be_d = -1.04, -0.176, 1.05*10**2, -4.36*10**2
    
	# Number of parameters Fisher
	n_param = 5 + nbins_dl + nbins_z
	Omega_b_true = 0.048

	# Compute power spectra (True)
	fourier = True

	# Define the redshift total range
	zm = 0.001
	zM = 7

	# Define the luminosity distance total range
	dlm = 1
	dlM = 100000

	# Luminosity distance interval, equal to the redshift one assuming fiducial cosmology
	fiducial_universe = FlatLambdaCDM(H0=H0_true, Om0=Omega_m_true, Ob0=Omega_b_true)

	dlm_bin = fiducial_universe.luminosity_distance(zm_bin_GW).value
	dlM_bin = fiducial_universe.luminosity_distance(zM_bin_GW).value

	z_gal = np.linspace(zm, zM, 1200)
	dl_GW = np.linspace(dlm, dlM, 1200)

	bin_int = np.linspace(zm_bin, zM_bin, nbins_z*1000)
	bin_int_GW = np.linspace(dlm_bin/1000, dlM_bin/1000, nbins_dl*1000)

	if gal_det=='euclid_photo':
		bg0, bg1, bg2, bg3 = 0.5125, 1.377, 0.222, -0.249
		sg0, sg1, sg2, sg3 = 0.0842, 0.0532, 0.298, -0.0113
		bin_centers_fit = np.array([0.001, 0.14, 0.26, 0.39, 0.53, 0.69, 0.84, 1.00, 1.14, 1.30, 1.44, 1.62, 1.78, 1.91, 2.1, 2.25])
		values_fit = np.array([0, 0.758, 2.607, 4.117, 3.837, 3.861, 3.730, 3.000, 2.827, 1.800, 1.078, 0.522, 0.360, 0.251, 0.1, 0])
		spline = UnivariateSpline(bin_centers_fit, values_fit, s=0.1)

	if gal_det=='euclid_spectro':
		bg0, bg1, bg2, bg3 = 0.853, 0.04, 0.713, -0.164
		sg0, sg1, sg2, sg3 = 1.231, -1.746, 1.810, -0.505
		bin_centers_fit = np.array([0.8, 1, 1.07, 1.14, 1.2, 1.35, 1.45, 1.56, 1.67, 1.9])
		values_fit = np.array([0., 0.2802, 0.2802, 0.2571, 0.2571, 0.2184, 0.2184, 0.2443, 0.2443, 0.])
		spline = UnivariateSpline(bin_centers_fit, values_fit, s=0)
		sig_gal = 0.001

	if gal_det=='ska':
		bg0, bg1, bg2, bg3 = 0.853, 0.04, 0.713, -0.164
		sg0, sg1, sg2, sg3 = 1.36, 1.76, -1.18, 0.28
		bin_centers_fit = np.array([0.01, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95])
		values_fit = np.array([0, 1.21872309, 1.74931326, 1.81914498, 1.6263191 , 1.33347361, 1.05034008, 0.79713276, 0.58895358, 0.42322164, 0.29564803, 0.20296989, 0.1366185 , 0.09011826, 0.0586648 , 0.03724468, 0.02323761, 0.01423011, 0.00848182, 0.00492732])
		spline = UnivariateSpline(bin_centers_fit, values_fit, s=0.001)
		f_sky = 0.7
		sig_gal = 0.001

	if bin_strategy=='right_cosmo':
		if nbins_dl<=nbins_z:
			print('number of bins in distance must be greater than bin in z, set automatically to nbins_z+1')
			nbins_dl = nbins_z+1
			
		gal_bin = spline(bin_int)
		gal_bin[gal_bin<0] = 0
		interval_gal = fcc.equal_interval(gal_bin, bin_int, nbins_z)
		bin_edges = bin_int[interval_gal]

		bin_edges_dl = np.zeros(nbins_dl+1)
		for i in range(len(bin_edges)):
			bin_edges_dl[i] = fiducial_universe.luminosity_distance(bin_edges[i]).value / 1000

		dlm_bin = fiducial_universe.luminosity_distance(zM_bin).value
		bin_int_GW = np.linspace(dlm_bin/1000, dlM_bin/1000, nbins_dl*100)

		GW_bin = A*(bin_int_GW/Z_0)**Alpha*np.exp(-(bin_int_GW/Z_0)**Beta)
		interval_GW = fcc.equal_interval(GW_bin, bin_int_GW, nbins_dl-nbins_z)

		bin_edges_dl[nbins_z:] = bin_int_GW[interval_GW]

	elif bin_strategy=='equal_space right_cosmo':
		if nbins_dl<=nbins_z:
			print('number of bins in distance must be greater than bin in z, set automatically to nbins_z+1')
			nbins_dl = nbins_z+1
			
		bin_edges = np.linspace(zm_bin, zM_bin, nbins_z+1)

		bin_edges_dl = np.zeros(nbins_dl+1)
		for i in range(len(bin_edges)):
			bin_edges_dl[i] = fiducial_universe.luminosity_distance(bin_edges[i]).value / 1000

		dlm_bin = fiducial_universe.luminosity_distance(zM_bin).value
		bin_int_GW = np.linspace(dlm_bin/1000, dlM_bin/1000, nbins_dl-nbins_z+1)

		bin_edges_dl[nbins_z:] = bin_int_GW

	elif bin_strategy=='wrong_cosmo':
		wrong_universe = FlatLambdaCDM(H0=65, Om0=0.32)

		if nbins_dl<=nbins_z:
			print('number of bins in distance must be greater than bin in z, set automatically to nbins_z+1')
			nbins_dl = nbins_z+1
			
		gal_bin = spline(bin_int)
		gal_bin[gal_bin<0] = 0
		interval_gal = fcc.equal_interval(gal_bin, bin_int, nbins_z)
		bin_edges = bin_int[interval_gal]

		bin_edges_dl = np.zeros(nbins_dl+1)
		for i in range(len(bin_edges)):
			bin_edges_dl[i] = wrong_universe.luminosity_distance(bin_edges[i]).value / 1000

		dlm_bin = wrong_universe.luminosity_distance(zM_bin).value
		bin_int_GW = np.linspace(dlm_bin/1000, dlM_bin/1000, nbins_dl*100)

		GW_bin = A*(bin_int_GW/Z_0)**Alpha*np.exp(-(bin_int_GW/Z_0)**Beta)
		interval_GW = fcc.equal_interval(GW_bin, bin_int_GW, nbins_dl-nbins_z)

		bin_edges_dl[nbins_z:] = bin_int_GW[interval_GW]

	elif bin_strategy=='equal_pop':
		gal_bin = spline(bin_int)
		gal_bin[gal_bin<0] = 0
		interval_gal = fcc.equal_interval(gal_bin, bin_int, nbins_z)
		bin_edges = bin_int[interval_gal]

		GW_bin = A*(bin_int_GW/Z_0)**Alpha*np.exp(-(bin_int_GW/Z_0)**Beta)
		interval_GW = fcc.equal_interval(GW_bin, bin_int_GW, nbins_dl)
		bin_edges_dl = bin_int_GW[interval_GW]

	elif bin_strategy=='equal_space':
		bin_edges_dl = np.linspace(dlm_bin/1000, dlM_bin/1000, nbins_dl+1)
		bin_edges = np.linspace(zm_bin, zM_bin, nbins_z+1)

	bin_z_fiducial = (bin_edges_dl*u.Gpc).to(cu.redshift, cu.redshift_distance(fiducial_universe, kind="luminosity")).value

	np.save(os.path.join(FLAGS.fout, 'bin_edges_GW_fiducial.npy'), bin_z_fiducial)
	np.save(os.path.join(FLAGS.fout, 'bin_edges_GW.npy'), bin_edges_dl)
	np.save(os.path.join(FLAGS.fout, 'bin_edges_gal.npy'), bin_edges)

	if gal_det=='euclid_photo':
		nz_gal = fcc.euclid_photo(z_gal, bin_edges, sig_gal)
		gal_tot = spline(z_gal)*8.35e7

	if gal_det=='euclid_spectro':
		nz_gal = fcc.euclid_spec(z_gal, bin_edges, sig_gal)
		gal_tot = spline(z_gal)*1.25e7

	if gal_det=='ska':
		nz_gal = fcc.ska(z_gal, bin_edges, sig_gal)
		gal_tot = spline(z_gal)*9.6e7

	gal_tot[gal_tot<0]=0

	n_tot_gal=np.trapz(gal_tot,z_gal)

	print('\nthe total number of galaxies across all redshift: ', n_tot_gal*4*np.pi*f_sky)

	bin_frac_gal = np.zeros(shape=(nbins_z))
	for i in range(nbins_z):
		bin_frac_gal[i] = sint.simps(nz_gal[i], z_gal)
	    
	n_gal_bins = np.sum(bin_frac_gal)

	print('the total number of galaxies in our bins: ', n_gal_bins*4*np.pi*f_sky)

	for i in range(nbins_z):
		plt.plot(z_gal, nz_gal[i])

	for i in range(nbins_z):
		plt.axvline(bin_edges[i], c='black', alpha=0.5)
	plt.axvline(bin_edges[-1], c='black', alpha=0.5, label='bin edges')
	    
	plt.xlabel(r'$z$')
	plt.ylabel(r'$w_i$')
	plt.title('Galaxy bin distribution')

	plt.xlim(zm_bin-0.3, zM_bin+0.3)

	plt.plot(z_gal, gal_tot, ls='--', alpha=0.8, color='red', label='total\ndistribution')

	plt.savefig( os.path.join(FLAGS.fout, 'gal_distr.pdf'), bbox_inches='tight')

	plt.close()

	print('number of galaxies per sterad in each bin', bin_frac_gal)

	shot_noise_gal = 1/bin_frac_gal

	print('shot noise per bin', shot_noise_gal)

	print('mean number of galaxies in each bin: ', np.mean(bin_frac_gal))
	print('mean shot noise in each bin: ', np.mean(shot_noise_gal))

	redshift = np.zeros(shape=nbins_z)
	for i in range(nbins_z):
		a = np.argmax(nz_gal[i])
		redshift[i]= z_gal[a]

	kk_nl = np.geomspace(1e-4, 1e2, 200)
	zz_nl = np.linspace(zm_bin_GW, zM_bin_GW, 100)
	C = cc.cosmo(Omega_m=Omega_m_true, h=H0_true/100)
	_, P_vals = C.camb_Pk(z = zz_nl, k = kk_nl, nonlinear = True, halofit = 'mead2020')
	P_interp = RectBivariateSpline(zz_nl, kk_nl, P_vals)
	zcenters_use = redshift
	
	def j1(x):
		return 3/(x**2)*(np.sin(x)/x-np.cos(x))
	
	kmax=[]
	
	for z_ in zcenters_use:
		def pk(k):
			return P_interp(z_, k)[0]
		
		def sig_sq(R):
			return integrate.quad(lambda x: 1/(2*np.pi**2)*x**2*( j1(x*R)**2 )*pk(x) , kk_nl[0], kk_nl[-1], limit=10000)[0]
		
		sol = optimize.root_scalar(lambda x: sig_sq(x)-0.25, bracket=[0.01, 20], method='bisect').root
		kmax_ = np.pi/sol/2
		kmax.append(kmax_)
		
	kmax=np.asarray(kmax)
	
	l_max_nl = np.asarray( [ fiducial_universe.comoving_distance(zcenters_use[i]).value*k_ for i,k_ in enumerate(kmax)] ).astype(int)
        
	sigma_sn_GW, l_max_loc = fcc.loc_error_param(bin_edges_dl, log_loc, log_dl, l_min, 10000)
	
	n = len(l_max_nl)
	m = len(l_max_loc)
	
	l_max_nl_ = np.concatenate((l_max_nl, l_max_loc[-(m-n):]))
	l_max_bin = np.minimum(l_max_loc, l_max_nl_)
	loc_or_nl = np.where(l_max_loc <= l_max_nl_, 0, 1)
	
	l_max = np.max(l_max_nl_)
	ll = np.sort(np.unique(np.concatenate([np.arange(l_min, 20, step=2), np.arange(20, 50, step=5), np.arange(50, 100, step=10), np.arange(100, l_max+1, step=25)])))
	ll[-1] = l_max
	ll_total = np.arange(l_min, l_max+1)

	c = ll*(ll+1.)/(2.*np.pi)

	print('l vector: ', ll)

	print('l max bin: ', l_max_bin)
	
	np.save(os.path.join(FLAGS.fout, 'ell_max.npy'), l_max_bin)
	np.save(os.path.join(FLAGS.fout, 'loc_nl.npy'), loc_or_nl)
	
	z_GW, bin_convert, ndl_GW, nGW, merger_rate_tot = fcc.merger_rate_dl(dl=dl_GW, bin_dl=bin_edges_dl, log_dl=log_dl, log_delta_dl=log_delta_dl, H0=H0_true, omega_m=Omega_m_true, omega_b=Omega_b_true, A=A, Z_0=Z_0, Alpha=Alpha, Beta=Beta, normalize=False)

	n_tot_GW=np.trapz(merger_rate_tot, dl_GW/1000)*4*np.pi

	print('\nthe total number of GW across all distance: ', n_tot_GW)

	bin_frac_GW = np.zeros(shape=(nbins_dl))
	for i in range(nbins_dl):
		bin_frac_GW[i] = np.trapz(ndl_GW[i], dl_GW/1000)
	    
	n_GW_bins = np.sum(bin_frac_GW)

	print('the total number of GW in our bins: ', n_GW_bins*4*np.pi)

	for i in range(nbins_dl):
		plt.plot(dl_GW/1000, ndl_GW[i])

	for i in range(nbins_dl):
		plt.axvline(bin_edges_dl[i], c='black', alpha=0.5)
	plt.axvline(bin_edges_dl[-1], c='black', alpha=0.5, label='bin edges')
	    
	plt.xlabel(r'$d_L[Gpc]$')
	plt.ylabel(r'$w_i$')
	plt.title('GW bin distribution')

	#plt.xlim(dlm_bin/1000-3, dlM_bin/1000+7)

	plt.plot(dl_GW/1000, merger_rate_tot, ls='--', alpha=0.8, color='red', label='total\ndistribution')

	plt.savefig( os.path.join(FLAGS.fout, 'GW_distr.pdf'), bbox_inches='tight')

	plt.close()

	print('fraction of GW per sterad in each bin', bin_frac_GW)

	shot_noise_GW = 1/bin_frac_GW

	print('shot noise per bin', shot_noise_GW)

	print('mean number of GW in each bin: ', np.mean(bin_frac_GW))
	print('mean shot noise in each bin: ', np.mean(shot_noise_GW))	
	
	fig = plt.figure(figsize=(18, 7), tight_layout=True)

	ax=fig.add_subplot(121)

	for i in range(nbins_z):
		plt.plot(z_gal, nz_gal[i])
	    
	for i in range(nbins_z+1):
		plt.axvline(bin_edges[i], c='black', alpha=0.5)

	plt.xlabel(r'$z$')
	plt.ylabel(r'$\frac{dN}{dzd\Omega}$')
	plt.title('Galaxy distribution')

	plt.xlim(zm_bin-0.3, zM_bin+0.3)

	plt.plot(z_gal, gal_tot, ls='--', alpha=0.5, color='red')

	ax=fig.add_subplot(122)

	for i in range(nbins_dl):
		plt.plot(z_GW, ndl_GW[i])

	for i in range(nbins_dl+1):
		plt.axvline(bin_convert[i], c='black', alpha=0.5)
	    
	plt.xlabel(r'$z$')
	plt.ylabel(r'$\frac{dN}{dzd\Omega}$')
	plt.title('Merger rate, fiducial model')

	plt.xlim(zm_bin-0.3, bin_convert[-1]+0.5)

	plt.plot(z_GW, merger_rate_tot, ls='--', alpha=0.5, color='red')

	plt.savefig( os.path.join(FLAGS.fout, 'distr_compare.pdf'), bbox_inches='tight')

	plt.close()	
	
	if Lensing:
	    # Evolution bias (b) and Magnification bias (s) from arXiv:2309.04391v1
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
		
		def Cl_func(H_0, Omega_m, Omega_b, A_s, n_s, b_gal, b_GW, npoints=13, npoints_x=20, grid_x='lin', zmin=1e-05, nlow=5, nhigh=5):
				
			C = cc.cosmo(Omega_m=Omega_m, Omega_b=Omega_b, h=H_0/100, As=1e-9*A_s, ns=n_s)
			S = LLG.limber(cosmology = C, z_limits = [zm, zM])

			kk = np.geomspace(1e-4, 1e2, 301)
			zz = np.linspace(0, zM, 101)

			# Compute nonlinear matter power spectra
			_, pkz = C.camb_Pk(z = zz, k = kk, nonlinear = True, halofit = 'mead2020')

			S.load_power_spectra(z = zz, k = kk, power_spectra = pkz)

			# Generate the GW distribution					
			z_GW, bin_GW_converted, ndl_GW, nGW, total = fcc.merger_rate_dl(dl=dl_GW, bin_dl=bin_edges_dl, log_dl=log_dl, log_delta_dl=log_delta_dl, H0=H_0, omega_m=Omega_m, omega_b=Omega_b, A=A, Z_0=Z_0, Alpha=Alpha, Beta=Beta, normalize=False)

			S.load_bin_edges(bin_edges, bin_z_fiducial, name_1='lensing_gal', name_2='lensing_GW')
			S.load_bin_edges(bin_edges, bin_z_fiducial, name_1='galaxy', name_2='GW')
			S.load_bin_edges(bin_edges, bin_z_fiducial, name_1='rsd', name_2='lsd')

			beta = compute_beta(H_0, Omega_m, Omega_b, s_a, s_b, s_c, s_d, be_a, be_b, be_c, be_d)
			
			# Load the window functions
			S.load_galaxy_clustering_window_functions(z=z_gal, nz=nz_gal, ll=ll, bias=b_gal, name='galaxy')
			S.load_rsd_window_functions(z=z_gal, nz=nz_gal, H_0=H_0, omega_m=Omega_m, omega_b=Omega_b, ll=ll, name='rsd')
			S.load_gravitational_wave_window_functions(z=z_GW, ndl=ndl_GW, H_0=H_0, omega_m=Omega_m, omega_b=Omega_b, ll=ll, bias=b_GW, name='GW')
			S.load_lsd_window_functions(z=z_GW, ndl=ndl_GW, H_0=H_0, omega_m=Omega_m, omega_b=Omega_b, ll=ll, name='lsd')
			S.load_galaxy_lensing_window_functions(z=z_gal, nz=nz_gal, H_0=H_0, omega_m=Omega_m, ll=ll, name='lensing_gal')
			S.load_gw_lensing_window_functions(z=z_GW, ndl=ndl_GW, H_0=H_0, omega_m=Omega_m, omega_b=Omega_b, ll=ll, name='lensing_GW')
			
			Cl = S.limber_angular_power_spectra(l=ll, windows=['galaxy', 'GW', 'rsd', 'lsd'])			
			Cl_lens = S.limber_angular_power_spectra_lensing_auto(l=ll, s_gal=s_gal, beta = beta, H_0=H_0, omega_m=Omega_m, omega_b=Omega_b, windows=['lensing_gal', 'lensing_GW'], npoints=npoints, npoints_x=npoints_x, zmin=zmin, grid_x=grid_x, nlow=nlow,  nhigh=nhigh)
			Cl_lens_cross = S.limber_angular_power_spectra_lensing_cross(l=ll, s_gal=s_gal, beta = beta , H_0=H_0, omega_m=Omega_m, omega_b=Omega_b, windows=None, npoints=npoints, npoints_x=npoints_x, zmin=zmin, grid_x=grid_x, nlow=nlow, nhigh=nhigh)
			
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
			
			Cl_delta_len_GWG = np.swapaxes(Cl_delta_len_GWG, 0,1)
			Cl_delta_RSD_GWG = np.swapaxes(Cl_delta_RSD_GWG, 0,1)
			Cl_RSD_len_GWG = np.swapaxes(Cl_RSD_len_GWG, 0,1)
								
			Cl_GG = Cl_delta_GG + Cl_len_GG + Cl_RSD_GG + 2*Cl_delta_len_GG + 2*Cl_delta_RSD_GG + 2*Cl_RSD_len_GG
			Cl_GWGW = Cl_delta_GWGW + Cl_len_GWGW + Cl_RSD_GWGW + 2*Cl_delta_len_GWGW + 2*Cl_delta_RSD_GWGW + 2*Cl_RSD_len_GWGW
			Cl_GGW = Cl_delta_GGW + Cl_len_GGW + Cl_RSD_GGW + Cl_delta_len_GGW + Cl_delta_len_GWG + Cl_delta_RSD_GGW + Cl_delta_RSD_GWG + Cl_RSD_len_GGW + Cl_RSD_len_GWG

			return Cl_GG, Cl_GWGW, Cl_GGW
	    
	else:
	
		def Cl_func(H_0, Omega_m, Omega_b, A_s, n_s, b_gal, b_GW, npoints=13, npoints_x=20, grid_x='mix', zmin=1e-05, nlow=5, nhigh=5):

			C = cc.cosmo(Omega_m=Omega_m, Omega_b=Omega_b, h=H_0/100, As=1e-9*A_s, ns=n_s)
			S = LLG.limber(cosmology = C, z_limits = [zm, zM])
			kk = np.geomspace(1e-4, 1e2, 500)
			zz = np.linspace(0, zM, 100)

			# Compute nonlinear matter power spectra
			_, pkz = C.camb_Pk(z = zz, k = kk, nonlinear = True, halofit = 'mead2020')

			S.load_power_spectra(z = zz, k = kk, power_spectra = pkz)

			# Generate the GW distribution
					
			z_GW, bin_GW_converted, ndl_GW, nGW, total = fcc.merger_rate_dl(dl=dl_GW, bin_dl=bin_edges_dl, log_dl=log_dl, log_delta_dl=log_delta_dl, H0=H_0, omega_m=Omega_m, omega_b=Omega_b, A=A, Z_0=Z_0, Alpha=Alpha, Beta=Beta, normalize=False)

			S.load_bin_edges(bin_edges, bin_GW_converted)
				    
			# Load the window functions
					
			S.load_galaxy_clustering_window_functions(z = z_gal, nz = nz_gal, ll=ll, bias = b_gal, name = 'galaxy')

			S.load_gravitational_wave_window_functions(z = z_GW, ndl = ndl_GW, H_0=H_0, omega_m=Omega_m, omega_b=Omega_b, ll=ll, bias = b_GW, name = 'GW')

			Cl = S.limber_angular_power_spectra(l = ll, windows = None)

			Cl_delta_GG = Cl['galaxy-galaxy']
			Cl_delta_GWGW = Cl['GW-GW']
			Cl_delta_GGW = Cl['galaxy-GW']

			return Cl_delta_GG, Cl_delta_GWGW, Cl_delta_GGW
		
	A_GW = 1.2
	gamma = 0.59

	z_mean_GW = (bin_z_fiducial[:-1]+bin_z_fiducial[1:])*0.5
	bias_GW = A_GW*(1.+z_mean_GW)**gamma

	np.save(os.path.join(FLAGS.fout, 'bias_fiducial_GW'), bias_GW)
	
	z_mean_gal = (bin_edges[:-1]+bin_edges[1:])*0.5
	bias_gal = bg0 + bg1*z_mean_gal + bg2*z_mean_gal**2 + bg3*z_mean_gal**3

	if gal_det=='ska':
		s_gal = (sg0 + sg1*z_gal + sg2*z_gal**2 + sg3*z_gal**3)*z_gal
	else:
		s_gal = sg0 + sg1*z_gal + sg2*z_gal**2 + sg3*z_gal**3
	
	np.save(os.path.join(FLAGS.fout, 'bias_fiducial_gal'), bias_gal)
	
	noise_gal = fcc.shot_noise_mat_auto(shot_noise_gal, ll_total)
	noise_GW = fcc.shot_noise_mat_auto(shot_noise_GW, ll_total)

	noise_loc = np.zeros(shape=(nbins_dl, len(ll_total)))
	noise_loc_auto = np.zeros(shape=(nbins_dl, len(ll_total)))
	
	for i in range(nbins_dl):
		for l in range(len(ll_total)): 
			if (ll_total[l]*(ll_total[l]+1)*(sigma_sn_GW[i]/(2*np.pi)**(3/2))) < 30:
				noise_loc[i,l] = np.exp(-ll_total[l]*(ll_total[l]+1)*(sigma_sn_GW[i]/(2*np.pi)**(3/2)))
				noise_loc_auto[i,l] = np.exp(-2*ll_total[l]*(ll_total[l]+1)*(sigma_sn_GW[i]/(2*np.pi)**(3/2)))
			else:
				noise_loc[i,l] = np.exp(-30)
				noise_loc_auto[i,l] = np.exp(-30)

	noise_loc_mat = np.zeros(shape=(nbins_z, nbins_dl, len(ll_total)))
	noise_loc_mat_auto = np.zeros(shape=(nbins_dl, nbins_dl, len(ll_total)))
	
	for i in range(nbins_z):
		for ii in range(nbins_dl):
			noise_loc_mat[i,ii,:] = noise_loc[ii,:]

	for i in range(nbins_dl):
		for ii in range(i, nbins_dl):
			noise_loc_mat_auto[i,ii,:] = noise_loc_auto[ii,:]

	for i in range(nbins_dl):
		for ii in range(i+1, nbins_dl):
			noise_loc_mat_auto[ii,i] = noise_loc_mat_auto[i,ii]
	
	print('\nComputing the Power Spectrum...\n')
	
	Cl_GG, Cl_GWGW, Cl_GGW = Cl_func(H_0=H0_true, Omega_m=Omega_m_true, Omega_b=Omega_b_true, A_s=2.12605, n_s=0.96, b_gal= bias_gal, b_GW = bias_GW)

	Cl_GG_total = np.zeros(shape=(nbins_z, nbins_z, len(ll_total)))
	Cl_GWGW_total = np.zeros(shape=(nbins_dl, nbins_dl, len(ll_total)))
	Cl_GGW_total = np.zeros(shape=(nbins_z, nbins_dl, len(ll_total)))

	for i in range(nbins_z):
		for ii in range(nbins_z):
			Cl_GG_interp = si.interp1d(ll, Cl_GG[i,ii])
			Cl_GG_total[i,ii] = Cl_GG_interp(ll_total)

	for i in range(nbins_dl):
		for ii in range(nbins_dl):
			Cl_GWGW_interp = si.interp1d(ll, Cl_GWGW[i,ii])
			Cl_GWGW_total[i,ii] = Cl_GWGW_interp(ll_total)
		
	for i in range(nbins_z):
		for ii in range(nbins_dl):
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

	vec = fcc.vector_cl(cl_cross=Cl_GGW_total, cl_auto1=Cl_GG_total, cl_auto2=Cl_GWGW_total)
	cov_mat = fcc.covariance_matrix(vec, nbins_z, nbins_dl)
	np.save(os.path.join(FLAGS.fout, 'cov_mat'), cov_mat)
	
	covariance_matrices = {}
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
	
	for param in parameters:
		print(f"\nComputing the derivative with respect to the {param['name']}...\n")
		
		step = param["step"]
		method = param.get("method", "central")
		
		partial_der_GG = nd.Derivative(lambda x: param["derivative_args"](x)[0], step=step, method=method)
		partial_der_GWGW = nd.Derivative(lambda x: param["derivative_args"](x)[1], step=step, method=method)
		partial_der_GGW = nd.Derivative(lambda x: param["derivative_args"](x)[2], step=step, method=method)
		
		der_GG = partial_der_GG(param["true_value"])
		der_GWGW = partial_der_GWGW(param["true_value"])
		der_GGW = partial_der_GGW(param["true_value"])
		
		der_vec = fcc.vector_cl(cl_cross=der_GGW, cl_auto1=der_GG, cl_auto2=der_GWGW)
		der_cov_mat = fcc.covariance_matrix(der_vec, nbins_z, nbins_dl)
		
		# Store the computed covariance matrix in memory
		covariance_matrices[param["key"]] = der_cov_mat
		
		# Save the covariance matrix to a file
		np.save(os.path.join(FLAGS.fout, f"{param['key']}.npy"), der_cov_mat)
	
	step=1e-3
	
	der_bgal = np.zeros(shape=(nbins_z, nbins_dl+nbins_z, nbins_dl+nbins_z, len(ll)))

	def compute_partial_derivatives_gal(bgal, der_bgal):

		for i in range(len(bgal)):
		
			print('\nComputing the derivative with respect to the galaxy bias in bin %i...\n' %i)
		
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
					
			der_bgal_GG = nd.Derivative(func_GG, step=step)(bgal[i])
			der_bgal_GWGW = nd.Derivative(func_GWGW, step=step)(bgal[i])
			der_bgal_GGW = nd.Derivative(func_GGW, step=step)(bgal[i])

			der_bgal_vec = fcc.vector_cl(cl_cross=der_bgal_GGW, cl_auto1=der_bgal_GG, cl_auto2=der_bgal_GWGW)
			der_bgal_cov_mat = fcc.covariance_matrix(der_bgal_vec, nbins_z, nbins_dl)
			
			der_bgal[i] = der_bgal_cov_mat
			
			np.save(os.path.join(FLAGS.fout, 'der_bgal_cov_mat_bin_%i.npy' %i), der_bgal_cov_mat)
			    
		return der_bgal

	der_bgal_cov_mat = compute_partial_derivatives_gal(bias_gal, der_bgal)	 
		
	step=1e-3
	
	der_bGW = np.zeros(shape=(nbins_dl, nbins_dl+nbins_z, nbins_dl+nbins_z, len(ll)))

	def compute_partial_derivatives_GW(bGW, der_bGW):

		for i in range(len(bGW)):
		
			print('\nComputing the derivative with respect to the GW bias in bin %i...\n' %i)
		
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
					
			der_bGW_GG = nd.Derivative(func_GG, step=step)(bGW[i])
			der_bGW_GWGW = nd.Derivative(func_GWGW, step=step)(bGW[i])
			der_bGW_GGW = nd.Derivative(func_GGW, step=step)(bGW[i])

			der_bGW_vec = fcc.vector_cl(cl_cross=der_bGW_GGW, cl_auto1=der_bGW_GG, cl_auto2=der_bGW_GWGW)
			der_bGW_cov_mat = fcc.covariance_matrix(der_bGW_vec, nbins_z, nbins_dl)
			
			der_bGW[i] = der_bGW_cov_mat
			
			np.save(os.path.join(FLAGS.fout, 'der_bGW_cov_mat_bin_%i.npy' %i), der_bGW_cov_mat)
			    
		return der_bGW

	der_bGW_cov_mat = compute_partial_derivatives_GW(bias_GW, der_bGW)
		
	print('\nAll derivative computed\n')
	
	all_der = np.zeros((n_param, nbins_z+nbins_dl, nbins_z+nbins_dl, len(ll)))

	all_der[0]=covariance_matrices['der_H0_cov_mat']
	all_der[1]=covariance_matrices['der_omega_cov_mat']
	all_der[2]=covariance_matrices['der_omega_b_cov_mat']
	all_der[3]=covariance_matrices['der_As_cov_mat']
	all_der[4]=covariance_matrices['der_ns_cov_mat']

	for i in range(nbins_z):
		index=n_param-nbins_dl-nbins_z+i
		all_der[index]=der_bgal_cov_mat[i]

	for i in range(nbins_dl):
		index=n_param-nbins_dl+i
		all_der[index]=der_bGW_cov_mat[i]
	    
	print(all_der.shape)
	
	all_der_total = np.zeros((n_param, nbins_z+nbins_dl, nbins_z+nbins_dl, len(ll_total)))
	
	for i in range(n_param):
		for ii in range(nbins_z+nbins_dl):
			for iii in range(nbins_z+nbins_dl):
				all_der_interp = si.interp1d(ll, all_der[i,ii,iii])
				all_der_total[i,ii,iii] = all_der_interp(ll_total)

	all_der_lmin = np.ones_like(all_der_total)

	bin_centers = np.concatenate((z_mean_gal, z_mean_GW), axis=0)

	ell_max_total = np.concatenate((l_max_nl, l_max_bin), axis=0)

	def compute_lmin(z):
		conditions = [z < 0.5, (z >= 0.5) & (z < 0.75), (z >= 0.75) & (z < 1.25), z >= 1.25]
		values = [0, 5, 10, 15]
		return np.select(conditions, values, default=np.nan)
	
	def generate_matrix(arr):
		n = len(arr)
		matrix = np.zeros((n, n))
		for i in range(n):
			matrix[i, :i+1] = arr[:i+1]
			matrix[i, i+1:] = arr[i] 
		return matrix
	
	def symm(matrix):
		return np.triu(matrix) + np.triu(matrix, k=1).T
	
	ell_matrix = generate_matrix(ell_max_total)

	for i in range(len(ell_max_total)):
		for ii in range(len(ell_max_total)):
			ell_matrix[i,ii] = min(ell_matrix[i,ii], ell_max_total[ii])

	ell_matrix = symm(ell_matrix)

	for i in range(n_param):
		for ii in range(nbins_z+nbins_dl):
			for iii in range(nbins_z+nbins_dl):
				z_temp = min(bin_centers[ii], bin_centers[iii])
				lmin_temp = compute_lmin(z_temp).astype(int)

				if lmin_temp != 0:
					for l in range(lmin_temp):
						all_der_lmin[i,ii,iii,l] = 0

				lmax_temp = (ell_matrix[ii,iii]-5).astype(int)
				all_der_lmin[i,ii,iii,lmax_temp:] = 0
	
	all_der_total = all_der_total * all_der_lmin

	np.save(os.path.join(FLAGS.fout, 'all_der_total.npy'), all_der_total)

	fisher = fcc.fisher_matrix(cov_mat, all_der_total, ll_total, f_sky)

	def rotate_fisher_Ob_to_ob(or_matrix, Ob=0.048, H0=67.7, pos={'H0':0, 'Ob':2}):
		nparams = or_matrix.shape[0]
		rotMatrix = np.identity(nparams)
		J_H0Ob_to_H0ob = np.array( [ [1, 0],[-2*Ob/H0, 1e04/H0**2] ] )
		rotMatrix[np.ix_([pos['H0'],pos['Ob']],[pos['H0'],pos['Ob']])] = J_H0Ob_to_H0ob
		matrix = rotMatrix.T@or_matrix@rotMatrix
		return matrix
	
	fisher = rotate_fisher_Ob_to_ob(fisher)

	np.save(os.path.join(FLAGS.fout, 'fisher_mat.npy'), fisher)
	
	fisher_inv=scipy.linalg.inv(fisher)
	fisher_marg=fisher_inv[:2, :2]	
	
	sigma_H0=np.sqrt(fisher_inv[0,0])
	sigma_omega=np.sqrt(fisher_inv[1,1])
	sigma_omega_b=np.sqrt(fisher_inv[2,2])
	sigma_As=np.sqrt(fisher_inv[3,3])
	sigma_ns=np.sqrt(fisher_inv[4,4])

	sigma_bias_gal=np.zeros(shape=(nbins_z))
	for i in range(nbins_z):
		index=n_param-nbins_dl-nbins_z+i
		sigma_bias_gal[i]=np.sqrt(fisher_inv[index,index])

	sigma_bias_GW=np.zeros(shape=(nbins_dl))
	for i in range(nbins_dl):
		index=n_param-nbins_dl+i
		sigma_bias_GW[i]=np.sqrt(fisher_inv[index,index])

	print('\nH_0 = ',sigma_H0)
	print('Omega_m = ',sigma_omega)
	print('Omega_b = ',sigma_omega_b)
	print('A_s = ',sigma_As)
	print('n_s = ',sigma_ns)

	for i in range(nbins_z):
		print('bias galaxy bin %i = '%(i+1),sigma_bias_gal[i])

	for i in range(nbins_dl):
		print('bias GW bin %i = '%(i+1),sigma_bias_GW[i])
	    
	rel_err_H0=2*sigma_H0/H0_true*100
	rel_err_omega=2*sigma_omega/Omega_m_true*100
	rel_err_omega_b=2*sigma_omega_b/Omega_b_true*100
	rel_err_As=2*sigma_As/2.12605*100
	rel_err_ns=2*sigma_ns/0.96*100

	rel_err_bias_gal=np.zeros(shape=(nbins_z))
	for i in range(nbins_z):
		rel_err_bias_gal[i]=2*sigma_bias_gal[i]/bias_gal[i]*100

	rel_err_bias_GW=np.zeros(shape=(nbins_dl))
	for i in range(nbins_dl):
		rel_err_bias_GW[i]=2*sigma_bias_GW[i]/bias_GW[i]*100

	print('\nrelative errors:\n')
	print('H_0 = ',rel_err_H0)
	print('Omega_m = ',rel_err_omega)
	print('Omega_b = ',rel_err_omega_b)
	print('A_s = ',rel_err_As)
	print('n_s = ',rel_err_ns)

	with open(os.path.join(FLAGS.fout, 'results_error.txt'), 'a') as file: 
		file.write('\ndetector: %s, year: %i, lensing:%s, bin strategy: %s, nbins_z: %i, nbins_dl: %i\nH_0 = %.2f, Omega_m = %.2f, Omega_b = %.2f, A_s = %.2f, n_s = %.2f\n' %(GW_det, yr, Lensing, bin_strategy, nbins_z, nbins_dl, rel_err_H0, rel_err_omega, rel_err_omega_b, rel_err_As, rel_err_ns))
	
	for i in range(nbins_z):
		print('bias galaxy bin %i = '%(i+1),rel_err_bias_gal[i])
		with open(os.path.join(FLAGS.fout, 'results_error.txt'), 'a') as file: 
			file.write('bias galaxy bin %i = %.2f\n' %(i+1, rel_err_bias_gal[i]))
	
	for i in range(nbins_dl):
		print('bias GW bin %i = '%(i+1),rel_err_bias_GW[i])
		with open(os.path.join(FLAGS.fout, 'results_error.txt'), 'a') as file: 
			file.write('bias GW bin %i = %.2f\n' %(i+1, rel_err_bias_GW[i]))		 

	# Define the mean (center) and covariance matrix for your 2D Gaussian distribution
	mean = np.array([H0_true, Omega_m_true])
	cov_matrix = fisher_marg

	# Create a grid of points to evaluate the Gaussian distribution

	scale = 0.05

	x, y = np.meshgrid(np.linspace(H0_true-scale*H0_true, H0_true+scale*H0_true, 200), np.linspace(Omega_m_true-scale*Omega_m_true, Omega_m_true+scale*Omega_m_true, 200))
	pos = np.dstack((x, y))  # Stack x and y grids to create (x, y) pairs

	# Compute the probability density function (PDF) of the Gaussian distribution
	pdf = multivariate_normal(mean, cov_matrix).pdf(pos)

	pdf /= np.max(pdf)

	# Calculate the 68% probability region
	confidence_level = 0.68

	# Create a contour plot of the Gaussian distribution
	contour = plt.contour(x, y, pdf, levels=[confidence_level], colors='blue')
	plt.clabel(contour, fontsize=10, fmt='%0.2f')
	contourf = plt.contourf(x, y, pdf, levels=[confidence_level,1000], cmap='Blues', alpha=0.3)

	perc_err_H0=2*sigma_H0/H0_true*100
	perc_err_Om=2*sigma_omega/Omega_m_true*100

	plt.scatter(H0_true, Omega_m_true, c='blue', s=15, label='$\sigma_{H_0}/H_0=%.1f\%%$\n$\sigma_{\Omega_m}/\Omega_m=%.1f\%%$' %(perc_err_H0, perc_err_Om))

	plt.grid(True, linestyle='--', alpha=0.5) 
	plt.legend(fontsize=15)
	plt.xlabel('$H_0$')
	plt.ylabel('$\Omega_m$')
	plt.title('%s' %GW_det)

	plt.savefig( os.path.join(FLAGS.fout, 'contour_plot.pdf'), bbox_inches='tight')

	plt.close()
		
	with open(os.path.join(FLAGS.fout, 'results.txt'), 'a') as file: 
		file.write('\ndetector: %s, year: %i, lensing:%s, bin strategy: %s, nbins_z: %i, nbins_dl: %i, z_min: %f, z_max: %f, err_gal: %f, l_max: %i, n_gal: %i, n_GW: %f, sigma_H0_perc: %.2f, sigma_omega_m_perc: %.2f\n' %(GW_det, yr, Lensing, bin_strategy, nbins_z, nbins_dl, zm_bin, zM_bin, sig_gal, l_max, n_gal_bins, n_GW_bins, perc_err_H0, perc_err_Om))
