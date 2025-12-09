#-----------------------------------------------------------------------------------------
#                                CONFIGURATION FILE: MODEL
#-----------------------------------------------------------------------------------------
COSMO_PARAMS = {
    # --- Background & Thermodynamics ---
    'h': 0.67810,
    'T_cmb': 2.7255,
    'YHe': 0.2454,  # replaced invalid 'BBN' , 0.2454
    'N_ur': 3.044,
    'N_ncdm':0,
    #'m_ncdm' : 0.0,
    'Omega_m': 0.31,
    'Omega_b': 0.048,
    'Omega_k': 0.0,
    'Omega_Lambda': 0,
    'Omega_fld': 0,
    'Omega_smg': -1,
    #'omega_ncdm' : 0.0,
    #'T_wdm': 0.71611,

    # --- Primordial Power Spectrum ---
    'A_s': 2.100549e-09,
    'n_s': 0.9660499,
    'tau_reio': 0.05430842,
    'Pk_ini_type': 'analytic_Pk',
    'k_pivot': 0.05,
    'alpha_s': 0.0,
    #'f_bi' : 1.,
    #'n_bi' : 1.5,
    #'f_cdi':1.,
    #'f_nid':1.,
    #'n_nid':2.,
    #'alpha_nid': 0.01,
    #'potential' : 'polynomial',
    #'full_potential' : 'polynomial',

    # --- Gravity Model ---
    # x_k, x_b, x_m, x_t, M*^2_ini
    'gravity_model': 'propto_omega',
    'parameters_smg': '0.2,0.0, 0.0, 0.0, 1.0',
    'expansion_model': 'wowa',
    #Omega_smg, w0, wa
    'expansion_smg': '0.6899, -1.0, 0.0',
    
    # --- quasi-static approximation ---
    'method_qs_smg' : 'quasi_static',
    'z_fd_qs_smg' : 10.,
    'trigger_mass_qs_smg' : 1.e3,
    'trigger_rad_qs_smg' : 1.e3,
    'eps_s_qs_smg' : 0.01,
    #'n_min_qs_smg' : 1e2,
    #'n_max_qs_smg' : 1e4,
    
    # ---- precision parameters ---
    'start_small_k_at_tau_c_over_tau_h' : 1e-4,
    'start_large_k_at_tau_h_over_tau_k' : 1e-4,
    'perturbations_sampling_stepsize' : 0.05,
    'l_logstep' : 1.045,
    'l_linstep' : 50,

    # --- Modified Gravity: Stability ---
    'output_background_smg': 1,
    'skip_stability_tests_smg': 'no',
    'cs2_safe_smg': 1e-4,
    'D_safe_smg': 0.0,
    'ct2_safe_smg': 0.0,
    'M2_safe_smg': 0.0,
    'a_min_stability_test_smg': 0,

    # --- MG Dynamics ---
    #'hubble_evolution': 'y',
    #'hubble_friction': 3.0,
    
    # --- DARK MATTER ---
    'DM_annihilation_efficiency': 0.,
    'DM_decay_fraction' : 0.,
    'DM_decay_Gamma' : 0.,
    'PBH_evaporation_fraction' : 0.,
    'PBH_evaporation_mass' : 0.,
    'PBH_accretion_fraction' : 0.,
    'PBH_accretion_mass' : 0.,
    'PBH_accretion_recipe' : 'disk_accretion',
    'PBH_accretion_ADAF_delta' : 1.e-3,
    'PBH_accretion_eigenvalue' : 0.1,
    'f_eff_type' : 'on_the_spot',
    'chi_type' : 'CK_2004',

    # --- Initial Conditions ---
    'pert_initial_conditions_smg': 'ext_field_attr',
    'pert_ic_ini_z_ref_smg': 1e10,
    'pert_ic_tolerance_smg': 2e-2,
    'pert_ic_regulator_smg': 1e-15,
    'pert_qs_ic_tolerance_test_smg': 10,
    'z_fd_qs_smg': 10.0,
    'trigger_mass_qs_smg': 1e3,
    'trigger_rad_qs_smg': 1e3,
    'eps_s_qs_smg': 0.01,
    'n_min_qs_smg': 1e2,
    'n_max_qs_smg': 1e4,

    # --- Sampling and Integration ---
    #'start_small_k_at_tau_c_over_tau_h': 1e-4,
    #'start_large_k_at_tau_h_over_tau_k': 1e-4,
    #'perturbations_sampling_stepsize': 0.05,
    #'l_logstep': 1.045,
    #'l_linstep': 50,

    # --- Modes, Gauge ---
    #'modes': 's',
    'ic': 'ad',
    'gauge': 'synchronous',

    # --- Reionization ---
    'recombination': 'RECFAST',
    'reio_parametrization': 'reio_camb',
    'reionization_exponent': 1.5,
    'reionization_width': 0.5,
    'helium_fullreio_redshift': 3.5,
    'helium_fullreio_width': 0.5,
    'compute_damping_scale' : 'no',
    'varying_fundamental_constants' : 'none',
    
    # --- Spectra parameters ---
    #'l_max_scalars' : 2500,
    #l_max_vectors = 500,
    #'l_max_tensors' : 500,
    

    # --- Fourier / Matter Power ---
    'P_k_max_h/Mpc': 100.0,
    'z_pk' : 0,
    'non_linear': 'halofit', 
    'z_max_pk':5.0,
    'lensing': 'no',
    'extra_metric_transfer_functions': 'no',
    'output': 'dTk,vTk,mPk',

    # --- Spectral Distortions (PIXIE etc.) ---
    #'sd_branching_approx': 'exact',
    #'sd_PCA_size': 2,
    #'sd_detector_name': 'PIXIE',
    #'sd_only_exotic': 'no',
    #'sd_include_g_distortion': 'no',
    #'sd_add_y': 0.0,
    #'sd_add_mu': 0.0,
    #'include_SZ_effect': 'no',

    # --- Verbosity ---
    'input_verbose': 1,
    'background_verbose': 1,
    'thermodynamics_verbose': 1,
    'perturbations_verbose': 1,
    'transfer_verbose': 1,
    'primordial_verbose': 1,
    'harmonic_verbose': 1,
    'fourier_verbose': 1,
    #'lensing_verbose': 1,
    #'distortions_verbose': 1,
    'output_verbose': 1
}




