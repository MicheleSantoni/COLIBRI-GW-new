#-----------------------------------------------------------------------------------------
#                                CONFIGURATION FILE: MODEL
#-----------------------------------------------------------------------------------------
COSMO_PARAMS = {
    # --- Background (CLASS wants Omega_b + Omega_cdm, not Omega_m directly) ---
    'h': 0.67810,
    'T_cmb': 2.7255,
    'YHe': 0.2454,
    'N_ur': 3.044,
    'N_ncdm': 0,
    'Omega_b': 0.048,
    'Omega_m': 0.31,   
    'Omega_k': 0.0,

    # --- Primordial ---
    'A_s': 2.100549e-09,
    'n_s': 0.9660499,
    'tau_reio': 0.05430842,

    # --- Power spectrum request ---
    'output': 'mPk',
    'P_k_max_h/Mpc': 10.0,
    'z_max_pk': 5.0,
    'non_linear': 'halofit',  # or 'HMcode'

    # --- MG wrapper knobs (LCDM limit) ---
    'gravity_model': 'propto_omega',
    # parameters_smg = 'x_k, x_b, x_m, x_t, M*^2_ini'  -> set all alphas=0, M*^2=1
    'parameters_smg': '0.0, 0.0, 0.0, 0.0, 1.0',
    'expansion_model': 'lcdm',
    'expansion_smg': '0.69',  # any value is ignored in LCDM in your wrapper; 0.69 is consistent with Ω_DE

    # (optional) verbosity
    'input_verbose': 1, 'background_verbose': 1, 'thermodynamics_verbose': 1,
    'perturbations_verbose': 1, 'transfer_verbose': 1, 'primordial_verbose': 1,
    'fourier_verbose': 1, 'output_verbose': 1,
}
