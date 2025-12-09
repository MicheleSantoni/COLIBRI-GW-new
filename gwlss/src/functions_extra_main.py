import numpy as np
from scipy import optimize
from scipy import integrate
from astropy.cosmology import FlatLambdaCDM
import functions_cross_correlation as fcc
from astropy import units as u
import astropy.constants as const
from scipy.integrate import quad, trapezoid
from scipy.optimize import minimize, root_scalar
from scipy.interpolate import UnivariateSpline
import scipy.interpolate as si
from concurrent.futures import ProcessPoolExecutor
import numdifftools as nd
import os
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

# ----
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

# ----
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
            'spline_s': 0.1,
            'sig_gal': 0.05
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

# ----
def compute_bin_edges_new(
        bin_strategy, n_bins_dl, n_bins_z,
        bin_int, z_M_bin, dlM_bin, z_m_bin,
        Hi_Cosmo, A, Z_0, Alpha, Beta,
        spline):
    """
    Compute bin edges in redshift and luminosity distance according to the specified strategy.

    Parameters:
        bin_strategy (str): Strategy name ('right_cosmo', 'equal_space right_cosmo', etc.)
        n_bins_dl (int): Number of bins for luminosity distance.
        n_bins_z (int): Number of bins for redshift.
        bin_int (array): z sampling for galaxy distribution.
        z_M_bin, dlM_bin, z_m_bin: z and dL bin boundaries.
        Hi_Cosmo: your colibri cosmology instance (cc.cosmo(**cosmo_params)).
        A, Z_0, Alpha, Beta: GW rate model parameters.
        spline: spline fitted to galaxy number density.

    Returns:
        bin_edges (array): Redshift bin edges.
        bin_edges_dl (array): Luminosity distance bin edges [Gpc].
    """
    def dL_from_C(Hi_Cosmo, z):
        """
        Luminosity distance d_L(z) in Mpc from your colibri cosmology `Hi_Cosmo`.
        Assumes Hi_Cosmo.comoving_distance(z) returns comoving distance in Mpc/h.
        """
        z = np.asarray(z, dtype=float)
        chi_Mpc = np.asarray(Hi_Cosmo.comoving_distance(z)) / Hi_Cosmo.h  # -> Mpc
        return (1.0 + z) * chi_Mpc

    # Helper: d_L in Gpc from your colibri cosmology
    def _dL_Gpc(cosmo, z):
        # dL_from_C returns Mpc; convert to Gpc
        return dL_from_C(cosmo, np.asarray(z, dtype=float)) / 1000.0

    if n_bins_dl <= n_bins_z:
        print('number of bins in distance must be greater than bins in z, set automatically to n_bins_z+1')
        n_bins_dl = n_bins_z + 1

    if bin_strategy == 'right_cosmo':
        gal_bin = spline(bin_int)
        gal_bin[gal_bin < 0] = 0
        interval_gal = np.asarray(fcc.equal_interval(gal_bin, bin_int, n_bins_z)).ravel()
        bin_edges = bin_int[interval_gal]

        # map those galaxy-z edges to d_L (Gpc)
        bin_edges_dl = np.zeros(n_bins_dl + 1, dtype=float)
        for i, z in enumerate(bin_edges):
            bin_edges_dl[i] = float(_dL_Gpc(Hi_Cosmo, z))

        # --- GW dL grid (Gpc) over [dL(z_m_bin), dlM_bin]  ---
        # (FIX: dlm was incorrectly using z_M_bin in the old code)
        dlm_bin_Gpc = float(_dL_Gpc(Hi_Cosmo, z_m_bin))
        dlM_bin_Gpc = float(dlM_bin) / 1000.0  # dlM_bin was passed in Mpc
        bin_int_GW = np.linspace(dlm_bin_Gpc, dlM_bin_Gpc, n_bins_dl * 100)

        GW_bin = A * (bin_int_GW / Z_0) ** Alpha * np.exp(-(bin_int_GW / Z_0) ** Beta)
        interval_GW = np.asarray(fcc.equal_interval(GW_bin, bin_int_GW, n_bins_dl - n_bins_z)).ravel()

        # append GW-only edges after the first n_bins_z+1 galaxy-mapped edges
        bin_edges_dl[n_bins_z:] = bin_int_GW[interval_GW]

    elif bin_strategy == 'equal_space right_cosmo':
        bin_edges = np.linspace(z_m_bin, z_M_bin, n_bins_z + 1)

        bin_edges_dl = np.zeros(n_bins_dl + 1, dtype=float)
        for i, z in enumerate(bin_edges):
            bin_edges_dl[i] = float(_dL_Gpc(Hi_Cosmo, z))

        # GW part: equal spacing in d_L over [dL(z_m_bin), dlM_bin]
        dlm_bin_Gpc = float(_dL_Gpc(Hi_Cosmo, z_m_bin))
        dlM_bin_Gpc = float(dlM_bin) / 1000.0
        bin_int_GW = np.linspace(dlm_bin_Gpc, dlM_bin_Gpc, n_bins_dl - n_bins_z + 1)

        bin_edges_dl[n_bins_z:] = bin_int_GW

    elif bin_strategy == 'wrong_cosmo':
        from astropy.cosmology import FlatLambdaCDM
        from astropy import units as u
        wrong_universe = FlatLambdaCDM(H0=65, Om0=0.32)

        gal_bin = spline(bin_int)
        gal_bin[gal_bin < 0] = 0
        interval_gal = np.asarray(fcc.equal_interval(gal_bin, bin_int, n_bins_z)).ravel()
        bin_edges = bin_int[interval_gal]

        bin_edges_dl = np.zeros(n_bins_dl + 1, dtype=float)
        for i, z in enumerate(bin_edges):
            bin_edges_dl[i] = wrong_universe.luminosity_distance(z).to_value(u.Gpc)

        # GW dL grid (Gpc) over [dL_wrong(z_m_bin), dlM_bin_wrong]
        dlm_bin_wrong = wrong_universe.luminosity_distance(z_m_bin).to_value(u.Gpc)  # min
        dlM_bin_wrong = float(dlM_bin) / 1000.0                                     # max (input Mpc → Gpc)
        bin_int_GW = np.linspace(dlm_bin_wrong, dlM_bin_wrong, n_bins_dl * 100)

        GW_bin = A * (bin_int_GW / Z_0) ** Alpha * np.exp(-(bin_int_GW / Z_0) ** Beta)
        interval_GW = np.asarray(fcc.equal_interval(GW_bin, bin_int_GW, n_bins_dl - n_bins_z)).ravel()

        bin_edges_dl[n_bins_z:] = bin_int_GW[interval_GW]

    elif bin_strategy == 'equal_pop':
        gal_bin = spline(bin_int)
        gal_bin[gal_bin < 0] = 0
        interval_gal = np.asarray(fcc.equal_interval(gal_bin, bin_int, n_bins_z)).ravel()
        bin_edges = bin_int[interval_gal]

        dlm_bin_Gpc = float(_dL_Gpc(Hi_Cosmo, z_m_bin))
        dlM_bin_Gpc = float(dlM_bin) / 1000.0
        bin_int_GW = np.linspace(dlm_bin_Gpc, dlM_bin_Gpc, n_bins_dl * 100)

        GW_bin = A * (bin_int_GW / Z_0) ** Alpha * np.exp(-(bin_int_GW / Z_0) ** Beta)
        interval_GW = np.asarray(fcc.equal_interval(GW_bin, bin_int_GW, n_bins_dl)).ravel()
        bin_edges_dl = bin_int_GW[interval_GW]

    elif bin_strategy == 'equal_space':
        bin_edges = np.linspace(z_m_bin, z_M_bin, n_bins_z + 1)
        dlm_bin_Gpc = float(_dL_Gpc(Hi_Cosmo, z_m_bin))
        dlM_bin_Gpc = float(dlM_bin) / 1000.0
        bin_edges_dl = np.linspace(dlm_bin_Gpc, dlM_bin_Gpc, n_bins_dl + 1)

    else:
        raise ValueError(f"Unknown binning strategy: {bin_strategy}")

    return bin_edges, bin_edges_dl


# ----
def compute_bin_edges(
        bin_strategy, n_bins_dl, n_bins_z,
        bin_int, z_M_bin, dlM_bin, z_m_bin,
        fiducial_universe, A, Z_0, Alpha, Beta,
        spline):
    """
    Compute bin edges in redshift and luminosity distance according to the specified strategy.

    Parameters:
        bin_strategy (str): Strategy name ('right_cosmo', 'equal_space right_cosmo', etc.)
        n_bins_dl (int): Number of bins for luminosity distance.
        n_bins_z (int): Number of bins for redshift.
        bin_int (array): z sampling for galaxy distribution.
        z_M_bin, dlM_bin, z_m_bin: z and dL bin boundaries.
        fiducial_universe: astropy cosmology instance.
        A, Z_0, Alpha, Beta: GW rate model parameters.
        spline: spline fitted to galaxy number density.

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
            bin_edges_dl[i] = fiducial_universe.luminosity_distance(z).value / 1000  #Gpc

        dlm_bin = fiducial_universe.luminosity_distance(z_M_bin).value
        bin_int_GW = np.linspace(dlm_bin / 1000, dlM_bin / 1000, n_bins_dl * 100)

        GW_bin = A * (bin_int_GW / Z_0) ** Alpha * np.exp(-(bin_int_GW / Z_0) ** Beta)
        interval_GW = fcc.equal_interval(GW_bin, bin_int_GW, n_bins_dl - n_bins_z)

        bin_edges_dl[n_bins_z:] = bin_int_GW[interval_GW]

    elif bin_strategy == 'equal_space right_cosmo':
        bin_edges = np.linspace(z_m_bin, z_M_bin, n_bins_z + 1)

        bin_edges_dl = np.zeros(n_bins_dl + 1)
        for i, z in enumerate(bin_edges):
            bin_edges_dl[i] = fiducial_universe.luminosity_distance(z).value / 1000

        dlm_bin = fiducial_universe.luminosity_distance(z_M_bin).value
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

        dlm_bin = wrong_universe.luminosity_distance(z_M_bin).value
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
        bin_edges = np.linspace(z_m_bin, z_M_bin, n_bins_z + 1)
        bin_edges_dl = np.linspace(dlm_bin / 1000, dlM_bin / 1000, n_bins_dl + 1)

    else:
        raise ValueError(f"Unknown binning strategy: {bin_strategy}")

    return bin_edges, bin_edges_dl

# ----
def plot_galaxy_bin_distributions(z_gal, nz_gal, gal_tot, bin_edges, n_bins_z, z_m_bin, z_M_bin, output_path):
    """
    Plot and save the galaxy redshift distribution and bin structure.

    Parameters:
        z_gal (array): Redshift grid.
        nz_gal (2D array): Redshift distribution per bin, shape (n_bins_z, len(z_gal)).
        gal_tot (array): Total redshift distribution.
        bin_edges (array): Edges of the redshift bins.
        n_bins_z (int): Number of galaxy bins.
        z_m_bin (float): Minimum redshift of the range.
        z_M_bin (float): Maximum redshift of the range.
        output_path (str): Path to save the output PDF.
    """

    # Plot each bin's distribution
    for i in range(n_bins_z):
        plt.plot(z_gal, nz_gal[i], label=f'bin {i+1}')

    # Plot vertical bin edges
    for edge in bin_edges[:-1]:
        plt.axvline(edge, color='black', alpha=0.5)
    plt.axvline(bin_edges[-1], color='black', alpha=0.5, label='bin edges')

    # Add total galaxy distribution
    plt.plot(z_gal, gal_tot, linestyle='--', alpha=0.8, color='red', label='total\ndistribution')

    # Configure plot
    plt.xlabel(r'$z$')
    plt.ylabel(r'$w_i$')
    plt.title('Galaxy bin distribution')
    plt.xlim(z_m_bin - 0.3, z_M_bin + 0.3)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save and close
    plt.savefig(os.path.join(output_path, 'gal_distr.pdf'), bbox_inches='tight')
    plt.close()


# ----
def compute_nz_gal_and_total(gal_det, z_gal, bin_edges, sig_gal, spline):
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

# ----
def compute_k_max(z_values, P_interp, k_range, sigma_target=0.25):
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

# ----

def compute_beta_new(cosmo,z_gal,Omega_m, s_a, s_b, s_c, s_d, be_a, be_b, be_c, be_d):
    """
    Compute the redshift distortion parameter 'beta' for galaxy clustering.

    Parameters:
    - H0: Hubble constant
    - Omega_m: Matter density parameter
    - Omega_b: Baryon density parameter
    - s_*: Coefficients for magnification bias (s)
    - be_*: Coefficients for evolution bias (b)

    Returns:
    - beta: Computed distortion parameter
    """
    z_bg = np.asarray(bg['z'])
    H_interp = interp1d(z_bg, bg['H [1/Mpc]'], kind='cubic', bounds_error=False, fill_value="extrapolate")  # [1/Mpc]
    #chi_interp = interp1d(z_bg, bg['comov. dist.'], kind='cubic', bounds_error=False, fill_value="extrapolate")  # Mpc
    #alpha_M_interp = interp1d(z_bg, bg['M2_running_smg'], kind='cubic', bounds_error=False, fill_value="extrapolate")

    H_gal=H_interp(1+z_gal)
    # Compute conformal Hubble parameter
    conf_H = H_gal / (1 + z_gal)

    # QUA MANCA

    # Derivative of the Hubble parameter with respect to redshift
    H_dot = -(3/2) * (cosmo.H(z_gal))**2 * Omega_m * (1 + z_gal)**3
    der_conf_H = 1 / ((1 + z_gal)**2) * (H_dot + (cosmo.H(z_gal))**2)

    # Intermediate variables
    r_conf_H = cosmo.comoving_distance(z_gal) * conf_H
    s = s_a + s_b * z_gal + s_c * z_gal**2 + s_d * z_gal**3
    gamma = r_conf_H / (1 + r_conf_H)
    b = be_a + be_b * z_gal + be_c * z_gal**2 + be_d * z_gal**3

    # Final beta expression
    beta = 5 * s - 1 + gamma * (2 / r_conf_H + gamma * ((der_conf_H / (conf_H**2)) - 1 / r_conf_H) - 1 - b)

    return beta


# ----
def compute_beta(H0, Omega_m, Omega_b,z_gal, s_a, s_b, s_c, s_d, be_a, be_b, be_c, be_d):
    """
    Compute the redshift distortion parameter 'beta' for galaxy clustering.

    Parameters:
    - H0: Hubble constant
    - Omega_m: Matter density parameter
    - Omega_b: Baryon density parameter
    - s_*: Coefficients for magnification bias (s)
    - be_*: Coefficients for evolution bias (b)

    Returns:
    - beta: Computed distortion parameter
    """
    cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m, Ob0=Omega_b, Tcmb0=2.7255)

    # Compute conformal Hubble parameter
    conf_H = cosmo.H(z_gal).value / (1 + z_gal)

    # Derivative of the Hubble parameter with respect to redshift
    H_dot = -(3/2) * (cosmo.H(z_gal).value)**2 * Omega_m * (1 + z_gal)**3
    der_conf_H = 1 / ((1 + z_gal)**2) * (H_dot + (cosmo.H(z_gal).value)**2)

    # Intermediate variables
    r_conf_H = cosmo.comoving_distance(z_gal).value * conf_H
    s = s_a + s_b * z_gal + s_c * z_gal**2 + s_d * z_gal**3
    gamma = r_conf_H / (1 + r_conf_H)
    b = be_a + be_b * z_gal + be_c * z_gal**2 + be_d * z_gal**3

    # Final beta expression
    beta = 5 * s - 1 + gamma * (2 / r_conf_H + gamma * ((der_conf_H / (conf_H**2)) - 1 / r_conf_H) - 1 - b)

    return beta


# ----
def plot_gw_bin_distributions(dl_GW, ndl_GW, merger_rate_tot, bin_edges_dl, n_bins_dl, output_path):

    for i in range(n_bins_dl):
        plt.plot(dl_GW / 1000, ndl_GW[i])
    for i in range(n_bins_dl):
        plt.axvline(bin_edges_dl[i], c='black', alpha=0.5)
    plt.axvline(bin_edges_dl[-1], c='black', alpha=0.5, label='bin edges')

    plt.xlabel(r'$d_L[Gpc]$')
    plt.ylabel(r'$w_i$')
    plt.title('GW bin distribution')
    plt.plot(dl_GW / 1000, merger_rate_tot, ls='--', alpha=0.8, color='red', label='total\ndistribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_path, 'GW_distr.pdf'), bbox_inches='tight')
    plt.close()

# ----
def plot_distribution_comparison(z_gal, nz_gal, gal_tot, bin_edges, n_bins_z, z_m_bin, z_M_bin,
                                 z_GW, ndl_GW, merger_rate_tot, bin_convert, n_bins_dl, output_path):

    fig = plt.figure(figsize=(18, 7), tight_layout=True)

    ax = fig.add_subplot(121)
    for i in range(n_bins_z):
        plt.plot(z_gal, nz_gal[i])
    for i in range(n_bins_z + 1):
        plt.axvline(bin_edges[i], c='black', alpha=0.5)
    plt.xlabel(r'$z$')
    plt.ylabel(r'$\frac{dN}{dzd\Omega}$')
    plt.title('Galaxy distribution')
    plt.xlim(z_m_bin - 0.3, z_M_bin + 0.3)
    plt.plot(z_gal, gal_tot, ls='--', alpha=0.5, color='red')

    ax = fig.add_subplot(122)
    for i in range(n_bins_dl):
        plt.plot(z_GW, ndl_GW[i])
    for i in range(n_bins_dl + 1):
        plt.axvline(bin_convert[i], c='black', alpha=0.5)
    plt.xlabel(r'$z$')
    plt.ylabel(r'$\frac{dN}{dzd\Omega}$')
    plt.title('Merger rate, fiducial model')
    plt.xlim(z_m_bin - 0.3, bin_convert[-1] + 0.5)
    plt.plot(z_GW, merger_rate_tot, ls='--', alpha=0.5, color='red')

    plt.savefig(os.path.join(output_path, 'distr_compare.pdf'), bbox_inches='tight')
    plt.close()


# ----
def compute_s_gal(z_gal, gal_det, sg0, sg1, sg2, sg3):
    """
    Compute the source evolution function s(z) for a galaxy population.

    Parameters:
    - z_gal: Redshift
    - gal_det: Galaxy detector type ('ska' or others)
    - sg0-sg3: Polynomial coefficients for s(z)

    Returns:
    - s_gal: Magnification bias s(z) value
    """
    if gal_det == 'ska':
        return (sg0 + sg1 * z_gal + sg2 * z_gal**2 + sg3 * z_gal**3) * z_gal
    else:
        return sg0 + sg1 * z_gal + sg2 * z_gal**2 + sg3 * z_gal**3


# ---


# ---------- 5-point stencils operating on a function returning (GG, GWGW, GGW) ----------
def _eval_triplet(F, x):
    out = F(x)
    if not (isinstance(out, (tuple, list)) and len(out) == 3):
        raise ValueError("derivative_args(x) must return (GG, GWGW, GGW).")
    return out

def deriv_5pt_forward_triplet(F, x0, h):
    f0_GG,   f0_GWGW,   f0_GGW   = _eval_triplet(F, x0)
    f1_GG,   f1_GWGW,   f1_GGW   = _eval_triplet(F, x0 + 1*h)
    f2_GG,   f2_GWGW,   f2_GGW   = _eval_triplet(F, x0 + 2*h)
    f3_GG,   f3_GWGW,   f3_GGW   = _eval_triplet(F, x0 + 3*h)
    f4_GG,   f4_GWGW,   f4_GGW   = _eval_triplet(F, x0 + 4*h)
    c = 1.0 / (12.0*h)
    der_GG   = c * (-25*f0_GG   + 48*f1_GG   - 36*f2_GG   + 16*f3_GG   - 3*f4_GG)
    der_GWGW = c * (-25*f0_GWGW + 48*f1_GWGW - 36*f2_GWGW + 16*f3_GWGW - 3*f4_GWGW)
    der_GGW  = c * (-25*f0_GGW  + 48*f1_GGW  - 36*f2_GGW  + 16*f3_GGW  - 3*f4_GGW)
    return der_GG, der_GWGW, der_GGW

def deriv_5pt_backward_triplet(F, x0, h):
    f0_GG,   f0_GWGW,   f0_GGW   = _eval_triplet(F, x0)
    f_1_GG,  f_1_GWGW,  f_1_GGW  = _eval_triplet(F, x0 - 1*h)
    f_2_GG,  f_2_GWGW,  f_2_GGW  = _eval_triplet(F, x0 - 2*h)
    f_3_GG,  f_3_GWGW,  f_3_GGW  = _eval_triplet(F, x0 - 3*h)
    f_4_GG,  f_4_GWGW,  f_4_GGW  = _eval_triplet(F, x0 - 4*h)
    c = 1.0 / (12.0*h)
    der_GG   = c * ( 25*f0_GG   - 48*f_1_GG   + 36*f_2_GG   - 16*f_3_GG   +  3*f_4_GG)
    der_GWGW = c * ( 25*f0_GWGW - 48*f_1_GWGW + 36*f_2_GWGW - 16*f_3_GWGW +  3*f_4_GWGW)
    der_GGW  = c * ( 25*f0_GGW  - 48*f_1_GGW  + 36*f_2_GGW  - 16*f_3_GGW  +  3*f_4_GGW)
    return der_GG, der_GWGW, der_GGW

def compute_parameter_derivatives(parameters, FLAGS, n_bins_z, n_bins_dl, covariance_matrices):
    """
    Uses 5-point O(h^4) ONLY for Alpha M, Alpha B (forward) and w_0 (backward).
    All other parameters use your existing nd.Derivative with their chosen method.
    """
    for param in parameters:
        print(f"\nComputing the derivative with respect to the {param['name']}...\n")

        step = abs(param["step"])
        name = param["name"]
        x0   = float(param["true_value"])
        F    = param["derivative_args"]  # F(x) -> (GG, GWGW, GGW)

        if name == "w_0":
            print(f"The method implemented for {name} is backward-5pt (O(h^4)).")
            der_GG, der_GWGW, der_GGW = deriv_5pt_backward_triplet(F, x0, step)

        elif name in ("Alpha M", "Alpha B"):
            print(f"The method implemented for {name} is forward-5pt (O(h^4)).")
            der_GG, der_GWGW, der_GGW = deriv_5pt_forward_triplet(F, x0, step)

        else:
            # Keep your previous numdifftools-based logic
            method = param.get("method",
                               "central" if name not in ("Alpha M","Alpha B","w_0") else "central")
            print(f"The method implemented for {name} is nd.Derivative with '{method}'.")
            partial_der_GG   = nd.Derivative(lambda x: F(x)[0], step=step, method=method)
            partial_der_GWGW = nd.Derivative(lambda x: F(x)[1], step=step, method=method)
            partial_der_GGW  = nd.Derivative(lambda x: F(x)[2], step=step, method=method)

            print(f"\n-------> Computing the derivative with respect to the {name}: G-G...")
            der_GG = partial_der_GG(x0)
            print(f"\n-------> Computing the derivative with respect to the {name}: GW-GW...")
            der_GWGW = partial_der_GWGW(x0)
            print(f"\n-------> Computing the derivative with respect to the {name}: G-GW...")
            der_GGW = partial_der_GGW(x0)


        # Save per-parameter derivative cubes for αB/αM (unchanged)
        if name == "Alpha B":
            np.save(os.path.join(FLAGS.fout, "dCl_GG_dalphaB.npy"),   der_GG)
            np.save(os.path.join(FLAGS.fout, "dCl_GWGW_dalphaB.npy"), der_GWGW)
            np.save(os.path.join(FLAGS.fout, "dCl_GGW_dalphaB.npy"),  der_GGW)

        if name == "Alpha M":
            np.save(os.path.join(FLAGS.fout, "dCl_GG_dalphaM.npy"),   der_GG)
            np.save(os.path.join(FLAGS.fout, "dCl_GWGW_dalphaM.npy"), der_GWGW)
            np.save(os.path.join(FLAGS.fout, "dCl_GGW_dalphaM.npy"),  der_GGW)


        # Build vector and covariance matrix (unchanged)
        der_vec     = fcc.vector_cl(cl_cross=der_GGW, cl_auto1=der_GG, cl_auto2=der_GWGW)
        der_cov_mat = fcc.covariance_matrix(der_vec, n_bins_z, n_bins_dl)

        covariance_matrices[param["key"]] = der_cov_mat
        np.save(os.path.join(FLAGS.fout, f"{param['key']}.npy"), der_cov_mat)
        print(f"\nThe derivative with respect to the {param['name']} has been computed.\n")

# ----
def generate_matrix(arr):
    """
    Generate a lower-triangular-like matrix based on input array.

    Parameters:
    - arr: 1D array of values

    Returns:
    - matrix: 2D matrix where each row is filled with earlier elements up to diagonal,
              and current element from diagonal onward
    """
    n = len(arr)
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i, :i + 1] = arr[:i + 1]  # Fill lower triangle including diagonal
        matrix[i, i + 1:] = arr[i]       # Fill upper triangle with diagonal value
    return matrix



# ----
def symm(matrix):
    """
    Symmetrize a square matrix using its upper triangle.

    Parameters:
    - matrix: 2D square matrix

    Returns:
    - symmetrized_matrix: Symmetric version of input matrix
    """
    return np.triu(matrix) + np.triu(matrix, k=1).T


# ----
def compute_lmin(z):
    """
    Compute the minimum ell based on redshift.

    Parameters:
    - z: redshift value

    Returns:
    - lmin: minimum multipole
    """
    conditions = [z < 0.5, (z >= 0.5) & (z < 0.75), (z >= 0.75) & (z < 1.25), z >= 1.25]
    values = [0, 5, 10, 15]
    return np.select(conditions, values, default=np.nan).astype(int)

# ----
def interpolate_all_derivatives(all_der, ll, ll_total):
    n_param, nb, _, _ = all_der.shape
    out = np.zeros((n_param, nb, nb, len(ll_total)))
    for p in range(n_param):
        for i in range(nb):
            for j in range(nb):
                f = si.interp1d(
                    ll, all_der[p, i, j],
                    kind='linear',
                    bounds_error=False,
                    # bounded fill (no extrapolation)
                    fill_value=(all_der[p, i, j, 0], all_der[p, i, j, -1])
                )
                out[p, i, j] = f(ll_total)
    return out

# ----
def apply_lmin_lmax_mask(all_der_total, bin_centers, ell_matrix):
    n_param, nb, _, _ = all_der_total.shape
    mask = np.ones_like(all_der_total)

    for p in range(n_param):
        for i in range(nb):
            for j in range(nb):
                z_min_ij = min(bin_centers[i], bin_centers[j])
                lmin = compute_lmin(z_min_ij)
                lmax = int(ell_matrix[i, j] - 5)

                if lmin > 0:
                    mask[p, i, j, :lmin] = 0
                if lmax <= lmin:
                    mask[p, i, j, :] = 0
                else:
                    mask[p, i, j, lmax:] = 0

    return all_der_total * mask

# ----
def process_derivatives(all_der, ll, ll_total, z_mean_gal, z_mean_GW, l_max_nl, l_max_bin, FLAGS):
    bin_centers = np.concatenate((z_mean_gal, z_mean_GW), axis=0)
    ell_max_total = np.concatenate((l_max_nl, l_max_bin), axis=0)

    # build then MIN, then SYMMETRIZE (order matches your inline block)
    ell_matrix = generate_matrix(ell_max_total)
    for i in range(len(ell_max_total)):
        for j in range(len(ell_max_total)):
            ell_matrix[i, j] = min(ell_matrix[i, j], ell_max_total[j])
    ell_matrix = symm(ell_matrix)

    all_der_total = interpolate_all_derivatives(all_der, ll, ll_total)
    all_der_total = apply_lmin_lmax_mask(all_der_total, bin_centers, ell_matrix)

    np.save(os.path.join(FLAGS.fout, 'all_der_total.npy'), all_der_total)
    print(f"Saved interpolated + masked derivative array to: {FLAGS.fout}/all_der_total.npy")
    return all_der_total

# ----

def rotate_fisher_Ob_to_ob(or_matrix, Ob=0.048, H0=67.7, pos={'H0': 0, 'Ob': 2}):
    """
    Rotate a Fisher matrix from (H0, Ob) basis to (H0, ob) basis,
    where ob = 1e4 * Ob / H0^2 is a derived parameter often used in cosmology.

    Parameters:
    - or_matrix: Original Fisher matrix (numpy.ndarray)
    - Ob: Fiducial value of baryon density parameter Ω_b
    - H0: Fiducial value of Hubble constant
    - pos: Dictionary indicating the indices of 'H0' and 'Ob' in the parameter vector

    Returns:
    - matrix: Rotated Fisher matrix in the (H0, ob) basis
    """
    nparams = or_matrix.shape[0]  # Total number of parameters
    rotMatrix = np.identity(nparams)  # Initialize with identity for untouched parameters

    # Jacobian for the change of variables: ob = 1e4 * Ob / H0^2
    # Partial derivatives:
    # ∂H0 = 1, ∂ob/∂H0 = -2 * Ob / H0
    # ∂Ob = 0, ∂ob/∂Ob = 1e4 / H0^2
    J_H0Ob_to_H0ob = np.array([
        [1, 0],
        [-2 * Ob / H0, 1e4 / H0 ** 2]
    ])

    # Insert Jacobian submatrix into full rotation matrix at appropriate indices
    rotMatrix[np.ix_([pos['H0'], pos['Ob']], [pos['H0'], pos['Ob']])] = J_H0Ob_to_H0ob

    # Rotate the Fisher matrix using F' = J^T F J
    matrix = rotMatrix.T @ or_matrix @ rotMatrix
    return matrix

def compute_and_print_sigma_params(fisher_inv):
    """
    Compute 1σ uncertainties (standard deviations) for cosmological parameters
    from the inverse of the Fisher matrix.

    Assumes parameters are ordered as:
    [H0, Omega_m, Omega_b, A_s, n_s, alpha_M, alpha_B, w_0, w_a]

    Parameters:
    - fisher_inv: Inverse Fisher matrix (covariance matrix)

    Returns:
    - Tuple of standard deviations: (σ_H0, σ_Omega_m, σ_Omega_b, σ_A_s, σ_n_s, σ_alpha_M, σ_alpha_B, σ_w_0, σ_w_a)
    """

    diag = np.diag(fisher_inv)

    def safe_sqrt(value, name):
        if value < 0:
            print(f"Warning: variance of {name} is negative ({value}), returning NaN.")
            return np.nan
        return np.sqrt(value)

    sigma_H0 = safe_sqrt(diag[0], "H0")
    sigma_Omega_m = safe_sqrt(diag[1], "Omega_m")
    sigma_Omega_b = safe_sqrt(diag[2], "Omega_b")
    sigma_A_s = safe_sqrt(diag[3], "A_s")
    sigma_n_s = safe_sqrt(diag[4], "n_s")
    sigma_alpha_M = safe_sqrt(diag[5], "alpha_M")
    sigma_alpha_B = safe_sqrt(diag[6], "alpha_B")
    sigma_w_0 = safe_sqrt(diag[7], "w_0")
    sigma_w_a = safe_sqrt(diag[8], "w_a")

    print("\nAbsolute 1σ uncertainties:\n")
    print(f"H_0       = {sigma_H0}")
    print(f"Omega_m   = {sigma_Omega_m}")
    print(f"Omega_b   = {sigma_Omega_b}")
    print(f"A_s       = {sigma_A_s}")
    print(f"n_s       = {sigma_n_s}")
    print(f"alpha_M   = {sigma_alpha_M}")
    print(f"alpha_B   = {sigma_alpha_B}")
    print(f"w_0       = {sigma_w_0}")
    print(f"w_a       = {sigma_w_a}")

    return (
        sigma_H0,
        sigma_Omega_m,
        sigma_Omega_b,
        sigma_A_s,
        sigma_n_s,
        sigma_alpha_M,
        sigma_alpha_B,
        sigma_w_0,
        sigma_w_a
    )



# ----
def compute_and_print_relative_errors(
    sigma_H0, sigma_omega, sigma_omega_b, sigma_As, sigma_ns,
    sigma_alpha_M, sigma_alpha_B, sigma_w_0, sigma_w_a,
    H0_true, Omega_m_true, Omega_b_true, A_s, n_s,
    alpha_M, alpha_B, w_0, w_a
):
    """
    Compute 2σ relative percentage errors for cosmological parameters.

    Parameters:
    - sigma_*: 1σ uncertainties of the respective parameters
    - *_true: Fiducial values of H0, Omega_m, Omega_b, etc.

    Returns:
    - Tuple of relative 2σ errors in percentage for each parameter.
    """

    def safe_relative_error(sigma, true_value, name):
        if true_value == 0:
            print(f"Warning: fiducial value of {name} is zero, cannot compute relative error.")
            return np.nan
        return 2 * sigma / true_value * 100

    # Compute relative errors safely
    rel_err_H0 = safe_relative_error(sigma_H0, H0_true, "H0")
    rel_err_Omega_m = safe_relative_error(sigma_omega, Omega_m_true, "Omega_m")
    rel_err_Omega_b = safe_relative_error(sigma_omega_b, Omega_b_true, "Omega_b")
    rel_err_As = safe_relative_error(sigma_As, A_s, "A_s")
    rel_err_ns = safe_relative_error(sigma_ns, n_s, "n_s")
    rel_err_alpha_M = safe_relative_error(sigma_alpha_M, alpha_M, "alpha_M")
    rel_err_alpha_B = safe_relative_error(sigma_alpha_B, alpha_B, "alpha_B")
    rel_err_w_0 = safe_relative_error(sigma_w_0, w_0, "w_0")
    rel_err_w_a = safe_relative_error(sigma_w_a, w_a, "w_a")

    # Print relative errors
    print("\nRelative 2σ errors (in %):\n")
    print(f"H_0       = {rel_err_H0}")
    print(f"Omega_m   = {rel_err_Omega_m}")
    print(f"Omega_b   = {rel_err_Omega_b}")
    print(f"A_s       = {rel_err_As}")
    print(f"n_s       = {rel_err_ns}")
    print(f"alpha_M   = {rel_err_alpha_M}")
    print(f"alpha_B   = {rel_err_alpha_B}")
    print(f"w_0       = {rel_err_w_0}")
    print(f"w_a       = {rel_err_w_a}")

    return (
        rel_err_H0,
        rel_err_Omega_m,
        rel_err_Omega_b,
        rel_err_As,
        rel_err_ns,
        rel_err_alpha_M,
        rel_err_alpha_B,
        rel_err_w_0,
        rel_err_w_a
    )



# ----
def plot_gaussian_contour(H0_true, Omega_m_true, sigma_H0, sigma_omega, fisher_marg, GW_det, output_path):

    mean = np.array([H0_true, Omega_m_true])
    cov_matrix = fisher_marg
    scale = 0.05

    x, y = np.meshgrid(
        np.linspace(H0_true - scale * H0_true, H0_true + scale * H0_true, 200),
        np.linspace(Omega_m_true - scale * Omega_m_true, Omega_m_true + scale * Omega_m_true, 200)
    )
    pos = np.dstack((x, y))
    pdf = multivariate_normal(mean, cov_matrix).pdf(pos)
    pdf /= np.max(pdf)

    confidence_level = 0.68
    contour = plt.contour(x, y, pdf, levels=[confidence_level], colors='blue')
    plt.clabel(contour, fontsize=10, fmt='%0.2f')
    plt.contourf(x, y, pdf, levels=[confidence_level, 1000], cmap='Blues', alpha=0.3)

    perc_err_H0 = 2 * sigma_H0 / H0_true * 100
    perc_err_Om = 2 * sigma_omega / Omega_m_true * 100
    plt.scatter(H0_true, Omega_m_true, c='blue', s=15,
                label='$\sigma_{H_0}/H_0=%.1f\%%$\n$\sigma_{\Omega_m}/\Omega_m=%.1f\%%$' % (perc_err_H0, perc_err_Om))

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=15)
    plt.xlabel('$H_0$')
    plt.ylabel('$\Omega_m$')
    plt.title('%s' % GW_det)
    plt.savefig(os.path.join(output_path, 'contour_plot.pdf'), bbox_inches='tight')
    plt.close()





'''
# ---------- 5-point stencils operating on a function returning (GG, GWGW, GGW) ----------
def _eval_triplet(F, x):
    """F(x) must return a tuple/list: (GG, GWGW, GGW)"""
    out = F(x)
    if not (isinstance(out, (tuple, list)) and len(out) == 3):
        raise ValueError("derivative_args(x) must return (GG, GWGW, GGW).")
    return out

def deriv_5pt_forward_triplet(F, x0, h):
    f0_GG,   f0_GWGW,   f0_GGW   = _eval_triplet(F, x0)
    f1_GG,   f1_GWGW,   f1_GGW   = _eval_triplet(F, x0 + 1*h)
    f2_GG,   f2_GWGW,   f2_GGW   = _eval_triplet(F, x0 + 2*h)
    f3_GG,   f3_GWGW,   f3_GGW   = _eval_triplet(F, x0 + 3*h)
    f4_GG,   f4_GWGW,   f4_GGW   = _eval_triplet(F, x0 + 4*h)
    c = 1.0 / (12.0*h)
    der_GG   = c * (-25*f0_GG   + 48*f1_GG   - 36*f2_GG   + 16*f3_GG   - 3*f4_GG)
    der_GWGW = c * (-25*f0_GWGW + 48*f1_GWGW - 36*f2_GWGW + 16*f3_GWGW - 3*f4_GWGW)
    der_GGW  = c * (-25*f0_GGW  + 48*f1_GGW  - 36*f2_GGW  + 16*f3_GGW  - 3*f4_GGW)
    return der_GG, der_GWGW, der_GGW

def deriv_5pt_backward_triplet(F, x0, h):
    f0_GG,   f0_GWGW,   f0_GGW   = _eval_triplet(F, x0)
    f_1_GG,  f_1_GWGW,  f_1_GGW  = _eval_triplet(F, x0 - 1*h)
    f_2_GG,  f_2_GWGW,  f_2_GGW  = _eval_triplet(F, x0 - 2*h)
    f_3_GG,  f_3_GWGW,  f_3_GGW  = _eval_triplet(F, x0 - 3*h)
    f_4_GG,  f_4_GWGW,  f_4_GGW  = _eval_triplet(F, x0 - 4*h)
    c = 1.0 / (12.0*h)
    der_GG   = c * ( 25*f0_GG   - 48*f_1_GG   + 36*f_2_GG   - 16*f_3_GG   +  3*f_4_GG)
    der_GWGW = c * ( 25*f0_GWGW - 48*f_1_GWGW + 36*f_2_GWGW - 16*f_3_GWGW +  3*f_4_GWGW)
    der_GGW  = c * ( 25*f0_GGW  - 48*f_1_GGW  + 36*f_2_GGW  - 16*f_3_GGW  +  3*f_4_GGW)
    return der_GG, der_GWGW, der_GGW

def deriv_5pt_central_triplet(F, x0, h):
    f_m2_GG,  f_m2_GWGW,  f_m2_GGW  = _eval_triplet(F, x0 - 2*h)
    f_m1_GG,  f_m1_GWGW,  f_m1_GGW  = _eval_triplet(F, x0 - 1*h)
    f_p1_GG,  f_p1_GWGW,  f_p1_GGW  = _eval_triplet(F, x0 + 1*h)
    f_p2_GG,  f_p2_GWGW,  f_p2_GGW  = _eval_triplet(F, x0 + 2*h)
    c = 1.0 / (12.0*h)
    der_GG   = c * ( f_m2_GG   - 8*f_m1_GG   + 8*f_p1_GG   - f_p2_GG)
    der_GWGW = c * ( f_m2_GWGW - 8*f_m1_GWGW + 8*f_p1_GWGW - f_p2_GWGW)
    der_GGW  = c * ( f_m2_GGW  - 8*f_m1_GGW  + 8*f_p1_GGW  - f_p2_GGW)
    return der_GG, der_GWGW, der_GGW

def _pick_h(x0, h_user, order=4):
    """Use user step if >0, else a scale-aware fallback."""
    if h_user is not None and h_user > 0:
        return float(h_user)
    eps = np.finfo(float).eps
    expo = {1:0.5, 2:1/3, 4:0.2}.get(order, 0.2)
    return (abs(x0) + 1.0) * (eps ** expo)

# ------------------------ Main function (5-point for αM, αB, w0) ------------------------
def compute_parameter_derivatives(parameters, FLAGS, n_bins_z, n_bins_dl, covariance_matrices):
    """
    Compute numerical derivatives of the angular power spectra with respect to cosmological parameters.
    Uses 5-point O(h^4) stencils:
      - forward for Alpha M / Alpha B
      - backward for w_0
      - central for all the others
    """
    for param in parameters:
        name = param["name"]
        x0   = float(param["true_value"])
        h    = _pick_h(x0, abs(param.get("step", 0.0)), order=4)
        F    = param["derivative_args"]  # F(x) -> (GG, GWGW, GGW)

        print(f"\nComputing the derivative with respect to {name} ...")
        if name == "w_0":
            method = "backward-5pt (O(h^4))"
            der_GG, der_GWGW, der_GGW = deriv_5pt_backward_triplet(F, x0, h)
        elif name in ("Alpha M", "Alpha B"):
            method = "forward-5pt (O(h^4))"
            # If the fiducial is exactly 0, forward is ideal; if >0, forward also keeps positivity.
            # (If you prefer, you can remap to log and use central; this keeps your original constraint logic.)
            der_GG, der_GWGW, der_GGW = deriv_5pt_forward_triplet(F, x0, h)
        else:
            method = "central-5pt (O(h^4))"
            der_GG, der_GWGW, der_GGW = deriv_5pt_central_triplet(F, x0, h)

        print(f"Method for {name}: {method} with h={h:.3e}")

        # Optional: save per-parameter derivative cubes for αB/αM (as in your code)
        if name == "Alpha B":
            np.save(os.path.join(FLAGS.fout, "dCl_GG_dalphaB.npy"),   der_GG)
            np.save(os.path.join(FLAGS.fout, "dCl_GWGW_dalphaB.npy"), der_GWGW)
            np.save(os.path.join(FLAGS.fout, "dCl_GGW_dalphaB.npy"),  der_GGW)

        if name == "Alpha M":
            np.save(os.path.join(FLAGS.fout, "dCl_GG_dalphaM.npy"),   der_GG)
            np.save(os.path.join(FLAGS.fout, "dCl_GWGW_dalphaM.npy"), der_GWGW)
            np.save(os.path.join(FLAGS.fout, "dCl_GGW_dalphaM.npy"),  der_GGW)

        # Build vector and covariance matrix (unchanged)
        der_vec     = fcc.vector_cl(cl_cross=der_GGW, cl_auto1=der_GG, cl_auto2=der_GWGW)
        der_cov_mat = fcc.covariance_matrix(der_vec, n_bins_z, n_bins_dl)

        covariance_matrices[param["key"]] = der_cov_mat
        np.save(os.path.join(FLAGS.fout, f"{param['key']}.npy"), der_cov_mat)
        print(f"Done: derivative wrt {name}.\n")

def compute_parameter_derivatives(parameters, FLAGS, n_bins_z, n_bins_dl, covariance_matrices):
    """
    Compute numerical derivatives of the angular power spectra with respect to cosmological parameters.

    Parameters:
    - parameters: List of parameter dicts with keys 'name', 'step', 'true_value', 'derivative_args'
    - FLAGS: Config object with output path
    - n_bins_z: Number of redshift bins
    - n_bins_dl: Number of luminosity distance bins
    - covariance_matrices: Dictionary to store results
    """
    for param in parameters:
        print(f"\nComputing the derivative with respect to the {param['name']}...\n")

        step = abs(param["step"])
        name = param["name"]

        if name == "w_0":
            method = param.get("method", "backward")  # keeps w_0<-1
        elif name in ("Alpha M", "Alpha B"):
            method = param.get("method", "forward")  # so they are positive define
        else:
            method = param.get("method", "central")

        print(f'\nThe method implemented for {name} is {method}.')
        # Define numerical derivatives for GG, GWGW, and GGW spectra
        partial_der_GG = nd.Derivative(lambda x: param["derivative_args"](x)[0], step=step, method=method)
        partial_der_GWGW = nd.Derivative(lambda x: param["derivative_args"](x)[1], step=step, method=method)
        partial_der_GGW = nd.Derivative(lambda x: param["derivative_args"](x)[2], step=step, method=method)

        # Evaluate the derivatives at the parameter's fiducial value
        print(f"\n-------> Computing the derivative with respect to the {param['name']}: G-G...")
        der_GG = partial_der_GG(param["true_value"])
        print(f"\n-------> Computing the derivative with respect to the {param['name']}: GW-GW...")
        der_GWGW = partial_der_GWGW(param["true_value"])
        print(f"\n-------> Computing the derivative with respect to the {param['name']}: G-GW...")
        der_GGW = partial_der_GGW(param["true_value"])


        if name=="Alpha B":
            np.save(os.path.join(FLAGS.fout, "dCl_GG_dalphaB.npy"), der_GG)
            np.save(os.path.join(FLAGS.fout, "dCl_GWGW_dalphaB.npy"), der_GWGW)
            np.save(os.path.join(FLAGS.fout, "dCl_GGW_dalphaB.npy"), der_GGW)

        if name=="Alpha M":
            np.save(os.path.join(FLAGS.fout, "dCl_GG_dalphaM.npy"), der_GG)
            np.save(os.path.join(FLAGS.fout, "dCl_GWGW_dalphaM.npy"), der_GWGW)
            np.save(os.path.join(FLAGS.fout, "dCl_GGW_dalphaM.npy"), der_GGW)

        # Build vector and covariance matrix
        der_vec = fcc.vector_cl(cl_cross=der_GGW, cl_auto1=der_GG, cl_auto2=der_GWGW)
        der_cov_mat = fcc.covariance_matrix(der_vec, n_bins_z, n_bins_dl)

        # Store in dictionary and save to disk
        covariance_matrices[param["key"]] = der_cov_mat
        np.save(os.path.join(FLAGS.fout, f"{param['key']}.npy"), der_cov_mat)
        print(f"\nThe derivative with respect to the {param['name']} has been computed.\n")
'''
