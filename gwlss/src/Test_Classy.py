import sys
sys.path.insert(0, "/home/cptsu4/santoni/home_cpt/Home_CPT/hi_class/python/build/lib.linux-x86_64-cpython-311")
import numpy as np
import os
os.makedirs("output", exist_ok=True)

import classy
#print("USING CLASSY FROM:", classy.__file__)

from classy import Class


import matplotlib.pyplot as plt
import numpy as np
import itertools
from copy import deepcopy
from classy import Class
from scipy.interpolate import RegularGridInterpolator

# -------------------------------
# 1. Scan di stabilità
# -------------------------------
def scan_stability(alpha_B, alpha_M, base_params):
    """
    Serial, stable scan. Returns structured numpy array:
    ('alpha_B', float), ('alpha_M', float), ('stable', int)
    """
    param_grid = list(itertools.product(alpha_B, alpha_M))
    out = np.empty(len(param_grid), dtype=[('alpha_B', float),
                                           ('alpha_M', float),
                                           ('stable', int)])

    for k, (b, m) in enumerate(param_grid):
        if k in [16000,32000,48000,64000,80000,96000,112000,128000,144000,159000]: #[250,500,750,1000,1500,2000,2250,2400]: #[2500,5000,6000,7000,8000,9000,9500]: #[5000,10000,15000,16000,17000,18000,19000]:
            print(f"Testing at", ((k)/10000)*100,"%")
            print(f"Testing alpha_B={b}, alpha_M={m}")

        # === Regions with output = 0 (Unstable) ===
        # ---- Blue regions (0) ----
        if (b >= -6 and b <= 6 and m >= -10 and m <= -3.39130435):
            out[k] = (b, m, 0); continue

        if (b >= -6 and b <= -4.08695652 and m >= -3.39130435 and m <= 6):
            out[k] = (b, m, 0); continue

        if (b >= 4.08695652 and b <= 6 and m >= 0.7826087 and m <= 6):
            out[k] = (b, m, 0); continue

        if (b >= 3.73913043 and b <= 4.08695652 and m >= 2.5217391 and m <= 6):
            out[k] = (b, m, 0); continue

        if (b >= 3.56521739 and b <= 3.73913043 and m >= 4.26086957 and m <= 6):
            out[k] = (b, m, 0); continue

        if (b >= -4.08695652 and b <= -2 and m >= -3.39130435 and m <= 2.86956522):
            out[k] = (b, m, 0); continue

        if (b >= -2 and b <= 3.91304348 and m >= -3.39130435 and m <= -2.5):
            out[k] = (b, m, 0); continue

        if (b >= 3.91304348 and b <= 4.43478261 and m >= -3.39130435 and m <= -2.86956522):
            out[k] = (b, m, 0); continue

        if (b >= 4.43478261 and b <= 4.7826087 and m >= -3.39130435 and m <= -3.04347826):
            out[k] = (b, m, 0); continue

        if (b >= -2 and b <= 1.65217391 and m >= -2.52173913 and m <= -1.13043478):
            out[k] = (b, m, 0); continue

        if (b >= 4.43478261 and b <= 6 and m >= -0.60869565 and m <= 0.7826087):
            out[k] = (b, m, 0); continue

        if (b >= 1.65217391 and b <= 2.86956522 and m >= -2.52173913 and m <= -1.82608696):
            out[k] = (b, m, 0); continue

        if (b >= 1.65217391 and b <= 2.34782609 and m >= -1.82608696 and m <= -1.47826087):
            out[k] = (b, m, 0); continue

        if (b >= 4.95652174 and b <= 6 and m >= -1.82608696 and m <= -0.60869565):
            out[k] = (b, m, 0); continue

        if (b >= -2 and b <= -0.4347826 and m >= -1.13043478 and m <= 0.4347826):
            out[k] = (b, m, 0); continue

        if (b >= -2 and b <= -1.13043478 and m >= 0.4347826 and m <= 1.65217391):
            out[k] = (b, m, 0); continue

        if (b >= -1.13043478 and b <= -0.7826087 and m >= 0.4347826 and m <= 1.13043478):
            out[k] = (b, m, 0); continue

        if (b >= -2 and b <= -1.65217391 and m >= 1.65217391 and m <= 2.34782609):
            out[k] = (b, m, 0); continue

        if (b >= -4.08695652 and b <= -3.2173913 and m >= 2.86956522 and m <= 4.7826087):
            out[k] = (b, m, 0); continue

        if (b >= -4.08695652 and b <= -3.56521739 and m >= 4.7826087 and m <= 5.30434783):
            out[k] = (b, m, 0); continue

        if (b >= -3.2173913 and b <= -2.69565217 and m >= 2.86956522 and m <= 3.91304348):
            out[k] = (b, m, 0); continue

        if (b >= -2.69565217 and b <= -2.34782609 and m >= 2.86956522 and m <= 3.39130435):
            out[k] = (b, m, 0); continue

        if (b >= 5.65217391 and b <= 6 and m >= -3.39130435 and m <= -1.82608696):
            out[k] = (b, m, 0); continue

        if (b >= 5.30434783 and b <= 5.65217391 and m >= -2.52173913 and m <= -1.82608696):
            out[k] = (b, m, 0); continue

        if (b >= 4.7826087 and b <= 4.95652174 and m >= -1.47826087 and m <= -0.60869565):
            out[k] = (b, m, 0); continue

        if (b >= -0.43478261 and b <= 0.95652174 and m >= -1.13043478 and m <= -0.60869565):
            out[k] = (b, m, 0); continue

        if (b >= 0.95652174 and b <= 1.30434783 and m >= -1.13043478 and m <= -0.7826087):
            out[k] = (b, m, 0); continue

        if (b >= -0.43478261 and b <= 0.43478261 and m >= -0.60869565 and m <= -0.26086957):
            out[k] = (b, m, 0); continue


        # ---- Black regions (1) ----
        if (b >= 0 and b <= 3.39130435 and m >= -0.08695652 and m <= 6):
            out[k] = (b, m, 1); continue

        if (b >= -1.65217391 and b <= 0.2 and m >= 2.5 and m <= 6):
            out[k] = (b, m, 1); continue

        if (b >= 2.5 and b <= 4.26086957 and m >= -1.47826087 and m <= -0.60869565):
            out[k] = (b, m, 1); continue

        if (b >= -3.23076923 and b <= -1.65217391 and m >= 4.95652174 and m <= 6):
            out[k] = (b, m, 1); continue

        if (b >= -2.69565217 and b <= -1.65217391 and m >= 4.08695652 and m <= 5):
            out[k] = (b, m, 1); continue

        if (b >= 4.08695652 and b <= 4.95652174 and m >= -2.52173913 and m <= -2):
            out[k] = (b, m, 1); continue

        if (b >= 3.39130435 and b <= 3.73913043 and m >= -0.60869565 and m <= 1.65217391):
            out[k] = (b, m, 1); continue

        if (b >= 1.13043478 and b <= 3.39130435 and m >= -0.60869565 and m <= -0.08695652):
            out[k] = (b, m, 1); continue

        if (b >= -0.60869565 and b <= 0 and m >= 0.95652174 and m <= 2.52173913):
            out[k] = (b, m, 1); continue

        if (b >= 2.86956522 and b <= 4.60869565 and m >= -1.65217391 and m <= -1.47826087):
            out[k] = (b, m, 1); continue

        if (b >= -2.17391304 and b <= -1.65217391 and m >= 3.39130435 and m <= 4.08695652):
            out[k] = (b, m, 1); continue

        if (b >= 3.73913043 and b <= 4.08695652 and m >= -0.60869565 and m <= 0.60869565):
            out[k] = (b, m, 1); continue

        if (b >= -3.73913043 and b <= -3.2173913 and m >= 5.65217391 and m <= 6):
            out[k] = (b, m, 1); continue

        if (b >= 4.95652174 and b <= 5.30434783 and m >= -3.04347826 and m <= -2.69565217):
            out[k] = (b, m, 1); continue

        if (b >= 3.04347826 and b <= 4.7826087 and m >= -1.82608696 and m <= -1.65217391):
            out[k] = (b, m, 1); continue

        if (b >= 1.65217391 and b <= 2.5 and m >= -0.95652174 and m <= -0.60869565):
            out[k] = (b, m, 1); continue

        if (b >= -1.30434783 and b <= -0.60869565 and m >= 2 and m <= 2.52173913):
            out[k] = (b, m, 1); continue



        try:
            params = deepcopy(base_params)
            params['parameters_smg'] = f"10.0,{b},{m},0.0,1.0"

            cosmo = Class()
            cosmo.set(params)
            cosmo.compute()
            flag = 1
        except Exception as e:
            #print(f"Failed at (b={b}, m={m}): {e}", flush=True)
            flag = 0
        finally:
            try:
                cosmo.struct_cleanup()
                cosmo.empty()
            except Exception:
                pass  # Safe cleanup

        out[k] = (b, m, flag)

    return out

# -------------------------------
# 2. Costruzione interpolante 2D
# -------------------------------
def build_stability_interpolator(results, alpha_B_grid, alpha_M_grid, method="linear"):
    """
    Costruisce un interpolante 2D della stabilità a partire dai risultati di scan_stability.
    """
    Z = np.zeros((len(alpha_B_grid), len(alpha_M_grid)))
    for entry in results:
        b, m, flag = entry
        i = np.where(np.isclose(alpha_B_grid, b))[0][0]
        j = np.where(np.isclose(alpha_M_grid, m))[0][0]
        Z[i, j] = flag

    interpolant = RegularGridInterpolator(
        (alpha_B_grid, alpha_M_grid), Z, method=method,
        bounds_error=False, fill_value=0.0
    )
    return interpolant, Z

# -------------------------------
# 3. utilizzo
# -------------------------------


# Parametri cosmologici di base (esempio minimale, aggiungi i tuoi)
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
    'Omega_Lambda': 0.0,
    'Omega_fld': 0.0,
    'Omega_smg': -1,
    #'omega_cdm': 0.262,
    #'omega_ncdm' : 0.0,
    #'T_wdm': 0.71611,

    # --- Primordial Power Spectrum ---
    'A_s': 2.100549e-09,
    'n_s': 0.9660499,
    'tau_reio': 0.05430842,
    'Pk_ini_type': 'analytic_Pk',
    'k_pivot': 0.05,
    'alpha_s': 0.0,

    # --- Gravity Model ---
    # x_k, x_b, x_m, x_t, M*^2_ini
    'gravity_model': 'propto_omega',
    'parameters_smg': '10.0, 0.0, 0.0, 0.0, 1.0',
    #'expansion_model' : 'wowa',
    #'expansion_smg' : '0.7, -1.0, 0.0',  #Omega_smg, w0, wa
    'expansion_model':'lcdm',
    'expansion_smg': '0.5',
    'want_lcmb_full_limber' : 'yes',

    # --- quasi-static approximation ---
    'method_qs_smg' : 'fully_dynamic',
    'z_fd_qs_smg' : 10.,
    'trigger_mass_qs_smg' : 1.e3,
    'trigger_rad_qs_smg' : 1.e3,
    'eps_s_qs_smg' : 0.01,
    'n_min_qs_smg' : 1e2,
    'n_max_qs_smg' : 1e4,

    # ---- precision parameters ---
    'start_small_k_at_tau_c_over_tau_h' : 1e-4,
    'start_large_k_at_tau_h_over_tau_k' : 1e-4,
    'perturbations_sampling_stepsize' : 0.05,
    'l_logstep' : 1.045,
    'l_linstep' : 50,

    # --- Modified Gravity: Stability ---
    'output_background_smg': 1,    #1 -> alpha functions, stability parameters (c_s^2, D)
    'skip_stability_tests_smg': 'no',
    'cs2_safe_smg': 0.0,
    'D_safe_smg': 0.0,
    'ct2_safe_smg': 0.0,
    'M2_safe_smg': 0.0,
    'a_min_stability_test_smg': 0,

    # --- Modes, Gauge ---,
    'ic': 'ad',
    'gauge': 'synchronous',


    # --- Fourier / Matter Power ---
    'P_k_max_h/Mpc': 100.0,
    'z_pk' : 0,
    'non_linear': 'halofit',
    'z_max_pk': 10.0,
    'lensing': 'no',
    'extra_metric_transfer_functions': 'yes',
    'output': 'mTk',

    # --- Verbosity ---
    'input_verbose': 1,
    'background_verbose': 1,
    'transfer_verbose': 1,
    'primordial_verbose': 1,
    'harmonic_verbose': 1,
}

# Griglie di parametri
alpha_B_grid = np.linspace(-6, 6, 400)
alpha_M_grid = np.linspace(-6, 6, 400)

# Calcola griglia di stabilità
results = scan_stability(alpha_B_grid, alpha_M_grid, COSMO_PARAMS)

# Quick tally
stable = results['stable'].sum()
print(f"Stable: {stable} / {len(results)}  (Unstable: {len(results)-stable})")


# Separate stable/unstable points
stable_mask = results['stable'] == 1
unstable_mask = results['stable'] == 0

plt.figure(figsize=(6, 6))

# Stable in green, unstable in red
plt.scatter(results['alpha_M'][stable_mask],
            results['alpha_B'][stable_mask],
            c='green', label='Stable', marker='o')

plt.scatter(results['alpha_M'][unstable_mask],
            results['alpha_B'][unstable_mask],
            c='red', label='Unstable', marker='x')

plt.xlabel(r'$\alpha_M$')
plt.ylabel(r'$\alpha_B$')
plt.title('Stability regions')
plt.legend()
plt.grid(True)
plt.savefig('stability_regions.png')
plt.show()



# Costruisci interpolante
interp_stab, Z = build_stability_interpolator(results, alpha_B_grid, alpha_M_grid)

# -------------------------------
# 4. Salvataggio griglia
# -------------------------------
np.savez("stability_grid_100x100.npz",
         alpha_B=alpha_B_grid,
         alpha_M=alpha_M_grid,
         stable=Z)

# -------------------------------
# 5. Ricarica e ricostruzione
# -------------------------------
# data = np.load("stability_grid.npz")
# alpha_B_grid = data["alpha_B"]
# alpha_M_grid = data["alpha_M"]
# stable = data["stable"]
# interp_stab = RegularGridInterpolator((alpha_B_grid, alpha_M_grid),
#                                       stable, method="linear",
#                                       bounds_error=False, fill_value=0.0)

# Supponiamo di avere già:
# interp_stab, Z, alpha_B_grid, alpha_M_grid

def plot_stability_region(alpha_B_grid, alpha_M_grid, Z, interp_stab=None,
                          title="Stability region", savefile=True):
    """
    Plotta la mappa di stabilità.

    Parameters
    ----------
    alpha_B_grid : array
        Griglia 1D di alpha_B usata nello scan.
    alpha_M_grid : array
        Griglia 1D di alpha_M usata nello scan.
    Z : 2D array
        Matrice di stabilità (0/1) di shape (len(alpha_B_grid), len(alpha_M_grid)).
    interp_stab : callable, opzionale
        Interpolante (se vuoi mostrare anche la versione lisciata).
    title : str
        Titolo del plot.
    savefile : str o None
        Se fornito, salva il plot su file.
    """
    fig, ax = plt.subplots(figsize=(6,5))

    # Heatmap discreta della griglia originale
    im = ax.imshow(Z.T, origin="lower",
                   extent=[alpha_B_grid.min(), alpha_B_grid.max(),
                           alpha_M_grid.min(), alpha_M_grid.max()],
                   aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Stability (0=unstable, 1=stable)")

    # Se vuoi mostrare una versione interpolata più liscia
    if interp_stab is not None:
        b_fine = np.linspace(alpha_B_grid.min(), alpha_B_grid.max(), 200)
        m_fine = np.linspace(alpha_M_grid.min(), alpha_M_grid.max(), 200)
        B, M = np.meshgrid(b_fine, m_fine, indexing="ij")
        Z_fine = interp_stab(np.column_stack([B.ravel(), M.ravel()])).reshape(B.shape)
        ax.contour(B, M, Z_fine, levels=[0.5], colors="k", linewidths=1.2)

    ax.set_xlabel(r"$\alpha_B$")
    ax.set_ylabel(r"$\alpha_M$")
    ax.set_title(title)

    if savefile:
        plt.savefig(savefile, bbox_inches="tight", dpi=150)
    plt.show()

# Uso:
plot_stability_region(alpha_B_grid, alpha_M_grid, Z, interp_stab)


'''
import multiprocessing

print("Number of CPU cores:", multiprocessing.cpu_count())

from concurrent.futures import ThreadPoolExecutor

import itertools
from functools import partial

def _compute_stability_full(bm, base_params):
    b, m = bm
    params = base_params.copy()  # Copy to avoid cross-process issues
    params['parameters_smg'] = f"10.0,{b},{m},0.0,1.0"

    cosmo = Class()
    cosmo.set(params)

    try:
        cosmo.compute()
        flag = 1
    except Exception:
        flag = 0
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()

    return (b, m, flag)

def scan_stability(alpha_B, alpha_M, base_params):
    """
    Parallelized stability scan. Returns structured numpy array with fields:
    ('alpha_B', float), ('alpha_M', float), ('stable', int)
    """
    param_grid = list(itertools.product(alpha_B, alpha_M))
    func = partial(_compute_stability_full, base_params=base_params)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(func, param_grid))

    out = np.array(results, dtype=[('alpha_B', float), ('alpha_M', float), ('stable', int)])
    return out


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
    'Omega_Lambda': 0.0,
    'Omega_fld': 0.0,
    'Omega_smg': -1,
    #'omega_cdm': 0.262,
    #'omega_ncdm' : 0.0,
    #'T_wdm': 0.71611,

    # --- Primordial Power Spectrum ---
    'A_s': 2.100549e-09,
    'n_s': 0.9660499,
    'tau_reio': 0.05430842,
    'Pk_ini_type': 'analytic_Pk',
    'k_pivot': 0.05,
    'alpha_s': 0.0,

    # --- Gravity Model ---
    # x_k, x_b, x_m, x_t, M*^2_ini
    'gravity_model': 'propto_omega',
    'parameters_smg': '10.0, 1.0, 1.5, 0.0, 1.0',
    #'expansion_model' : 'wowa',
    #'expansion_smg' : '0.7, -1.0, 0.0',  #Omega_smg, w0, wa
    'expansion_model':'lcdm',
    'expansion_smg': '0.5',
    'want_lcmb_full_limber' : 'yes',

    # --- quasi-static approximation ---
    'method_qs_smg' : 'fully_dynamic',
    'z_fd_qs_smg' : 10.,
    'trigger_mass_qs_smg' : 1.e3,
    'trigger_rad_qs_smg' : 1.e3,
    'eps_s_qs_smg' : 0.01,
    'n_min_qs_smg' : 1e2,
    'n_max_qs_smg' : 1e4,

    # ---- precision parameters ---
    'start_small_k_at_tau_c_over_tau_h' : 1e-4,
    'start_large_k_at_tau_h_over_tau_k' : 1e-4,
    'perturbations_sampling_stepsize' : 0.05,
    'l_logstep' : 1.045,
    'l_linstep' : 50,

    # --- Modified Gravity: Stability ---
    'output_background_smg': 1,    #1 -> alpha functions, stability parameters (c_s^2, D)
    'skip_stability_tests_smg': 'no',
    'cs2_safe_smg': 0.0,
    'D_safe_smg': 0.0,
    'ct2_safe_smg': 0.0,
    'M2_safe_smg': 0.0,
    'a_min_stability_test_smg': 0,

    # --- Modes, Gauge ---,
    'ic': 'ad',
    'gauge': 'synchronous',


    # --- Fourier / Matter Power ---
    'P_k_max_h/Mpc': 100.0,
    'z_pk' : 0,
    'non_linear': 'halofit',
    'z_max_pk': 10.0,
    'lensing': 'no',
    'extra_metric_transfer_functions': 'yes',
    'output': 'mTk',

    # --- Verbosity ---
    'input_verbose': 1,
    'background_verbose': 1,
    'transfer_verbose': 1,
    'primordial_verbose': 1,
    'harmonic_verbose': 1,
}
# Reuse your COSMO_PARAMS as the base:
results = scan_stability(
    alpha_B=np.linspace(0, 6, 2),
    alpha_M=np.linspace(0, 6,2),
    base_params=COSMO_PARAMS
)

# Quick tally
stable = results['stable'].sum()
print(f"Stable: {stable} / {len(results)}  (Unstable: {len(results)-stable})")


import matplotlib.pyplot as plt

# Separate stable/unstable points
stable_mask = results['stable'] == 1
unstable_mask = results['stable'] == 0

plt.figure(figsize=(6, 6))

# Stable in green, unstable in red
plt.scatter(results['alpha_M'][stable_mask],
            results['alpha_B'][stable_mask],
            c='green', label='Stable', marker='o')

plt.scatter(results['alpha_M'][unstable_mask],
            results['alpha_B'][unstable_mask],
            c='red', label='Unstable', marker='x')

plt.xlabel(r'$\alpha_M$')
plt.ylabel(r'$\alpha_B$')
plt.title('Stability regions')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('stability_regions.png')


# Instantiate Class object
cosmo = Class()

#print(dir(Class))
#print()
#help(cosmo.comoving_distance)
#help(cosmo.get_qs_functions_at_k_and_z_qs_smg)
#help(cosmo.Hubble)
#help(cosmo.set_default)
#help(cosmo.state)
#help(cosmo.get_current_derived_parameters)
#help(cosmo.get_background)
#help(cosmo.get_perturbations)
#help(cosmo.get_sources)

COSMO_PARAMS = {
    # --- Background & Thermodynamics ---
    'h': 0.67810,
    'T_cmb': 2.7255,
    'YHe': 0.2454,  # replaced invalid 'BBN' , 0.2454
    'N_ur': 3.044,
    'N_ncdm':0,
    #'m_ncdm' : 0.0,
    'Omega_m': 0.31,
    'Omega_b': 0.02238280,
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
    'parameters_smg': '1.0, 0.92, 1.05, 0.0, 1.0',
    'expansion_model': 'wowa',
    #Omega_smg, w0, wa
    'expansion_smg': '0.5, -1.0, 0.0',
    
    # --- quasi-static approximation ---
    'method_qs_smg' : 'fully_dynamic',
    'z_fd_qs_smg' : 10.,
    'trigger_mass_qs_smg' : 1.e3,
    'trigger_rad_qs_smg' : 1.e3,
    'eps_s_qs_smg' : 0.01,
    'n_min_qs_smg' : 1e2,
    'n_max_qs_smg' : 1e4,
    
    # ---- precision parameters ---
    'start_small_k_at_tau_c_over_tau_h' : 1e-4,
    'start_large_k_at_tau_h_over_tau_k' : 1e-4,
    'perturbations_sampling_stepsize' : 0.05,
    'l_logstep' : 1.045,
    'l_linstep' : 50,

    # --- Modified Gravity: Stability ---
    'output_background_smg': 1,
    'skip_stability_tests_smg': 'no',
    'cs2_safe_smg': 0.0,
    'D_safe_smg': 0.0,
    'ct2_safe_smg': 0.0,
    'M2_safe_smg': 0.0,
    'a_min_stability_test_smg': 0,

    # --- MG Dynamics ---
    'hubble_evolution': 'y',
    'hubble_friction': 3.0,
    
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
    'start_small_k_at_tau_c_over_tau_h': 1e-4,
    'start_large_k_at_tau_h_over_tau_k': 1e-4,
    'perturbations_sampling_stepsize': 0.05,
    'l_logstep': 1.045,
    'l_linstep': 50,

    # --- Modes, Gauge ---
    'modes': 's',
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
    'P_k_max_h/Mpc': 1.0,
    'z_pk' : 0,
    'non_linear': 'halofit',
    'lensing': 'no',
    'extra_metric_transfer_functions': 'yes',
    'output': 'dTk,vTk,mPk',

    # --- Spectral Distortions (PIXIE etc.) ---
    'sd_branching_approx': 'exact',
    'sd_PCA_size': 2,
    'sd_detector_name': 'PIXIE',
    'sd_only_exotic': 'no',
    'sd_include_g_distortion': 'no',
    'sd_add_y': 0.0,
    'sd_add_mu': 0.0,
    'include_SZ_effect': 'no',

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

cosmo.set(COSMO_PARAMS)
cosmo.compute()
help(cosmo.comoving_distance)

print(cosmo.get_background())
#print(cosmo.get_current_derived_parameters())


cosmo.struct_cleanup()
cosmo.empty()
'''


