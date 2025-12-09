import numpy as np
import time
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d
import astropy.cosmology.units as cu
from astropy import units as u
import astropy.constants as const
import scipy.special as ss
import scipy.integrate as sint
from scipy.integrate import trapezoid
from scipy.integrate import cumulative_trapezoid
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
from concurrent.futures import ProcessPoolExecutor

def merger_rate_dl_new(bg,dl, bin_dl, log_dl,log_delta_dl, A, Z_0, Alpha, Beta,C, normalize=True):

    def dL_from_C(C, z,alpha_interp):
        """
        Luminosity distance d_L(z) in Mpc from your colibri cosmology `C`.
        Assumes C.comoving_distance(z) returns comoving distance in Mpc/h.
        """
        z = np.asarray(z, dtype=float)
        chi_Mpc = np.asarray(C.comoving_distance(z))/C.h  # -> Mpc
        exp= exp_factor_alphaM(z,alpha_interp)

        return (1.0 + z) * chi_Mpc * exp

    def exp_factor_alphaM(z, alpha_interp):
        """
        Compute exp( ∫_0^z 0.5 * alpha_M(z')/(1+z') dz' ).
        Vectorised in z. z can be scalar or 1D array (not necessarily sorted).
        """

        z = np.atleast_1d(z).astype(float)

        # Sort z so we can do a cumulative integral once
        order = np.argsort(z)
        z_sorted = z[order]

        # Build integrand on sorted grid
        vals = 0.5 * alpha_interp(z_sorted) / (1.0 + z_sorted)

        # Cumulative integral from min(z) upward
        I_sorted = cumulative_trapezoid(vals, z_sorted, initial=0.0)

        # Map back to original order
        I = np.empty_like(I_sorted)
        I[order] = I_sorted

        out = np.exp(I)
        # Return scalar if input was scalar
        return float(out[0]) if np.isscalar(z) and out.size == 1 else out

    def z_from_dL(C, dL_Mpc,alpha_interp, z_max=10.0, ngrid=1001):
        """
        Invert d_L(z) -> z using a precomputed grid and linear interpolation.
        dL_Mpc: array-like in Mpc.
        Returns z with same shape as dL_Mpc.
        """
        # Build a monotonic (z, dL) grid
        #t0 = time.time()
        z_grid = np.linspace(0.0, float(z_max), int(ngrid))
        #t1 = time.time()
        #print(f"\t \t \t Time merger_rate_dl_new, Z_GRID: {t1 - t0:.4f} s")

        #t0 = time.time()
        dL_grid = dL_from_C(C, z_grid,alpha_interp) # [Mpc]
        #t1 = time.time()
        #print(f"\t \t \tTime merger_rate_dl_new, DL_GRID: {t1 - t0:.4f} s")

        # Ensure strict monotonicity for interp1d by uniquifying dL_grid
        # (d_L is monotonic in standard cosmologies; numerical noise can cause ties)
        order = np.argsort(dL_grid)
        dL_sorted = dL_grid[order]
        z_sorted = z_grid[order]
        # Drop any duplicates in dL_sorted
        mask = np.concatenate(([True], np.diff(dL_sorted) > 0))
        dL_unique = dL_sorted[mask]
        z_unique = z_sorted[mask]

        inv = interp1d(dL_unique, z_unique, kind='linear',bounds_error=False, fill_value='extrapolate',assume_sorted=True)
        return inv(np.asarray(dL_Mpc, dtype=float))

    # t0 = time.time()
    z_bg = np.asarray(bg['z'])
    alpha_M_interp = interp1d(z_bg, bg['M2_running_smg'], kind='cubic', bounds_error=False,
                              fill_value="extrapolate")
    #print('alpha',alpha_M_interp.shape)
    # t1 = time.time()
    # print(f"\t \t Time merger_rate_dl_new, ALPHA: {t1 - t0:.4f} s")

    #t0 = time.time()
    z = z_from_dL(C, dL_Mpc=np.asarray(dl, dtype=float), alpha_interp=alpha_M_interp)  # dl in Mpc
    #dz = np.diff(z)
    #t1 = time.time()
    #print(f"\t \t Time merger_rate_dl_new, DZ: {t1 - t0:.4f} s")
    #print("max dz =", dz.max())

    #t0 = time.time()
    bin_z_converted = z_from_dL(C, dL_Mpc= 1e3*np.asarray(bin_dl), alpha_interp=alpha_M_interp)  # bin_dl given in Mpc
    #t1 = time.time()
    #print(f"\t \t Time merger_rate_dl_new, BIN_Z: {t1 - t0:.4f} s")

    n_bins = len(bin_dl) - 1

    #t0 = time.time()
    merger_rate_GW = A * (dl / 1000 / Z_0) ** Alpha * np.exp(-(dl / 1000 / Z_0) ** Beta)
    #t1 = time.time()
    #print(f"\t \t Time merger_rate_dl_new, MERGER_RATE: {t1 - t0:.4f} s")

    def norm(x):
        return A * (x / Z_0) ** Alpha * np.exp(-(x / Z_0) ** Beta)

    #t0 = time.time()
    I, err = sint.quad(norm, 0.001, 200)
    #t1 = time.time()
    #print(f"\t \t Time merger_rate_dl_new, I, ERR: {t1 - t0:.4f} s")

    merger_rate = np.zeros(shape=(n_bins, len(z)))

    bin_centers = 0.5 * (bin_dl[1:] + bin_dl[:-1])

    n_bins = len(bin_dl) - 1

    sigma = np.zeros(n_bins)

    delta_dl = 10 ** log_delta_dl
    dl_error = 10 ** log_dl

    #t0 = time.time()
    for i in range(n_bins):
        mask = np.logical_and(dl_error > bin_dl[i], dl_error < bin_dl[i + 1])
        delta_dl_bin = delta_dl[mask]
        sigma[i] = np.percentile(delta_dl_bin, 50) / bin_centers[i]

    #    sigma = 10**beta_1*bin_centers**(beta_2-1)
    #t1 = time.time()
    #print(f"\t \t Time merger_rate_dl_new, LOOP 1: {t1 - t0:.4f} s")

    #t0 = time.time()
    for i in range(n_bins):
        if normalize:
            merger_rate[i] = merger_rate_GW / I
        else:
            merger_rate[i] = merger_rate_GW

        S = 1 / 2 * (ss.erf((np.log(dl / 1000) - np.log(bin_dl[i])) / (2 ** (1 / 2) * sigma[i]))
                     - ss.erf((np.log(dl / 1000) - np.log(bin_dl[i + 1])) / (2 ** (1 / 2) * sigma[i])))
        merger_rate[i] = merger_rate[i] * S
    #t1 = time.time()
    #print(f"\t \t Time merger_rate_dl_new, LOOP 2: {t1 - t0:.4f} s")

    return z, bin_z_converted, merger_rate, I, merger_rate_GW   # merger_rate_GW dimensionless; merger_rate= [1/Gpc]


def merger_rate_dl(dl, bin_dl, log_dl,log_delta_dl,H0, omega_m, omega_b, A, Z_0, Alpha, Beta, normalize=True):

    conversion = FlatLambdaCDM(H0=H0, Om0=omega_m, Ob0=omega_b)
    z = (dl * u.Mpc).to(cu.redshift, cu.redshift_distance(conversion, kind="luminosity")).value
    bin_z_converted = (bin_dl * u.Gpc).to(cu.redshift, cu.redshift_distance(conversion, kind="luminosity")).value

    n_bins = len(bin_dl) - 1

    merger_rate_GW = A * (dl / 1000 / Z_0) ** Alpha * np.exp(-(dl / 1000 / Z_0) ** Beta)

    def norm(x):
        return A * (x / Z_0) ** Alpha * np.exp(-(x / Z_0) ** Beta)

    I, err = sint.quad(norm, 0.001, 200)

    merger_rate = np.zeros(shape=(n_bins, len(z)))

    bin_centers = 0.5 * (bin_dl[1:] + bin_dl[:-1])

    n_bins = len(bin_dl) - 1

    sigma = np.zeros(n_bins)

    delta_dl = 10 ** log_delta_dl
    dl_error = 10 ** log_dl

    for i in range(n_bins):
        mask = np.logical_and(dl_error > bin_dl[i], dl_error < bin_dl[i + 1])
        delta_dl_bin = delta_dl[mask]

        sigma[i] = np.percentile(delta_dl_bin, 50) / bin_centers[i]

    #    sigma = 10**beta_1*bin_centers**(beta_2-1)

    for i in range(n_bins):
        if normalize:
            merger_rate[i] = merger_rate_GW / I
        else:
            merger_rate[i] = merger_rate_GW

        S = 1 / 2 * (ss.erf((np.log(dl / 1000) - np.log(bin_dl[i])) / (2 ** (1 / 2) * sigma[i]))
                     - ss.erf((np.log(dl / 1000) - np.log(bin_dl[i + 1])) / (2 ** (1 / 2) * sigma[i])))
        merger_rate[i] = merger_rate[i] * S

    return z, bin_z_converted, merger_rate, I, merger_rate_GW


# Euclid photometric distribution

def euclid_photo(z, bin_z, sigma):
    
    n_bins=len(bin_z)-1
    #print(sigma,bin_z,z)
    #gal_fit=z**2*np.exp(-(z/(0.894/1.412))**1.705)*2.57e9
    bin_centers = np.array([0.001, 0.14, 0.26, 0.39, 0.53, 0.69, 0.84, 1.00, 1.14, 1.30, 1.44, 1.62, 1.78, 1.91, 2.1, 2.25])
    values = np.array([0, 0.758, 2.607, 4.117, 3.837, 3.861, 3.730, 3.000, 2.827, 1.800, 1.078, 0.522, 0.360, 0.251, 0.1, 0])

    spline = UnivariateSpline(bin_centers, values, s=0.1)

    gal_fit = spline(z)*8.35e7
    gal_fit[gal_fit<0] = 0
    
    gal_distribution = np.zeros(shape=(n_bins, len(z)))
    
    for i in range(n_bins):
        gal_distribution[i]=gal_fit
        
        S=1/2*(ss.erf((np.log(z)-np.log(bin_z[i]))/(2**(1/2)*sigma))
                        -ss.erf((np.log(z)-np.log(bin_z[i+1]))/(2**(1/2)*sigma)))
        gal_distribution[i]=gal_distribution[i]*S
    
    return gal_distribution

def euclid_spec(z, bin_z, sigma):
    
    n_bins=len(bin_z)-1
    
    bin_centers = np.array([0.8, 1, 1.07, 1.14, 1.2, 1.35, 1.45, 1.56, 1.67, 1.9])
    values = np.array([0., 0.2802, 0.2802, 0.2571, 0.2571, 0.2184, 0.2184, 0.2443, 0.2443, 0.])
    
    spline = UnivariateSpline(bin_centers, values, s=0)

    gal_fit = spline(z)*1.25e7
    gal_fit[gal_fit<0] = 0
    
    gal_distribution = np.zeros(shape=(n_bins, len(z)))
    
    for i in range(n_bins):
        gal_distribution[i]=gal_fit
        
        S=1/2*(ss.erf((np.log(z)-np.log(bin_z[i]))/(2**(1/2)*sigma))
                        -ss.erf((np.log(z)-np.log(bin_z[i+1]))/(2**(1/2)*sigma)))
        gal_distribution[i]=gal_distribution[i]*S
    
    return gal_distribution

def ska(z, bin_z, sigma):
    
    n_bins=len(bin_z)-1
    
    bin_centers = np.array([0.01, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95])
    values = np.array([0, 1.21872309, 1.74931326, 1.81914498, 1.6263191 , 1.33347361, 1.05034008, 0.79713276, 0.58895358, 0.42322164, 0.29564803, 0.20296989, 0.1366185 , 0.09011826, 0.0586648 , 0.03724468, 0.02323761, 0.01423011, 0.00848182, 0.00492732])
    
    spline = UnivariateSpline(bin_centers, values, s=0.001)

    gal_fit = spline(z)*9.6e7
    gal_fit[gal_fit<0] = 0
    
    gal_distribution = np.zeros(shape=(n_bins, len(z)))
    
    for i in range(n_bins):
        gal_distribution[i]=gal_fit
        
        S=1/2*(ss.erf((np.log(z)-np.log(bin_z[i]))/(2**(1/2)*sigma))
                        -ss.erf((np.log(z)-np.log(bin_z[i+1]))/(2**(1/2)*sigma)))
        gal_distribution[i]=gal_distribution[i]*S
    
    return gal_distribution

# Shot noise functions
def sn_sigma_GW(bin_edges_dl, alpha_1, alpha_2):
    bin_centers = 0.5 * (bin_edges_dl[1:] + bin_edges_dl[:-1])
    
    sigma = 10**alpha_1*bin_centers**alpha_2
    return sigma

def compute_sn_slice(args):
    l_index, shot_noise, n = args
    slice_matrix = np.zeros((n, n))
    for i in range(n):
        slice_matrix[i, i] = shot_noise[i]  # Only diagonal is nonzero
    return l_index, slice_matrix

def shot_noise_mat_auto(shot_noise, ll):
    n = len(shot_noise)
    sn_matrix = np.zeros((n, n, len(ll)))

    # Prepare arguments for each ℓ
    tasks = [(l, shot_noise, n) for l in range(len(ll))]

    # Run in parallel
    with ProcessPoolExecutor() as executor:
        for l_index, slice_matrix in executor.map(compute_sn_slice, tasks):
            sn_matrix[:, :, l_index] = slice_matrix

    return sn_matrix



# Find the 50 and 99 percentile for the best angular resolution and the average resolution of GWs
def loc_error_param(bin_edges_dl, log_loc, log_dl, l_min, l_max):
    n_bins = len(bin_edges_dl)-1
    
    loc_50 = np.zeros(n_bins)
    loc_99 = np.zeros(n_bins)
    
    loc = 10**log_loc
    dl = 10**log_dl
    
    for i in range(n_bins):
        mask = np.logical_and(dl > bin_edges_dl[i], dl < bin_edges_dl[i+1])
        loc_bin = loc[mask]
        
        loc_50[i] = np.percentile(loc_bin, 50)
        loc_99[i] = np.percentile(loc_bin, 1)
    
    max_l = np.pi/np.sqrt(loc_99)
    max_l = max_l.astype(int)
    
    for i in range(len(max_l)):
        if max_l[i]>=l_max:
            max_l[i]=l_max
        if max_l[i]<=l_min:
            max_l[i]=l_min
            
    return loc_50, max_l


# Find the maximum multipole from the fit
def find_l_max(bin_edges, gamma_1, gamma_2, l_min, l_max):
    
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    min_resolution = 10**gamma_1*bin_centers**gamma_2
    
    max_l = np.pi/np.sqrt(min_resolution)
    max_l = max_l.astype(int)
    
    for i in range(len(max_l)):
        if max_l[i]>=l_max:
            max_l[i]=l_max
        if max_l[i]<=l_min:
            max_l[i]=l_min
    
    return max_l


# Compute the C_l vector
def vector_cl(cl_cross, cl_auto1, cl_auto2):
    n_bins_1 = len(cl_auto1)
    n_bins_2 = len(cl_auto2)

    lenght = n_bins_1 ** 2 - np.sum(range(n_bins_1)) + (n_bins_1 * n_bins_2) + n_bins_2 ** 2 - np.sum(range(n_bins_2))

    vec_cl = np.zeros(shape=(lenght, len(cl_auto1[0, 0])))
    i = 0

    for j in range(n_bins_1):
        for k in range(j, n_bins_1):
            vec_cl[i] = cl_auto1[j, k]
            i += 1
    for j in range(n_bins_1):
        for k in range(n_bins_2):
            vec_cl[i] = cl_cross[j, k]
            i += 1
    for j in range(n_bins_2):
        for k in range(j, n_bins_2):
            vec_cl[i] = cl_auto2[j, k]
            i += 1

    return vec_cl



def covariance_matrix(vec_cl, n_bins_1, n_bins_2):
    dim = n_bins_1 + n_bins_2
    cov_matrix = np.zeros(shape=(dim, dim, len(vec_cl[0])))
    i = 0
    for j in range(n_bins_1):
        for k in range(j, n_bins_1):
            cov_matrix[j, k] = vec_cl[i]
            i += 1
    for j in range(n_bins_1):
        for k in range(n_bins_1, dim):
            cov_matrix[j, k] = vec_cl[i]
            i += 1
    for j in range(n_bins_1, dim):
        for k in range(j, dim):
            cov_matrix[j, k] = vec_cl[i]
            i += 1

    for i in range(dim):
        for j in range(i + 1, dim):
            cov_matrix[j][i] = cov_matrix[i][j]

    return cov_matrix



def fisher_matrix(Cl, Cl_der, ll, f_sky=0.3):
    n_par = len(Cl_der)

    Cl_inv = np.zeros_like(Cl)
    product = np.zeros_like(Cl)
    trace = np.zeros(len(ll))
    fisher = np.zeros((n_par, n_par))

    for i in range(len(ll)):
        Cl_inv[:, :, i] = np.linalg.inv(Cl[:, :, i])

    for a in range(n_par):
        for b in range(n_par):
            for i in range(len(ll)):
                product[:, :, i] = np.dot(np.dot(np.dot(Cl_inv[:, :, i], Cl_der[a, :, :, i]), Cl_inv[:, :, i]),
                                          Cl_der[b, :, :, i])

                trace[i] = np.trace(product[:, :, i])

            fisher[a, b] = f_sky * np.sum(((2 * ll + 1) / 2) * trace)

    return fisher


def fisher_matrix_different_l(Cl, Cl_der, l_min_tot, l_max_bin, f_sky=0.3):
    l_max_bin = l_max_bin[::-1]
    n_par = len(Cl_der)
    fisher = np.zeros((len(l_max_bin), n_par, n_par))

    for j in range(len(l_max_bin)):

        if j == 0:
            ll_aux = np.arange(l_min_tot, l_max_bin[j] + 1)
        else:
            ll_aux = np.arange(l_max_bin[j - 1] + 1, l_max_bin[j] + 1)

        Cl_inv = np.zeros((len(Cl[:]), len(Cl[0, :]), len(ll_aux)))
        product = np.zeros((len(Cl[:]), len(Cl[0, :]), len(ll_aux)))
        trace = np.zeros(len(ll_aux))

        for i in range(len(ll_aux)):
            Cl_inv[:, :, i] = np.linalg.inv(Cl[:, :, ll_aux[i] - l_min_tot])

        for a in range(n_par):
            for b in range(n_par):
                for i in range(len(ll_aux)):
                    product[:, :, i] = np.dot(
                        np.dot(np.dot(Cl_inv[:, :, i], Cl_der[a, :, :, ll_aux[i] - l_min_tot]), Cl_inv[:, :, i]),
                        Cl_der[b, :, :, ll_aux[i] - l_min_tot])

                    trace[i] = np.trace(product[:, :, i])

                fisher[j, a, b] = np.sum(((2 * ll_aux + 1) / 2) * trace)

        Cl = Cl[:-1, :-1, :]
        Cl_der = Cl_der[:, :-1, :-1, :]

    fisher_tot = f_sky * np.sum(fisher, axis=0)

    return fisher_tot

#---------------------------------

def fisher_matrix_different_l_nl(Cl, Cl_der, l_min_tot, l_max_bin, limit, n_bins_z, f_sky=0.3):
    
    l_max_bin=np.sort(l_max_bin)
    loc_or_nl = limit[np.argsort(l_max_bin)]
    n_par=len(Cl_der)
    fisher=np.zeros((len(l_max_bin), n_par, n_par))
    
    for j in range(len(l_max_bin)):
        
        if j==0:
            ll_aux = np.arange(l_min_tot, l_max_bin[j]+1)
        else:
            ll_aux = np.arange(l_max_bin[j-1]+1, l_max_bin[j]+1)
        
        Cl_inv=np.zeros((len(Cl[:]), len(Cl[0,:]), len(ll_aux)))
        product=np.zeros((len(Cl[:]), len(Cl[0,:]), len(ll_aux)))
        trace=np.zeros(len(ll_aux))

        for i in range(len(ll_aux)):
            Cl_inv[:,:,i]=np.linalg.inv(Cl[:,:,ll_aux[i]-l_min_tot])
        
        for a in range(n_par):
            for b in range(n_par):
                for i in range(len(ll_aux)):
                    
                    product[:,:,i] = np.dot(np.dot(np.dot(Cl_inv[:,:,i], Cl_der[a,:,:,ll_aux[i]-l_min_tot]), Cl_inv[:,:,i]), 
                                        Cl_der[b,:,:,ll_aux[i]-l_min_tot])
                
                    trace[i]=np.trace(product[:,:,i])
    
                fisher[j,a,b]=np.sum(((2*ll_aux+1)/2)*trace)
        
        if loc_or_nl[j]==0:
            Cl = Cl[:-1, :-1, :]
            Cl_der=Cl_der[:, :-1, :-1, :]

        else:
            Cl = np.delete(np.delete(Cl, [0, n_bins_z], axis=0), [0, n_bins_z], axis=1)
            Cl_der = np.delete(np.delete(Cl_der, [0, n_bins_z], axis=1), [0, n_bins_z], axis=2)
            n_bins_z = n_bins_z-1

        
    fisher_tot=f_sky*np.sum(fisher, axis=0)
    
    return fisher_tot



# Return the bin edges for bins equally populated
def equal_interval(distr, z, n):
    """
    Return indices of `z` that split the 1D distribution `distr(z)` into `n` bins
    with (approximately) equal integral under the curve.
    """
    # --- sanitize inputs ---
    z     = np.asarray(z, dtype=float).ravel()
    distr = np.asarray(distr, dtype=float).ravel()
    n     = int(n)

    if z.ndim != 1 or distr.ndim != 1 or z.size != distr.size:
        raise ValueError("`z` and `distr` must be 1D arrays of the same length.")
    if n < 1:
        raise ValueError("`n` must be >= 1.")
    if not np.all(np.diff(z) > 0):
        raise ValueError("`z` must be strictly increasing.")

    # --- cumulative integral (same length as z) ---
    cum = cumulative_trapezoid(distr, z, initial=0.0)   # shape (N,)
    total = cum[-1]
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Total integral must be positive and finite.")

    # --- target cumulative values for n bins ---
    targets = np.linspace(0.0, total, n + 1)            # includes 0 and total
    # indices where cum >= target
    idx = np.searchsorted(cum, targets, side="left")

    # clamp to valid range and ensure strictly increasing indices
    idx = np.clip(idx, 0, len(z) - 1)
    # de-duplicate in case of flat regions
    _, unique_first = np.unique(idx, return_index=True)
    if unique_first.size != idx.size:
        # fallback: spread duplicates minimally
        idx = np.minimum(np.arange(len(idx)), len(z) - 1)

    return idx.astype(int)




# Compute the cumulative signal to noise ratio

def SNR_bins_auto(Cl, noise, f_sky, l_min, l_max):
    nbin = len(Cl)
    ll=np.arange(l_min, l_max+1)
    
    Cl_noise = np.copy(Cl)
    SNR = np.zeros(shape=(nbin, nbin, len(ll)))
    
    Cl_noise += noise
    
    for i in range(len(ll)):
        for j in range(nbin):
            for k in range(nbin):
                SNR[j,k,i] = (Cl[j,k,i]**2)/(Cl_noise[j,j,i]*Cl_noise[k,k,i]+(Cl_noise[j,k,i]**2))
    SNR = SNR * f_sky * (2*ll+1)
    
    SNR_cum = np.sqrt(np.sum(SNR, axis=2))
    
    return SNR_cum


def SNR_bins_cross(Cl_auto_1, Cl_auto_2, Cl_cross, noise_1, noise_2, f_sky, l_min, l_max):
    nbin_1 = len(Cl_auto_1)
    nbin_2 = len(Cl_auto_2)
    ll=np.arange(l_min, l_max+1)
    
    Cl_noise_1 = np.copy(Cl_auto_1)
    Cl_noise_2 = np.copy(Cl_auto_2)
    SNR = np.zeros(shape=(nbin_1, nbin_2, len(ll)))
    
    Cl_noise_1 += noise_1
    Cl_noise_2 += noise_2
    
    for i in range(len(ll)):
        for j in range(nbin_1):
            for k in range(nbin_2):
                SNR[j,k,i] = (Cl_cross[j,k,i]**2)/(Cl_noise_1[j,j,i]*Cl_noise_2[k,k,i]+(Cl_cross[j,k,i]**2))
    SNR = SNR * f_sky * (2*ll+1)
    
    SNR_cum = np.sqrt(np.sum(SNR, axis=2))
    
    return SNR_cum

