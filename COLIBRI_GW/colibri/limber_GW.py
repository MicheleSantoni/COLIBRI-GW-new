import os
import colibri.constants as const
import colibri.cosmology as cc
import numpy as np
import scipy.special as ss
import scipy.interpolate as si
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import cumulative_trapezoid
from scipy.integrate import trapezoid, simpson
import scipy.integrate as sint
import scipy.optimize as so
import colibri.fourier as FF
from six.moves import xrange
from math import sqrt
from astropy import units as u
import astropy.cosmology.units as cu
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
import time
from sklearn import preprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

#==================
# CLASS LIMBER
#==================
class limber():
    """
    The class :func:`colibri.limber.limber` contains all the functions useful to compute
    the angular power spectra and correlation functions in the flat sky and Limber's approximation.
    It also contains routines to compute widely-used window functions as well as a routine to add custom ones.
    At initialization it takes as inputs a redshift range for integration and a
    :func:`colibri.cosmology.cosmo` instance.


    :param cosmology: Fixes the cosmological parameters. If not declared, the default values are chosen (see :func:`colibri.cosmology.cosmo` documentation).
    :type cosmology: ``cosmo`` instance, default = ``cosmology.cosmo()``

    :param z_limits: Lower and upper limit of integration along the line of sight. Both numbers must be non-negative and the first number must be smaller than the second. If the lower limit is set to 0, it will be enhanced by 1e-10 to avoid divergences at the origin of the lightcone.
    :type z_limits: 2-uple or list/array of length 2, default = (0., 5.)


    .. warning::

     All the power spectra are computed in the Limber approximation and the window function are assumed to be dependent only on redshift and not on scales
     (see e.g. :func:`colibri.limber.limber.load_lensing_window_functions`).
     Typically the scale dependence of the window functions can be factorized out (e.g. ISW effect, different orders of cosmological perturbation theory...)
     and in this code it can be added to the power spectrum (see :func:`colibri.limber.limber.load_power_spectra`).

    """

    #-----------------------------------------------------------------------------------------
    # INITIALIZATION FUNCTION
    #-----------------------------------------------------------------------------------------
    def __init__(self, cosmology, z_limits=(0., 5.)):
        self.cosmology = cosmology
        self.class_obj = cosmology.class_Cl

        # Redshifts
        assert len(z_limits) == 2, "Limits of integration must be a 2-uple or a list of length 2, with z_min at 1st place and z_max at 2nd place"
        assert z_limits[0] < z_limits[1], "z_min (lower limit of integration) must be smaller than z_max (upper limit)"

        # Minimum and maximum redshifts
        self.z_min = z_limits[0]
        self.z_max = z_limits[1]

        # Remove possible infinity at z = 0.
        if self.z_min == 0.: self.z_min += 1e-10    # Remove singularity of 1/chi(z) at z = 0

        # Set the array of integration
        self.dz_min = 0.0625
        self.n_z_min = int((self.z_max-self.z_min)/self.dz_min+1)
        #self.n_z_integration = int((self.z_max - self.z_min)*self.n_z_min + 2)
        
        self.n_z_integration = 100
        
        self.z_integration  = np.linspace(self.z_min, self.z_max, self.n_z_integration)

        # Array of redshifts for computing window integrals (set number of redshift so that there is an integral at least each dz = 0.025)
        self.dz_windows = 0.025
        self.z_windows  = np.arange(self.z_min, self.z_max+self.dz_windows, self.dz_windows)
        self.n_z_windows = len(np.atleast_1d(self.z_windows))

        # Distances (in Mpc/h)
        self.geometric_factor         = self.geometric_factor_f_K(self.z_integration)  #[Mpc/h]
        self.geometric_factor_windows = self.geometric_factor_f_K(self.z_windows)

        # Hubble parameters (in km/s/(Mpc/h))
        self.Hubble         = self.cosmology.H(self.z_integration)/self.cosmology.h
        self.Hubble_windows = self.cosmology.H(self.z_windows)    /self.cosmology.h

        # Factor c/H(z)/f_K(z)^2  [Mpc/h]*[h/Mpc]**2
        self.c_over_H_over_chi_squared = const.c/self.Hubble/self.geometric_factor**2.

        # Initialize window functions
        self.window_function = {}
        self.bin_edges = {}

    #-----------------------------------------------------------------------------------------
    # LOAD_PK
    #-----------------------------------------------------------------------------------------
    def load_power_spectra(self, k, z, power_spectra):
        """
        This routine interpolates the total matter power spectrum (using the CDM prescription) in scales (units of :math:`h/\mathrm{Mpc}`) and redshifts.
        `power_spectra` and `galaxy_bias` must be a 2D array of shape ``(len(z), len(k))`` which contains the power spectrum (in units of :math:`(\mathrm{Mpc}/h)^3`) and
        galaxy bias evaluated at the scales and redshifts specified above.

        :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
        :type k: array

        :param z: Redshifts at which power spectrum must be/is computed.
        :type z: array

        :param power_spectra: It must be a 2D array of shape ``(len(z), len(k))`` which contains the power spectrum (in units of :math:`(\mathrm{Mpc}/h)^3`) evaluated at the scales and redshifts specified above.
        :type power_spectra: 2D NumPy array

        :return: Nothing, but two 2D-interpolated object ``self.power_spectra_interpolator`` and ``self.galaxy_bias_interpolator`` containing :math:`P(k,z)` in units of :math:`(\mathrm{Mpc}/h)^3` and :math:`b(k,z)` are created
        """
     	# Select scales and redshifts
        self.k     = np.atleast_1d(k)
        self.z     = np.atleast_1d(z)
        self.k_min = k.min()
        self.k_max = k.max()

    	# Interpolate
        kind_of_interpolation = 'cubic' if (len(self.z) > 3) and (len(self.k) > 3) else 'linear'
        #print("DEBUG: interpolation method =", kind_of_interpolation, type(kind_of_interpolation))
    	# Transpose if needed: (z, k) order
        interp_func = RegularGridInterpolator((self.k, self.z),power_spectra,kind_of_interpolation,bounds_error=False,fill_value=0.)
        self.power_spectra_interpolator = interp_func


    def load_bin_edges(self, bin_edges_1, bin_edges_2, name_1='galaxy', name_2='GW'):
        self.bin_edges[name_1] = bin_edges_1
        self.bin_edges[name_2] = bin_edges_2

    #-----------------------------------------------------------------------------------------
    # EXAMPLES OF GALAXY PDF'S
    #-----------------------------------------------------------------------------------------
    def euclid_distribution(self, z, z_min, z_max, a = 2.0, b = 1.5, z_med = 0.9, step = 5e-3):
        """
        Example function for the distribution of source galaxy. This distribution in particular is expected to be used in the Euclid mission:

        .. math::

            n(z) \propto z^a \ \exp{\left[-\left(\\frac{z}{z_{med}/\sqrt 2}\\right)^b\\right]}

        This distribution will eventually be normalized such that its integral on all redshifts is 1.

        :param z: Redshifts.
        :type z: array

        :param z_min: Lower edge of the bin (a small width of 0.005 will be applied for convergence reasons).
        :type z_min: float

        :param z_max: Upper edge of the bin (a small width of 0.005 will be applied for convergence reasons).
        :type z_max: float

        :param a: Parameter of the distribution.
        :type a: float, default = 2.0

        :param b: Parameter of the distribution.
        :type b: float, default = 1.5

        :param z_med: Median redshift of the distribution.
        :type z_med: float, default = 0.9

        :param step: width of the cutoff (better to avoid a sharp one for numerical reasons, better set it to be at least 0.001)
        :type step: float, default = 0.005

        :return: array
        """
        # from median redshift to scale-redshift
        z_0 = z_med/sqrt(2.)
        # Heaviside-like function
        lower = 0.5*(1.+np.tanh((z-z_min)/step))
        upper = 0.5*(1.+np.tanh((z_max-z)/step))
        # Galaxy distribution
        n = (z/z_0)**a*np.exp(-(z/z_0)**b)*lower*upper
        return n

    def euclid_distribution_with_photo_error(self, z, z_min, z_max, a = 2.0, b = 1.5, z_med = 0.9, f_out = 0.1, c_b = 1.0, z_b = 0.0, sigma_b = 0.05, c_o = 1.0, z_o = 0.1, sigma_o = 0.05, A=200000, normalize=True):
        """
        Example function for the distribution of source galaxy. This distribution in particular is expected to be used in the Euclid mission. Here also the effect of photometric errors is included.

        .. math::

         n^{(i)}(z) \propto \int_{z_i^-}^{z_i^+} dy \ z^a \ \exp{\left[-\left(\\frac{z}{z_{med}/\sqrt 2}\\right)^b\\right]} \ p_\mathrm{ph}(y|z)

        where

        .. math::


         p_\mathrm{ph}(y|z) = \\frac{1-f_\mathrm{out}}{\sqrt{2\pi}\sigma_b(1+z)} \ \exp\left[-\\frac{1}{2} \left(\\frac{z-c_b y -z_b}{\sigma_b(1+z)}\\right)^2\\right] +

         + \\frac{f_\mathrm{out}}{\sqrt{2\pi}\sigma_o(1+z)} \ \exp\left[-\\frac{1}{2} \left(\\frac{z-c_o y -z_o}{\sigma_o(1+z)}\\right)^2\\right]

        :param z: Redshifts.
        :type z: array

        :param z_min: Lower edge of the bin.
        :type z_min: float

        :param z_max: Upper edge of the bin.
        :type z_max: float

        :param a: Parameter of the distribution.
        :type a: float, default = 1.5

        :param b: Parameter of the distribution.
        :type b: float, default = 1.5

        :param z_med: Median redshift of the distribution.
        :type z_med: float, default = 0.9

        :param f_out: Fraction of outliers
        :type f_out: float, default = 0.1

        :param c_b: Parameter of the Gaussian (normalization) representing the uncertainty on the photometric error for in-liers.
        :type c_b: float, default = 1.0

        :param z_b: Parameter of the Gaussian (scale-redshift) representing the uncertainty on the photometric error for in-liers.
        :type z_b: float, default = 0.0

        :param sigma_b: Parameter of the Gaussian (width) representing the uncertainty on the photometric error for in-liers.
        :type sigma_b: float, default = 0.05

        :param c_o: Parameter of the Gaussian (normalization) representing the uncertainty on the photometric error for out-liers.
        :type c_o: float, default = 1.0

        :param z_o: Parameter of the Gaussian (scale-redshift) representing the uncertainty on the photometric error for out-liers.
        :type z_o: float, default = 0.1

        :param sigma_o: Parameter of the Gaussian (width) representing the uncertainty on the photometric error for out-liers.
        :type sigma_o: float, default = 0.05

        :return: array
        """
        # from median redshift to scale-redshift
        z_0       = z_med/sqrt(2.)
        gal_distr = A*(z/z_0)**a*np.exp(-(z/z_0)**b)
        
        def norm(x):
            return (x/z_0)**a*np.exp(-(x/z_0)**b)

        I,err = sint.quad(norm, 0.001, 6)
        
        if normalize:
            gal_distr=gal_distr/I
        
        # Photometric error function
        distr_in  = (1.-f_out)/(2.*c_b)*(ss.erf((z - c_b*z_min - z_b)/(sqrt(2.)*sigma_b*(1.+z)))-ss.erf((z - c_b*z_max - z_b)/(sqrt(2.)*sigma_b*(1.+z))))
        distr_out =    (f_out)/(2.*c_o)*(ss.erf((z - c_o*z_min - z_o)/(sqrt(2.)*sigma_o*(1.+z)))-ss.erf((z - c_o*z_max - z_o)/(sqrt(2.)*sigma_o*(1.+z))))
        photo_err_func = distr_in+distr_out
        return photo_err_func*gal_distr

    def gaussian_distribution(self, z, mean, sigma):
        """
        Example function for the distribution of source galaxy. Here we use a Gaussian galaxy distribution

        :param z: Redshifts.
        :type z: array

        :param mean: Mean redshift of the distribution.
        :type mean: float

        :param sigma: Width of the Gaussian
        :type sigma: float

        :return: array
        """
        exponent = -0.5*((z-mean)/sigma)**2.
        return np.exp(exponent)


    def constant_distribution( z, z_min, z_max, step = 5e-3):
        """
        Example function for the distribution of source galaxy. Here we use a constant distribution of sources.

        :param z: Redshifts.
        :type z: array

        :param z_min: Lower edge of the bin.
        :type z_min: float

        :param z_max: Upper edge of the bin.
        :type z_max: float

        :param step: width of the cutoff (better to avoid a sharp one for numerical reasons, better set it to be at least 0.001)
        :type step: float, default = 0.005

        :return: array
        """
        
        # Heaviside-like function
        lower = 0.5*(1.+np.tanh((z-z_min)/step))
        upper = 0.5*(1.+np.tanh((z_max-z)/step))
        # Galaxy distribution
        n = z**0.*lower*upper
        return n
        

    #-------------------------------------------------------------------------------
    # GEOMETRIC FACTOR
    #-------------------------------------------------------------------------------
    def geometric_factor_f_K(self, z, z0 = 0.):
        """
        Geometric factor (distance) between two given redshifts ``z`` and ``z0``.
        It assumes neutrinos as matter, which is a good approximation at low redshifts.
        In fact, this latter assumption introduces a bias of less than 0.02% at :math:`z<10`
        for even the lowest neutrino masses allowed by particle physics.

        :param z: Redshifts.
        :type z: array

        :param z0: Pivot redshift.
        :type z0: float, default = 0

        :return: array
        """
        # Curvature in (h/Mpc)^2 units. Then I will take the sqrt and it will
        # go away with comoving_distance(z), giving a final result in units of Mpc/h
        K = self.cosmology.K
        chi_z0 = self.cosmology.comoving_distance(z0)
        chi_z  = self.cosmology.comoving_distance(z)-chi_z0
        # Change function according to sign of K
        if K == 0.:  return chi_z #Mpc/h
        elif K > 0.: return 1./K**0.5*np.sin(K**0.5*chi_z) #Mpc/h
        else:        return 1./np.abs(K)**0.5*np.sinh(np.abs(K)**0.5*chi_z) #Mpc/h

    #-----------------------------------------------------------------------------------------
    # WINDOW FUNCTIONS IMPLEMENTATION
    #-----------------------------------------------------------------------------------------
        
    #-----------------------------------------------------------------------------------------
    # GRAVITATIONAL WAVES LENSING WINDOW FUNCTION - Background
    #-----------------------------------------------------------------------------------------
    def load_gw_lensing_window_functions(self,bg,C, z,h, n_dl, H_0, omega_m, ll, name = 'lensing_GW'):
        def dL_from_C(C, z, alpha_interp):
            """
            Luminosity distance d_L(z) in Mpc from your colibri cosmology `C`.
            Assumes C.comoving_distance(z) returns comoving distance in Mpc/h.
            """
            z = np.asarray(z, dtype=float)
            chi_Mpc = np.asarray(C.comoving_distance(z))/C.h  # -> Mpc
            return (1.0 + z) * chi_Mpc * exp_factor_alphaM(z,alpha_interp)

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

        z_bg = np.asarray(bg['z'])
        alpha_M_interp = interp1d(z_bg, bg['M2_running_smg'], kind='cubic', bounds_error=False,fill_value="extrapolate")

        z  = np.array(z)
        n_dl = np.asarray(n_dl)  # shape (n_bins, len(z))
        n_bins = len(n_dl)

        norm_const = simpson(n_dl , x=dL_from_C(C, z, alpha_M_interp), axis=1)

        #print('Hubble',H_0)
        constant = 3.*omega_m*(H_0/const.c)**2. #[1/Mpc]
        z_int= self.z_integration

        # Initialize window
        self.window_function[name]  = []
        # Set windows
        for galaxy_bin in xrange(n_bins):
            # Select which is the function and which are the arguments
            n_z_interp = si.interp1d(z, n_dl[galaxy_bin], 'cubic', bounds_error = False, fill_value = 0.)
            self.window_function[name].append(si.interp1d(z_int, constant*n_z_interp(z_int)/norm_const[galaxy_bin], 'cubic', bounds_error = False, fill_value = 0.))

        self.window_function[name]=np.tile(np.array(self.window_function[name]).reshape(-1,1), (1,len(ll)))



    #-----------------------------------------------------------------------------------------
    # GALAXY LENSING WINDOW FUNCTION
    #-----------------------------------------------------------------------------------------
    def load_galaxy_lensing_window_functions(self, z, n_z, H_0, omega_m, ll, name = 'galaxy_lensing'):
        constant = 3.*omega_m*(H_0/const.c)**2.
        n_z = np.array(n_z)
        z  = np.array(z)
        n_bins = len(n_z)
        norm_const = simpson(n_z, x = z, axis = 1)

        self.window_function[name] = []
        # Compute window
        for galaxy_bin in xrange(n_bins):
            tmp_interp = si.interp1d(z,n_z[galaxy_bin],'cubic', bounds_error = False, fill_value = 0.)
            self.window_function[name].append(si.interp1d(self.z_integration, constant*tmp_interp(self.z_integration)/norm_const[galaxy_bin], 'cubic', bounds_error = False, fill_value = 0.))

        self.window_function[name]=np.tile(np.array(self.window_function[name]).reshape(-1,1), (1,len(ll)))

    # -----------------------------------------------------------------------------------------
    # GALAXY CLUSTERING WINDOW FUNCTION - Background
    # -----------------------------------------------------------------------------------------
    def load_galaxy_clustering_window_functions(self,bg, z, n_z, ll, bias=1.0, name='galaxy'):
        """
        Builds galaxy clustering window functions W_G^i(z) = b(z) n^i(z) H(z)/c
        using:

        """

        # --- inputs & guards
        z_bg = np.asarray(bg['z'])
        H_interp = interp1d(z_bg, bg['H [1/Mpc]'], kind='cubic', bounds_error=False, fill_value="extrapolate") # [1/Mpc]
        #print('H(z)')

        n_z = np.asarray(n_z)
        z = np.asarray(z)
        n_bins = len(n_z)
        norm_const = simpson(n_z, x=z, axis=1)

        assert np.all(np.diff(z) < self.dz_windows), \
            "For convergence reasons, dz must be <= %.3f" % (self.dz_windows)
        assert n_z.ndim == 2, "'n_z' must be 2-dimensional"
        assert n_z.shape[1] == z.shape[0], "Each n_z[i] must have length len(z)"
        assert z.min() <= self.z_min, "z.min() must be <= z_min integration bound"
        assert z.max() >= self.z_max, "z.max() must be >= z_max integration bound"

        if isinstance(bias, (float, int, np.floating, np.integer)):
            bias = float(bias) * np.ones(n_bins)
        else:
            assert len(bias) == n_bins, "Number of bias factors must equal number of bins"

        # --- Restrict to the correct range
        z_int = np.asarray(self.z_integration)

        H_on_grid = H_interp(z_int)  # in 1/Mpc
        #print('H(z) GAL', H_on_grid)

        #print('GALAXY NEW:\n')
        #print('nz', n_z)
        #print('z', z)
        #print('bias', bias)
        #print('n_bins', n_bins)
        #print('norm_const', norm_const)
        #print('Hubble ', H_on_grid.tolist())

        windows = []
        for i_bin in xrange(n_bins):
            nz_i_interp = si.interp1d(z, n_z[i_bin], kind='cubic', bounds_error=False, fill_value=0.0)
            nz_on_grid = nz_i_interp(z_int)
            factor = (nz_on_grid / norm_const[i_bin]) * bias[i_bin]
            w_vals = factor * H_on_grid  # (1/Mpc)

            windows.append(si.interp1d(z_int, w_vals, kind='cubic', bounds_error=False, fill_value=0.0))

        # --- store the window
        self.window_function[name] = np.tile(np.array(windows).reshape(-1, 1), (1, len(ll)))


    # -----------------------------------------------------------------------------------------
    # GRAVITATIONAL WAVES CLUSTERING WINDOW FUNCTION — Background
    # -----------------------------------------------------------------------------------------
    def load_gravitational_wave_window_functions(self,bg, C, z, n_dl, ll, bias=1.0, name='GW'):
        """
        Build GW *clustering* windows coherent with:
            Δ_i^GW(z) = [ddL_GW/dz] * b_GW(z) * (H(z)/c) * w_i^GW(z),
        with w_i^GW(z) = n_dl_i(dL_GW(z)) / N_i  and  N_i = ∫ n_dl_i(dL_GW(z)) [ddL_GW/dz] dz.

        This function returns W_i(z) = b_GW(z) * [ n_dl_i(dL_GW(z)) / N_i ] * [ddL_GW/dz],
        i.e. WITHOUT the (H/c) factor (the projector must provide it once).
        """

        def dL_from_C(C, z, alpha_interp):
            """
            Luminosity distance d_L(z) in Mpc from your colibri cosmology `C`.
            Assumes C.comoving_distance(z) returns comoving distance in Mpc/h.
            """
            z = np.asarray(z, dtype=float)
            chi_Mpc = np.asarray(C.comoving_distance(z))/C.h # -> Mpc https://classylss.readthedocs.io/en/stable/api/classylss.binding.html
            return (1.0 + z) * chi_Mpc * exp_factor_alphaM(z,alpha_interp)

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

        # --- bg interpolators
        #h=C.h
        z_bg = np.asarray(bg['z'])
        H_interp = interp1d(z_bg, bg['H [1/Mpc]'], kind='cubic', bounds_error=False, fill_value="extrapolate") #[1/Mpc]
        chi_interp = interp1d(z_bg, bg['comov. dist.'], kind='cubic', bounds_error=False,fill_value="extrapolate") # Mpc
        alpha_M_interp = interp1d(z_bg, bg['M2_running_smg'], kind='cubic', bounds_error=False, fill_value="extrapolate")

        # inputs & guards
        z = np.asarray(z)
        n_dl = np.asarray(n_dl)  # shape (n_bins, len(z))
        n_bins = len(n_dl)

        # n_dl [1/Gpc]
        norm = simpson(n_dl, x= dL_from_C(C, z, alpha_M_interp), axis=1) #


        assert np.all(np.diff(z) < self.dz_windows), "For convergence reasons, the distribution function arrays must be sampled with frequency of at least dz<=%.3f" % (self.dz_windows)
        assert n_dl.ndim == 2, "'n_z' must be 2-dimensional"
        assert (n_dl.shape)[1] == z.shape[0], "Length of each 'n_z[i]' must be the same of 'z'"
        assert z.min() <= self.z_min, "Minimum input redshift must be < %.3f, the minimum redshift of integration" % (self.z_min)
        assert z.max() >= self.z_max, "Maximum input redshift must be > %.3f, the maximum redshift of integration" % (self.z_max)

        if isinstance(bias, float):
            bias = bias * np.ones(n_bins)
        elif isinstance(bias, int):
            bias = np.float(bias) * np.ones(n_bins)
        else:
            assert len(bias) == n_bins, "Number of bias factors different from number of bins"

        # --- Build both windows on the *same* integration grid
        z_int = np.asarray(self.z_integration)

        # bg branch: H and r from bg
        H_bg_on_z = H_interp(z_int)  # [1/Mpc]
        r_bg_on_z = chi_interp(z_int)  # [Mpc/1]
        #print('H(z) GW', H_bg_on_z)

        alpha_M_bg_on_z = alpha_M_interp(z_int)
        exp_val = exp_factor_alphaM(z_int, alpha_M_interp)

        jac_dL_bg = ((1 + z_int) *1 / H_bg_on_z + r_bg_on_z +  r_bg_on_z * 0.5 * alpha_M_bg_on_z) * exp_val # [Mpc]


        windows= []
        for i_bin in xrange(n_bins):
            ndl_i = si.interp1d(z, n_dl[i_bin], kind='cubic', bounds_error=False, fill_value=0.0)
            ndl_on_z = ndl_i(z_int)

            # W_i(z) = b_i * n_dl_i(d_L(z)) * (ddL/dz) * H(z) / N_i
            #        =  [1]*     [1/Mpc]        * [Mpc]    * [1/Mpc] / [1/Mpc]
            w_vals = bias[i_bin] * ndl_on_z * jac_dL_bg * H_bg_on_z / norm[i_bin]
            windows.append(si.interp1d(z_int, w_vals, kind='cubic', bounds_error=False, fill_value=0.0))

        # --- store
        self.window_function[name] = np.tile(np.array(windows).reshape(-1, 1), (1, len(ll)))


    #-----------------------------------------------------------------------------------------
    # RSD WINDOW FUNCTION
    #-----------------------------------------------------------------------------------------
    def load_rsd_window_functions(self,bg,h, z, n_z, ll, name = 'RSD'):

        # --- bg interpolators
        z_bg = np.asarray(bg['z'])
        H_interp = interp1d(z_bg, bg['H [1/Mpc]'], kind='cubic', bounds_error=False, fill_value="extrapolate")
        rho_b = interp1d(z_bg, bg['(.)rho_b'], kind='cubic', bounds_error=False, fill_value="extrapolate")
        rho_cdm = interp1d(z_bg, bg['(.)rho_cdm'], kind='cubic', bounds_error=False, fill_value="extrapolate")
        rho_crit = interp1d(z_bg, bg['(.)rho_crit'], kind='cubic', bounds_error=False, fill_value="extrapolate")

        n_z = np.array(n_z)
        z  = np.array(z)
        n_bins = len(n_z)
        z_int=self.z_integration
        norm_const = simpson(n_z, x = z, axis = 1)

        H_z=H_interp(z_int)/h # [h/Mpc]

        #omega_gamma= interp1d(z_bg, bg['gr.fac. f'], kind='cubic', bounds_error=False, fill_value="extrapolate")
        #omega_gamma_z= omega_gamma(z)

        factor = 6 / 11
        Om_z = (rho_b(z) + rho_cdm(z)) / rho_crit(z)

        assert np.all(np.diff(z)<self.dz_windows), "For convergence reasons, the distribution function arrays must be sampled with frequency of at least dz<=%.3f" %(self.dz_windows)
        assert n_z.ndim == 2, "'n_z' must be 2-dimensional"
        assert (n_z.shape)[1] == z.shape[0], "Length of each 'n_z[i]' must be the same of 'z'"
        assert z.min() <= self.z_min, "Minimum input redshift must be < %.3f, the minimum redshift of integration" %(self.z_min)
        assert z.max() >= self.z_max, "Maximum input redshift must be > %.3f, the maximum redshift of integration" %(self.z_max)

        def L0 (ll):
            return (2*ll**2+2*ll-1)/((2*ll-1)*(2*ll+3))
        def Lm1 (ll):
            return -ll*(ll-1)/((2*ll-1)*np.sqrt((2*ll-3)*(2*ll+1)))
        def Lp1 (ll):
            return -(ll-1)*(ll+2)/((2*ll+3)*np.sqrt((2*ll+1)*(2*ll+5)))

        # Initialize window
        self.window_function[name] = []
        # Compute window
        for galaxy_bin in xrange(n_bins):

            n_z_interp = si.interp1d(z,n_z[galaxy_bin],'cubic', bounds_error = False, fill_value = 0.)
            f_interp = si.interp1d(z,Om_z**(factor),'cubic', bounds_error = False, fill_value = 0.)

            def Wm1 (ll):
                return Lm1(ll)*n_z_interp(((2*ll+1-4)/(2*ll+1))*z_int)*f_interp(((2*ll+1-4)/(2*ll+1))*z_int)
            def Wz (ll):
                return L0(ll)*n_z_interp(((2*ll+1)/(2*ll+1))*z_int)*f_interp(((2*ll+1)/(2*ll+1))*z_int)
            def Wp1 (ll):
                return Lp1(ll)*n_z_interp(((2*ll+1+4)/(2*ll+1))*z_int)*f_interp(((2*ll+1+4)/(2*ll+1))*z_int)


            self.window_function[name].append([si.interp1d(z_int, (Wm1(l)+Wz(l)+Wp1(l))*H_z*1/norm_const[galaxy_bin], 'cubic', bounds_error = False, fill_value = 0.) for l in ll])

        self.window_function[name]=np.array(self.window_function[name])
            
    #-----------------------------------------------------------------------------------------
    # LSD WINDOW FUNCTION
    #-----------------------------------------------------------------------------------------
    def load_lsd_window_functions(self,bg, h,C, z, n_dl, ll, name = 'LSD'):
        def dL_from_C(C, z, alpha_interp):
            """
            Luminosity distance d_L(z) in Mpc from your colibri cosmology `C`.
            Assumes C.comoving_distance(z) returns comoving distance in Mpc/h.
            """
            z = np.asarray(z, dtype=float)
            chi_Mpc_h = np.asarray(C.comoving_distance(z)) # Mpc/h
            return (1.0 + z) * chi_Mpc_h * exp_factor_alphaM(z,alpha_interp) # Mpc/h

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

        # --- bg interpolators
        z_bg = np.asarray(bg['z'])
        H_interp = interp1d(z_bg, bg['H [1/Mpc]']/C.h, kind='cubic', bounds_error=False, fill_value="extrapolate") #[h/Mpc]
        chi_interp = interp1d(z_bg, bg['comov. dist.']*C.h, kind='cubic', bounds_error=False, fill_value="extrapolate") # [Mpc/h]
        alpha_M_interp = interp1d(z_bg, bg['M2_running_smg'], kind='cubic', bounds_error=False,fill_value="extrapolate")
        rho_b = interp1d(z_bg, bg['(.)rho_b'], kind='cubic', bounds_error=False, fill_value="extrapolate")
        rho_cdm = interp1d(z_bg, bg['(.)rho_cdm'], kind='cubic', bounds_error=False, fill_value="extrapolate")
        rho_crit = interp1d(z_bg, bg['(.)rho_crit'], kind='cubic', bounds_error=False, fill_value="extrapolate")

        n_dl = np.array(n_dl)
        z  = np.array(z)
        z_int= self.z_integration
        n_bins = len(n_dl)
        factor= 6/11

        H_z = H_interp(z) #[h/Mpc]
        Om_z = (rho_b(z) + rho_cdm(z)) / rho_crit(z)

        #print('omega_z NEW', Om_z)

        # H is already in [h/Mpc] and chi in [Mpc/h]  → product is dimensionless
        conf_H = H_interp(z) / (1.0 + z)  # ≡ H(z)/(1+z) H(z) in h/Mpc
        r_conf_H = chi_interp(z) * conf_H  # ≡ χ(z) * H(z)/(1+z)   Mpc/h
        gamma = r_conf_H / (1.0 + r_conf_H)

        # bg branch: H and r from bg
        r_z = chi_interp(z)  # [Mpc/h]
        alpha_M_z = alpha_M_interp(z)
        exp_val = exp_factor_alphaM(z, alpha_M_interp)

        jj = ((1 + z) *1 / H_z + r_z + r_z  * 0.5 * alpha_M_z) * exp_val  # [Mpc/h]

        # n_dl [1/Gpc]
        norm_const = simpson(n_dl *1/h, x=dL_from_C(C, z, alpha_M_interp), axis=1)  #

        #print('conf_H NEW', conf_H) # * C.h * const.c)
        #print('r_conf_H NEW', r_conf_H) # / C.h) * C.h * const.c)
        #print('gamma NEW', gamma)
        #print('jj NEW', jj)
        #print('norm_const NEW', norm_const)


        assert np.all(np.diff(z)<self.dz_windows), "For convergence reasons, the distribution function arrays must be sampled with frequency of at least dz<=%.3f" %(self.dz_windows)
        assert n_dl.ndim == 2, "'n_z' must be 2-dimensional"
        assert (n_dl.shape)[1] == z.shape[0], "Length of each 'n_z[i]' must be the same of 'z'"
        assert z.min() <= self.z_min, "Minimum input redshift must be < %.3f, the minimum redshift of integration" %(self.z_min)
        assert z.max() >= self.z_max, "Maximum input redshift must be > %.3f, the maximum redshift of integration" %(self.z_max)

        def L0 (ll):
            return (2*ll**2+2*ll-1)/((2*ll-1)*(2*ll+3))
        def Lm1 (ll):
            return -ll*(ll-1)/((2*ll-1)*np.sqrt((2*ll-3)*(2*ll+1)))
        def Lp1 (ll):
            return -(ll-1)*(ll+2)/((2*ll+3)*np.sqrt((2*ll+1)*(2*ll+5)))

        H_z=H_interp(z_int) #[h/Mpc]
        # Initialize window
        self.window_function[name] = []
        # Compute window
        for galaxy_bin in xrange(n_bins):
            n_z_interp = si.interp1d(z, n_dl[galaxy_bin],'cubic', bounds_error = False, fill_value = 0.)
            f_interp = si.interp1d(z, Om_z**(factor) * 2 * gamma * jj,'cubic', bounds_error = False, fill_value = 0.)

            def Wm1 (ll):
                return Lm1(ll)*n_z_interp(((2*ll+1-4)/(2*ll+1))*z_int)*f_interp(((2*ll+1-4)/(2*ll+1))*z_int)
            def Wz (ll):
                return L0(ll)*n_z_interp(((2*ll+1)/(2*ll+1))*z_int)*f_interp(((2*ll+1)/(2*ll+1))*z_int)
            def Wp1 (ll):
                return Lp1(ll)*n_z_interp(((2*ll+1+4)/(2*ll+1))*z_int)*f_interp(((2*ll+1+4)/(2*ll+1))*z_int)


            self.window_function[name].append([si.interp1d(z_int, (Wm1(l)+Wz(l)+Wp1(l))*H_z*1/norm_const[galaxy_bin], 'cubic', bounds_error = False, fill_value = 0.) for l in ll])

        self.window_function[name]=np.array(self.window_function[name])

    #-----------------------------------------------------------------------------------------
    # OTHER WINDOW FUNCTION
    #-----------------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------------
    # SHEAR WINDOW FUNCTION
    #-----------------------------------------------------------------------------------------
    def load_shear_window_functions(self, z, n_z, name = 'shear'):
        """
        This function computes the window function for cosmic shear given the galaxy distribution in input.
        The function automatically normalizes the galaxy distribution such that the integral over
        redshifts is 1.
        The routine adds a key (specified in the ``name`` argument) to the ``self.window_function`` dictionary.
        Given a galxy distruibution in a redshift bin :math:`n^{(i)}(z)`, the equation is:

        .. math::

           W^{(i)}_\gamma(z) = \\frac{3}{2}\Omega_m \ \\frac{H_0^2}{c^2} \ f_K[\chi(z)] (1+z) \int_z^\infty dx \ n^{(i)}(x) \ \\frac{f_K[\chi(z-x)]}{f_K[\chi(z)]}


        :param z: array or list of redshift at which the galaxy distribution ``n_z`` is evaluated
        :type z: 1-D array, default = None

        :param n_z: 2-D array or 2-D list where each sublist is the galaxy distribution of a given redshift bin
        :type n_z: 2-D array with shape ``(n_bins, len(z))``, default = None

        :param name: name of the key to add to the dictionary
        :type name: string, default = 'shear'

        An example call can be, for 3 bins all with a :func:`colibri.limber.limber.euclid_distribution` with default arguments for ``a`` and ``b`` but different bin edges ``z_min``, ``z_max``:

        .. code-block:: python

           bin_edges = [0.00, 0.72, 1.11, 5.00]
           nbins     = len(bin_edges)-1
           z_w       = np.linspace(0., 6., 1001)
           n_z_w      = [S.euclid_distribution(z = z_w, a = 2.0, b = 1.5, z_min = bin_edges[i], z_max = bin_edges[i+1]) for i in range(nbins)]
           S.load_shear_window_functions(z = z_w, n_z = n_z_w)

        :return: A key of a given name is added to the ``self.window_function`` dictionary

        """
        n_z = np.array(n_z)
        z  = np.array(z)
        assert np.all(np.diff(z)<self.dz_windows), "For convergence reasons, the distribution function arrays must be sampled with frequency of at least dz<=%.3f" %(self.dz_windows)
        assert n_z.ndim == 2, "'n_z' must be 2-dimensional"
        assert (n_z.shape)[1] == z.shape[0], "Length of each 'n_z[i]' must be the same of 'z'"
        assert z.min() <= self.z_min, "Minimum input redshift must be < %.3f, the minimum redshift of integration" %(self.z_min)
        assert z.max() >= self.z_max, "Maximum input redshift must be > %.3f, the maximum redshift of integration" %(self.z_max)

        # Call a simpler function if Omega_K == 0.
        if self.cosmology.Omega_K == 0.:
            self.load_shear_window_functions_flat(z,n_z,name)
        # Otherwise compute window function in curved geometry
        else:
            # Set number of bins, normalize them, find constant in front
            n_bins = len(n_z)
            norm_const = simpson(n_z, x = z, axis = 1)
            constant = 3./2.*self.cosmology.Omega_m*(self.cosmology.H0/const.c)**2.*(1.+self.z_windows)*self.geometric_factor_windows

            # Initialize windows
            self.window_function[name]  = []

            # Set windows
            for galaxy_bin in xrange(n_bins):
                # Select the n(z) array and do the integral for window function
                n_z = si.interp1d(z,n_z[galaxy_bin],'cubic', bounds_error = False, fill_value = 0.)
                integral = list(map(lambda z_i: sint.quad(lambda x: n_z(x)*
                                                                    self.geometric_factor_f_K(x,z_i)/
                                                                    self.geometric_factor_f_K(x),
                                                                    z_i, self.z_max,
                                                                    epsrel = 1.e-3)[0], self.z_windows))
                
                # Fill temporary window functions with real values
                window_function_tmp    = constant*integral/norm_const[galaxy_bin]
                # Interpolate (Akima interpolator avoids oscillations around the zero due to spline)
                try:               self.window_function[name].append(si.interp1d(self.z_windows,
                                                                                 window_function_tmp,
                                                                                 'cubic',
                                                                                 bounds_error=False,
                                                                                 fill_value=0.))
                except ValueError: self.window_function[name].append(si.Akima1DInterpolator(self.z_windows,
                                                                                 window_function_tmp))

                    
    def load_shear_window_functions_flat(self, z, n_z, name = 'shear'):
        # Set number of bins, normalize them, find constant in front
        n_bins = len(n_z)
        norm_const = simpson(n_z, x = z, axis = 1)
        constant = 3./2.*self.cosmology.Omega_m*(self.cosmology.H0/const.c)**2.*(1.+self.z_windows)*self.geometric_factor_windows

        # Initialize window
        self.window_function[name]  = []
        # Set windows
        chi_max = self.cosmology.comoving_distance(self.z_max,False)
        for galaxy_bin in xrange(n_bins):
            # Select which is the function and which are the arguments
            tmp_interp = si.interp1d(z,n_z[galaxy_bin],'cubic', bounds_error = False, fill_value = 0.)
            n_z_array  = tmp_interp(self.z_windows)
            n_z_interp = si.interp1d(self.geometric_factor_windows, n_z_array*(self.Hubble_windows/const.c), 'cubic', bounds_error = False, fill_value = 0.)
            # Do the integral for window function
            integral = list(map(lambda chi_i: sint.quad(lambda chi: n_z_interp(chi)*(1.-chi_i/chi), chi_i, chi_max, epsrel = 1.e-3)[0], self.geometric_factor_windows))
            # Fill temporary window functions with real values
            window_function_tmp    = constant*integral/norm_const[galaxy_bin]
            # Interpolate (the Akima interpolator avoids oscillations around the zero due to the cubic spline)
            try:
                self.window_function[name].append(si.interp1d(self.z_windows, window_function_tmp, 'cubic', bounds_error = False, fill_value = 0.))
            except ValueError:
                self.window_function[name].append(si.Akima1DInterpolator(self.z_windows, window_function_tmp))


    #-----------------------------------------------------------------------------------------
    # INTRINSIC ALIGNMENT WINDOW FUNCTION
    #-----------------------------------------------------------------------------------------
    def load_IA_window_functions(self, z, n_z, A_IA = 1.0, eta_IA = 0.0, beta_IA = 0.0, lum_IA=1.0, name = 'IA'):
        """
        This function computes the window function for intrinsic alignment given a galaxy distribution.
        The function automatically normalizes the galaxy distribution such that the integral over
        redshifts is 1.
        The routine adds a key (specified in the ``name`` argument) to the ``self.window_function`` dictionary.
        Given a galxy distruibution in a redshift bin :math:`n^{(i)}(z)`, the equation is:

        .. math::

           W^{(i)}_\mathrm{IA}(z) = -\\frac{A_\mathrm{IA} \mathcal C_1 \Omega_\mathrm m}{D_1(k,z)}(1+z)^{\eta_\mathrm{IA}} \left[\\frac{L(z)}{L_*(z)}\\right]^{\\beta_\mathrm{IA}} \ n^{(i)}(z) \\frac{H(z)}{c}


        :param z: array or list of redshift at which the galaxy distribution ``n_z`` is evaluated
        :type z: 1-D array, default = None

        :param n_z: 2-D array or 2-D list where each sublist is the galaxy distribution of a given redshift bin
        :type n_z: 2-D array with shape ``(n_bins, len(z))``, default = None

        :param A_IA: Intrinsic alignment amplitude.
        :type A_IA: float, default = 1

        :param eta_IA: Exponent for redshift dependence of intrinsic alignment.
        :type eta_IA: float, default = 0

        :param beta_IA: Exponent for luminosity dependence of intrinsic alignment.
        :type beta_IA: float, default = 0

        :param lum_IA: Relative luminosity of galaxies :math:`L(z)/L_*(z)`.
        :type lum_IA: float or callable whose **only** argument is :math:`z`, default = 1

        :param name: name of the key to add to the dictionary
        :type name: string, default = 'IA'

        An example call can be, for 3 bins all with a :func:`colibri.limber.limber.euclid_distribution` with default arguments for ``a`` and ``b`` but different bin edges ``z_min``, ``z_max``:

        .. code-block:: python

           bin_edges = [0.00, 0.72, 1.11, 5.00]
           nbins     = len(bin_edges)-1
           z_w       = np.linspace(0., 6., 1001)
           n_z_w      = [S.euclid_distribution(z = z_w, a = 2.0, b = 1.5, z_min = bin_edges[i], z_max = bin_edges[i+1]) for i in range(nbins)]
           S.load_IA_window_functions(z = z_w, n_z = n_z_w, A_IA = 1, eta_IA = 0, beta_IA = 0, lum_IA = 1)

        :return: A key of a given name is added to the ``self.window_function`` dictionary

        """
        n_z = np.array(n_z)
        z  = np.array(z)
        n_bins = len(n_z)
        norm_const = simpson(n_z, x = z, axis = 1)
        assert np.all(np.diff(z)<self.dz_windows), "For convergence reasons, the distribution function arrays must be sampled with frequency of at least dz<=%.3f" %(self.dz_windows)
        assert n_z.ndim == 2, "'n_z' must be 2-dimensional"
        assert (n_z.shape)[1] == z.shape[0], "Length of each 'n_z[i]' must be the same of 'z'"
        assert z.min() <= self.z_min, "Minimum input redshift must be < %.3f, the minimum redshift of integration" %(self.z_min)
        assert z.max() >= self.z_max, "Maximum input redshift must be > %.3f, the maximum redshift of integration" %(self.z_max)
        # Initialize window
        self.window_function[name] = []
        # IA kernel
        F_IA = self.intrinsic_alignment_kernel(self.z_integration,A_IA,eta_IA,beta_IA,lum_IA)
        # Compute window
        for galaxy_bin in xrange(n_bins):
            tmp_interp = si.interp1d(z,n_z[galaxy_bin],'cubic', bounds_error = False, fill_value = 0.)
            self.window_function[name].append(si.interp1d(self.z_integration, tmp_interp(self.z_integration)*F_IA*self.Hubble/const.c/norm_const[galaxy_bin], 'cubic', bounds_error = False, fill_value = 0.))

    #-----------------------------------------------------------------------------------------
    # LENSING WINDOW FUNCTION
    #-----------------------------------------------------------------------------------------
    def load_lensing_window_functions(self, z, n_z, A_IA = 0.0, eta_IA = 0.0, beta_IA = 0.0, lum_IA=1.0, name = 'lensing'):
        """
        This function computes the window function for lensing (comprehensive of shear and intrinsic
        alignment) given a galaxy distribution.
        The function automatically normalizes the galaxy distribution such that the integral over
        redshifts is 1.
        The routine adds a key (specified in the ``name`` argument) to the ``self.window_function`` dictionary.
        See :func:`colibri.limber.limber.load_shear_window_functions` 
        and :func:`colibri.limber.limber.load_IA_window_functions` for the equations.


        :param z: array or list of redshift at which the galaxy distribution ``n_z`` is evaluated
        :type z: 1-D array, default = None

        :param n_z: 2-D array or 2-D list where each sublist is the galaxy distribution of a given redshift bin
        :type n_z: 2-D array with shape ``(n_bins, len(z))``, default = None

        :param A_IA: Intrinsic alignment amplitude.
        :type A_IA: float, default = 1

        :param eta_IA: Exponent for redshift dependence of intrinsic alignment.
        :type eta_IA: float, default = 0

        :param beta_IA: Exponent for luminosity dependence of intrinsic alignment.
        :type beta_IA: float, default = 0

        :param lum_IA: Relative luminosity of galaxies :math:`L(z)/L_*(z)`.
        :type lum_IA: float or callable whose **only** argument is :math:`z`, default = 1

        :param name: name of the key to add to the dictionary
        :type name: string, default = 'lensing'

        An example call can be, for 3 bins all with a :func:`colibri.limber.limber.euclid_distribution` with default arguments for ``a`` and ``b`` but different bin edges ``z_min``, ``z_max``:

        .. code-block:: python

           bin_edges = [0.00, 0.72, 1.11, 5.00]
           nbins     = len(bin_edges)-1
           z_w       = np.linspace(0., 6., 1001)
           n_z_w      = [S.euclid_distribution(z = z_w, a = 2.0, b = 1.5, z_min = bin_edges[i], z_max = bin_edges[i+1]) for i in range(nbins)]
           S.load_lensing_window_functions(z = z_w, n_z = n_z_w, A_IA = 1, eta_IA = 0, beta_IA = 0, lum_IA = 1)

        :return: A key of a given name is added to the ``self.window_function`` dictionary

        """
        # Compute shear and IA
        self.load_shear_window_functions(z,n_z,name='shear_temporary_for_lensing')
        self.load_IA_window_functions(z,n_z,A_IA,eta_IA,beta_IA,lum_IA,name='IA_temporary_for_lensing')
        n_bins = len(self.window_function['shear_temporary_for_lensing'])

        # Initialize window
        self.window_function[name] = []
        for galaxy_bin in xrange(n_bins):
            WL = self.window_function['shear_temporary_for_lensing'][galaxy_bin](self.z_windows)+self.window_function['IA_temporary_for_lensing'][galaxy_bin](self.z_windows)
            try:
                self.window_function[name].append(si.interp1d(self.z_windows, WL, 'cubic', bounds_error = False, fill_value = 0.))
            except ValueError:
                self.window_function[name].append(si.Akima1DInterpolator(self.z_windows, WL))   
        del self.window_function['shear_temporary_for_lensing']
        del self.window_function['IA_temporary_for_lensing']



    #-----------------------------------------------------------------------------------------
    # HI WINDOW FUNCTION
    #-----------------------------------------------------------------------------------------
    def load_HI_window_functions(self, z, n_z, Omega_HI = 6.25e-4, bias = 1., name = 'HI'):
        """
        This function computes the window function for HI brightness temperature given a galaxy distribution.
        The function automatically normalizes the galaxy distribution such that the integral over
        redshifts is 1.
        The routine adds a key (specified in the ``name`` argument) to the ``self.window_function`` dictionary.
        Given a galxy distruibution in a redshift bin :math:`n^{(i)}(z)`, the equation is:

        .. math::

           W^{(i)}_\mathrm{HI}(z) = b(z) \ T_b(z) \ D(z) \ \ n^{(i)}(z) \\frac{H(z)}{c}


        :param z: array or list of redshift at which the galaxy distribution ``n_z`` is evaluated
        :type z: 1-D array, default = None

        :param n_z: 2-D array or 2-D list where each sublist is the galaxy distribution of a given redshift bin
        :type n_z: 2-D array with shape ``(n_bins, len(z))``, default = None

        :param Omega_HI: HI density parameter.
        :type Omega_HI: float, default = 6.25e-4

        :param bias: Galaxy bias.
        :type bias: float or array, same length of ``n_z``, default = 1

        :param name: name of the key to add to the dictionary
        :type name: string, default = 'HI'


        An example call can be, for 3 bins all with a :func:`colibri.limber.limber.euclid_distribution` with default arguments for ``a`` and ``b`` but different bin edges ``z_min``, ``z_max``:

        .. code-block:: python

           bin_edges = [0.00, 0.72, 1.11, 5.00]
           nbins     = len(bin_edges)-1
           z_w       = np.linspace(0., 6., 1001)
           n_z_w      = [S.euclid_distribution(z = z_w, a = 2.0, b = 1.5, z_min = bin_edges[i], z_max = bin_edges[i+1]) for i in range(nbins)]
           S.load_HI_window_functions(z = z_w, n_z = n_z_w, Omega_HI = 0.00063, bias = 1)

        :return: A key of a given name is added to the ``self.window_function`` dictionary

        """

        n_z = np.array(n_z)
        z  = np.array(z)
        n_bins = len(n_z)
        norm_const = simpson(n_z, x = z, axis = 1)
        assert np.all(np.diff(z)<self.dz_windows), "For convergence reasons, the distribution function arrays must be sampled with frequency of at least dz<=%.3f" %(self.dz_windows)
        assert n_z.ndim == 2, "'n_z' must be 2-dimensional"
        assert (n_z.shape)[1] == z.shape[0], "Length of each 'n_z[i]' must be the same of 'z'"
        assert z.min() <= self.z_min, "Minimum input redshift must be < %.3f, the minimum redshift of integration" %(self.z_min)
        assert z.max() >= self.z_max, "Maximum input redshift must be > %.3f, the maximum redshift of integration" %(self.z_max)

        if   isinstance(bias, float): bias = bias*np.ones(n_bins)
        elif isinstance(bias, int)  : bias = np.float(bias)*np.ones(n_bins)
        else:                         assert len(bias)==n_bins, "Number of bias factors different from number of bins"
        # Initialize window
        self.window_function[name] = []
        # Compute window
        Dz = self.cosmology.growth_factor_scale_independent(self.z_integration)
        Tz = self.brightness_temperature_HI(self.z_integration,Omega_HI)
        for galaxy_bin in xrange(n_bins):
            tmp_interp = si.interp1d(z,n_z[galaxy_bin],'cubic', bounds_error = False, fill_value = 0.)
            self.window_function[name].append(si.interp1d(self.z_integration, tmp_interp(self.z_integration)*self.Hubble/const.c/norm_const[galaxy_bin]*bias[galaxy_bin]*Tz*Dz, 'cubic', bounds_error = False, fill_value = 0.))

    #-----------------------------------------------------------------------------------------
    # CMB LENSING WINDOW FUNCTION
    #-----------------------------------------------------------------------------------------
    def load_CMB_lensing_window_functions(self, z, n_z, z_LSS = 1089., name = 'CMB lensing'):
        """
        This function computes the window function for CMB lensing given a galaxy distribution.
        The function automatically normalizes the galaxy distribution such that the integral over
        redshifts is 1.
        The routine adds a key (specified in the ``name`` argument) to the ``self.window_function`` dictionary.
        Given a galxy distruibution in a redshift bin :math:`n^{(i)}(z)`, the equation is:

        .. math::

           W^{(i)}_\mathrm{CMB}(z) = \\frac{3}{2}\Omega_m \ \\frac{H_0^2}{c^2} \ f_K[\chi(z)] (1+z) \ n^{(i)}(z) \\frac{H(z)}{c} \ \\frac{f_K[\chi(z_{LSS})]-f_K[\chi(z)]}{f_K[\chi(z_{LSS})]}


        :param z: array or list of redshift at which the galaxy distribution ``n_z`` is evaluated
        :type z: 1-D array, default = None

        :param n_z: 2-D array or 2-D list where each sublist is the galaxy distribution of a given redshift bin
        :type n_z: 2-D array with shape ``(n_bins, len(z))``, default = None

        :param z_LSS: last-scattering surface redshift.
        :type z_LSS: float, default = 1089

        :param name: name of the key to add to the dictionary
        :type name: string, default = 'CMB lensing'


        An example call can be, for 3 bins all with a :func:`colibri.limber.limber.euclid_distribution` with default arguments for ``a`` and ``b`` but different bin edges ``z_min``, ``z_max``:

        .. code-block:: python

           bin_edges = [0.00, 0.72, 1.11, 5.00]
           nbins     = len(bin_edges)-1
           z_w       = np.linspace(0., 6., 1001)
           n_z_w      = [S.euclid_distribution(z = z_w, a = 2.0, b = 1.5, z_min = bin_edges[i], z_max = bin_edges[i+1]) for i in range(nbins)]
           S.load_CMB_window_functions(z = z_w, n_z = n_z_w, z_LSS = 1089.)

        :return: A key of a given name is added to the ``self.window_function`` dictionary

        """
        n_z = np.array(n_z)
        z  = np.array(z)
        n_bins = len(n_z)
        norm_const = simpson(n_z, x = z, axis = 1)
        constant = 3./2.*self.cosmology.Omega_m*(self.cosmology.H0/self.cosmology.h/const.c)**2.*(1.+self.z_windows)*self.geometric_factor_windows
        assert np.all(np.diff(z)<self.dz_windows), "For convergence reasons, the distribution function arrays must be sampled with frequency of at least dz<=%.3f" %(self.dz_windows)
        assert n_z.ndim == 2, "'n_z' must be 2-dimensional"
        assert (n_z.shape)[1] == z.shape[0], "Length of each 'n_z[i]' must be the same of 'z'"
        assert z.min() <= self.z_min, "Minimum input redshift must be < %.3f, the minimum redshift of integration" %(self.z_min)
        assert z.max() >= self.z_max, "Maximum input redshift must be > %.3f, the maximum redshift of integration" %(self.z_max)

        # Initialize window
        self.window_function[name] = []
        
        # Comoving distance to last scattering surface
        com_dist_LSS = self.geometric_factor_f_K(z_LSS)
        # Comoving distances to redshifts
        for galaxy_bin in xrange(n_bins):
            tmp_interp = si.interp1d(z,n_z[galaxy_bin],'cubic', bounds_error = False, fill_value = 0.)
            self.window_function[name].append(si.interp1d(self.z_windows, constant*tmp_interp(self.z_windows)/norm_const[galaxy_bin]*self.Hubble_windows/const.c*(com_dist_LSS-self.geometric_factor_windows)/com_dist_LSS, 'cubic', bounds_error = False, fill_value = 0.))


    #-----------------------------------------------------------------------------------------
    # ADD WINDOW FUNCTION
    #-----------------------------------------------------------------------------------------
    def load_custom_window_functions(self, z, window, name):
        """
        This function loads a custom window function and adds the key to the dictionary
        The window function in input must already be normalized.

        :param z: array or list of redshift at which the galaxy distribution ``n_z`` is evaluated
        :type z: 1-D array, default = None

        :param window: 2-D array or 2-D list where each sublist is the galaxy distribution of a given redshift bin
        :type window: 2-D array with shape ``(n_bins, len(z))``, default = None

        :param name: name of the key to add to the dictionary
        :type name: string

        :return: A key of a given name is added to the ``self.window_function`` dictionary

        """
        n_z = np.array(window)
        z  = np.array(z)
        n_bins = len(n_z)
        assert np.all(np.diff(z)<self.dz_windows), "For convergence reasons, the distribution function arrays must be sampled with frequency of at least dz<=%.3f" %(self.dz_windows)
        assert n_z.ndim == 2, "'n_z' must be 2-dimensional"
        assert (n_z.shape)[1] == z.shape[0], "Length of each 'n_z[i]' must be the same of 'z'"
        assert z.min() <= self.z_min, "Minimum input redshift must be < %.3f, the minimum redshift of integration" %(self.z_min)
        assert z.max() >= self.z_max, "Maximum input redshift must be > %.3f, the maximum redshift of integration" %(self.z_max)   
         # Initialize window
        self.window_function[name] = []
        # Compute window
        for galaxy_bin in xrange(n_bins):
            self.window_function[name].append(si.interp1d(z,window[galaxy_bin],'cubic', bounds_error = False, fill_value = 0.))

    #-----------------------------------------------------------------------------------------
    # CORRECTION FUNCTION FOR INTRINSIC ALIGNMENT
    #-----------------------------------------------------------------------------------------
    def intrinsic_alignment_kernel(self, z, A_IA = 0., eta_IA = 0., beta_IA = 0., lum_IA = 1.):
        # Constants in front
        C1     = 0.0134
        front  = -A_IA*C1*self.cosmology.Omega_m
        # Growth factors
        growth = self.cosmology.growth_factor_scale_independent(z = z)
        # Relative luminosity is either a function or a float
        if   callable(lum_IA):          rel_lum = lum_IA(z)
        elif isinstance(lum_IA, float): rel_lum = lum_IA
        else:                           raise TypeError("'lum_IA' must be either a float or a function with redshift as the only argument.")
        return front/growth*(1.+z)**eta_IA*rel_lum**beta_IA

    #-----------------------------------------------------------------------------------------
    # BRIGHTNESS TEMPERATURE OF HI
    #-----------------------------------------------------------------------------------------
    def brightness_temperature_HI(self,z,Omega_HI):
        TB0        = 44. # micro-K
        Hz         = self.cosmology.H_massive(z)
        Omega_HI_z = Omega_HI*(1.+z)**3.*self.cosmology.H0**2./Hz**2.
        return TB0*Omega_HI_z*self.cosmology.h/2.45e-4*(1.+z)**2.*self.cosmology.H0/Hz

    # -----------------------------------------------------------------------------------------
    # NEW GEOMETRIC FACTOR FROM THE BACKGROUND
    # -----------------------------------------------------------------------------------------



    #-----------------------------------------------------------------------------------------
    # ANGULAR SPECTRA
    #-----------------------------------------------------------------------------------------
    def limber_angular_power_spectra(self,bg,h, l, windows = None):
        """
        This function computes the angular power spectra (using the Limber's and the flat-sky approximations) for the window function specified.
        Given two redshift bins `i` and `j` the equation is

        .. math::

          C^{(ij)}(\ell) = \int_0^\infty dz \ \\frac{1}{H(z)} \ \\frac{W^{(i)}(z) W^{(j)}(z)}{f_K^2[\chi(z)]} \ P\left(\\frac{\ell}{f_K[\chi(z)]}, z\\right),

        where :math:`P(k,z)` is the matter power spectrum and :math:`W^{(i)}(z)` are the window functions.

        :param alpha_M:
        :param l: Multipoles at which to compute the shear power spectra.
        :type l: array

        :param windows: which spectra (auto and cross) must be computed. If set to ``None`` all the spectra will be computed.
        :type windows: list of strings, default = ``None``

        :return: dictionary whose keys are combinations of window functions specified in ``windows``. Each key is a 3-D array whose entries are ``Cl[bin i, bin j, multipole l]``
        """

        def geometric_factor_bg(chi_interp, z, h, K_h=0.0, z0=0.0):
            """
            Return f_K(chi[z]-chi[z0]) in units of Mpc/h.
            bg['comov. dist.'] is assumed in Mpc (flat comoving distance).
            K_h must be in (h/Mpc)^2.
            """
            chi_Mpc = chi_interp(z) - chi_interp(z0)  # Mpc
            #chi_h = h * chi_Mpc  # Mpc/h

            if np.allclose(K_h, 0.0):
                return  chi_Mpc #chi_h  flat: f_K(chi)=chi  [Mpc/1]

            if np.any(K_h > 0):
                # closed: f_K = sin(sqrt(K)*chi)/sqrt(K)
                return np.sin(np.sqrt(K_h) * chi_Mpc) / np.sqrt(K_h)  # Mpc/1
            else:
                # open:   f_K = sinh(sqrt(|K|)*chi)/sqrt(|K|)
                Kabs = np.abs(K_h)
                return np.sinh(np.sqrt(Kabs) * chi_Mpc) / np.sqrt(Kabs)  # Mpc/1

        # 1) Define lengths and quantities
        zz = self.z_integration
        n_l = len(np.atleast_1d(l))
        n_z = self.n_z_integration

        ################################################
        # the previous term was: cH_chi2  = self.c_over_H_over_chi_squared where the curvature is considered
        # Factor c/H(z)/f_K(z)^2
        # self.c_over_H_over_chi_squared = const.c/self.Hubble/self.geometric_factor**2

        # --- bg interpolators
        z_bg = np.asarray(bg['z'])
        H_interp = interp1d(z_bg, bg['H [1/Mpc]'], kind='cubic', bounds_error=False, fill_value="extrapolate") # 1/Mpc
        chi_interp = interp1d(z_bg, bg['comov. dist.'], kind='cubic', bounds_error=False,fill_value="extrapolate")  # [Mpc]

        # set the right range
        #H_interp_z_int = H_interp(zz)  # 1/Mpc
        geom_int = geometric_factor_bg(chi_interp, zz, h)  # shape: (n_z,)  in Mpc
        H_inverse = (1 / H_interp(zz)) * (1 / (geom_int ** 2)) #[Mpc/1]*[1/Mpc^2] = [1/Mpc]

        #print('H_inverse', H_inverse)

        # Check existence of power spectrum
        try: self.power_spectra_interpolator
        except AttributeError: raise AttributeError("Power spectra have not been loaded yet")

        # Check convergence with (l, k, z):
        assert np.atleast_1d(l).min() > self.k_min*h*geometric_factor_bg(chi_interp, self.z_min,h), "Minimum 'l' is too low. Extend power spectra to lower k_min? Use lower z_min for power spectrum?"
        assert np.atleast_1d(l).max() < self.k_max*h*geometric_factor_bg(chi_interp, self.z_max,h), "Maximum 'l' is too high. Extend power spectra to higher k_max? Use higher z_max for power spectrum?"

        # Check window functions to use
        if windows is None:
            windows_to_use = self.window_function
        else:
            windows_to_use = {}
            for key in windows:
                try:             windows_to_use[key] = self.window_function[key]
                except KeyError: raise KeyError("Requested window function '%s' not known" %key)

        # Check window functions 
        keys   = windows_to_use.keys()
        if len(keys) == 0: raise AttributeError("No window function has been computed!")
        n_keys  = len(keys)
        n_bins=[len(windows_to_use[key]) for idx,key in enumerate(keys)]

        Cl = {}
        for i,ki in enumerate(keys):
            for j,kj in enumerate(keys):
                Cl['%s-%s' %(ki,kj)] = np.zeros((n_bins[i],n_bins[j], n_l))


        #start = time.time()
        # 2) Load power spectra: use geom_int instead of self.geometric_factor
        power_spectra = self.power_spectra_interpolator
        PS_lz = np.zeros((n_l, n_z))
        for il in xrange(n_l):
            for iz in xrange(n_z):
                k_Mpc = (l[il] + 0.5) / geom_int[iz]  # k in 1/Mpc
                PS_lz[il, iz] = power_spectra([(k_Mpc, zz[iz])]) #PS_lz in (Mpc/1)^3
        #end = time.time()
        #print(f"Time took {end - start:.4f} seconds  for LOADING PS \n")

        # --- curvature correction ---
        #print('curvature',self.cosmology.K)
        if self.cosmology.K != 0.:
            print('I am in the curvature!')
            KK = self.cosmology.K  # should be in (1/Mpc)^2 to match geom_int in Mpc/1
            factor = np.zeros((n_l, n_z))
            for il, ell in enumerate(l):
                # note: ((ell+0.5)/geom_int)**2 broadcasts over z
                factor[il] = (1 - np.sign(KK) * ell ** 2 / (((ell + 0.5) / geom_int) ** 2 + KK)) ** -0.5
            PS_lz *= factor

        #start = time.time()
        # 3) load Cls given the source functions
        # 1st key (from 1 to N_keys)
        for index_X in xrange(n_keys):
            key_X = list(keys)[index_X]
            W_X = np.array([[windows_to_use[key_X][i,j](zz) for j in xrange(n_l)] for i in xrange(n_bins[index_X])])
            # 2nd key (from 1st key to N_keys)
            for index_Y in xrange(index_X,n_keys):
                key_Y = list(keys)[index_Y]
                W_Y = np.array([[windows_to_use[key_Y][i,j](zz) for j in xrange(n_l)] for i in xrange(n_bins[index_Y])])

                '''
                if {key_X, key_Y} == {"GW", "galaxy"}:
                    for bin_i in range(n_bins[index_X]):
                        for bin_j in range(n_bins[index_Y]):
                            if bin_i==bin_j:
                                xx = 0  # any ell index, windows are z-only
                                WG = W_X[bin_i, xx] if key_X == "galaxy" else W_Y[bin_j, xx]
                                WGW = W_X[bin_i, xx] if key_X == "GW" else W_Y[bin_j, xx]

                                plt.figure(figsize=(6, 4))
                                plt.plot(zz, WGW, label=f"W_GW (bin {bin_i if key_X == 'GW' else bin_j})")
                                plt.plot(zz, WG, label=f"W_gal (bin {bin_i if key_X == 'galaxy' else bin_j})")
                                plt.plot(zz, H_inverse*WGW * WG, "--", label="1/H * 1/r^2 *product")
                                plt.xlabel("z")
                                plt.ylabel("window")
                                plt.legend()
                                plt.tight_layout()
                                outpath = os.path.join(outdir,f"windows_GW_gal_bin{bin_i}_bin{bin_j}_0.png")
                                plt.savefig(outpath, dpi=300, bbox_inches="tight")

                                plt.close()

                if {key_X, key_Y} == {"GW", "GW"}:
                    for bin_i in range(n_bins[index_X]):
                        for bin_j in range(n_bins[index_Y]):
                            if bin_i==bin_j:
                                xx = 0  # any ell index, windows are z-only
                                #WG = W_X[bin_i, xx] if key_X == "galaxy" else W_Y[bin_j, xx]
                                WGW = W_X[bin_i, xx] if key_X == "GW" else W_Y[bin_j, xx]

                                plt.figure(figsize=(6, 4))
                                plt.plot(zz, WGW, label=f"W_GW (bin {bin_i if key_X == 'GW' else bin_j})")
                                #plt.plot(zz, WG, label=f"W_gal (bin {bin_i if key_X == 'galaxy' else bin_j})")
                                plt.plot(zz, H_inverse*WGW * WGW, "--", label="1/H * 1/r^2 *product")
                                plt.xlabel("z")
                                plt.ylabel("window")
                                plt.legend()
                                plt.tight_layout()
                                outpath = os.path.join(outdir,f"windows_GW_GW_bin{bin_i}_bin{bin_j}_0.png")
                                plt.savefig(outpath, dpi=300, bbox_inches="tight")
                                plt.close()
                '''
                # Symmetry C_{AA}^{ij} == C_{AA}^{ji}
                if key_X == key_Y:
                    for bin_i in xrange(n_bins[index_X]):
                        for bin_j in xrange(bin_i, n_bins[index_Y]):
                            Cl['%s-%s' %(key_X,key_Y)][bin_i,bin_j] = [simpson(H_inverse*W_X[bin_i,xx]*W_Y[bin_j,xx]*PS_lz[xx], x = zz) for xx in xrange(n_l)]
                            Cl['%s-%s' %(key_X,key_Y)][bin_j,bin_i] = Cl['%s-%s' %(key_X,key_Y)][bin_i,bin_j]
                # Symmetry C_{AB}^{ij} == C_{BA}^{ji}
                else:
                    for bin_i in xrange(n_bins[index_X]):
                        for bin_j in xrange(n_bins[index_Y]):
                            #                                                    [1/Mpc] *  [1/Mpc] * [1/Mpc] * (Mpc)^3
                            Cl['%s-%s' %(key_X,key_Y)][bin_i,bin_j] = [simpson(H_inverse*W_X[bin_i,xx]*W_Y[bin_j,xx]*PS_lz[xx], x = zz) for xx in xrange(n_l)]
                            Cl['%s-%s' %(key_Y,key_X)][bin_j,bin_i] = Cl['%s-%s' %(key_X,key_Y)][bin_i,bin_j]

        #end = time.time()
        #print(f"Time took {end - start:.4f} seconds for FULL Cl FINAL \n")
        return Cl

    #-----------------------------------------------------------------------------------------
    # ANGULAR POWER SPECTRA AUTO CORRELATION LENSING
    #-----------------------------------------------------------------------------------------   
    def limber_angular_power_spectra_lensing_auto(self,bg, l,  s_gal, beta, windows = None, n_points=20, n_points_x=20, grid_x='mix', n_low=5, n_high=5, Delta_z=0.05, z_min=1e-05):

        def geometric_factor_bg(bg, z, z_bg, z0=0.):
            """
            Geometric factor (distance) between two given redshifts ``z`` and ``z0``.
            It assumes neutrinos as matter, which is a good approximation at low redshifts.
            In fact, this latter assumption introduces a bias of less than 0.02% at :math:`z<10`
            for even the lowest neutrino masses allowed by particle physics.

            :param z: Redshifts.
            :type z: array

            :param z0: Pivot redshift.
            :type z0: float, default = 0

            :return: array
            """
            # Mpc
            chi_interp = interp1d(z_bg, bg['comov. dist.'], kind='cubic', bounds_error=False, fill_value="extrapolate")

            # Curvature in (h/Mpc)^2 units. Then I will take the sqrt and it will
            # go away with comoving_distance(z), giving a final result in units of Mpc/h
            K = 0
            chi_z0 = chi_interp(z0)
            chi_z = chi_interp(z) - chi_z0
            # Change function according to sign of K
            if K == 0.:
                return chi_z  # Mpc
            elif K > 0.:
                return 1. / K ** 0.5 * np.sin(K ** 0.5 * chi_z)  # Mpc
            else:
                return 1. / np.abs(K) ** 0.5 * np.sinh(np.abs(K) ** 0.5 * chi_z)  # Mpc

        z_bg = bg['z']
        H_interp = interp1d(z_bg, bg['H [1/Mpc]'], kind='cubic', bounds_error=False, fill_value="extrapolate") # [1/Mpc]
        comoving_distance_interp = interp1d(z_bg, bg["comov. dist."], kind='cubic', bounds_error=False,fill_value="extrapolate") # [Mpc]
        alpha_M_interp = interp1d(z_bg, bg['M2_running_smg'], kind='cubic', bounds_error=False,fill_value="extrapolate")

        # Check existence of power spectrum
        try: self.power_spectra_interpolator
        except AttributeError: raise AttributeError("Power spectra have not been loaded yet")

        # Check convergence with (l, k, z):
        assert np.atleast_1d(l).min() > self.k_min * geometric_factor_bg(bg, self.z_min,z_bg), "Minimum 'l' is too low. Extend power spectra to lower k_min? Use lower z_min for power spectrum?"
        assert np.atleast_1d(l).max() < self.k_max * geometric_factor_bg(bg, self.z_max,z_bg), "Maximum 'l' is too high. Extend power spectra to higher k_max? Use higher z_max for power spectrum?"

        # Check window functions to use
        if windows is None:
            windows_to_use = self.window_function
        else:
            windows_to_use = {}
            for key in windows:
                try:             windows_to_use[key] = self.window_function[key]
                except KeyError: raise KeyError("Requested window function '%s' not known" %key)

        # Check window functions 
        keys   = windows_to_use.keys()
        if len(keys) == 0: raise AttributeError("No window function has been computed!")
        n_keys  = len(keys)
        n_bins = [len(windows_to_use[key]) for key in keys]
        
        # 1) Define lengths and quantities
        #zz       = self.z_integration
        n_l      = len(np.atleast_1d(l))
        #n_z      = self.n_z_integration # this is now n_points

        Cl       = {}
        for i,ki in enumerate(keys):
            for j,kj in enumerate(keys):
                Cl['%s-%s' %(ki,kj)] = np.zeros((n_bins[i],n_bins[j], n_l))

        z_interp_bias = np.linspace(z_min, 10, 2000)

        #print('z',z_bg)
        #print('H',H_interp)
        #print('comoving distance',comoving_distance_interp)
        #print('alpha_M',alpha_M_interp)

        def exponential_value(x, alpha_interp):
            integrand = lambda z_: 0.5 * alpha_interp(z_) / (1 + z_)
            integral_val, _ = sint.quad(integrand, 0, x, epsabs=1e-6)
            return np.exp(integral_val)

        # JJ is here 1D
        print('CONTROLLA QUA LA JJ, POTREBBE ESSERE CHE SONO TUTTE UGUALI')
        def JJ(tracer, x, H_interp, comoving_distance_interp, alpha_M_interp):
                z = np.atleast_1d(x)
                print('z',x)
                if 'gal' in tracer:
                    return np.ones_like(z)
                else:
                    H_z = H_interp(z)
                    r_z = comoving_distance_interp(z)
                    alphaM_z = alpha_M_interp(z)
                    exp_val = np.array([exponential_value(zi, alpha_M_interp) for zi in z])
                    jj=((1.0 + z) / H_z + r_z + 0.5 * r_z * alphaM_z) * exp_val
                    print('jj',jj)
                    return jj

        def A_L(chi, z, tracer, r_z1, bs, H_vals):
            if 'gal' in tracer:
                return 0.5 * (5 * bs - 2) * (r_z1 - chi) / chi
            else:
                conf_H = H_vals / (1 + z)
                #print('conf_H NEW', conf_H)
                return 0.5 * (((r_z1 - chi) / chi) * (bs - 2) + (1 / (1 + r_z1 * conf_H)))


        # 2) Load power spectra
        power_spectra = self.power_spectra_interpolator

        '''
        #Add curvature correction (see arXiv:2302.04507)
        if self.cosmology.K != 0.:
            KK = self.cosmology.K
            factor = np.zeros((n_l,n_z))
            for il,ell in enumerate(l):
                factor[il]=(1-np.sign(KK)*ell**2/(((ell+0.5)/self.geometric_factor)**2+KK))**-0.5
            PS_lz *= factor
        '''

        def lensing_int(H_interp,comoving_distance_interp,alpha_M_interp,z1, z2, ll, bin_i, bin_j, WX_, WY_, tX, tY,b1,b2, n_points=30, p=1, z_min=1e-04, grid='mix', show_plot=False):

            if grid == 'mix':
                x1 = np.linspace(z_min * (1 + 0.01), z2, n_points)
                x2 = np.geomspace(z_min, z2 * (1 - 0.01), n_points)
                x = np.zeros((2 * x1.shape[0], x1.shape[1], x1.shape[2]))
                for i in xrange(x1.shape[1]):
                    for j in xrange(x1.shape[2]):
                        x_ = np.sort(np.unique(np.concatenate([x1[:, i, j], x2[:, i, j]])))
                        if len(x_) < len(x):
                            n_ = len(x) - len(x_)
                            x_ = np.sort(np.unique(
                                np.concatenate([x_, np.linspace(z_min * (1 + 0.05), z2[i, j] * (1 - 0.05), n_)])))
                        x[:, i, j] = x_
            elif grid == 'lin':
                x = np.linspace(z_min, z2, n_points)
            elif grid == 'geom':
                # not recommended
                x = np.geomspace(z_min, z2, n_points)

            # [1 / Mpc]
            r_1 =  comoving_distance_interp(z1)
            r_2 =  comoving_distance_interp(z2)
            r_x =  comoving_distance_interp(x)
            H_x = H_interp(x)

            #print('NEW')
            #print('A_lx', A_L(r_x, x, tX, r_1, b1(x), H_x))
            #print('A_ly', A_L(r_x, x, tY, r_2, b2(x), H_x))
            #print('rx', r_x)
            #print('H', 1 / H_x)

            # t1 = np.transpose(A_L(r_x, x, tX, r_1, b1(x), H_x) * A_L( r_x, x, tY, r_2, b2(x), H_x) * (1 + x)**2 *1 / H_x * r_x**2, (1,2,0) )
            t1 = np.transpose(
                A_L(r_x, x, tX, r_1, b1(x), H_x) * A_L(r_x, x, tY, r_2, b2(x), H_x)
                * ((1.0 + x) ** 2)
                *1 / H_x * (r_x**2),
                (1,2, 0)
            )

            if show_plot:
                idx = 9
    
                plt.plot( z1, WX_)
                plt.scatter( z1, WX_)
                plt.plot( z2[:, idx], WY_[:, idx], ls='--')
                plt.scatter( z2[:, idx], WY_[:, idx], ls='--')
                plt.axvline(z1[idx])
                plt.title( str(bin_i)+str(bin_j)+', '+tX+'-'+tY)
                #plt.yscale('log')
                plt.show()
                plt.close()
    
                plt.plot( x[:, idx, idx], t1[idx, idx, :] )
                plt.scatter( x[:, idx, idx], t1[idx, idx, :] )
                plt.yscale('log')
                plt.xscale('log')
                plt.show()
                plt.close()
                plt.plot(x[:, idx, idx], t1[idx, idx, :] )
                plt.scatter(x[:, idx, idx], t1[idx, idx, :] )
                #plt.yscale('log')
                #plt.xscale('log')
                plt.show()
                plt.close()

            PS_ = np.squeeze(np.asarray([[[[power_spectra((xx, yy)) for xx, yy in
                                            zip((ll[l] + 0.5) / r_x[:, k, i], x[:, k, i])] for i in range(len(z1))] for k
                                          in range(len(z2))] for l in range(len(ll))]))

            my_int = t1[None, :, :, :, ] * PS_

            I1_ = np.asarray( [[[ trapezoid( my_int[l, i2, i1], x=x[:, i2, i1], axis=0 ) for i1 in range(len(z1))] for i2 in range(len(z2))] for l in range(len(ll))])


            if show_plot:
                plt.plot(x[:, idx, idx], my_int[15, idx, idx, :] )
                plt.scatter(x[:, idx, idx], my_int[15, idx, idx, :] )
                plt.yscale('log')
                plt.xscale('log')
                plt.show()
                plt.close()
                plt.plot(x[:, idx, idx], my_int[15, idx, idx, :] )
                plt.scatter(x[:, idx, idx], my_int[15, idx, idx, :] )
                plt.show()
                plt.close()
            
            I2_ = np.asarray( [[ trapezoid( WY_[:, i1] * JJ(tY, z2[:, i1],H_interp, comoving_distance_interp, alpha_M_interp)*I1_[l, :, i1]/r_2[:, i1], x=z2[:, i1], axis=0 ) for i1 in range(len(z1))] for l in range(len(ll))])

            I3_ =  trapezoid( WX_ * JJ(tX, z1,H_interp, comoving_distance_interp, alpha_M_interp)*I2_/r_1, x=z1, axis=1 )

            return I3_

        # 3) load Cls given the source functions
        b_interp_gal = si.interp1d(z_interp_bias, s_gal, 'cubic', bounds_error=False, fill_value=0.)
        b_interp_beta = si.interp1d(z_interp_bias, beta, 'cubic', bounds_error=False, fill_value=0.)

        # 1st key (from 1 to N_keys)
        for index_X in xrange(n_keys):
            key_X = list(keys)[index_X]

            # choose z grid in each bin so that it is denser around the peak
            bins_ = self.bin_edges[key_X]
            #bins_centers_ = (bins_[:-1] + bins_[1:]) / 2
            zzs = []
            ll_ = 0
            for bin_i in xrange(n_bins[index_X]):
                if 'gal' in key_X:
                    max_z = 5 # check this
                    n_pts = n_high
                else:
                    max_z = 10 # check this
                    n_pts = 2*n_high

                '''
                if max(z_min, bins_[bin_i]*(1-5*Delta_z)-0.01)==z_min:
                    n1 = n_points+n_low
                else:
                    n1 = n_points
                '''
                if bin_i<n_bins[index_X]: 
                    my_arr = np.sort( np.unique(np.concatenate( [np.linspace( max(z_min, bins_[bin_i]*(1-5*Delta_z) ), bins_[bin_i+1]*(1+5*Delta_z), n_points ), np.linspace( z_min, max(z_min, bins_[bin_i]*(1-5*Delta_z)-0.01), n_low ), np.linspace( bins_[bin_i+1]*(1+0.05)+0.01, max_z, n_pts )] )))
                else:
                    my_arr = np.sort( np.unique(np.concatenate( [np.linspace( max(z_min, bins_[bin_i]*(1-5*Delta_z) ), max_z, n_points+n_pts ), np.linspace( z_min, max(z_min, bins_[bin_i]*(1-5*Delta_z)-0.01), n_low )] )))
                l_ = len(my_arr)
                if l_>ll_:
                    ll_=l_
                zzs.append( my_arr )
            
            for i,a in enumerate(zzs):
                #print('len %s: %s'%(i, len(a)))
                if not len(a)==l_:
                    n_ = l_-len(a)
                    #print('adding %s'%n_)
                    zzs[i] =  np.sort( np.unique(np.concatenate([zzs[i], np.linspace(z_min*(1+0.01), max(a)*(1-0.01), n_ ) ] )))
                    #print('new len %s: %s'%(i, len(zzs[i])))
            try:
                zzs = np.asarray(zzs)
            except Exception as e:
                print(zzs)
                print(z_max)
                print(e)

            # now compute window functions
            W_X = np.array([[windows_to_use[key_X][i,j](zzs[i]) for j in range(n_l)] for i in range(n_bins[index_X])])

            # 2nd key (from 1st key to N_keys)
            for index_Y in xrange(n_keys):
                key_Y = list(keys)[index_Y]
                     
                # Symmetry C_{AA}^{ij} == C_{AA}^{ji}
                # NOW THIS IS NOT TRUE FOR LENSING! 
                for bin_i in xrange(n_bins[index_X]):
                    my_range = xrange(n_bins[index_Y])
                    for bin_j in my_range:
                        if  (bin_j==bin_i): #and (key_Y==key_X):                      and ('gal' in key_Y) and (bin_j<=2)
                            # print( 'computing %s %s %s %s'%(key_Y, key_X, bin_i, bin_j))
                            # this could also made finer by adapting the grid. 
                            # For now we leave as it is
                            z2s_ = np.linspace(z_min, zzs[bin_i], n_points )
                            W_Y = np.array([[windows_to_use[key_Y][i,j](z2s_) for j in range(n_l)] for i in range(n_bins[index_Y])])
                            WY = W_Y[bin_j, 0]
                            WX = W_X[bin_i, 0]

                            b1 = b_interp_gal if 'gal' in key_X else b_interp_beta
                            b2 = b_interp_gal if 'gal' in key_Y else b_interp_beta

                            Cl['%s-%s' %(key_X,key_Y)][bin_i,bin_j,:] = l**2*(l+1)**2/((l+0.5)**4)*lensing_int(H_interp,comoving_distance_interp,alpha_M_interp ,zzs[bin_i], z2s_, l, bin_i, bin_j, WX, WY, key_X, key_Y,b1,b2,
                                                z_min=z_min, n_points=n_points_x, grid=grid_x)
        return Cl

    # -----------------------------------------------------------------------------------------
    # ANGULAR POWER SPECTRA CROSS CORRELATION WITH LENSING
    # -----------------------------------------------------------------------------------------
    def limber_angular_power_spectra_lensing_cross(
            self,bg, l, s_gal, beta, windows=None,
            n_points=20, n_points_x=20, grid_x='mix', n_low=5, n_high=5,
            Delta_z=0.05, z_min=1e-5):


        z_interp_bias = np.linspace(z_min, 10, 2000)

        z_bg = bg['z']
        H_interp = interp1d(z_bg, bg['H [1/Mpc]'], kind='cubic', bounds_error=False, fill_value="extrapolate")  # 1/Mpc
        comoving_distance_interp = interp1d(z_bg, bg["comov. dist."], kind='cubic', bounds_error=False,
                                            fill_value="extrapolate")  # Mpc
        alpha_M_interp = interp1d(z_bg, bg['M2_running_smg'], kind='cubic', bounds_error=False,
                                  fill_value="extrapolate")

        # Check existence of power spectrum
        try:
            self.power_spectra_interpolator
        except AttributeError:
            raise AttributeError("Power spectra have not been loaded yet")

        # Convergence checks on (l, k, z)
        assert np.atleast_1d(l).min() > self.k_min * self.geometric_factor_f_K(self.z_min), \
            "Minimum 'l' is too low. Extend power spectra to lower k_min? Use lower z_min for power spectrum?"
        assert np.atleast_1d(l).max() < self.k_max * self.geometric_factor_f_K(self.z_max), \
            "Maximum 'l' is too high. Extend power spectra to higher k_max? Use higher z_max for power spectrum?"

        # Window functions to use
        if windows is None:
            windows_to_use = self.window_function
        else:
            windows_to_use = {}
            for key in windows:
                try:
                    windows_to_use[key] = self.window_function[key]
                except KeyError:
                    raise KeyError("Requested window function '%s' not known" % key)

        keys = list(windows_to_use.keys())
        if len(keys) == 0:
            raise AttributeError("No window function has been computed!")
        nkeys = len(keys)
        n_bins = [len(windows_to_use[key]) for key in keys]

        n_l = len(np.atleast_1d(l))
        Cl = {}
        for i, ki in enumerate(keys):
            for j, kj in enumerate(keys):
                Cl['%s-%s' % (ki, kj)] = np.zeros((n_bins[i], n_bins[j], n_l))

        def exponential_value(x, alpha_interp):
            integrand = lambda z_: 0.5 * alpha_interp(z_) / (1 + z_)
            integral_val, _ = sint.quad(integrand, 0, x, epsabs=1e-6)
            return np.exp(integral_val)

        def JJ(tracer, x, H_interp, comoving_distance_interp, alpha_M_interp):

            # x shape = (13, 23)
            Nz, Nl = x.shape
            jj = np.zeros((Nz, Nl))

            if 'gal' in tracer:
                return np.ones((Nz, Nl))
            else:
                for j in range(Nl):  # loop  over the second shape
                    for i in range(Nz):  # loop over the bins
                        xi = x[i, j]  # scalar
                        #print('xi',xi)

                        H_x = H_interp(xi)
                        r_x = comoving_distance_interp(xi)
                        exp_val = exponential_value(xi, alpha_M_interp)
                        alphaM_x = alpha_M_interp(xi)

                        jj[i, j] = ((1 + xi) / H_x + r_x + r_x * 0.5 * alphaM_x) * exp_val

                return jj  # shape (13,23)

        def A_L(chi, z, tracer, r_z1, bs, H_vals):
            if 'gal' in tracer:
                return 0.5 * (5 * bs - 2) * (r_z1 - chi) / chi
            else:
                conf_H = H_vals / (1 + z)
                return 0.5 * (((r_z1 - chi) / chi) * (bs - 2) + (1 / (1 + r_z1 * conf_H)))

        power_spectra = self.power_spectra_interpolator  # expects points=(k,z)

        # ---------------- core integrand ----------------
        def lensing_int(H_interp, comoving_distance_interp,alpha_M_interp, z1, z2, ll, bin_i, bin_j, WX_, WY_, tX, tY, n_points=30, p=1, z_min=1e-04, grid='mix', show_plot=False):

            #print('z2 before',z2.shape)
            #print('z1',z1)
            #print('z2',z2)

            r_1 = comoving_distance_interp(z1)  # (Nz1,) Mpc
            z2 = np.asarray(z2)  # (Nz2, Nz1)
            r_2 = comoving_distance_interp(z2)  # (Nz2, Nz1) Mpc
            H_x = H_interp(z2)  # (Nz2, Nz1) #1/Mpc

            #print('z2 after', z2.shape)

            if 'gal' in tX:
                b1 = si.interp1d(z_interp_bias, s_gal, kind='cubic', bounds_error=False, fill_value=0.0)
            else:
                b1 = si.interp1d(z_interp_bias, beta, kind='cubic', bounds_error=False, fill_value=0.0)

            if 'gal' in tY:
                b2 = si.interp1d(z_interp_bias, s_gal, kind='cubic', bounds_error=False, fill_value=0.0)
            else:
                b2 = si.interp1d(z_interp_bias, beta, kind='cubic', bounds_error=False, fill_value=0.0)

            #print("Shapes:")
            #print("r_2:", np.shape(r_2))
            #print("z2:", np.shape(z2))
            #print("r_1:", np.shape(r_1))
            #print("b2(z2):", np.shape(b2(z2)))
            #print("H_x:", np.shape(H_x))

            #print('A_L', A_L(r_2, z2, tY, r_1, b2(z2), H_x).shape)
            #print('1+z',(1.0 + z2).shape)
            #print('JJ',JJ(tY, z2, H_interp, comoving_distance_interp, alpha_M_interp).shape)
            #check=A_L(r_2, z2, tY, r_1, b2(z2), H_x)* (1.0 + z2)* JJ(tY, z2, H_interp, comoving_distance_interp, alpha_M_interp)
            #print('check.shape',check.shape)

            t1 = np.transpose(
                A_L(r_2, z2, tY, r_1, b2(z2), H_x)
                * (1.0 + z2)
                * JJ(tY, z2, H_interp, comoving_distance_interp, alpha_M_interp),
                (1, 0)
            )

            '''
            # Build P(k,z) on (n_l, L, Npts)
            PS_list = []
            for lidx in range(len(ll)):
                row_list = []
                for i in range(len(z1)):
                    kline = (ll[lidx] + 0.5) / r_2[:, i]  # (Npts,)
                    zline = z2[:, i]  # (Npts,)
                    pts = np.column_stack([kline, zline])  # (Npts, 2)
                    row_list.append(power_spectra(pts))  # (Npts,)
                PS_list.append(np.asarray(row_list))  # (L, Npts)
            PS_ = np.asarray(PS_list)  # (n_l, L, Npts)
            '''
            PS_ = np.squeeze(np.asarray([[[power_spectra((xx, yy)) for xx, yy in zip((ll[l] + 0.5) / r_2[:, i], z2[:, i])] for i in range(len(z1))] for l in range(len(ll))]))

            # Windows: WY_ passed as (n_l, Npts, L); transpose to (n_l, L, Npts)
            WY_ = np.transpose(WY_, (0, 2, 1))  # (n_l, L, Npts)

            # Broadcast multiply -> (n_l, L, Npts)
            my_int = t1[None, :, :] * WY_ * PS_

            # ∫ dz2
            I1_ = np.asarray([
                [np.trapz(my_int[lidx, i1], x=z2[:, i1], axis=0)  # -> scalar per (lidx,i1)
                 for i1 in range(len(z1))]
                for lidx in range(len(ll))
            ])  # (n_l, L)

            # ∫ dz1
            I2_ = np.trapz(WX_ * I1_ / r_1 * (1 / H_interp(z1)), x=z1, axis=1)  # (n_l,)

            return I2_

        # ---------------- end integrand ----------------
        # Truncate or validate `l` to expected number of multipoles
        l = np.atleast_1d(l)
        n_l = l.shape[0]
        # 3) Assemble Cl
        for index_X in range(nkeys):
            key_X = keys[index_X]

            # z grid per bin, denser near peaks
            bins_ = self.bin_edges[key_X]
            bins_centers_ = (bins_[:-1] + bins_[1:]) / 2.0
            zzs = []
            ll_max = 0
            for bin_i in range(n_bins[index_X]):
                if 'gal' in key_X:
                    max_z = 5.0
                    n_pts = n_high
                else:
                    max_z = 10.0
                    n_pts = 2 * n_high

                if max(z_min, bins_[bin_i] * (1 - 5 * Delta_z) - 0.01) == z_min:
                    n1 = n_points + n_low
                else:
                    n1 = n_points

                if bin_i < n_bins[index_X] :
                    myarr = np.sort(np.unique(np.concatenate([
                        np.linspace(max(z_min, bins_[bin_i] * (1 - 5 * Delta_z)),
                                    bins_[bin_i + 1] * (1 + 5 * Delta_z), n_points),
                        np.linspace(z_min, max(z_min, bins_[bin_i] * (1 - 5 * Delta_z) - 0.01), n_low),
                        np.linspace(bins_[bin_i + 1] * (1 + 0.05) + 0.01, max_z, n_pts)
                    ])))
                else:
                    myarr = np.sort(np.unique(np.concatenate([
                        np.linspace(max(z_min, bins_[bin_i] * (1 - 5 * Delta_z)), max_z, n_points + n_pts),
                        np.linspace(z_min, max(z_min, bins_[bin_i] * (1 - 5 * Delta_z) - 0.01), n_low)
                    ])))
                ll_max = max(ll_max, len(myarr))
                zzs.append(myarr)

            for i, a in enumerate(zzs):
                if len(a) != ll_max:
                    n_ = ll_max - len(a)
                    zzs[i] = np.sort(np.unique(np.concatenate([
                        zzs[i],
                        np.linspace(z_min * (1 + 0.01), max(a) * (1 - 0.01), n_)
                    ])))
            zzs = np.asarray(zzs)  # (n_bins_X, L)

            # Windows for X
            W_X = np.array([
                [windows_to_use[key_X][i, j](zzs[i]) for j in range(n_l)]
                for i in range(n_bins[index_X])
            ])  # (n_bins_X, n_l, L)

            for index_Y in range(nkeys):
                key_Y = keys[index_Y]

                for bin_i in range(n_bins[index_X]):
                    for bin_j in range(n_bins[index_Y]):
                        if bin_j == bin_i:
                            # 2D z2 grid up to each z1 element (Npts, L)
                            z2s_ = np.linspace(z_min, zzs[bin_i], n_points)

                            # Windows for Y on z2 grid
                            W_Y = np.array([
                                [windows_to_use[key_Y][i, j](z2s_) for j in range(n_l)]
                                for i in range(n_bins[index_Y])
                            ])  # (n_bins_Y, n_l, Npts, L)

                            WY = W_Y[bin_j]  # (n_l, Npts, L)
                            WX = W_X[bin_i]  # (n_l, L)

                            # cross prefactor (keep as in your original cross routine)
                            #pref = l * (l + 1) / ((l + 0.5) ** 2)  # (n_l,)
                            l_used = np.atleast_1d(l)
                            pref = l_used * (l_used + 1) / ((l_used + 0.5) ** 2)
                            #print("pref shape:", pref.shape)

                            I_vec = lensing_int(
                                H_interp, comoving_distance_interp, alpha_M_interp,
                                zzs[bin_i], z2s_, l,
                                bin_i, bin_j, WX, WY,
                                key_X, key_Y,
                                z_min=z_min, n_points=n_points_x, grid=grid_x
                            )

                            assert len(pref) == len(I_vec), f"Mismatch: pref={len(pref)}, I_vec={len(I_vec)}"

                            #print("I_vec shape:", I_vec.shape)

                            #I_vec = lensing_int(
                                #zzs[bin_i], z2s_, np.atleast_1d(l), bin_i, bin_j,
                                #WX, WY, key_X, key_Y,
                                #z_min=z_min, n_points=n_points_x, grid=grid_x
                            #)  # (n_l,)

                            Cl['%s-%s' % (key_X, key_Y)][bin_i, bin_j, :] = pref * I_vec

        return Cl



    #-----------------------------------------------------------------------------------------
    # CORRELATION FUNCTIONS
    #-----------------------------------------------------------------------------------------
    def limber_angular_correlation_functions(self, theta, l, Cl, order):
        """
        This function computes the angular correlation function from an angular power spectrum. The equation is as follows

        .. math::

            \\xi^{(ij)}_{XY}(\\theta) = \int_0^\infty \\frac{d\ell}{2\pi} \ \ell \ C_{XY}^{(ij)}(\ell) \ J_{\\nu} (\ell\\theta),

        where :math:`\\nu` is the order of the transform and it changes from observable to observable.

        .. warning::

         For example, for shear :math:`\\nu=0` or :math:`\\nu=4`, for galaxy clustering :math:`\\nu=0` and for galaxy-galaxy lensing :math:`\\nu=2`.

        :param theta: Angles (in :math:`\mathrm{arcmin}`) where to compute the shear correlation functions
        :type theta: array

        :param l: Multipoles at which the spectrum is computed
        :type l: array

        :param Cl: 3D array, where first and second dimensions are the bins and the third is the multipoles, i.e. ``Cl[bin i, bin j, multipole l]``. The last dimension has to have the same length as ``l``.
        :type Cl: 3D array

        :param order: Order of Hankel transform.
        :type order: float

        :return: 3D array containing ``xi[bin i, bin j, angle theta]``

        """
        # 1) Define and check lengths and quantities
        l,theta,Cl = np.atleast_1d(l), np.atleast_1d(theta), np.atleast_3d(Cl)
        n_theta,n_l,nbins_i,nbins_j = len(theta),len(l),len(Cl),len(Cl[0])
        assert len(Cl[0,0]) == n_l, "Shapes of multipoles and spectra do not match"

        # 2) Check consistency of angles and multipoles
        ratio            = 30.  #(1/30 of minimum and maximum to avoid oscillations)
        theta_min_from_l = 60*180/np.pi/np.max(l)  # arcmin
        theta_max_from_l = 60*180/np.pi/np.min(l)  # arcmin
        assert theta.min() > theta_min_from_l, "Minimum theta is too small to obtain convergent results for the correlation function"
        assert theta.max() < theta_max_from_l, "Maximum theta is too large to obtain convergent results for the correlation function"


        # 3) Initialize arrays
        NN = 8192
        xi = np.zeros((nbins_i,nbins_j,n_theta))

        # 4) Hankel transform (ORDER!!!!)
        for bin_i in xrange(nbins_i):
            for bin_j in xrange(nbins_j):
                theta_tmp, xi_tmp = FF.Hankel(l, Cl[bin_i,bin_j]/(2.*np.pi), order = order, N = NN)
                xi_interp = si.interp1d(theta_tmp*180./np.pi*60., xi_tmp, 'cubic', bounds_error = False, fill_value = 0.)
                xi[bin_i,bin_j] = xi_interp(theta)
        del xi_tmp

        return xi

    # -----------------------------------------------------------------------------------------
    #							COMPARE GALAXY WINDOWS
    # -----------------------------------------------------------------------------------------
    def compare_galaxy_windows(self, z, ll, outdir=".", prefix="W_G",
                               name_bg="bg", name_legacy="legacy",
                               save_overlays=False, save_rel=False, title='GALAXY'):
        """
        Compare W_G from:
          - self.window_function[name_bg]
          - self.window_function[name_legacy]

        Save per-bin figures to outdir; return (per_bin_max, global_max).
        """

        def _rel_change(new, old, threshold=1e-10):
            new = np.asarray(new, float)
            old = np.asarray(old, float)
            out = np.full_like(new, np.nan)

            print('diff:\n', np.round(new - old, 2))

            # Only compute relative difference where either value is above the threshold
            valid = (np.abs(old) > threshold) | (np.abs(new) > threshold)

            with np.errstate(divide='ignore', invalid='ignore'):
                np.divide(new, old, out=out, where=valid & (old != 0))

            out[~valid] = np.nan  # Set low-value regions to NaN
            return out - 1.0

        os.makedirs(outdir, exist_ok=True)

        # Integration grid
        z_int = np.asarray(self.z_integration)
        n_bins = self.window_function[name_bg].shape[0]

        # Evaluate the window functions
        W_bg = np.array([self.window_function[name_bg][i, 0](z_int) for i in range(n_bins)])
        W_legacy = np.array([self.window_function[name_legacy][i, 0](z_int) for i in range(n_bins)])

        # Compute relative difference
        rel = _rel_change(W_bg, W_legacy)
        per_bin_max = np.nanmax(np.abs(rel), axis=1)
        global_max = np.nanmax(np.abs(rel))

        # Save summary to file
        with open(os.path.join(outdir, f"{prefix}_diff_summary.txt"), "w") as f:
            for i in range(n_bins):
                f.write(f"bin {i:02d}: max|Δ/legacy| = {per_bin_max[i]:.6e}\n")
            f.write(f"GLOBAL max|Δ/legacy| = {global_max:.6e}\n")

        print(f'\n {title}')
        for i in range(n_bins):
            print(f"bin {i:02d}: max|Δ/legacy| = {per_bin_max[i]:.6e}")
        print(f"GLOBAL max|Δ/legacy| = {global_max:.6e}\n")

        # Save plots
        for i in range(n_bins):
            if save_overlays:
                fig = plt.figure()
                plt.plot(z_int, W_legacy[i], label='legacy', lw=1.6)
                plt.plot(z_int, W_bg[i], label='bg', lw=1.3, ls='--')
                plt.title(f"{prefix} overlay – bin {i}")
                plt.xlabel("z")
                plt.ylabel(r"$W_G(z)$ [1/Mpc]")
                plt.grid(True, ls=':')
                plt.legend()
                fig.savefig(os.path.join(outdir, f"{prefix}_bin{i:02d}_overlay.png"),
                            dpi=160, bbox_inches='tight')
                plt.close(fig)

            if save_rel:
                fig = plt.figure()
                plt.plot(z_int, rel[i], lw=1.3)
                plt.title(f"{prefix} rel diff – bin {i}")
                plt.xlabel("z")
                plt.ylabel("(bg/legacy − 1)")
                plt.grid(True, ls=':')
                fig.savefig(os.path.join(outdir, f"{prefix}_bin{i:02d}_reldiff.png"),
                            dpi=160, bbox_inches='tight')
                plt.close(fig)

        # Optional: return tiled arrays for further analysis
        W_bg_tiled = np.tile(W_bg.reshape(-1, 1), (1, len(ll)))
        W_legacy_tiled = np.tile(W_legacy.reshape(-1, 1), (1, len(ll)))

        return W_bg_tiled, W_legacy_tiled, per_bin_max, global_max

    # -----------------------------------------------------------------------------------------
    #							COMPARE ANGULAR POWER SPECTRA
    # -----------------------------------------------------------------------------------------

    def compare_angular_power_spectra(
            self, bg, h, ell, outdir="Cl_comparison", prefix="Cl",
            spectra=None, atol=1e-10, rtol_warn=1e-2,
            save_overlay=True, save_diff=True, save_rel=True
    ):
        """
        Compare angular power spectra from new vs old Limber implementation.
        Saves per-bin plots and a summary of max differences.

        Parameters
        ----------
        self : class instance
            Must provide .limber_angular_power_spectra() and .limber_angular_power_spectra_old()
        bg : dict
            Background cosmology from CLASS (must include 'z', 'H [1/Mpc]', 'comov. dist.')
        h : float
            Reduced Hubble constant (H0 / 100)
        ell : array-like
            Multipoles at which to evaluate the Cls
        outdir : str
            Output directory for plots and summary
        prefix : str
            File prefix for plots and output
        spectra : list of str, optional
            Which spectra (e.g. ['galaxy', 'GW']) to compare. If None, compare all in self.window_function
        atol : float
            Absolute tolerance below which values are ignored in relative difference
        rtol_warn : float
            Threshold for flagging large relative differences
        """
        os.makedirs(outdir, exist_ok=True)

        # Compute both versions
        Cl_new = self.limber_angular_power_spectra(bg=bg, h=h, l=ell, windows=spectra)
        Cl_old = self.limber_angular_power_spectra_old(l=ell, windows=spectra)

        # Compare all keys
        all_keys = sorted(set(Cl_new.keys()) & set(Cl_old.keys()))
        if not all_keys:
            raise ValueError("No overlapping spectra keys between new and old Cl dictionaries.")

        with open(os.path.join(outdir, f"{prefix}_Cl_diff_summary.txt"), "w") as f:
            f.write(f"# Comparing angular power spectra for keys: {all_keys}\n")
            f.write(f"# Each Cl is shape (n_bin_i, n_bin_j, n_ell)\n")

            for key in all_keys:
                new = Cl_new[key]
                old = Cl_old[key]

                if new.shape != old.shape:
                    raise ValueError(f"Shape mismatch in key {key}: {new.shape} vs {old.shape}")

                n_i, n_j, n_l = new.shape

                # Mask: only consider elements above threshold
                mask = (np.abs(new) > 1e-10) | (np.abs(old) > 1e-10)

                if np.any(mask):
                    abs_diff = np.abs(new - old)[mask]
                    rel_diff = abs_diff / np.maximum(np.abs(old[mask]), atol)
                    max_abs = np.max(abs_diff)
                    max_rel = np.max(rel_diff)
                else:
                    max_abs, max_rel = 0.0, 0.0

                print(f"\nSpectrum: {key}\n")
                print(f"  Max absolute diff: {max_abs:.6e}\n")
                print(f"  Max relative diff: {max_rel:.6e}\n")

                f.write(f"\nSpectrum: {key}\n")
                f.write(f"  Max absolute diff: {max_abs:.6e}\n")
                f.write(f"  Max relative diff: {max_rel:.6e}\n")

                for i in range(n_i):
                    for j in range(n_j):
                        cl_new = new[i, j, :]
                        cl_old = old[i, j, :]

                        # Mask values above threshold
                        mask = (np.abs(cl_new) > 1e-10) | (np.abs(cl_old) > 1e-10)

                        abs_diff = np.zeros_like(cl_new)
                        rel_diff = np.zeros_like(cl_new)

                        if np.any(mask):
                            abs_diff[mask] = np.abs(cl_new[mask] - cl_old[mask])
                            rel_diff[mask] = abs_diff[mask] / np.maximum(np.abs(cl_old[mask]), atol)

                        max_rel_ij = np.max(rel_diff[mask]) if np.any(mask) else 0.0
                        if max_rel_ij > rtol_warn:
                            f.write(f"  Bin ({i},{j}): MAX rel diff = {max_rel_ij:.3e}\n")

                        # Plot overlays
                        if save_overlay:
                            plt.figure()
                            plt.plot(ell, cl_old, label='legacy', lw=1.3)
                            plt.plot(ell, cl_new, '--', label='new', lw=1.3)
                            #plt.xscale('log')
                            #plt.yscale('log')
                            plt.title(f"{key} Cl (bin {i},{j})")
                            plt.xlabel(r"$\ell$")
                            plt.ylabel(r"$C_\ell$")
                            plt.grid(True, ls=':')
                            plt.legend()
                            plt.savefig(os.path.join(outdir, f"{prefix}_{key}_bin{i}{j}_overlay.png"), dpi=160)
                            plt.close()

                        # Plot abs difference
                        if save_diff:
                            plt.figure()
                            plt.plot(ell, abs_diff)
                            #plt.xscale('log')
                            #plt.yscale('log')
                            plt.title(f"{key} |ΔCₗ| (bin {i},{j})")
                            plt.xlabel(r"$\ell$")
                            plt.ylabel(r"$|C_\ell^{new} - C_\ell^{old}|$")
                            plt.grid(True, ls=':')
                            plt.savefig(os.path.join(outdir, f"{prefix}_{key}_bin{i}{j}_absdiff.png"), dpi=160)
                            plt.close()

                        # Plot rel difference
                        if save_rel:
                            plt.figure()
                            plt.plot(ell, rel_diff)
                            #plt.xscale('log')
                            plt.title(f"{key} ΔCₗ/Cₗ (bin {i},{j})")
                            plt.xlabel(r"$\ell$")
                            plt.ylabel(r"$(new / old - 1)$")
                            plt.grid(True, ls=':')
                            plt.savefig(os.path.join(outdir, f"{prefix}_{key}_bin{i}{j}_reldiff.png"), dpi=160)
                            plt.close()

        return Cl_new, Cl_old

    # -----------------------------------------------------------------------------------------
    #							COMPARE WINDOWS
    # -----------------------------------------------------------------------------------------
    def compare_windows(self,
                        name_a: str,
                        name_b: str,
                        z_eval=None,  # 1D array of z values to evaluate on (defaults to self.z_integration)
                        ells=None,
                        # list/array of ell indices to compare (indices, not values); default = all available
                        outdir=".",
                        prefix="Wcmp",
                        rel_threshold=1e-12,  # values below this are treated as ~0 in rel diff
                        plot_overlays=True,
                        plot_rel_lines=True,
                        plot_rel_heatmap=True):
        """
        Compare two window-function collections stored in self.window_function[name].
        Works when each entry is a callable f(z) or a sampled array over z_eval.

        Assumed structure: self.window_function[name] has shape [n_bins, n_ell] of callables/arrays.
        If arrays are provided, they must be sampled on the same z grid (z_eval).

        Returns
        -------
        results : dict with keys:
            'W_a', 'W_b'              -> arrays [n_bins, n_ell_sel, n_z]
            'rel'                     -> (W_a / W_b - 1) with NaNs where W_b ~ 0
            'per_bin_max_abs'         -> max over (ell,z) per bin
            'per_ell_max_abs'         -> max over (bin,z) per ell
            'global_max_abs'          -> scalar max abs rel
            'selected_ell_indices'    -> the ell indices compared
            'z_eval'                  -> evaluation grid used
            'summary_path'            -> path to text summary (if written)
        """
        os.makedirs(outdir, exist_ok=True)

        # --- pick z grid
        if z_eval is None:
            z_eval = np.asarray(self.z_integration, float)
        else:
            z_eval = np.asarray(z_eval, float)
            if z_eval.ndim != 1:
                raise ValueError("z_eval must be a 1-D array")

        Wa = self.window_function[name_a]
        Wb = self.window_function[name_b]

        # shape checks
        if Wa.ndim != 2 or Wb.ndim != 2:
            raise ValueError("Expected self.window_function[name] with shape [n_bins, n_ell].")

        n_bins_a, n_ell_a = Wa.shape
        n_bins_b, n_ell_b = Wb.shape
        if n_bins_a != n_bins_b:
            raise ValueError(f"Different number of bins: {n_bins_a} vs {n_bins_b}")

        # choose ells (indices in the second axis)
        if ells is None:
            n_ell = min(n_ell_a, n_ell_b)
            ell_idx = np.arange(n_ell, dtype=int)
        else:
            ell_idx = np.asarray(ells, int)
            if np.any(ell_idx < 0):
                raise ValueError("ell indices must be non-negative")
            if np.max(ell_idx) >= min(n_ell_a, n_ell_b):
                raise ValueError("Some ell indices exceed available range")

        n_bins = n_bins_a
        n_ell_sel = len(ell_idx)
        n_z = z_eval.size

        # --- helper to evaluate either callables or arrays
        def _eval_entry(entry):
            # callable f(z) → evaluate; array → verify shape and return
            if callable(entry):
                return np.asarray(entry(z_eval), float)
            arr = np.asarray(entry, float)
            if arr.ndim != 1 or arr.size != n_z:
                raise ValueError("Found array window not matching z_eval length; "
                                 "either pass callables or arrays sampled on z_eval.")
            return arr

        # --- evaluate both windows on [bin, ell_idx, z]
        W_a = np.empty((n_bins, n_ell_sel, n_z), dtype=float)
        W_b = np.empty_like(W_a)

        for ib in range(n_bins):
            for j, jell in enumerate(ell_idx):
                W_a[ib, j, :] = _eval_entry(Wa[ib, jell])
                W_b[ib, j, :] = _eval_entry(Wb[ib, jell])

        # --- relative difference (a/b - 1), robust near zeros of b
        rel = np.full_like(W_a, np.nan)
        denom = np.where(np.abs(W_b) > rel_threshold, W_b, np.nan)
        rel = W_a / denom - 1.0

        # --- metrics
        per_bin_max_abs = np.nanmax(np.abs(rel), axis=(1, 2))  # over (ell,z)
        per_ell_max_abs = np.nanmax(np.abs(rel), axis=(0, 2))  # over (bin,z)
        global_max_abs = np.nanmax(np.abs(rel))

        # --- write summary
        summary_path = os.path.join(outdir, f"{prefix}_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Compare windows: {name_a} vs {name_b}\n")
            f.write(f"z range: [{z_eval.min():.6g}, {z_eval.max():.6g}]  n_z={n_z}\n")
            f.write(f"ell indices compared: {ell_idx.tolist()}\n\n")
            for ib in range(n_bins):
                f.write(f"bin {ib:02d}: max|Δ/Ref| over (ell,z) = {per_bin_max_abs[ib]:.6e}\n")
            f.write("\n")
            for j, jell in enumerate(ell_idx):
                f.write(f"ell index {jell:03d}: max|Δ/Ref| over (bin,z) = {per_ell_max_abs[j]:.6e}\n")
            f.write(f"\nGLOBAL max|Δ/Ref| = {global_max_abs:.6e}\n")

        # --- plotting
        for ib in range(n_bins):
            for j, jell in enumerate(ell_idx):
                if plot_overlays:
                    fig = plt.figure()
                    plt.plot(z_eval, W_b[ib, j], label=f"{name_b}", lw=1.6)
                    plt.plot(z_eval, W_a[ib, j], label=f"{name_a}", lw=1.3, ls='--')
                    plt.title(f"{prefix} overlay – bin {ib}, ell#{jell}")
                    plt.xlabel("z")
                    plt.ylabel("W(z) [dimensionless]")
                    plt.grid(True, ls=':')
                    plt.legend()
                    fig.savefig(os.path.join(outdir, f"{prefix}_bin{ib:02d}_ell{jell:03d}_overlay.png"),
                                dpi=160, bbox_inches='tight')
                    plt.close(fig)

                if plot_rel_lines:
                    fig = plt.figure()
                    plt.plot(z_eval, rel[ib, j], lw=1.3)
                    plt.title(f"{prefix} rel diff – bin {ib}, ell#{jell}")
                    plt.xlabel("z")
                    plt.ylabel(f"({name_a}/{name_b} − 1)")
                    plt.grid(True, ls=':')
                    fig.savefig(os.path.join(outdir, f"{prefix}_bin{ib:02d}_ell{jell:03d}_reldiff.png"),
                                dpi=160, bbox_inches='tight')
                    plt.close(fig)

        if plot_rel_heatmap:
            # heatmap of max |rel| over z, shown as [bin, ell]
            max_over_z = np.nanmax(np.abs(rel), axis=2)  # [n_bins, n_ell_sel]
            fig = plt.figure()
            plt.imshow(max_over_z, aspect='auto', origin='lower',
                       extent=[ell_idx.min() - 0.5, ell_idx.max() + 0.5, -0.5, n_bins - 0.5])
            cbar = plt.colorbar()
            cbar.set_label("max_z |Δ/Ref|")
            plt.xlabel("ell index")
            plt.ylabel("bin")
            plt.title(f"{prefix} – heatmap of max |rel| over z")
            fig.savefig(os.path.join(outdir, f"{prefix}_heatmap_maxabs_over_z.png"),
                        dpi=160, bbox_inches='tight')
            plt.close(fig)

        results = dict(
            W_a=W_a, W_b=W_b, rel=rel,
            per_bin_max_abs=per_bin_max_abs,
            per_ell_max_abs=per_ell_max_abs,
            global_max_abs=global_max_abs,
            selected_ell_indices=ell_idx,
            z_eval=z_eval,
            summary_path=summary_path,
        )
        print(f"GLOBAL max|Δ/Ref| = {global_max_abs:.6e}  (summary: {summary_path})")
        return results

    # -----------------------------------------------------------------------------------------
    #							COMPARE GW WINDOWS
    # -----------------------------------------------------------------------------------------
    def compare_gw_windows(
            self, z, ll, name_bg='bg', name_legacy='legacy',
            outdir=".", prefix="W_GW", save_overlays=True, save_rel=True
    ):
        """
        Compare GW windows already loaded into self.window_function[name_bg] and name_legacy.
        Saves overlay and relative difference plots, and writes per-bin max differences.

        Parameters
        ----------
        z : array
            Redshift array for integration/evaluation.
        ll : array
            Multipole array (used to tile window shapes).
        name_bg : str
            Name key for background GW window in self.window_function.
        name_legacy : str
            Name key for legacy GW window in self.window_function.
        """

        def _rel_change(new, old, threshold=1e-13):
            new = np.asarray(new, float)
            old = np.asarray(old, float)
            out = np.full_like(new, np.nan)

            # Only compute relative difference where either value is above the threshold
            valid = (np.abs(old) > threshold) | (np.abs(new) > threshold)

            with np.errstate(divide='ignore', invalid='ignore'):
                np.divide(new, old, out=out, where=valid & (old != 0))

            out[~valid] = np.nan  # Set low-value regions to NaN
            return out - 1.0

        os.makedirs(outdir, exist_ok=True)

        z_int = np.asarray(self.z_integration)
        n_bins = self.window_function[name_bg].shape[0]

        # Evaluate both window functions
        W_bg = np.array([self.window_function[name_bg][i, 0](z_int) for i in range(n_bins)])
        W_legacy = np.array([self.window_function[name_legacy][i, 0](z_int) for i in range(n_bins)])

        print("max |W_bg|    =", np.max(np.abs(W_bg)))
        print("max |W_legacy|=", np.max(np.abs(W_legacy)))

        # Compute relative difference
        rel = _rel_change(W_bg, W_legacy)
        per_bin_max = np.nanmax(np.abs(rel), axis=1)
        global_max = np.nanmax(np.abs(rel))

        # Save summary
        with open(os.path.join(outdir, f"{prefix}_diff_summary.txt"), "w") as f:
            for i in range(n_bins):
                f.write(f"bin {i:02d}: max|Δ/legacy| = {per_bin_max[i]:.6e}\n")
            f.write(f"GLOBAL max|Δ/legacy| = {global_max:.6e}\n")

        print('\n GW')
        for i in range(n_bins):
            print(f"bin {i:02d}: max|Δ/legacy| = {per_bin_max[i]:.6e}")
        print(f"GLOBAL max|Δ/legacy| = {global_max:.6e}\n")

        # Save figures
        for i in range(n_bins):
            if save_overlays:
                fig = plt.figure()
                plt.plot(z_int, W_legacy[i], label='legacy', lw=1.6)
                plt.plot(z_int, W_bg[i], label='bg', lw=1.3, ls='--')
                plt.title(f"{prefix} overlay – bin {i}")
                plt.xlabel("z")
                plt.ylabel(r"$W_{\rm GW}(z)$ [unitless]")
                plt.grid(True, ls=':')
                plt.legend()
                fig.savefig(os.path.join(outdir, f"{prefix}_bin{i:02d}_overlay.png"),
                            dpi=160, bbox_inches='tight')
                plt.close(fig)
            if save_rel:
                fig = plt.figure()
                plt.plot(z_int, rel[i], lw=1.3)
                plt.title(f"{prefix} rel diff – bin {i}")
                plt.xlabel("z")
                plt.ylabel("(bg / legacy − 1)")
                plt.grid(True, ls=':')
                fig.savefig(os.path.join(outdir, f"{prefix}_bin{i:02d}_reldiff.png"),
                            dpi=160, bbox_inches='tight')
                plt.close(fig)

        W_bg_tiled = np.tile(W_bg.reshape(-1, 1), (1, len(ll)))
        W_legacy_tiled = np.tile(W_legacy.reshape(-1, 1), (1, len(ll)))
        return W_bg_tiled, W_legacy_tiled, per_bin_max, global_max


    ######################################## OLD


    # -----------------------------------------------------------------------------------------
    # GRAVITATIONAL WAVES CLUSTERING WINDOW FUNCTION
    # -----------------------------------------------------------------------------------------
    def load_gravitational_wave_window_functions_old(self, z, ndl, H0, omega_m, omega_b, ll, bias=1.0, name='GW'):

        conversion = FlatLambdaCDM(H0=H0, Om0=omega_m, Ob0=omega_b, Tcmb0=2.7255)
        ndl = np.array(ndl)
        z = np.array(z)
        n_bins = len(ndl)

        norm_const = simpson(ndl, x=conversion.luminosity_distance(z).value, axis=1)  # [Mpc]
        assert np.all(np.diff(
            z) < self.dz_windows), "For convergence reasons, the distribution function arrays must be sampled with frequency of at least dz<=%.3f" % (
            self.dz_windows)
        assert ndl.ndim == 2, "'nz' must be 2-dimensional"
        assert (ndl.shape)[1] == z.shape[0], "Length of each 'nz[i]' must be the same of 'z'"
        assert z.min() <= self.z_min, "Minimum input redshift must be < %.3f, the minimum redshift of integration" % (
            self.z_min)
        assert z.max() >= self.z_max, "Maximum input redshift must be > %.3f, the maximum redshift of integration" % (
            self.z_max)

        if isinstance(bias, float):
            bias = bias * np.ones(n_bins)
        elif isinstance(bias, int):
            bias = np.float(bias) * np.ones(n_bins)
        else:
            assert len(bias) == n_bins, "Number of bias factors different from number of bins"

        # conversion.luminosity_distance in [Mpc]; const.c / self.Hubble in [Mpc/h]   ---> / h --->[Mpc]
        jac_dl = ((1 + self.z_integration) * ((const.c / self.Hubble)/0.6781) + (conversion.luminosity_distance(self.z_integration).value) / (1 + self.z_integration))  # [Mpc]
        # print('\n GW OLD:')
        # print('Hubble [h/Mpc] ', self.Hubble/const.c)
        # print('r_n_z [Mpc/h]', conversion.luminosity_distance(self.z_integration).value)
        # print('jac_dL mix', jac_dL.tolist())
        # print('norm',norm_const)
        # print('\n-------------------------------------------------------------\n')

        # Initialize window
        self.window_function[name] = []
        # Compute window
        for galaxy_bin in xrange(n_bins):
            tmp_interp = si.interp1d(z, ndl[galaxy_bin], 'cubic', bounds_error=False, fill_value=0.)
            #                                                                         1/Gpc * [Mpc]* [1/Mpc] *[Mpc]
            self.window_function[name].append(si.interp1d(self.z_integration,tmp_interp(self.z_integration) * (jac_dl) * ((self.Hubble / const.c) * 0.6781) / norm_const[galaxy_bin] * bias[galaxy_bin], 'cubic',bounds_error=False, fill_value=0.))

        self.window_function[name] = np.tile(np.array(self.window_function[name]).reshape(-1, 1), (1, len(ll)))

    # -----------------------------------------------------------------------------------------
    # GALAXY CLUSTERING WINDOW FUNCTION
    # -----------------------------------------------------------------------------------------
    def load_galaxy_clustering_window_functions_old(self, z, nz, ll, bias=1.0, name='galaxy'):
        """
        This function computes the window function for galaxy clustering given a galaxy distribution.
        The function automatically normalizes the galaxy distribution such that the integral over
        redshifts is 1.
        The routine adds a key (specified in the ``name`` argument) to the ``self.window_function`` dictionary.
        Given a galxy distruibution in a redshift bin :math:`n^{(i)}(z)`, the equation is:

        .. math::

           W^{(i)}_\mathrm{G}(z) = b(z) \ n^{(i)}(z) \\frac{H(z)}{c}


        :param z: array or list of redshift at which the galaxy distribution ``nz`` is evaluated
        :type z: 1-D array, default = None

        :param nz: 2-D array or 2-D list where each sublist is the galaxy distribution of a given redshift bin
        :type nz: 2-D array with shape ``(n_bins, len(z))``, default = None

        :param bias: Galaxy bias.
        :type bias: float or array, same length of ``nz``, default = 1

        :param name: name of the key to add to the dictionary
        :type name: string, default = 'galaxy'


        An example call can be, for 3 bins all with a :func:`colibri.limber.limber.euclid_distribution` with default arguments for ``a`` and ``b`` but different bin edges ``zmin``, ``zmax``:

        .. code-block:: python

           bin_edges = [0.00, 0.72, 1.11, 5.00]
           nbins     = len(bin_edges)-1
           z_w       = np.linspace(0., 6., 1001)
           nz_w      = [S.euclid_distribution(z = z_w, a = 2.0, b = 1.5, zmin = bin_edges[i], zmax = bin_edges[i+1]) for i in range(nbins)]
           S.load_galaxy_clustering_window_functions(z = z_w, nz = nz_w, bias = 1)

        :return: A key of a given name is added to the ``self.window_function`` dictionary

        """
        nz = np.array(nz)
        z = np.array(z)
        n_bins = len(nz)
        norm_const = simpson(nz, x=z, axis=1)
        assert np.all(np.diff(
            z) < self.dz_windows), "For convergence reasons, the distribution function arrays must be sampled with frequency of at least dz<=%.3f" % (
            self.dz_windows)
        assert nz.ndim == 2, "'nz' must be 2-dimensional"
        assert (nz.shape)[1] == z.shape[0], "Length of each 'nz[i]' must be the same of 'z'"
        assert z.min() <= self.z_min, "Minimum input redshift must be < %.3f, the minimum redshift of integration" % (
            self.z_min)
        assert z.max() >= self.z_max, "Maximum input redshift must be > %.3f, the maximum redshift of integration" % (
            self.z_max)

        if isinstance(bias, float):
            bias = bias * np.ones(n_bins)
        elif isinstance(bias, int):
            bias = np.float(bias) * np.ones(n_bins)
        else:
            assert len(bias) == n_bins, "Number of bias factors different from number of bins"
        # Initialize window
        self.window_function[name] = []
        # Compute window
        for galaxy_bin in xrange(n_bins):
            tmp_interp = si.interp1d(z, nz[galaxy_bin], 'cubic', bounds_error=False, fill_value=0.)
            # to_append = si.interp1d(self.z_integration, tmp_interp(self.z_integration)*self.Hubble/const.c/norm_const[galaxy_bin]*bias[galaxy_bin], 'cubic', bounds_error = False, fill_value = 0.)
            #                                                                      [...]                    *    [Mpc/h]->[Mpc]
            self.window_function[name].append(si.interp1d(self.z_integration,tmp_interp(self.z_integration) * ((self.Hubble / const.c)*0.6781) * 1 /norm_const[galaxy_bin] * bias[galaxy_bin], 'cubic',bounds_error=False, fill_value=0.))

        self.window_function[name] = np.tile(np.array(self.window_function[name]).reshape(-1, 1), (1, len(ll)))

    # -----------------------------------------------------------------------------------------
    # ANGULAR SPECTRA
    # -----------------------------------------------------------------------------------------
    def limber_angular_power_spectra_old(self, l, windows=None):
        """
        This function computes the angular power spectra (using the Limber's and the flat-sky approximations) for the window function specified.
        Given two redshift bins `i` and `j` the equation is

        .. math::

          C^{(ij)}(\ell) = \int_0^\infty dz \ \\frac{c}{H(z)} \ \\frac{W^{(i)}(z) W^{(j)}(z)}{f_K^2[\chi(z)]} \ P\left(\\frac{\ell}{f_K[\chi(z)]}, z\\right),

        where :math:`P(k,z)` is the matter power spectrum and :math:`W^{(i)}(z)` are the window functions.

        :param l: Multipoles at which to compute the shear power spectra.
        :type l: array

        :param windows: which spectra (auto and cross) must be computed. If set to ``None`` all the spectra will be computed.
        :type windows: list of strings, default = ``None``

        :return: dictionary whose keys are combinations of window functions specified in ``windows``. Each key is a 3-D array whose entries are ``Cl[bin i, bin j, multipole l]``
        """

        # Check existence of power spectrum
        try:
            self.power_spectra_interpolator
        except AttributeError:
            raise AttributeError("Power spectra have not been loaded yet")

        # Check convergence with (l, k, z):
        assert np.atleast_1d(l).min() > self.k_min * self.geometric_factor_f_K(
            self.z_min), "Minimum 'l' is too low. Extend power spectra to lower k_min? Use lower z_min for power spectrum?"
        assert np.atleast_1d(l).max() < self.k_max * self.geometric_factor_f_K(
            self.z_max), "Maximum 'l' is too high. Extend power spectra to higher k_max? Use higher z_max for power spectrum?"

        # Check window functions to use
        if windows is None:
            windows_to_use = self.window_function
        else:
            windows_to_use = {}
            for key in windows:
                try:
                    windows_to_use[key] = self.window_function[key]
                except KeyError:
                    raise KeyError("Requested window function '%s' not known" % key)

        # Check window functions
        keys = windows_to_use.keys()
        if len(keys) == 0: raise AttributeError("No window function has been computed!")
        nkeys = len(keys)
        n_bins = [len(windows_to_use[key]) for key in keys]

        # 1) Define lengths and quantities
        zz = self.z_integration
        n_l = len(np.atleast_1d(l))
        n_z = self.n_z_integration
        HH = self.Hubble
        cH_chi2 = self.c_over_H_over_chi_squared #[h/Mpc]
        Cl = {}
        for i, ki in enumerate(keys):
            for j, kj in enumerate(keys):
                Cl['%s-%s' % (ki, kj)] = np.zeros((n_bins[i], n_bins[j], n_l))

        # 2) Load power spectra
        power_spectra = self.power_spectra_interpolator
        PS_lz = np.zeros((n_l, n_z))
        for il in xrange(n_l):
            for iz in range(n_z):
                k_Mpc = ((l[il] + 0.5) / self.geometric_factor[iz])*0.6781  # k in h/Mpc
                PS_lz[il, iz] = power_spectra([(k_Mpc, zz[iz])])  # PS_lz in (Mpc/1)^3

        # Add curvature correction (see arXiv:2302.04507)
        if self.cosmology.K != 0.:
            KK = self.cosmology.K
            factor = np.zeros((n_l, n_z))
            for il, ell in enumerate(l):
                factor[il] = (1 - np.sign(KK) * ell ** 2 / (
                            ((ell + 0.5) / self.geometric_factor) ** 2 + KK)) ** -0.5
            PS_lz *= factor

        # 3) load Cls given the source functions
        # 1st key (from 1 to N_keys)
        for index_X in xrange(nkeys):
            key_X = list(keys)[index_X]
            W_X = np.array([[windows_to_use[key_X][i, j](zz) for j in range(n_l)] for i in range(n_bins[index_X])])
            # 2nd key (from 1st key to N_keys)
            for index_Y in xrange(index_X, nkeys):
                key_Y = list(keys)[index_Y]
                W_Y = np.array(
                    [[windows_to_use[key_Y][i, j](zz) for j in range(n_l)] for i in range(n_bins[index_Y])])
                # Symmetry C_{AA}^{ij} == C_{AA}^{ji}
                if key_X == key_Y:
                    for bin_i in xrange(n_bins[index_X]):
                        for bin_j in xrange(bin_i, n_bins[index_Y]):
                            Cl['%s-%s' % (key_X, key_Y)][bin_i, bin_j] = [
                                simpson((cH_chi2*0.6781) * W_X[bin_i, xx] * W_Y[bin_j, xx] * PS_lz[xx], x=zz) for xx in
                                range(n_l)]
                            Cl['%s-%s' % (key_X, key_Y)][bin_j, bin_i] = Cl['%s-%s' % (key_X, key_Y)][bin_i, bin_j]
                # Symmetry C_{AB}^{ij} == C_{BA}^{ji}
                else:
                    for bin_i in xrange(n_bins[index_X]):
                        for bin_j in xrange(n_bins[index_Y]):
                            Cl['%s-%s' % (key_X, key_Y)][bin_i, bin_j] = [
                                simpson((cH_chi2*0.6781)  * W_X[bin_i, xx] * W_Y[bin_j, xx] * PS_lz[xx], x=zz) for xx in
                                range(n_l)]
                            Cl['%s-%s' % (key_Y, key_X)][bin_j, bin_i] = Cl['%s-%s' % (key_X, key_Y)][bin_i, bin_j]
        return Cl
