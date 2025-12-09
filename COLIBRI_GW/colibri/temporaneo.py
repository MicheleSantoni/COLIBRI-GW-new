# -------------------------------------------------------------------------------
# CAMB_Pk
# -------------------------------------------------------------------------------
def camb_Pk(self,
            z=0.,
            k=np.logspace(-4., 2., 1001),
            nonlinear=False,
            halofit='mead2020',
            var_1='tot',
            var_2='tot',
            share_delta_neff=True,
            **kwargs
            ):  ######################################  QUI
    """
    This routine uses the CAMB Boltzmann solver to return power spectra for the chosen cosmology.
    Depending on the value of 'nonlinear', the power spectrum is linear or non-linear; the 'halofit'
    argument chooses the non-linear model.

    :param z: Redshifts.
    :type z: array, default = 0

    :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
    :type k: array, default = ``np.logspace(-4., 2., 1001)``

    :param nonlinear: Whether to return non-linear power spectra.
    :type nonlinear: boolean, default = False

    :param halofit: Which version of Halofit to use. See CAMB documentation for further info.
    :type halofit: string, default = 'mead2020'

    :param var_1: Density field for the first component of the power spectrum.
    :type var_1: string, default = 'tot'

    :param var_2: Density field for the second component of the power spectrum.

     - `'tot'`   : total matter
     - `'cdm'`   : cold dark matter
     - `'b'`     : baryons
     - `'nu'`    : neutrinos
     - `'cb'`    : cold dark matter + baryons
     - `'gamma'` : photons
     - `'v_cdm'` : cdm velocity
     - `'v_b'`   : baryon velocity
     - `'Phi'`   : Weyl potential
    :type var_2: string, default = `'tot'`


    :param kwargs: Keyword arguments to be passed to ``camb.set_params``. See CAMB documentation for further info: https://camb.readthedocs.io/en/latest/

    Returns
    -------

    k: array
        Scales in :math:`h/\mathrm{Mpc}`. Basically the same 'k' of the input.

    pk: 2D array of shape ``(len(z), len(k))``
        Power spectrum in units of :math:`(\mathrm{Mpc}/h)^3`.
    """
    # Neutrino part
    # num_nu_massless – (float64) Effective number of massless neutrinos
    # num_nu_massive – (integer) Total physical (integer) number of massive neutrino species
    # nu_mass_eigenstates – (integer) Number of non-degenerate mass eigenstates
    # nu_mass_degeneracies – (float64 array) Degeneracy of each distinct eigenstate
    # nu_mass_fractions – (float64 array) Mass fraction in each distinct eigenstate
    # nu_mass_numbers – (integer array) Number of physical neutrinos per distinct eigenstate
    nu_mass_eigen = len(np.unique([mm for mm in self.M_nu])) if np.any(self.M_nu != 0.) else 0
    nu_mass_numbers = [list(self.M_nu).count(x) for x in set(list(self.M_nu))]
    nu_mass_numbers = sorted(nu_mass_numbers, reverse=True) if np.any(self.M_nu != 0.) else [0]
    # Set parameters
    cambparams = {
        'num_nu_massive': self.massive_nu,
        'num_nu_massless': self.massless_nu,
        'nu_mass_eigenstates': nu_mass_eigen,
        'nu_mass_numbers': nu_mass_numbers,
        'nnu': self.N_eff,
        'omnuh2': self.omega_nu_tot,
        'ombh2': self.omega_b,
        'omch2': self.omega_cdm + self.omega_wdm_tot,
        'omk': self.Omega_K,
        'H0': 100. * self.h,
        'As': self.As,
        'ns': self.ns,
        'w': self.w0,
        'wa': self.wa,
        'TCMB': self.T_cmb,
        'tau': self.tau,
        'share_delta_neff': True,
        'dark_energy_model': 'DarkEnergyPPF'}
    # kwargs
    for key, value in kwargs.items():
        if not key in cambparams: cambparams[key] = value
    params = camb.set_params(**cambparams)

    # Redshifts
    z = np.atleast_1d(z)
    nz = len(z)

    # Possible components to use
    components = {'tot': 'delta_tot',
                  'cdm': 'delta_cdm',
                  'b': 'delta_baryon',
                  'nu': 'delta_nu',
                  'cb': 'delta_nonu',
                  'gamma': 'delta_photon',
                  'v_cdm': 'v_newtonian_cdm',
                  'v_b': 'v_newtonian_baryon',
                  'Phi': 'Weyl'}  # Weyl: (phi+psi)/2 is proportional to lensing potential

    # Number of points (according to logint)
    logint = 100
    npoints = int(logint * np.log10(k.max() / k.min()))
    dlogk = 2. * np.log10(k.max() / k.min()) / npoints

    # Halofit version
    if nonlinear == True:
        params.NonLinearModel.set_params(halofit_version=halofit)
        params.NonLinear = camb.model.NonLinear_both

    # Computing spectra ######################################  QUI
    params.set_matter_power(redshifts=z, kmax=k.max() * 10 ** dlogk, silent=True, k_per_logint=0,
                            accurate_massive_neutrino_transfers=True)
    results = camb.get_results(params)
    kh, z, pkh = results.get_matter_power_spectrum(minkh=k.min() * 10. ** -dlogk, maxkh=k.max() * 10 ** dlogk,
                                                   npoints=npoints, var1=components[var_1], var2=components[var_2])

    # Interpolation to the required scales k's
    # I use UnivariateSpline because it makes good extrapolation
    pk = np.zeros((nz, len(np.atleast_1d(k))))
    for iz in range(nz):
        lnpower = si.InterpolatedUnivariateSpline(kh, np.log(pkh[iz]), k=3, ext=0, check_finite=False)
        pk[iz] = np.exp(lnpower(k))

    return k, pk


# -------------------------------------------------------------------------------
# CAMB_XPk
# -------------------------------------------------------------------------------
def camb_XPk(self,
             z=0.,
             k=np.logspace(-4., 2., 1001),
             nonlinear=False,
             halofit='mead2020',
             var_1=['tot'],
             var_2=['tot'],
             share_delta_neff=True,
             **kwargs
             ):
    """
    The function CAMB_XPk() runs the Python wrapper of CAMB and returns auto- and
    cross-spectra for all the quantities specified in 'var_1' and 'var_2'.
    Depending on the value of 'nonlinear', the power spectrum is linear or non-linear.
    It returns scales in units of :math:`h/\mathrm{Mpc}` and power spectra in units of (:math:`(\mathrm{Mpc}/h)^3`.

    :param z: Redshifts.
    :type z: array, default = 0

    :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
    :type k: array, default = ``np.logspace(-4., 2., 1001)``

    :param nonlinear: Whether to return non-linear power spectra.
    :type nonlinear: boolean, default = False

    :param halofit: Which version of Halofit to use. See CAMB documentation for further info.
    :type halofit: string, default = 'mead2020'

    :param var_1: Density field for the first component of the power spectrum.
    :type var_1: list of strings, default = ['tot']

    :param var_2: Density field for the second component of the power spectrum.

     - `'tot'`   : total matter
     - `'cdm'`   : cold dark matter
     - `'b'`     : baryons
     - `'nu'`    : neutrinos
     - `'cb'`    : cold dark matter + baryons
     - `'gamma'` : photons
     - `'v_cdm'` : cdm velocity
     - `'v_b'`   : baryon velocity
     - `'Phi'`   : Weyl potential
    :type var_2: list of strings, default = ['tot']


    :param kwargs: Keyword arguments to be passed to ``camb.set_params``. See CAMB documentation for further info: https://camb.readthedocs.io/en/latest/

    Returns
    -------

    k: array
        Scales in :math:`h/\mathrm{Mpc}`. Basically the same 'k' of the input.

    pk: dictionary
        Keys are given by `'var_1-var_2'`. Each of these is a 2D array of shape ``(len(z), len(k))`` containing :math:`P_\mathrm{var_1-var_2}(z,k)` in units of :math:`(\mathrm{Mpc}/h)^3`.

    """

    # Neutrino part
    nu_mass_eigen = len(np.unique([mm for mm in self.M_nu])) if np.any(self.M_nu != 0.) else 0
    nu_mass_numbers = [list(self.M_nu).count(x) for x in set(list(self.M_nu))]
    nu_mass_numbers = sorted(nu_mass_numbers, reverse=True) if np.any(self.M_nu != 0.) else [0]
    # Set parameters
    cambparams = {
        'num_nu_massive': self.massive_nu,
        'num_nu_massless': self.massless_nu,
        'nu_mass_eigenstates': nu_mass_eigen,
        'nu_mass_numbers': nu_mass_numbers,
        'nnu': self.N_eff,
        'omnuh2': self.omega_nu_tot,
        'ombh2': self.omega_b,
        'omch2': self.omega_cdm + self.omega_wdm_tot,
        'omk': self.Omega_K,
        'H0': 100. * self.h,
        'As': self.As,
        'ns': self.ns,
        'w': self.w0,
        'wa': self.wa,
        'TCMB': self.T_cmb,
        'tau': self.tau,
        'share_delta_neff': True,
        'dark_energy_model': 'DarkEnergyPPF'}
    # kwargs
    for key, value in kwargs.items():
        if not key in cambparams: cambparams[key] = value
    params = camb.set_params(**cambparams)

    # Redshifts and scales
    k = np.atleast_1d(k)
    nk = len(k)
    z = np.atleast_1d(z)
    nz = len(z)
    if nz > 3:
        spline = 'cubic'
    else:
        spline = 'linear'

    # Possible components to use
    components = {'tot': 'delta_tot',
                  'cdm': 'delta_cdm',
                  'b': 'delta_baryon',
                  'nu': 'delta_nu',
                  'cb': 'delta_nonu',
                  'gamma': 'delta_photon',
                  'v_cdm': 'v_newtonian_cdm',
                  'v_b': 'v_newtonian_baryon',
                  'Phi': 'Weyl'}

    # Number of points (according to logint)
    npoints = int(100 * np.log10(k.max() / k.min()))
    dlogk = 2. * np.log10(k.max() / k.min()) / npoints

    # Halofit version
    if nonlinear == True:
        # camb.nonlinear.Halofit(halofit_version = halofit)
        params.NonLinearModel.set_params(halofit_version=halofit)
        params.NonLinear = camb.model.NonLinear_both

    # Initialize power spectrum as a dictionary and compute it
    pk = {}
    params.set_matter_power(redshifts=z, kmax=k.max() * 10 ** dlogk, silent=True,
                            accurate_massive_neutrino_transfers=True)
    results = camb.get_results(params)

    # Fill the power spectrum array
    for c1 in var_1:
        for c2 in var_2:
            string = c1 + '-' + c2
            kh, zz, ppkk = results.get_matter_power_spectrum(minkh=k.min() * 10. ** -dlogk,
                                                             maxkh=k.max() * 10 ** dlogk,
                                                             npoints=npoints,
                                                             var1=components[c1],
                                                             var2=components[c2])

            pk[string] = np.zeros((nz, nk))
            for iz in range(nz):
                lnpower = si.InterpolatedUnivariateSpline(kh, np.log(ppkk[iz]), k=3, ext=0, check_finite=False)
                pk[string][iz] = np.exp(lnpower(k))

            # if nz != 1:
            #    power = si.interp2d(kh, zz, ppkk, kind = spline)
            #    pk[string] = power(k, z)
            #    pk[string] = np.nan_to_num(pk[string])
            # else:
            #    power = si.interp1d(kh, ppkk, kind = spline)
            #    pk[string] = power(k)
            #    pk[string] = np.nan_to_num(pk[string])

    return k, pk


# -------------------------------------------------------------------------------
# CLASS_Pk
# -------------------------------------------------------------------------------
def class_Pk(self,
             z=0.,
             k=np.logspace(-4., 2., 1001),
             nonlinear=False,
             halofit='halofit',
             **kwargs):
    """
    This routine uses CLASS to return power spectra for the chosen cosmology. Depending
    on the value of 'nonlinear', the power spectrum is linear or non-linear.

    :param z: Redshifts.
    :type z: array, default = 0

    :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
    :type k: array, default = ``np.logspace(-4., 2., 1001)``

    :param nonlinear: Whether to return non-linear power spectra.
    :type nonlinear: boolean, default = False

    :param kwargs: Keyword arguments of ``classy.pyx`` (see the file `explanatory.ini` in Class or https://github.com/lesgourg/class_public/blob/master/python/classy.pyx)

    Returns
    -------

    k: array
        Scales in :math:`h/\mathrm{Mpc}`. Basically the same 'k' of the input.

    pk: 2D array of shape ``(len(z), len(k))``
        Power spectrum in units of :math:`(\mathrm{Mpc}/h)^3`.
    """

    # Set halofit for non-linear computation
    if nonlinear == True:
        halofit = halofit
    else:
        halofit = 'none'

    # Setting lengths
    nk = len(np.atleast_1d(k))
    nz = len(np.atleast_1d(z))
    z = np.atleast_1d(z)
    k = np.atleast_1d(k)
    kmax = max(k.max(), 500.)
    zmax = max(z.max(), 101.)
    tau = self.tau
    params = {
        'output': 'mPk dTk',
        'n_s': self.ns,
        'h': self.h,
        'omega_b': self.Omega_b * self.h ** 2.,
        'omega_cdm': self.Omega_cdm * self.h ** 2.,
        'Omega_k': self.Omega_K,
        'tau_reio': self.tau,
        'T_cmb': self.T_cmb,
        'P_k_max_h/Mpc': kmax,
        'z_max_pk': zmax,
        'non_linear': halofit}
    if self.use_EFT:
        params['use_EFT'] = 'yes'
        params['gravity_model'] = self.gravity_model
        params['parameters_smg'] = ', '.join(str(x) for x in self.parameters_smg)
        params['M_pl_evolution'] = 'yes'

    # Set initial conditions
    if self.sigma_8 is not None:
        params['sigma8'] = self.sigma_8
    else:
        params['A_s'] = self.As
    # Set dark energy
    if self.w0 != -1. or self.wa != 0.:
        params['Omega_fld'] = self.Omega_lambda
        params['w0_fld'] = self.w0
        params['wa_fld'] = self.wa
    # Set neutrino masses
    params['N_ur'] = self.massless_nu
    params['N_ncdm'] = self.massive_nu
    if self.massive_nu != 0:
        params['m_ncdm'] = ', '.join(str(x) for x in self.M_nu)
        params['T_ncdm'] = ', '.join(str(self.Gamma_nu) for x in self.M_nu)
    # Set WDM masses (remove UR species cause Class treats WDM and neutrinos the same way)
    params['N_ncdm'] += self.N_wdm
    if self.N_wdm > 0 and self.massive_nu > 0.:
        params['m_ncdm'] += ', ';
        params['T_ncdm'] += ', '
        params['m_ncdm'] += ', '.join(str(x) for x in self.M_wdm)
        params['T_ncdm'] += ', '.join(str(x) for x in self.Gamma_wdm)
    elif self.N_wdm > 0:
        params['m_ncdm'] = ', '.join(str(x) for x in self.M_wdm)
        params['T_ncdm'] = ', '.join(str(x) for x in self.Gamma_wdm)
    # Add the keyword arguments
    for key, value in kwargs.items():
        if not key in params:
            params[key] = value
        else:
            raise KeyError("Parameter %s already exists in the dictionary, impossible to substitute it." % key)

    # Compute
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    # I change to k/h since CLASS uses k in units of 1/Mpc
    k *= self.h

    # Storing Pk
    pk = np.zeros((nz, nk))
    for i in range(nk):
        for j in range(nz):
            pk[j, i] = cosmo.pk(k[i], z[j]) * self.h ** 3.
    # Re-switching to (Mpc/h) units
    k /= self.h

    cosmo.struct_cleanup()
    cosmo.empty()

    return k, pk


# -------------------------------------------------------------------------------
# CLASS_XPk
# -------------------------------------------------------------------------------
def class_XPk(self,
              z=0.,
              k=np.logspace(-4., 2., 1001),
              nonlinear=False,
              halofit='halofit',
              var_1=['tot'],
              var_2=['tot'],
              **kwargs
              ):
    """
    The function class_XPk() runs the Python wrapper of CLASS and returns auto- and
    cross-spectra for all the quantities specified in 'var_1' and 'var_2'.
    Depending on the value of 'nonlinear', the power spectrum is linear or non-linear.
    Halofit by Takahashi is empoyed.

    :param z: Redshifts.
    :type z: array, default = 0

    :param k: Scales in units of :math:`h/\mathrm{Mpc}`.
    :type k: array, default = ``np.logspace(-4., 2., 1001)``

    :param nonlinear: Whether to return non-linear power spectra.
    :type nonlinear: boolean, default = False

    :param var_1: Density field for the first component of the power spectrum.
    :type var_1: list of strings, default = ['tot']

    :param var_2: Density field for the second component of the power spectrum.

     - `'tot'`   : total matter
     - `'cdm'`   : cold dark matter
     - `'b'`     : baryons
     - `'nu'`    : massive neutrinos
     - `'ur'`    : massless neutrinos
     - `'cb'`    : cold dark matter + baryons
     - `'cold'`  : cold dark matter + baryons + warm dark matter
     - `'gamma'` : photons
     - `'Phi'`   : Weyl potential
     - `'Psi'`   : Weyl potential
    :type var_2: list of strings, default = ['tot']


    :param kwargs: Keyword arguments of ``classy.pyx`` (see the file `explanatory.ini` in Class or https://github.com/lesgourg/class_public/blob/master/python/classy.pyx)

    Returns
    -------

    k: array
        Scales in :math:`h/\mathrm{Mpc}`. Basically the same 'k' of the input.

    pk: dictionary
        Keys are given by `'var_1-var_2'`. Each of these is a 2D array of shape ``(len(z), len(k))`` containing :math:`P_\mathrm{var_1-var_2}(z,k)` in units of :math:`(\mathrm{Mpc}/h)^3`.
    """
    components = {'tot': 'd_tot',
                  'cdm': 'd_cdm',
                  'wdm': 'd_wdm',
                  'b': 'd_b',
                  'cb': 'd_cb',
                  'cold': 'd_cold',
                  'nu': 'd_nu',
                  'ur': 'd_ur',
                  'gamma': 'd_g',
                  'Phi': 'phi',
                  'Psi': 'psi'}

    # Set halofit for non-linear computation
    if nonlinear == True:
        halofit = halofit
    else:
        halofit = 'none'

    # Setting lengths
    nk = len(np.atleast_1d(k))
    nz = len(np.atleast_1d(z))
    z = np.atleast_1d(z)
    k = np.atleast_1d(k)
    kmax = max(k.max(), 500.)
    zmax = max(z.max(), 100.)
    # Parameters
    params = {
        'output': 'mPk dTk',
        'n_s': self.ns,
        'h': self.h,
        'omega_b': self.Omega_b * self.h ** 2.,
        'omega_cdm': self.Omega_cdm * self.h ** 2.,
        'Omega_k': self.Omega_K,
        'tau_reio': self.tau,
        'T_cmb': self.T_cmb,
        'P_k_max_h/Mpc': kmax,
        'z_max_pk': zmax,
        'non_linear': halofit}
    if self.use_EFT:
        params['use_EFT'] = 'yes'
        params['gravity_model'] = self.gravity_model
        params['parameters_smg'] = ', '.join(str(x) for x in self.parameters_smg)
        params['M_pl_evolution'] = 'yes'

    # Set initial conditions
    if self.sigma_8 is not None:
        params['sigma8'] = self.sigma_8
    else:
        params['A_s'] = self.As
    # Set dark energy
    if self.w0 != -1. or self.wa != 0.:
        params['Omega_fld'] = self.Omega_lambda
        params['w0_fld'] = self.w0
        params['wa_fld'] = self.wa
    # Set neutrino masses
    params['N_ur'] = self.massless_nu
    params['N_ncdm'] = self.massive_nu
    if self.massive_nu != 0:
        params['m_ncdm'] = ', '.join(str(x) for x in self.M_nu)
        params['T_ncdm'] = ', '.join(str(self.Gamma_nu) for x in self.M_nu)
    # Set WDM masses (remove UR species cause Class treats WDM and neutrinos the same way)
    params['N_ncdm'] += self.N_wdm
    if self.N_wdm > 0 and self.massive_nu > 0.:
        params['m_ncdm'] += ', ';
        params['T_ncdm'] += ', '
        params['m_ncdm'] += ', '.join(str(x) for x in self.M_wdm)
        params['T_ncdm'] += ', '.join(str(x) for x in self.Gamma_wdm)
    elif self.N_wdm > 0:
        params['m_ncdm'] = ', '.join(str(x) for x in self.M_wdm)
        params['T_ncdm'] = ', '.join(str(x) for x in self.Gamma_wdm)
    # Add the keyword arguments
    for key, value in kwargs.items():
        if not key in params:
            params[key] = value
        else:
            raise KeyError("Parameter %s already exists in the dictionary, impossible to substitute it." % key)

    # Compute
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    # Setting lengths
    n1 = len(var_1)
    n2 = len(var_2)

    # I change to k/h since CLASS uses k in units of 1/Mpc
    k *= self.h

    # Storing Pk
    pk_m = np.zeros((nz, nk))
    for i in range(nk):
        for j in range(nz):
            pk_m[j, i] = cosmo.pk(k[i], z[j]) * self.h ** 3.

    # Re-switching to (Mpc/h) units
    k /= self.h

    # Get transfer functions and rescale the power spectrum
    pk = {}
    # Loop over variables
    for c1 in var_1:
        for c2 in var_2:
            string = c1 + '-' + c2
            pk[string] = np.zeros((nz, nk))
            # Loop over redshifts
            for ind_z in range(nz):
                # Get transfer functions at z
                TF = cosmo.get_transfer(z=z[ind_z])
                TF['d_nu'] = np.zeros_like(TF['k (h/Mpc)'])
                for inu in range(self.massive_nu):
                    index = inu
                    TF['d_nu'] += self.M_nu[inu] * TF['d_ncdm[%i]' % index] / np.sum(self.M_nu)
                TF['d_wdm'] = np.zeros_like(TF['k (h/Mpc)'])
                for inw in range(self.N_wdm):
                    index = inw + self.massive_nu
                    TF['d_wdm'] += self.Omega_wdm[inw] / self.Omega_wdm_tot * TF['d_ncdm[%i]' % index]
                TF['d_cold'] = (self.Omega_cdm * TF['d_cdm'] +
                                self.Omega_wdm_tot * TF['d_wdm'] +
                                self.Omega_b * TF['d_b']) / self.Omega_cold
                TF['d_cb'] = (self.Omega_cdm * TF['d_cdm'] +
                              self.Omega_b * TF['d_b']) / self.Omega_cb
                # !!!!!!!!!!!
                # For reasons unknown, for non-standard cosmological constant, the amplitude is off...
                # !!!!!!!!!!!
                if self.w0 != -1. or self.wa != 0.:
                    TF['d_tot'] = (self.Omega_cold * TF['d_cold'] +
                                   self.Omega_nu_tot * TF['d_nu']) / self.Omega_m
                # !!!!!!!!!!!
                # Interpolation of matter T(k)
                tm_int = si.interp1d(TF['k (h/Mpc)'], TF['d_tot'],
                                     kind='cubic', fill_value="extrapolate", bounds_error=False)
                transf_m = tm_int(k)
                # Interpolate them to required k
                t1_int = si.interp1d(TF['k (h/Mpc)'], TF[components[c1]],
                                     kind='cubic', fill_value="extrapolate", bounds_error=False)
                t2_int = si.interp1d(TF['k (h/Mpc)'], TF[components[c2]],
                                     kind='cubic', fill_value="extrapolate", bounds_error=False)
                transf_1 = t1_int(k)
                transf_2 = t2_int(k)
                # Rescaling
                pk[string][ind_z] = pk_m[ind_z] * transf_1 * transf_2 / transf_m ** 2.
    cosmo.struct_cleanup()
    cosmo.empty()

    return k, pk