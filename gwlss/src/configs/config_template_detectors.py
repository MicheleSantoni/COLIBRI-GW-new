#-----------------------------------------------------------------------------------------
#                                CONFIGURATION FILE: SETTINGS
#-----------------------------------------------------------------------------------------

# path to local version of colibri
colibri_path ="/home/cptsu4/santoni/home_cpt/Home_CPT/gwlss/src/COLIBRI_GW"

# GW detector (ET_Delta_2CE, ET_2L_2CE, ET_Delta, ET_2L, LVK)
GW_det = 'ET_2L_2CE'

# galaxy survey (euclid_photo, euclid_spectro, ska)
gal_det = 'euclid_photo'

# Years of observation
yr = 10

# Define the number of bins
n_bins_z = 13
n_bins_dl = 16

# Define the redshift total range
z_m = 0.001
z_M = 7

# Define the luminosity distance total range
dlm = 1
dlM = 100000

# Define the galaxy bin range
z_m_bin = 0.0001
z_M_bin = 2.5

# Define the GW bin range in redshift (will be converted in dl using the fiducial model)
z_m_bin_GW = 0.0001
z_M_bin_GW = 8

# Set the binning strategy (right_cosmo, wrong_cosmo, equal_pop, equal_space)
bin_strategy = 'right_cosmo'

# Include the lensing
Lensing = False
# Compute power spectra (True)
fourier = True

# Parameters for GW bias model
A_GW = 1.2
gamma = 0.59

# Fraction of the sky covered from the survey
f_sky = 0.35
f_sky_GW = 1

# Errors on the galaxy distribution
sig_gal = 0.05

# l min
l_min = 5

# Define step size for numerical differentiation
step = 1e-3






