import matplotlib.pyplot as plt
import numpy as np
import os


import re

def build_vector(n_bins_z,n_bins_dl,Cl_GGW,Cl_GWGW,Cl_GG):
    # vector_idx: labels for all spectra
    vec_idx = vector_idx(n_bins_z, n_bins_dl)

    # vector_cl: flatten all Cℓ into a single data vector
    vec = vector_cl(cl_cross=Cl_GGW, cl_auto1=Cl_GG, cl_auto2=Cl_GWGW)

    # Build a dict of Cℓ by label   ---> Associa ad ogni termine la sua corretta etichetta
    vec_dict = {label: vec[i] for i, label in enumerate(vec_idx)}
    return vec, vec_dict,vec_idx

def vector_idx(nbins_1, nbins_2):
    '''
     Crea un array con le etichettte per i tracers X/Y e i bins: X0X1 -> tracer X e X, bins 0 e 1
    '''
    lenght = nbins_1 ** 2 - np.sum(range(nbins_1)) + (nbins_1 * nbins_2) + nbins_2 ** 2 - np.sum(range(nbins_2))
    vec_cl = np.zeros(shape=(lenght), dtype=object)
    i = 0
    for j in range(nbins_1):
        for k in range(j, nbins_1):
            vec_cl[i] = 'X%iX%i' % (j, k)
            i += 1
    for j in range(nbins_1):
        for k in range(nbins_2):
            vec_cl[i] = 'X%iY%i' % (j, k)
            i += 1
    for j in range(nbins_2):
        for k in range(j, nbins_2):
            vec_cl[i] = 'Y%iY%i' % (j, k)
            i += 1
    return vec_cl


def vector_cl(cl_cross, cl_auto1, cl_auto2):
    '''
     Crea una matrice di dimensione (Lunghezza tutte le Cl, n_ell)
    '''
    nbins_1 = len(cl_auto1)
    nbins_2 = len(cl_auto2)
    lenght = nbins_1 ** 2 - np.sum(range(nbins_1)) + (nbins_1 * nbins_2) + nbins_2 ** 2 - np.sum(range(nbins_2))
    vec_cl = np.zeros(shape=(lenght, len(cl_auto1[0, 0])))
    i = 0
    for j in range(nbins_1):
        for k in range(j, nbins_1):
            vec_cl[i] = cl_auto1[j, k]
            i += 1
    for j in range(nbins_1):
        for k in range(nbins_2):
            vec_cl[i] = cl_cross[j, k]
            i += 1
    for j in range(nbins_2):
        for k in range(j, nbins_2):
            vec_cl[i] = cl_auto2[j, k]
            i += 1
    return vec_cl


# -----------------------------------------------------------------------------------------
#                                COVARIANCE BUILDING AND INVERTING
# -----------------------------------------------------------------------------------------
def build_cov_and_inv(n_bins_z,n_bins_dl,vec_fid,vec_idx,ll):
    """
        Build and Invert the 3×3 covariance block at each multipole ℓ.

        Parameters
        ----------
        Cov : array, shape (3, 3, N_ell)
            Covariance matrix for [GG, GWGW, GGW] at each ℓ.
        ell : array, shape (N_ell,)
            Multipole values (only used for sanity checks / loops).

        Returns
        -------
        Cov_inv : array, shape (3, 3, N_ell)
            Inverse covariance for each ℓ.
    """
    # Building the full covariance matrix
    tot_len = n_bins_z ** 2 - np.sum(range(n_bins_z)) + n_bins_z * n_bins_dl + n_bins_dl ** 2 - np.sum(range(n_bins_dl))
    cov_mat = np.zeros(shape=(tot_len, tot_len, len(ll)))
    for i in range(tot_len):
        for j in range(tot_len):
            # sort_lab: Label reshuffling helpers
            cov_mat[i, j] = vec_fid[sort_lab(lab_I1J1(vec_idx, i, j))] * vec_fid[sort_lab(lab_I2J2(vec_idx, i, j))] + vec_fid[sort_lab(lab_I1J2(vec_idx, i, j))] * vec_fid[sort_lab(lab_I2J1(vec_idx, i, j))]

    cov_inv = np.zeros_like(cov_mat)
    for l in range(len(ll)):
        cov_inv[:, :, l] = np.linalg.inv(cov_mat[:, :, l])

    return cov_mat, cov_inv


def lab_I1J1(vec_idx, i, j):
    pattern = r'([XY])(\d+)([XY])(\d+)'
    match_i = re.match(pattern, vec_idx[i])
    match_j = re.match(pattern, vec_idx[j])
    if match_i and match_j:
        letter1, num1 = match_i.group(1), match_i.group(2)
        letter2, num2 = match_j.group(1), match_j.group(2)
        return f'{letter1}{num1}{letter2}{num2}'
    else:
        raise ValueError(f"Invalid label format: {vec_idx[i]} or {vec_idx[j]}")


def lab_I2J2(vec_idx, i, j):
    pattern = r'([XY])(\d+)([XY])(\d+)'
    match_i = re.match(pattern, vec_idx[i])
    match_j = re.match(pattern, vec_idx[j])
    if match_i and match_j:
        letter1, num1 = match_i.group(3), match_i.group(4)
        letter2, num2 = match_j.group(3), match_j.group(4)
        return f'{letter1}{num1}{letter2}{num2}'
    else:
        raise ValueError(f"Invalid label format: {vec_idx[i]} or {vec_idx[j]}")


def lab_I1J2(vec_idx, i, j):
    pattern = r'([XY])(\d+)([XY])(\d+)'  # Match: letter-number-letter-number
    match_i = re.match(pattern, vec_idx[i])
    match_j = re.match(pattern, vec_idx[j])
    if match_i and match_j:
        # Take first pair from i, second pair from j
        letter1, num1 = match_i.group(1), match_i.group(2)
        letter2, num2 = match_j.group(3), match_j.group(4)
        return f'{letter1}{num1}{letter2}{num2}'
    else:
        raise ValueError(f"Invalid label format: {vec_idx[i]} or {vec_idx[j]}")


def lab_I2J1(vec_idx, i, j):
    pattern = r'([XY])(\d+)([XY])(\d+)'
    match_i = re.match(pattern, vec_idx[i])
    match_j = re.match(pattern, vec_idx[j])
    if match_i and match_j:
        letter1, num1 = match_i.group(3), match_i.group(4)
        letter2, num2 = match_j.group(1), match_j.group(2)
        return f'{letter1}{num1}{letter2}{num2}'
    else:
        raise ValueError(f"Invalid label format: {vec_idx[i]} or {vec_idx[j]}")


def sort_lab(label):
    '''
        rimette in ordine le labels del tipo: Prima X e poi Y; prima numeri piccoli e poi più alti:
        sort_lab('X2X0') → X0X2
        sort_lab('Y3X1') → X1Y3 (since X < Y )
    '''
    pattern = r'([XY])(\d+)([XY])(\d+)'
    match = re.match(pattern, label)
    if match:
        part1 = (match.group(1), int(match.group(2)))  # e.g., ('X', 10)
        part2 = (match.group(3), int(match.group(4)))  # e.g., ('X', 0)
        # Sort using letter + number
        first, second = sorted([part1, part2])
        return f'{first[0]}{first[1]}{second[0]}{second[1]}'
    else:
        raise ValueError(f"Invalid label format: {label}")

def plot_SNR_xL(ll, snr_l,output_path):
    plt.figure(figsize=(7, 5))
    plt.plot(ll, snr_l, marker='o')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'${\rm SNR}_\ell$')
    plt.title('Signal-to-noise per multipole')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'SNR_per_multipole.pdf'), bbox_inches='tight')
    plt.close()

def plot_SNR_cumulative(ll, snr_l,output_path):
    snr_cum = np.sqrt(np.cumsum(snr_l))

    plt.figure(figsize=(7, 5))
    plt.plot(ll, snr_cum)
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'${\rm SNR}(<\ell)$')
    plt.title('SNR cumulative')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'SNR_per_cumulative.pdf'), bbox_inches='tight')
    plt.close()






