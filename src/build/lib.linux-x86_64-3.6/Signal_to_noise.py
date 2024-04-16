import numpy as np
import matplotlib.pyplot as plt
import cython_mylibc as pippo
import os
import healpy as hp

def S_2_N(beta, cov_matrix):

    cov_inv = np.linalg.inv(cov_matrix)
    s_n = 0
    temp = np.zeros(len(cov_matrix[0]))
    for i in range(len(cov_matrix[0])):
        for j in range(len(beta)):
            temp[i] += cov_inv[i][j]*beta[j]
        s_n += beta[i].T*temp[i]
    return np.sqrt(np.sum(s_n))

def S_2_N_th(beta, variance):
    s_n = np.divide((beta)**2, variance)
    return np.sqrt(np.sum(s_n))

def Variance_cl_shot_noise(ell, cl_tg,cl_tt, cl_gg, fsky=1.):
    variance = np.zeros_like(cl_tg)
    for l,ell in enumerate(ell):
        variance[l]=(cl_tt[l]*cl_gg[l]+cl_tg[l]**2)/((2*ell+1)*fsky)
    return variance

if __name__ == "__main__":

    jmax = 12
    lmax = 256#782
    B = pippo.mylibpy_jmax_lmax2B(jmax, lmax)
    
    simparams = {'nside'   : 512,
                 'ngal'    : 5.76e5, 
     	     	 'ngal_dim': 'ster',
    	     	 'pixwin'  : False}

    nside=128

    mask = hp.read_map(f'mask/EUCLID/mask_rsd2022g-wide-footprint-year-6-equ-order-13-moc_ns0128_G_filled_2deg2.fits')
    fsky = np.mean(mask)
    print(fsky)


    #beta_theory = np.loadtxt(f'/ehome/bdecaro/xcmbneed/src/output_needlet_TG_OmL/Grid_spectra_30_planck_2_lmin0/beta_TS_galS_theoretical_OmL_fiducial_B{B}.dat')
    cl_theory = np.loadtxt(f'spectra/inifiles/EUCLID_fiducial.dat')
    cl_theory_tg = cl_theory[2]
    cl_theory_tt = cl_theory[1]
    cl_theory_gg = cl_theory[3]
    ell_theory = cl_theory[0]
    delta_ell = 1

    Cls_TG_bin_ell = np.array([cl_theory_tg[l:(l+delta_ell)] for l in range(0,lmax-delta_ell,delta_ell)])
    Cls_TG_bin_mean = np.array([np.mean(Cls_TG_bin_ell[l]) for l in range(Cls_TG_bin_ell.shape[0])])

    cov_matrix = np.loadtxt(f'/ehome/bdecaro/xcmbneed/src/output_Cl_TG/EUCLID/Mask_noise/TG_128_fsky0.35/cov_TS_galT_lmax256_nside128.dat')
    cov_matrix_mask = np.loadtxt(f'/ehome/bdecaro/xcmbneed/src/output_Cl_TG/EUCLID/Mask_noise/TG_128_fsky0.35/cov_TS_galT_lmax256_nside128_fsky0.35480745633443195.dat')
    #cov_matrix = np.loadtxt(f'/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/TG_512_planck_2_lmin0/cov_TS_galS_jmax12_B{B}_nside{simparams["nside"]}.dat')
    variance=Variance_cl_shot_noise(ell=ell_theory,cl_tg=cl_theory_tg, cl_tt=cl_theory_tt, cl_gg=cl_theory_gg)
    variance_mask=Variance_cl_shot_noise(ell=ell_theory,cl_tg=cl_theory_tg, cl_tt=cl_theory_tt, cl_gg=cl_theory_gg, fsky=fsky)
    
    #variance = np.loadtxt(f'/ehome/bdecaro/xcmbneed/src/output_needlet_TG_OmL/Grid_spectra_30_planck_2_lmin0/variance_TS_galS_theoretical_OmL_fiducial_B{B}.dat') #questa varianza qua è con la radice
    #cov_matrix_mask = np.loadtxt(f'/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/Mask_noise/TG_256_mask_shot_noise/cov_TS_galS_jmax12_B = 1.74 $_nside256_mask.dat')
    
    #S_N = S_2_N(beta=beta_theory[1:jmax], cov_matrix=cov_matrix[1:(jmax),1:(jmax)])
    S_N = S_2_N(beta=Cls_TG_bin_mean, cov_matrix=cov_matrix)
    S_N_theory = S_2_N_th(beta=cl_theory_tg, variance=variance)
    #S_N_theory = S_2_N_th(beta=beta_theory, variance=variance**2)
    print(f'S/N={S_N}, S/N variance={S_N_theory}')

    #S_N_mask = S_2_N(beta=fsky*beta_theory[1:jmax], cov_matrix=cov_matrix_mask[1:jmax][1:jmax]) #mettere beta mascherati dalle sim e fare il rapporto senza maschera
    S_N_mask = S_2_N(beta=Cls_TG_bin_mean, cov_matrix=cov_matrix_mask) #mettere beta mascherati dalle sim e fare il rapporto senza maschera
    S_N_theory_mask = S_2_N_th(beta=cl_theory_tg, variance=(variance_mask))

    print(f'S/N mask={S_N_mask}, S/N variance mask={S_N_theory_mask}')
    print(S_N_theory_mask/S_N_theory)
