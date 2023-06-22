#!/usr/bin/env python
import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import healpy as hp
import argparse, os, sys, warnings, glob
import cython_mylibc as mylibc
import analysis, utils, spectra, sims
from IPython import embed
import master


import seaborn as sns

import os
path = os.path.abspath(spectra.__file__)

# Matplotlib defaults ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#rc('text',usetex=True)
#rc('font',**{'family':'serif','serif':['Computer Modern']})
plt.rcParams['axes.linewidth']  = 3.
plt.rcParams['axes.labelsize']  = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['xtick.major.size'] = 7
plt.rcParams['ytick.major.size'] = 7
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['legend.fontsize']  = 15
plt.rcParams['legend.frameon']  = False

plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
#plt.rcParams['backend'] = 'WX'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# embed()

# Parameters
simparams = {   'nside'   : 128 ,
                'ngal'    : 5.76e5, #dovrebbe importare solo per lo shot noise (noise poissoniano)
                'ngal_dim': 'ster',
                'pixwin'  : False}
nside=simparams['nside']
lmax = 256
nsim = 500
delta_ell = 1

mask = hp.read_map(f'/ehome/bdecaro/xcmbneed/src/mask/EUCLID/mask_planck_comm_2018_x_euclid_fsky0p35_nside={nside}.fits')
fsky = np.mean(mask)
print(fsky)

# Paths
fname_xcspectra = '/ehome/bdecaro/xcmbneed/src/spectra/inifiles/EUCLID_fiducial.dat'
sims_dir        = f'/ehome/bdecaro/xcmbneed/src/sims/Euclid_sims/TG_sims_{nside}/'
out_dir         = f'output_Cl_TG/EUCLID/Mask_noise/TG_{nside}_fsky{fsky:.2f}_nsim{nsim}/'
path_inpainting = 'inpainting/inpainting.py'
cov_dir 		= f'covariance/Cl/EUCLID/Mask_noise/TG_{nside}/'
if not os.path.exists(cov_dir):
        os.makedirs(cov_dir)


# Loading theory spectra
xcspectra = spectra.XCSpectraFile(fname_xcspectra, nltt = None,WantTG = True)

# Simulations class
simulations = sims.KGsimulations(xcspectra, sims_dir, simparams, WantTG = True)
simulations.Run(nsim, WantTG = True)

# Cl Analysis

myanalysis = analysis.HarmAnalysis(lmax, out_dir, simulations, lmin=2, delta_ell=delta_ell, pixwin=simparams['pixwin'], fsky_approx=True)
lbins = myanalysis.ell_binned
myanalysis_mask = analysis.HarmAnalysis(lmax, out_dir, simulations, lmin=2, delta_ell=delta_ell,mask=mask, pixwin=simparams['pixwin'], fsky_approx=True)


# Computing simulated Cls 
print("...computing Betajs for simulations...")
fname_cl_sims_TS_galT      = f'cl_sims_TS_galT_lmax{lmax}_nside{nside}.dat'
fname_cl_sims_TS_galT_mask      = f'cl_sims_TS_galT_lmax{lmax}_nside{nside}_fsky{fsky}.dat'

cl_TS_galT = myanalysis.GetClSimsFromMaps(field1='TS', nsim=nsim, field2='galT', fname=fname_cl_sims_TS_galT)
cl_TS_galT_mask = myanalysis_mask.GetClSimsFromMaps(field1='TS', nsim=nsim, field2='galT', fname=fname_cl_sims_TS_galT_mask)

cl_TS_galT_mean =np.mean(cl_TS_galT, axis=0)
cl_TS_galT_mask_mean=np.mean(cl_TS_galT_mask, axis=0)


# Covariances
print("...computing Cov Matrices...")
fname_cov_TS_galT      = f'cov_TS_galT_lmax{lmax}_nside{nside}.dat'
fname_cov_TS_galT_mask      = f'cov_TS_galT_lmax{lmax}_nside{nside}_fsky{fsky}.dat'

cov_TS_galT, corr_TS_galT           = myanalysis.GetCovMatrixFromMaps(field1='TS', nsim=nsim, field2='galT', fname=fname_cov_TS_galT, fname_sims=fname_cl_sims_TS_galT)
cov_TS_galT_mask, corr_TS_galT_mask           = myanalysis_mask.GetCovMatrixFromMaps(field1='TS', nsim=nsim, field2='galT', fname=fname_cov_TS_galT_mask, fname_sims=fname_cl_sims_TS_galT_mask)

print("...done...")

# Some plots
print("...here come the plots...")

# Covariances
#fig, axs = plt.subplots(ncols=2, nrows=1,figsize=(52,25))  
#
#mask_ = np.tri(corr_TS_galT.shape[0],corr_TS_galT.shape[1],0)
#
##plt.subplot(131)
#axs[0].set_title(r'Corr $T^S\times gal^T$')
#sns.color_palette("crest", as_cmap=True)
#sns.heatmap(corr_TS_galT, annot=True, fmt='.2f', cmap  = 'viridis',mask=mask_, ax=axs[0])
#
#axs[1].set_title(r'Corr $T^S\times gal^T$ Masked')
#sns.color_palette("crest", as_cmap=True)
#sns.heatmap(corr_TS_galT_mask, annot=True, fmt='.2f', cmap  = 'viridis',mask=mask_, ax=axs[1])
#
#fig.tight_layout()
#plt.savefig(cov_dir+f'corr_TS_galT_lmax{lmax}_nsim{nsim}_nside{nside}.png')

# Cls 
cltg_theory = xcspectra.cltg[:lmax-1]
ell_theory = xcspectra.ell[:lmax-1]


print(f'ratio cl mean  / cl theory = {np.mean(cl_TS_galT_mean/cltg_theory)}')
print(f'ratio cl mean mask / cl theory = {np.mean(cl_TS_galT_mask_mean/cltg_theory)}')
print(f'ratio cl mean mask - cl mean = {np.mean(cl_TS_galT_mask_mean-cl_TS_galT_mean)}')

print(f'ratio cl mean mask  = {np.mean(cl_TS_galT_mask_mean)}')
print(f'ratio cl mean  = {np.mean(cl_TS_galT_mean)}')

fig = plt.figure(figsize=(25,10))
ax  = fig.add_subplot(1, 2, 1)

ax.plot(ell_theory,cl_TS_galT_mean/cltg_theory,'o' ,color='firebrick' , label = 'Full sky')
ax.plot(ell_theory,cl_TS_galT_mask_mean/cltg_theory,'o', color='seagreen' ,label = 'Masked')
ax.axhline(y=fsky,ls='--', color='k')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\langle C_{\ell}^{Tgal} \rangle /C_{\ell}^{Tgal, th}$')
ax.legend(loc='best')
plt.savefig(out_dir+f'ratio_cl_sim_cl_theory_T_gal_lmax{lmax}_nside{nside}_fsky_{fsky}.png', bbox_inches='tight')

lbins = myanalysis.ell_binned

ell_bin = np.array([(xcspectra.ell[l+delta_ell] -xcspectra.ell[l])//2 + xcspectra.ell[l]  for l in range(0,lmax-delta_ell,delta_ell)])


kgb = myanalysis.bin_spectra(xcspectra.cltg)
fig = plt.figure(figsize=(25,10))
ax  = fig.add_subplot(1, 2, 1)

cl_theory_binned=np.array([xcspectra.cltg[l-1] for l in lbins])


ax.plot(ell_theory, ell_theory*(ell_theory+1)/(2*np.pi)*cltg_theory, color='k', lw=2, label='Theory')
#ax.plot(lbins,(lbins)*(lbins+1)/(2*np.pi)*kgb, 'ko',label='Theory')
#ax.errorbar(lbins-5, np.mean(cl_TS_galT, axis=0), yerr=np.diag(cov_TS_galT/nsim)**.5, fmt='o', color='seagreen', capsize=0, label=r'$T^S \times gal^T$')
ax.errorbar(ell_bin, (ell_bin)*(ell_bin+1)/(2*np.pi)*cl_TS_galT_mean, yerr=np.diag(((ell_bin*(ell_bin+1)/(2*np.pi))**2)*cov_TS_galT/nsim)**.5, fmt='o', color='firebrick', capsize=0, label=r'$T^S \times gal^T$')
#ax.set_xticks(xcspectra.ell[:lmax])
#ax.set_xlim(0,20)
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\langle D_{\ell}^{Tgal} \rangle$')
ax.legend(loc='best')
plt.savefig(out_dir+f'cl_T_gal_lmax{lmax}_nside{nside}_fsky_{fsky}.png', bbox_inches='tight')


fig = plt.figure(figsize=(10,8))

plt.suptitle(r'$\ell_{max} =$'+str(lmax) + r'$~N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

kgb = myanalysis.bin_spectra(xcspectra.cltg)

ax.axhline(ls='--', color='k')
ax.errorbar(xcspectra.ell[:lmax-1], (cl_TS_galT_mean-cltg_theory)/cltg_theory, yerr=np.sqrt(np.diag(cov_TS_galT))/(np.sqrt(nsim)*cltg_theory), color='firebrick', fmt='o', capsize=0, label=r'$T^S \times gal^T$')
ax.errorbar(xcspectra.ell[:lmax-1], (cl_TS_galT_mask_mean-cltg_theory)/cltg_theory, yerr=np.sqrt(np.diag(cov_TS_galT_mask))/(np.sqrt(nsim)*cltg_theory), color='seagreen', fmt='o', capsize=0, label=r'$T^S \times gal^T$ Masked')

#ax.errorbar(lbins, (np.mean(cl_TS_galT_mask, axis=0)-kgb)/kgb, yerr=np.sqrt(np.diag(cov_TS_galT_mask))/(np.sqrt(nsim)*kgb), color='seagreen', fmt='o', capsize=0, label=r'$T^S \times gal^T$ Masked')

ax.set_xlabel(r'$\ell$')
ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_ylabel(r'$(\langle C_{\ell}^{Tgal} \rangle - C_{\ell}^{Tgal, th})/C_{\ell}^{Tgal, th}$')
#ax.set_ylim([-0.2,0.3])

fig.tight_layout()
plt.savefig(out_dir+f'cl_T_gal_mean_lmax{lmax}_nsim{nsim}_nside{nside}.png', bbox_inches='tight')


