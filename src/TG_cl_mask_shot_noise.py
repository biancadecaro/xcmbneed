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
simparams = {'nside'   : 128,
             'ngal'    : 5.76e5, #dovrebbe importare solo per lo shot noise (noise poissoniano)
 	        'ngal_dim': 'ster',
	        'pixwin'  : False}
nside = simparams['nside']

# Paths
fname_xcspectra = '/ehome/bdecaro/xcmbneed/src/spectra/inifiles/CAMBSpectra_OmL_fiducial_lmin0.dat' #'spectra/CAMBSpectra_planck.dat'#
sims_dir        = f'sims/Needlet/Planck/Mask_noise/TGsims_{nside}_mask_shot_noise/'
out_dir         = f'output_Cl_TG/Planck/Mask_noise/TG_{nside}_mask_shot_noise/'
path_inpainting = 'inpainting/inpainting.py'
cov_dir         = f'covariance/Cl/Planck/Mask_noise/covariance_TG_{nside}_mask_shot_noise/'
if not os.path.exists(cov_dir):
        os.makedirs(cov_dir)

lmax = 512
nsim = 50
delta_ell = 50


mask = hp.read_map(f'mask/mask70_gal_nside={nside}.fits', verbose=False)
fsky = np.mean(mask)

# Loading theory spectra
xcspectra = spectra.XCSpectraFile(fname_xcspectra, nltt = None,WantTG = True)

# Simulations class
simulations = sims.KGsimulations(xcspectra, sims_dir, simparams, WantTG = True)
simulations.Run(nsim, WantTG = True)

# Cl Analysis
myanalysis = analysis.HarmAnalysis(lmax, out_dir, simulations, lmin=2, delta_ell=delta_ell, pixwin=simparams['pixwin'], fsky_approx=True)
myanalysis_mask   = analysis.HarmAnalysis(lmax, out_dir, simulations, lmin=2, delta_ell=delta_ell, mask=mask, pixwin=simparams['pixwin'], fsky_approx=True)

lbins = myanalysis.ell_binned


# Computing simulated Cls 
print("...computing Cls for simulations...")
fname_cl_sims_TS_galS      = f'cl_sims_TS_galS_lmax{lmax}_nside{nside}.dat'
#fname_cl_sims_TS_galT      = f'cl_sims_TS_galT_lmax{lmax}_nside{nside}.dat'

fname_cl_sims_TS_galS_mask      = f'cl_sims_TS_galS_lmax{lmax}_nside{nside}_fsky_{fsky}.dat'
#fname_cl_sims_TS_galT_mask      = f'cl_sims_TS_galT_lmax{lmax}_nside{nside}_fsky_{fsky}.dat'

#fname_cl_sims_TS_TS      = 'cl_sims_TS_TS_lmax'+str(lmax)+'_nside'+str(simparams['nside'])+'.dat'
#fname_cl_sims_galS_galS      = 'cl_sims_galS_galS_lmax'+str(lmax)+'_nside'+str(simparams['nside'])+'.dat'

cl_TS_galS = myanalysis.GetClSimsFromMaps('TS', nsim, field2='galS', fname=fname_cl_sims_TS_galS)
#cl_TS_galT = myanalysis.GetClSimsFromMaps('TS', nsim, field2='galT', fname=fname_cl_sims_TS_galT)

cl_TS_galS_mask = myanalysis_mask.GetClSimsFromMaps('TS', nsim, field2='galS', fname=fname_cl_sims_TS_galS_mask)
#cl_TS_galT_mask = myanalysis_mask.GetClSimsFromMaps('TS', nsim, field2='galT', fname=fname_cl_sims_TS_galT_mask)

#cl_TS_TS = myanalysis.GetClSimsFromMaps('TS', nsim, field2='TS', fname=fname_cl_sims_TS_TS)
#cl_galS_galS = myanalysis.GetClSimsFromMaps('galS', nsim, field2='galS', fname=fname_cl_sims_galS_galS)

# Covariances
print("...computing Cov Matrices...")
fname_cov_TS_galS      = f'cov_TS_galS_lmax{lmax}_nside{nside}.dat'
fname_cov_TS_galT      = f'cov_TS_galT_lmax{lmax}_nside{nside}.dat'

fname_cov_TS_galS_mask      = f'cov_TS_galS_lmax{lmax}_nside{nside}_fsky_{fsky}.dat'
fname_cov_TS_galT_mask      = f'cov_TS_galT_lmax{lmax}_nside{nside}_fsky_{fsky}.dat'

cov_TS_galS, corr_TS_galS           = myanalysis.GetCovMatrixFromMaps(field1='TS', nsim=nsim, field2='galS', fname=fname_cov_TS_galS, fname_sims=fname_cl_sims_TS_galS)
#cov_TS_galT, corr_TS_galT           = myanalysis.GetCovMatrixFromMaps(field1='TS', nsim=nsim, field2='galT', fname=fname_cov_TS_galT, fname_sims=fname_cl_sims_TS_galT)

cov_TS_galS_mask, corr_TS_galS_mask           = myanalysis_mask.GetCovMatrixFromMaps(field1='TS', nsim=nsim, field2='galS', fname=fname_cov_TS_galS_mask, fname_sims=fname_cl_sims_TS_galS_mask)
#cov_TS_galT_mask, corr_TS_galT_mask           = myanalysis_mask.GetCovMatrixFromMaps(field1='TS', nsim=nsim, field2='galT', fname=fname_cov_TS_galT_mask, fname_sims=fname_cl_sims_TS_galT_mask)

print("...done...")

# Some plots
print("...here come the plots...")

# Covariances
fig, axs = plt.subplots(ncols=2, nrows=2,figsize=(38,23))   
fig.suptitle(r'$\ell_{max} = %1.2f $' %lmax + r'$N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

mask_ = np.tri(corr_TS_galS.shape[0],corr_TS_galS.shape[1],0)

#plt.subplot(131)
axs[0,0].set_title(r'Corr $T^S\times gal^S$')
sns.color_palette("crest", as_cmap=True)
sns.heatmap(corr_TS_galS, annot=True, fmt='.2f', cmap  = 'viridis',mask=mask_, ax=axs[0,0])

##plt.subplot(132)
axs[0,1].set_title(r'Corr $T^S\times gal^S$ Masked')
sns.heatmap(corr_TS_galS_mask, annot=True, fmt='.2f',cmap  = 'viridis', mask=mask_, ax=axs[0,1])
#
##plt.subplot(133)
#ax3.set_title(r'Corr $\delta^T \times \kappa^T$ Masked')
#sns.heatmap(corr_kappaT_deltaT_mask, annot=True, fmt='.2f', mask=mask_,ax=ax3)
axs[1,0].set_title(r'Corr $T^T\times gal^T$')
#sns.heatmap(corr_TS_galT, annot=True, fmt='.2f',cmap  = 'viridis', mask=mask_, ax=axs[1,0])

axs[1,1].set_title(r'Corr $T^T\times gal^T$ Masked')
#sns.heatmap(corr_TS_galT_mask, annot=True, fmt='.2f',cmap  = 'viridis', mask=mask_, ax=axs[1,1])


fig.tight_layout()

plt.savefig(cov_dir+f'corr_cl_T_gal_lmax{lmax}_nside{nside}_fsky_{fsky}.png', bbox_inches='tight')

# Cls 

fig = plt.figure(figsize=(25,10))
ax  = fig.add_subplot(1, 2, 1)

# embed()

#for i in range(cl_TS_galT.shape[0]):
#	if i != 0:
#		ax.plot(lbins, cl_TS_galT[i,:], color='grey', alpha=0.05)
#	else:
#		ax.plot(lbins, cl_TS_galT[i,:], color='grey', alpha=0.05, label=r'$\delta^T \times \kappa^T$ Sims')

ax.plot(xcspectra.cltg, color='k', lw=2, label='Theory')
#ax.errorbar(lbins-5, np.mean(cl_TS_galT, axis=0), yerr=np.diag(cov_TS_galT/nsim)**.5, fmt='o', color='seagreen', capsize=0, label=r'$T^S \times gal^T$')
ax.errorbar(lbins+5, np.mean(cl_TS_galS, axis=0), yerr=np.diag(cov_TS_galS/nsim)**.5, fmt='o', color='firebrick', capsize=0, label=r'$T^S \times gal^S$')
ax.errorbar(lbins,   np.mean(cl_TS_galS_mask, axis=0), yerr=np.diag(cov_TS_galS_mask/nsim)**.5, fmt='o', color='darkorange', capsize=0, label=r'$T^S \times gal^S$ Masked')
#ax.errorbar(lbins,   np.mean(cl_TS_galT_mask, axis=0), yerr=np.diag(cov_TS_galT_mask/nsim)**.5, fmt='o', color='grey', capsize=0, label=r'$T^S \times gal^T$ Masked')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\langle C_{\ell}^{Tgal} \rangle$')
ax.legend(loc='best')
plt.savefig(out_dir+f'cl_T_gal_mask_noise_lmax{lmax}_nside{nside}_fsky_{fsky}.png', bbox_inches='tight')

fig  = plt.figure(figsize=(25,10))
ax1  = fig.add_subplot(1, 2, 1)
kgb = myanalysis.bin_spectra(xcspectra.cltg)

ax1.axhline(ls='--', color='k')
ax1.errorbar(lbins-5, (np.mean(cl_TS_galS, axis=0)-kgb)/kgb, yerr=np.sqrt(np.diag(cov_TS_galS))/(np.sqrt(nsim)*kgb), color='firebrick', fmt='o', capsize=0, label=r'$T^S \times gal^S$')
#ax1.errorbar(lbins+5, (np.mean(cl_TS_galT, axis=0)-kgb)/kgb, yerr=np.sqrt(np.diag(cov_TS_galT))/(np.sqrt(nsim)*kgb), color='seagreen', fmt='o', capsize=0, label=r'$T^S \times gal^T$')
ax1.errorbar(lbins, (np.mean(cl_TS_galS_mask, axis=0)-kgb)/kgb, yerr=np.sqrt(np.diag(cov_TS_galS_mask))/(np.sqrt(nsim)*kgb), color='darkorange', fmt='o', capsize=0, label=r'$T^S \times gal^S$ Masked')
#ax1.errorbar(lbins, (np.mean(cl_TS_galT_mask, axis=0)-kgb)/kgb, yerr=np.sqrt(np.diag(cov_TS_galT_mask))/(np.sqrt(nsim)*kgb), color='grey', fmt='o', capsize=0, label=r'$T^S \times gal^T$ Masked')

ax1.set_xlabel(r'$\ell$')
ax1.set_ylabel(r'$(\langle C_{\ell}^{Tgal} \rangle - C_{\ell}^{Tgal,th})/C_{\ell}^{Tgal,th}$')
ax1.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig(out_dir+f'cl_mean_T_gal_mask_noise_lmax{lmax}_nside{nside}_fsky_{fsky}.png', bbox_inches='tight')

#ax = fig.add_subplot(1, 3, 3)

# for i in xrange(cl_sims_deltaT_kappaN_null.shape[0]):
# 	if i != 0:
# 		ax.plot(lbins, cl_sims_deltaT_kappaN_null[i,:], color='grey', alpha=0.05)
# 	else:
# 		ax.plot(lbins, cl_sims_deltaT_kappaN_null[i,:], color='grey', alpha=0.05, label=r'$\delta^T \times \kappa^N$ Sims')
#ax.axhline(ls='--', color='k')
#ax.errorbar(lbins, np.mean(cl_sims_deltaT_kappaN_null, axis=0), yerr=np.diag(cov_cl_deltaT_kappaN_null/nsim)**.5, fmt='o', color='seagreen', capsize=0, label=r'$\delta^T \times \kappa^N$ Null-test')
#ax.set_xlabel(r'$\ell$')
#ax.set_ylabel(r'$\langle C_{\ell}^{\kappa g} \rangle$')
#ax.legend(loc='best')
