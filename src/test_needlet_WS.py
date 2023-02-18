#!/usr/bin/env python
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')

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
print(path)

# Matplotlib defaults ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
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
# plt.rcParams['backend'] = 'WX'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# embed()


# Paths
fname_xcspectra = 'spectra/XCSpectra.dat'
sims_dir        = 'sims/Needlet/sims_256/'
out_dir         = 'output_needlet/'
path_inpainting = 'inpainting/inpainting.py'

# Parameters
simparams = {'nside'   : 256,
             'ngal'    : 5.76e5,
 	     	 'ngal_dim': 'ster',
	     	 'pixwin'  : False}

jmax = 12
lmax = 600
nsim = 500

mask = utils.GetGalMask(simparams['nside'], lat=20.)
fsky = np.mean(mask)

# Loading theory spectra
xcspectra = spectra.XCSpectraFile(fname_xcspectra, b=3, WantTG = False)

# Simulations class
simulations = sims.KGsimulations(xcspectra, sims_dir, simparams)
simulations.Run(nsim, WantTG = False)

# Needlet Analysis
myanalysis = analysis.NeedAnalysis(jmax, lmax, out_dir, simulations)

# Theory Needlet spectra
need_theory = spectra.NeedletTheory(myanalysis.B)
print(myanalysis.B)

# Picking one lensing noise sim for null tests
noisemap = simulations.GetSimField('kappaN', 9)#49)

# Computing simulated Cls 
print("...computing Betajs for simulations...")
fname_betaj_sims_kappaS_deltaS      = 'betaj_sims_kappaS_deltaS.dat'
fname_betaj_sims_kappaT_deltaT      = 'betaj_sims_kappaT_deltaT.dat'
fname_betaj_sims_kappaT_deltaT_mask = 'betaj_sims_kappaT_deltaT_mask.dat'
fname_betaj_sims_deltaT_kappaN_null = 'betaj_sims_deltaT_kappaN_null.dat'
fname_betaj_sims_kappaT_deltaT_mask_inpaint = 'betaj_sims_kappaT_deltaT_mask_inpaint.dat'

betaj_sims_kappaS_deltaS      = myanalysis.GetBetajSimsFromMaps('kappaS', nsim, field2='deltaS', fname=fname_betaj_sims_kappaS_deltaS)
betaj_sims_kappaT_deltaT      = myanalysis.GetBetajSimsFromMaps('kappaT', nsim, field2='deltaT', fname=fname_betaj_sims_kappaT_deltaT)
betaj_sims_kappaT_deltaT_mask = myanalysis.GetBetajSimsFromMaps('kappaT', nsim, field2='deltaT', mask=mask, fname=fname_betaj_sims_kappaT_deltaT_mask)
betaj_sims_deltaT_kappaN_null = myanalysis.GetBetajSimsFromMaps('deltaT', nsim, fix_field=noisemap, fname=fname_betaj_sims_deltaT_kappaN_null)
betaj_sims_kappaT_deltaT_mask_inpaint = myanalysis.GetBetajSimsFromMaps('kappaT', nsim, field2='deltaT', mask=mask, inpainting=True, fname=fname_betaj_sims_kappaT_deltaT_mask_inpaint, path_inpainting=path_inpainting)

print("...done...")

# Covariances
print("...computing Cov Matrices...")
fname_cov_kappaS_deltaS      = 'cov_kappaS_deltaS.dat'
fname_cov_kappaT_deltaT      = 'cov_kappaT_deltaT.dat'
fname_cov_deltaT_kappaN_null = 'cov_deltaT_kappaN_null.dat'
fname_cov_kappaT_deltaT_mask = 'cov_kappaT_deltaT_mask.dat'
fname_cov_kappaT_deltaT_mask_inpaint = 'cov_kappaT_deltaT_mask_inpaint.dat'

cov_kappaS_deltaS, corr_kappaS_deltaS           = myanalysis.GetCovMatrixFromMaps(field1='kappaS', nsim=nsim, field2='deltaS', fname=fname_cov_kappaS_deltaS, fname_sims=fname_betaj_sims_kappaS_deltaS)
#cov_kappaS_deltaS, corr_kappaS_deltaS           = myanalysis.GetCovMatrixFromMaps('kappaS', nsim, field2='deltaS', corr=True, fname=fname_cov_kappaS_deltaS, fname_sims=fname_betaj_sims_kappaS_deltaS)
cov_kappaT_deltaT ,  corr_kappaT_deltaT       = myanalysis.GetCovMatrixFromMaps('kappaT', nsim, field2='deltaT',  fname=fname_cov_kappaT_deltaT, fname_sims=fname_betaj_sims_kappaT_deltaT)
#cov_kappaT_deltaT, corr_kappaT_deltaT           = myanalysis.GetCovMatrixFromMaps('kappaT', nsim, field2='deltaT', corr=True, fname=fname_cov_kappaT_deltaT, fname_sims=fname_betaj_sims_kappaT_deltaT)
cov_deltaT_kappaN_null , corr_deltaT_kappaN_null = myanalysis.GetCovMatrixFromMaps('deltaT', nsim, fix_field=noisemap,  fname=fname_cov_deltaT_kappaN_null, fname_sims=fname_betaj_sims_deltaT_kappaN_null)
#cov_deltaT_kappaN_null, corr_deltaT_kappaN_null = myanalysis.GetCovMatrixFromMaps('deltaT', nsim, fix_field=noisemap, corr=True, fname=fname_cov_deltaT_kappaN_null, fname_sims=fname_betaj_sims_deltaT_kappaN_null)
cov_kappaT_deltaT_mask , corr_kappaT_deltaT_mask = myanalysis.GetCovMatrixFromMaps('kappaT', nsim, field2='deltaT', mask=mask,  fname=fname_cov_kappaT_deltaT_mask, fname_sims=fname_betaj_sims_kappaT_deltaT_mask)
#cov_kappaT_deltaT_mask, corr_kappaT_deltaT_mask = myanalysis.GetCovMatrixFromMaps('kappaT', nsim, field2='deltaT', mask=mask, corr=True, fname=fname_cov_kappaT_deltaT_mask, fname_sims=fname_betaj_sims_kappaT_deltaT_mask)
cov_kappaT_deltaT_mask_inpaint, corr_kappaT_deltaT_mask_inpaint = myanalysis.GetCovMatrixFromMaps('kappaT', nsim, field2='deltaT', fname=fname_cov_kappaT_deltaT_mask_inpaint, fname_sims=fname_betaj_sims_kappaT_deltaT_mask_inpaint)

print("...done...")

# <Beta_j>_MC
betaj_kappaS_deltaS_mean      = myanalysis.GetBetajMeanFromMaps('kappaS', nsim, field2='deltaS', fname_sims=fname_betaj_sims_kappaS_deltaS)
betaj_kappaT_deltaT_mean      = myanalysis.GetBetajMeanFromMaps('kappaT', nsim, field2='deltaT', fname_sims=fname_betaj_sims_kappaT_deltaT)
betaj_kappaT_deltaT_mask_mean = myanalysis.GetBetajMeanFromMaps('kappaT', nsim, field2='deltaT', fname_sims=fname_betaj_sims_kappaT_deltaT_mask)
betaj_deltaT_kappaN_null_mean = myanalysis.GetBetajMeanFromMaps('deltaT', nsim, fix_field='noisemap', fname_sims=fname_betaj_sims_deltaT_kappaN_null)
betaj_kappaT_deltaT_mask_mean_inpaint = myanalysis.GetBetajMeanFromMaps('dsf', nsim, fname_sims=fname_betaj_sims_kappaT_deltaT_mask_inpaint)


#sys.exit()


# Some plots
print("...here come the plots...")

# Covariances
fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(25,18))   
fig.suptitle(r'$B = %1.2f $' %myanalysis.B + r'$N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

mask_ = np.tri(corr_kappaS_deltaS.shape[0],corr_kappaS_deltaS.shape[1],0)

#plt.subplot(131)
ax1.set_title(r'Corr $\delta^S \times \kappa^S$')
sns.heatmap(corr_kappaS_deltaS, annot=True, fmt='.2f', mask=mask_, ax=ax1)

#plt.subplot(132)
ax2.set_title(r'Corr $\delta^T \times \kappa^T$')
sns.heatmap(corr_kappaT_deltaT, annot=True, fmt='.2f', mask=mask_, ax=ax2)

#plt.subplot(133)
ax3.set_title(r'Corr $\delta^T \times \kappa^T$ Masked')
sns.heatmap(corr_kappaT_deltaT_mask, annot=True, fmt='.2f', mask=mask_,ax=ax3)

plt.savefig('corr_nside256.png')

#plt.show()

#sys.exit()


fig = plt.figure(figsize=(17,10))
#ax  = fig.add_subplot(1, 2, 1)

# Theory + Normalization Needlet power spectra
betakg    = need_theory.cl2betaj(jmax=jmax, cl=xcspectra.clkg_tot)
beta_norm = need_theory.cl2betaj(jmax=jmax, cl=np.ones(xcspectra.clkg_tot.size))

plt.suptitle(r'$B = %1.2f $' %myanalysis.B + r'$  N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

# ax.axhline(ls='--', color='k')
#ax.plot(myanalysis.jvec[1:], betakg[1:]/beta_norm[1:], color='royalblue', label='Theory')
#ax.errorbar(myanalysis.jvec-0.1, betaj_kappaS_deltaS_mean/beta_norm, yerr=np.diag(cov_kappaS_deltaS/nsim)**.5/beta_norm, color='gold', fmt='o', capsize=0, label=r'$\delta^S \times \kappa^S$')
#ax.errorbar(myanalysis.jvec, betaj_kappaT_deltaT_mask_mean/beta_norm/fsky, yerr=np.diag(cov_kappaT_deltaT_mask/nsim)**.5/beta_norm/fsky, color='firebrick', fmt='o', capsize=0, label=r'$\delta^T \times \kappa^T$ Masked')
#ax.errorbar(myanalysis.jvec+0.1, betaj_kappaT_deltaT_mean/beta_norm, yerr=np.diag(cov_kappaT_deltaT/nsim)**.5/beta_norm, fmt='o', color='seagreen', capsize=0, label=r'$\delta^T \times \kappa^T$')

#ax.set_xlabel(r'$j$')
#ax.set_ylabel(r'$\langle\beta_j^{\kappa g}\rangle$')
#ax.set_yscale('log')
#ax.legend(loc='best')
#ax.set_ylim([1e-9,1e-3])
#ax.set_xlim([-0.2,18])
#
#ax = fig.add_subplot(1, 2, 1)
#
#ax.axhline(ls='--', color='k')
#ax.plot(myanalysis.jvec[1:], betakg[1:]/beta_norm[1:], color='royalblue', label='Theory')
#ax.errorbar(myanalysis.jvec-0.15, betaj_kappaS_deltaS_mean/beta_norm, yerr=np.diag(cov_kappaS_deltaS/nsim)**.5/beta_norm, color='gold', fmt='o', capsize=0, label=r'$\delta^S \times \kappa^S$')
#ax.errorbar(myanalysis.jvec-0.05, betaj_kappaT_deltaT_mask_mean/beta_norm, yerr=np.diag(cov_kappaT_deltaT_mask/nsim)**.5/beta_norm, color='firebrick', fmt='o', capsize=0, label=r'$\delta^T \times \kappa^T$ Masked')
#ax.errorbar(myanalysis.jvec+0.05, betaj_kappaT_deltaT_mean/beta_norm, yerr=np.diag(cov_kappaT_deltaT/nsim)**.5/beta_norm, fmt='o',color='seagreen', capsize=0, label=r'$\delta^T \times \kappa^T$')
#ax.errorbar(myanalysis.jvec+0.15, betaj_kappaT_deltaT_mask_mean_inpaint/beta_norm, yerr=np.diag(cov_kappaT_deltaT_mask_inpaint/nsim)**.5/beta_norm, fmt='o',color='darkorange', capsize=0, label=r'$\delta^T \times \kappa^T$ Masked Inpainted')
#ax.legend(loc='best')
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#ax.set_xlabel(r'$j$')
#ax.set_ylabel(r'$\langle\beta_j^{\kappa g}\rangle$')
#ax.set_xlim([-0.2,18])

ax = fig.add_subplot(1, 1, 1)
#ax = fig.add_subplot(1, 2, 2)

ax.axhline(ls='--', color='k')
ax.errorbar(myanalysis.jvec-0.15, (betaj_kappaS_deltaS_mean-betakg)/betakg, yerr=np.sqrt(np.diag(cov_kappaS_deltaS))/(np.sqrt(nsim)*betakg), color='gold', fmt='o', capsize=0, label=r'$\delta^S \times \kappa^S$')
ax.errorbar(myanalysis.jvec-0.05, (betaj_kappaT_deltaT_mean-betakg)/betakg, yerr=np.sqrt(np.diag(cov_kappaT_deltaT))/(np.sqrt(nsim)*betakg), color='seagreen', fmt='o', capsize=0,label=r'$\delta^T \times \kappa^T$')
#ax.errorbar(myanalysis.jvec+0.05, (betaj_kappaT_deltaT_mask_mean-betakg)/betakg, yerr=np.sqrt(np.diag(cov_kappaT_deltaT_mask))/(np.sqrt(nsim)*betakg), color='firebrick', fmt='o', capsize=0, label=r'$\delta^T \times \kappa^T$ Masked')
#ax.errorbar(myanalysis.jvec+0.15, (betaj_kappaT_deltaT_mask_mean_inpaint-betakg)/betakg, yerr=np.sqrt(np.diag(cov_kappaT_deltaT_mask_inpaint))/(np.sqrt(nsim)*betakg), color='darkorange', fmt='o', capsize=0, label=r'$\delta^T \times \kappa^T$ Masked Inpainted')
ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$(\langle \beta_j^{\kappa g} \rangle - \beta_j^{\kappa g, th})/\beta_j^{\kappa g, th}$')
ax.set_ylim([-0.2,0.3])

plt.savefig('betaj_mean_delta_kappa_jmax12_nside256.png', bbox_inches='tight')

#plt.show()

# Null-test
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(1, 1, 1)

plt.suptitle(r'Null-tests $B = %1.2f $' %myanalysis.B + r'$  N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

ax.axhline(ls='--', color='k')
ax.errorbar(myanalysis.jvec, betaj_deltaT_kappaN_null_mean, yerr=np.diag(cov_deltaT_kappaN_null/nsim)**.5, color='royalblue', fmt='o', capsize=0, label=r'$\delta^T \times \kappa^N$')
#ax.errorbar(myanalysis.jvec+0.1, betaj_deltaT_null_mask, yerr=np.diag(cov_deltaT_kappaN/nsim)**.5, color='firebrick', fmt='o', capsize=0, label=r'$\delta^T \times \kappa^N$ Masked')
ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$(\langle \beta_j^{\kappa g} \rangle - \beta_j^{\kappa g, th})/\beta_j^{\kappa g, th}$')

plt.savefig('null_test.png')

#plt.show()

