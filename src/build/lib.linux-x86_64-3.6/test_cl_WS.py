import numpy as np
import matplotlib
#matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import healpy as hp
import argparse, os, sys, warnings, glob
#import cython_mylibc as mylibc
import analysis, utils, spectra, sims
from IPython import embed

import seaborn as sns

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
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Paths
fname_xcspectra = 'spectra/XCSpectra.dat'
sims_dir        = 'sims/Needlet/sims_256/'
out_dir         = 'output_MYSIMS_parallel_newSIMS/'

# Parameters
simparams = {'nside'   : 256,
             'ngal'    : 5.76e5,
 	     	 'ngal_dim': 'ster',
	     	 'pixwin'  : False}

jmax = 12
lmax = 600
nsim = 10
delta_ell = 50

mask = utils.GetGalMask(simparams['nside'], lat=20.)
fsky = np.mean(mask)

# Loading theory spectra
xcspectra = spectra.XCSpectraFile(fname_xcspectra, b=1)

# Simulations class
simulations = sims.KGsimulations(xcspectra, sims_dir, simparams)
simulations.Run(nsim)

# Setting up the Harmonic Analysis classes
myanalysis_nomask = analysis.HarmAnalysis(lmax, out_dir, simulations, lmin=2, delta_ell=delta_ell, pixwin=simparams['pixwin'], fsky_approx=True)
myanalysis_mask   = analysis.HarmAnalysis(lmax, out_dir, simulations, lmin=2, delta_ell=delta_ell, pixwin=simparams['pixwin'], fsky_approx=False, mask=mask)

lbins = myanalysis_nomask.ell_binned

# Picking one lensing noise sim for null tests
noisemap = simulations.GetSimField('kappaN', 9) #simulations.GetSimField('kappaN', 50)

# Computing simulated Cls 
print("...computing Cls for simulations...")
fname_cl_sims_kappaS_deltaS      = 'cl_sims_kappaS_deltaS.dat'
fname_cl_sims_kappaT_deltaT      = 'cl_sims_kappaT_deltaT.dat'
fname_cl_sims_kappaT_deltaT_mask = 'cl_sims_kappaT_deltaT_mask.dat'
fname_cl_sims_deltaT_kappaN_null = 'cl_sims_deltaT_kappaN_null.dat'

cl_sims_kappaS_deltaS      = myanalysis_nomask.GetClSimsFromMaps('kappaS', nsim, field2='deltaS', fname=fname_cl_sims_kappaS_deltaS)
cl_sims_kappaT_deltaT      = myanalysis_nomask.GetClSimsFromMaps('kappaT', nsim, field2='deltaT', fname=fname_cl_sims_kappaT_deltaT)
cl_sims_kappaT_deltaT_mask = myanalysis_mask.GetClSimsFromMaps('kappaT', nsim, field2='deltaT', fname=fname_cl_sims_kappaT_deltaT_mask)
cl_sims_deltaT_kappaN_null = myanalysis_nomask.GetClSimsFromMaps('deltaT', nsim, fix_field=noisemap, fname=fname_cl_sims_deltaT_kappaN_null)

print("...done...")


# Cov matrices
print("...computing Cov Matrices...")
fname_cov_kappaS_deltaS      = 'cov_cl_kappaS_deltaS.dat'
fname_cov_kappaT_deltaT      = 'cov_cl_kappaT_deltaT.dat'
fname_cov_kappaT_deltaT_mask = 'cov_cl_kappaT_deltaT_mask.dat'
fname_cov_deltaT_kappaN_null = 'cov_cl_deltaT_kappaN_null.dat'


#cov_cl_kappaT_deltaT          = myanalysis_nomask.GetCovMatrixFromMaps(field1='kappaT', nsim=nsim, field2='deltaT', fname=fname_cov_kappaT_deltaT, fname_sims=fname_cl_sims_kappaT_deltaT)#, corr=True)
cov_cl_kappaT_deltaT, corr_cl_kappaT_deltaT           = myanalysis_nomask.GetCovMatrixFromMaps(field1='kappaT', nsim=nsim, field2='deltaT', fname=fname_cov_kappaT_deltaT, fname_sims=fname_cl_sims_kappaT_deltaT)#, corr=True)
#cov_cl_kappaS_deltaS           = myanalysis_nomask.GetCovMatrixFromMaps(field1='kappaS', nsim=nsim, field2='deltaS', fname=fname_cov_kappaS_deltaS, fname_sims=fname_cl_sims_kappaS_deltaS)#, corr=True)
cov_cl_kappaS_deltaS, corr_cl_kappaS_deltaS           = myanalysis_nomask.GetCovMatrixFromMaps(field1='kappaS', nsim=nsim, field2='deltaS', fname=fname_cov_kappaS_deltaS, fname_sims=fname_cl_sims_kappaS_deltaS)#, corr=True)
#cov_cl_kappaT_deltaT_mask = myanalysis_mask.GetCovMatrixFromMaps(field1='kappaT', nsim=nsim, field2='deltaT', fname=fname_cov_kappaT_deltaT_mask, fname_sims=fname_cl_sims_kappaT_deltaT_mask)#, corr=True)
cov_cl_kappaT_deltaT_mask, corr_cl_kappaT_deltaT_mask = myanalysis_mask.GetCovMatrixFromMaps(field1='kappaT', nsim=nsim, field2='deltaT', fname=fname_cov_kappaT_deltaT_mask, fname_sims=fname_cl_sims_kappaT_deltaT_mask)#, corr=True)
#cov_cl_deltaT_kappaN_null = myanalysis_nomask.GetCovMatrixFromMaps(field1='kappaT', nsim=nsim, fix_field=noisemap, fname=fname_cov_deltaT_kappaN_null, fname_sims=fname_cl_sims_deltaT_kappaN_null)#, corr=True)
cov_cl_deltaT_kappaN_null, corr_cl_deltaT_kappaN_null = myanalysis_nomask.GetCovMatrixFromMaps(field1='kappaT', nsim=nsim, fix_field=noisemap, fname=fname_cov_deltaT_kappaN_null, fname_sims=fname_cl_sims_deltaT_kappaN_null)#, corr=True)
print("...done...")

# sys.exit()

# !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# !~~~~~~~~~~~~ PLOTS ~~~~~~~~~~~~~~~~~~~~~~~~
# !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from matplotlib import cm
cool_cmap = cm.RdGy
#cool_cmap = cm.RdGyt
cool_cmap.set_under("w") # sets background to white

# Covariances
plt.suptitle(r'$N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim) + r'$\Delta\ell = $'+str(delta_ell))

mask_ = np.tri(corr_cl_kappaS_deltaS.shape[0],corr_cl_kappaS_deltaS.shape[1],0)

plt.subplot(131)
plt.title(r'Corr $\delta^S \times \kappa^S$')
sns.heatmap(corr_cl_kappaS_deltaS, annot=True, fmt='.2f', mask=mask_)

plt.subplot(132)
plt.title(r'Corr $\delta^T \times \kappa^T$')
sns.heatmap(corr_cl_kappaT_deltaT, annot=True, fmt='.2f', mask=mask_)

plt.subplot(133)
plt.title(r'Corr $\delta^T \times \kappa^T$ Masked')
sns.heatmap(corr_cl_kappaT_deltaT_mask, annot=True, fmt='.2f', mask=mask_)

plt.show()

# Cls 
lbins = myanalysis_nomask.ell_binned

fig = plt.figure(figsize=(10,8))
ax  = fig.add_subplot(1, 2, 1)

# embed()

for i in range(cl_sims_kappaT_deltaT.shape[0]):
	if i != 0:
		ax.plot(lbins, cl_sims_kappaT_deltaT[i,:], color='grey', alpha=0.05)
	else:
		ax.plot(lbins, cl_sims_kappaT_deltaT[i,:], color='grey', alpha=0.05, label=r'$\delta^T \times \kappa^T$ Sims')

ax.plot(xcspectra.clkg_tot, color='k', lw=2, label='Theory')
ax.errorbar(lbins-5, np.mean(cl_sims_kappaT_deltaT, axis=0), yerr=np.diag(cov_cl_kappaT_deltaT/nsim)**.5, fmt='o', color='seagreen', capsize=0, label=r'$\delta^T \times \kappa^T$')
ax.errorbar(lbins+5, np.mean(cl_sims_kappaS_deltaS, axis=0), yerr=np.diag(cov_cl_kappaS_deltaS/nsim)**.5, fmt='o', color='firebrick', capsize=0, label=r'$\delta^S \times \kappa^S$')
ax.errorbar(lbins,   np.mean(cl_sims_kappaT_deltaT_mask, axis=0), yerr=np.diag(cov_cl_kappaT_deltaT_mask/nsim)**.5, fmt='o', color='darkorange', capsize=0, label=r'$\delta^T \times \kappa^T$ Masked')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\langle C_{\ell}^{\kappa g} \rangle$')
ax.legend(loc='best')

ax  = fig.add_subplot(1, 3, 2)
kgb = myanalysis_nomask.bin_spectra(xcspectra.clkg_tot)

ax.axhline(ls='--', color='k')
ax.errorbar(lbins-5, (np.mean(cl_sims_kappaS_deltaS, axis=0)-kgb)/kgb, yerr=np.sqrt(np.diag(cov_cl_kappaS_deltaS))/(np.sqrt(nsim)*kgb), color='firebrick', fmt='o', capsize=0, label=r'$\delta^S \times \kappa^S$')
ax.errorbar(lbins+5, (np.mean(cl_sims_kappaT_deltaT, axis=0)-kgb)/kgb, yerr=np.sqrt(np.diag(cov_cl_kappaT_deltaT))/(np.sqrt(nsim)*kgb), color='seagreen', fmt='o', capsize=0, label=r'$\delta^T \times \kappa^T$')
ax.errorbar(lbins, (np.mean(cl_sims_kappaT_deltaT_mask, axis=0)-kgb)/kgb, yerr=np.sqrt(np.diag(cov_cl_kappaT_deltaT_mask))/(np.sqrt(nsim)*kgb), color='darkorange', fmt='o', capsize=0, label=r'$\delta^T \times \kappa^T$ Masked')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$(\langle C_{\ell}^{\kappa g} \rangle - C_{\ell}^{\kappa g,th})/C_{\ell}^{\kappa g,th}$')
ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

ax = fig.add_subplot(1, 3, 3)

# for i in xrange(cl_sims_deltaT_kappaN_null.shape[0]):
# 	if i != 0:
# 		ax.plot(lbins, cl_sims_deltaT_kappaN_null[i,:], color='grey', alpha=0.05)
# 	else:
# 		ax.plot(lbins, cl_sims_deltaT_kappaN_null[i,:], color='grey', alpha=0.05, label=r'$\delta^T \times \kappa^N$ Sims')
ax.axhline(ls='--', color='k')
#ax.errorbar(lbins, np.mean(cl_sims_deltaT_kappaN_null, axis=0), yerr=np.diag(cov_cl_deltaT_kappaN_null/nsim)**.5, fmt='o', color='seagreen', capsize=0, label=r'$\delta^T \times \kappa^N$ Null-test')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\langle C_{\ell}^{\kappa g} \rangle$')
ax.legend(loc='best')

plt.show()

# 
# plt.suptitle(r'$B = %1.2f$' %myanalysis.B + r'$  N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

# plt.subplot(221)

# plt.plot(myanalysis.jvec-0.1, betaj_deltaS_mean, '-x',label=r'$\delta^S \times \delta^S$')
# # plt.plot(myanalysis.jvec,     betaj_deltaT_mean, label=r'$\delta^T \times \delta^T$')
# plt.plot(myanalysis.jvec+0.1, betaj_deltaS_mean_mask, '-o', label=r'$\delta^S \times \delta^S$ w/ mask')
# # plt.xlabel(r'$j$')
# plt.yscale('log')
# plt.ylabel(r'$<\beta_j>_{MC}$')
# plt.legend()
# # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# plt.xlim([0.9,16.2])

# plt.subplot(222)

# plt.plot(myanalysis.jvec-0.1, betaj_kappaS_mean, '-x', label=r'$\kappa^S \times \kappa^S$')
# # plt.plot(myanalysis.jvec,     betaj_kappaT_mean, label=r'$\kappa^T \times \kappa^T$')
# plt.plot(myanalysis.jvec+0.1, betaj_kappaS_mean_mask, '-o', label=r'$\kappa^S \times \kappa^S$ w/ mask')
# # plt.xlabel(r'$j$')
# # plt.ylabel(r'$\beta_j$')
# plt.yscale('log')
# plt.legend()
# # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# plt.xlim([0.9,16.2])

# plt.subplot(223)

# plt.plot(myanalysis.jvec-0.05, betaj_deltaT_null, '-x', label=r'$\delta^T \times \kappa^N$ (Null-test)')
# plt.plot(myanalysis.jvec+0.05, betaj_deltaT_null_mask, '-o', label=r'$\delta^T \times \kappa^N$ w/ mask  (Null-test)')
# plt.xlabel(r'$j$')
# plt.ylabel(r'$<\beta_j>_{MC}$')
# plt.legend()
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# plt.xlim([0.9,16.2])
# plt.axhline(ls='--', color='k')

# plt.subplot(224)

# plt.plot(myanalysis.jvec-0.05, betaj_kappaS_deltaS_mean, '-x', label=r'$\delta^S \times \kappa^S$')
# plt.plot(myanalysis.jvec+0.05, betaj_kappaS_deltaS_mean_mask, '-o', label=r'$\kappa^S \times \delta^S$ w/ mask')
# plt.xlabel(r'$j$')
# # plt.ylabel(r'$\beta_j$')
# plt.yscale('log')
# plt.legend()
# # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# plt.xlim([0.9,16.2])

# plt.show()


