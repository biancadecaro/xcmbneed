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
simparams = {'nside'   : 256 ,
             'ngal'    : 5.76e5, #dovrebbe importare solo per lo shot noise (noise poissoniano)
 	     	 'ngal_dim': 'ster',
	     	 'pixwin'  : False}
nside=simparams['nside']
lmax = 700
nsim = 10
delta_ell = 40


# Paths
fname_xcspectra = '/ehome/bdecaro/xcmbneed/src/spectra/inifiles/CAMBSpectra_planck_fiducial_lmin0_2050.dat'#'/ehome/bdecaro/xcmbneed/src/spectra/inifiles/CAMBSpectra_OmL_fiducial_lmin0.dat' 
sims_dir        = f'sims/Cl/Planck/Mask_noise/TGsims_{nside}_mask/'#'sims/Needlet/Planck/TGsims_'+str(simparams['nside'])+'_planck_2_lmin0/'
out_dir         = f'output_Cl_TG/Planck/Mask_noise/TG_{nside}_lmax{lmax}_mask/'
path_inpainting = 'inpainting/inpainting.py'
cov_dir 	= f'covariance/Cl/Planck/covariance_TG_{nside}_lmax{lmax}_mask/'
if not os.path.exists(cov_dir):
        os.makedirs(cov_dir)



mask = hp.read_map(f'mask/mask70_gal_nside={nside}.fits', verbose=False)#utils.GetGalMask(simparams['nside'], lat=20.)
fsky = np.mean(mask)

# Loading theory spectra
xcspectra = spectra.XCSpectraFile(fname_xcspectra, nltt = None,WantTG = True)

# Simulations class
simulations = sims.KGsimulations(xcspectra, sims_dir, simparams, WantTG = True)
simulations.Run(nsim, WantTG = True)

# Cl Analysis

myanalysis = analysis.HarmAnalysis(lmax, out_dir, simulations, lmin=2, delta_ell=delta_ell,mask=mask, pixwin=simparams['pixwin'], fsky_approx=True)
lbins = myanalysis.ell_binned


# Computing simulated Cls 
print("...computing Betajs for simulations...")
fname_cl_sims_TS_galS      = f'cl_sims_TS_galS_lmax{lmax}_nside{nside}.dat'
fname_cl_sims_TS_TS      = f'cl_sims_TS_TS_lmax{lmax}_nside{nside}.dat'
fname_cl_sims_galS_galS      = f'cl_sims_galS_galS_lmax{lmax}_nside{nside}.dat'

cl_TS_galS = myanalysis.GetClSimsFromMaps(field1='TS', nsim=nsim, field2='galS', fname=fname_cl_sims_TS_galS)
#cl_TS_TS = myanalysis.GetClSimsFromMaps('TS', nsim, field2='TS', fname=fname_cl_sims_TS_TS)
#cl_galS_galS = myanalysis.GetClSimsFromMaps('galS', nsim, field2='galS', fname=fname_cl_sims_galS_galS)

# Covariances
print("...computing Cov Matrices...")
fname_cov_TS_galS      = 'cov_TS_galS_lmax'+str(lmax)+'_nside'+str(simparams['nside'])+'.dat'

cov_TS_galS, corr_TS_galS           = myanalysis.GetCovMatrixFromMaps(field1='TS', nsim=nsim, field2='galS', fname=fname_cov_TS_galS, fname_sims=fname_cl_sims_TS_galS)

print("...done...")

# Some plots
print("...here come the plots...")

# Covariances
fig, ax1 = plt.subplots(1,1,figsize=(25,18))   

mask_ = np.tri(corr_TS_galS.shape[0],corr_TS_galS.shape[1],0)

#plt.subplot(131)
ax1.set_title(r'Corr $T^S\times gal^S$')
sns.color_palette("crest", as_cmap=True)
sns.heatmap(corr_TS_galS, annot=True, fmt='.2f', cmap  = 'viridis',mask=mask_, ax=ax1)

fig.tight_layout()
plt.savefig(cov_dir+f'corr_TS_galS_lmax{lmax}_nsim{nsim}_nside{nside}.png')


# Cls 
lbins = myanalysis.ell_binned
print(lbins.shape)

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
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\langle C_{\ell}^{Tgal} \rangle$')
ax.legend(loc='best')
plt.savefig(out_dir+f'cl_T_gal_lmax{lmax}_nside{nside}_fsky_{fsky}.png', bbox_inches='tight')


fig = plt.figure(figsize=(10,8))

plt.suptitle(r'$\ell_{max} =$'+str(lmax) + r'$~N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

#for i in range(cl_TS_galS.shape[0]):
#	if i != 0:
#		ax.plot(lbins, cl_TS_galS[i,:], color='grey', alpha=0.05)
#	else:
#		ax.plot(lbins, cl_TS_galS[i,:], color='grey', alpha=0.05, label=r'$T^S \times gal^S$ Sims')

#print(np.sqrt(np.diag(cov_TS_galS))/(np.sqrt(nsim)*betatg) - delta/(np.sqrt(nsim)*betatg))
#ax.plot(xcspectra.cltg, color='k', lw=2, label='Theory')
kgb = myanalysis.bin_spectra(xcspectra.cltg)


ax.axhline(ls='--', color='k')
ax.errorbar(lbins, (np.mean(cl_TS_galS, axis=0)-kgb)/kgb, yerr=np.sqrt(np.diag(cov_TS_galS))/(np.sqrt(nsim)*kgb), color='firebrick', fmt='o', capsize=0, label=r'$T^S \times gal^S$')
ax.set_xlabel(r'$\ell$')
ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_ylabel(r'$(\langle C_{\ell}^{Tgal} \rangle - C_{\ell}^{Tgal, th})/C_{\ell}^{Tgal, th}$')
#ax.set_ylim([-0.2,0.3])

fig.tight_layout()
plt.savefig(out_dir+f'cl_T_gal_mean_lmax{lmax}_nsim{nsim}_nside{nside}.png', bbox_inches='tight')

#plt.show()
print(np.mean(cl_TS_galS, axis=0)/kgb - 1)

#pixwin = []#np.zeros(jmax+1)
#for j in range(jmax+1):
#	print(j)
#	l_min = np.floor(myanalysis.B**(j-1))
#	l_max = np.floor(myanalysis.B**(j+1))
#	print(l_max)
#	ell = np.arange(l_min,l_max+1,dtype=int)
#	pixwin.append(np.sum(hp.sphtfunc.pixwin(simparams['nside'], lmax = int(l_max))))
#pixwin = np.asarray(pixwin)
#print(pixwin)
#print(betaj_TS_galS_mean)

