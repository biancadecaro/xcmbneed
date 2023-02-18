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
#plt.rcParams['axes.linewidth']  = 3.
#plt.rcParams['axes.labelsize']  = 16
#plt.rcParams['xtick.labelsize'] = 12
#plt.rcParams['ytick.labelsize'] = 12
#plt.rcParams['xtick.major.size'] = 7
#plt.rcParams['ytick.major.size'] = 7
#plt.rcParams['xtick.minor.size'] = 3
#plt.rcParams['ytick.minor.size'] = 3
#plt.rcParams['legend.fontsize']  = 15
#plt.rcParams['legend.frameon']  = False
#
#plt.rcParams['xtick.major.width'] = 1
#plt.rcParams['ytick.major.width'] = 1
#plt.rcParams['xtick.minor.width'] = 1
#plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['font.size'] = '25'
#plt.rcParams['backend'] = 'WX'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# embed()

# Parameters
simparams = {'nside'   : 256,
             'ngal'    : 5.76e5, #dovrebbe importare solo per lo shot noise (noise poissoniano)
 	     	 'ngal_dim': 'ster',
	     	 'pixwin'  : False}

# Paths
fname_xcspectra = 'spectra/CAMBSpectra.dat'
sims_dir        = 'sims/Needlet/TGsims_'+str(simparams['nside'])+'_mask/'
out_dir         = 'output_needlet_TG/TG_'+str(simparams['nside'])+'_mask/'
path_inpainting = 'inpainting/inpainting.py'
cov_dir 		= 'covariance/covariance_TG'+str(simparams['nside'])+'_mask/'


jmax = 12
lmax = 782
nsim = 250
#B = 1.47

#jmax = round(np.log(lmax*1.00000000000000222045e+00)/np.log(B))
#lmax = round(B**(jmax+1))
#mylibc.debug_needlets()

mask = utils.GetGalMask(simparams['nside'], lat=20.)
fsky = np.mean(mask)

# Loading theory spectra
xcspectra = spectra.XCSpectraFile(fname_xcspectra, WantTG = True)

# Simulations class
simulations = sims.KGsimulations(xcspectra, sims_dir, simparams, WantTG = True)
simulations.Run(nsim, WantTG = True)

# Needlet Analysis
myanalysis = analysis.NeedAnalysis(jmax, lmax, out_dir, simulations)

# Theory Needlet spectra
need_theory = spectra.NeedletTheory(myanalysis.B)
print(myanalysis.B)

# Computing simulated Cls 
print("...computing Betajs for simulations...")
fname_betaj_sims_TS_galS      = f'betaj_sims_TS_galS_jmax{jmax}_B_{myanalysis.B}_nside'+str(simparams['nside'])+'.dat'
fname_betaj_sims_TS_galS_mask      = f'betaj_sims_TS_galS_jmax{jmax}_B_{myanalysis.B}_nside'+str(simparams['nside'])+f'_fsky_{fsky}.dat'

#fname_betaj_sims_galS_galS      = 'betaj_sims_galS_galS.dat'

betaj_sims_TS_galS = myanalysis.GetBetajSimsFromMaps('TS', nsim, field2='galS', fname=fname_betaj_sims_TS_galS)
betaj_sims_TS_galS_mask = myanalysis.GetBetajSimsFromMaps('TS', nsim, field2='galS', mask= mask, fname=fname_betaj_sims_TS_galS_mask)

#betaj_sims_G_G = myanalysis.GetBetajSimsFromMaps('galS', nsim, field2='galS', fname=fname_betaj_sims_G_G)

# Covariances
print("...computing Cov Matrices...")
fname_cov_TS_galS      = f'cov_TS_galS_jmax{jmax}_B = %1.2f $' %myanalysis.B +'_nside'+str(simparams['nside'])+'.dat'
fname_cov_TS_galS_mask      = f'cov_TS_galS_jmax{jmax}_B = %1.2f $' %myanalysis.B +'_nside'+str(simparams['nside'])+'_mask.dat'
#fname_cov_galS_galS      = 'cov_galS_galS.dat'

#step = 100
#for n in range(100,nsim,step):
#	fname_corr_n = 'corr_'+str(n)+'_jmax'+str(jmax)+'_B = %1.2f ' %myanalysis.B +'.dat'
#	fname_cov_n = 'cov_'+str(n)+'_jmax'+str(jmax)+'_B = %1.2f ' %myanalysis.B +'.dat'
#	fname_sims_n = 'betaj_sims_TS_galS_'+str(n)+'_jmax'+str(jmax)+'_B = %1.2f ' %myanalysis.B +'.dat'
#	betaj_sims_n = myanalysis.GetBetajSimsFromMaps(field1='TS', nsim=n, field2='galS', fname=fname_sims_n)
#	cov_betaj_n = np.cov(betaj_sims_n.T)
#	corr_betaj_n = np.corrcoef(cov_betaj_n)
#	np.savetxt(cov_dir+fname_cov_n, cov_betaj_n, header='cov_'+str(n))
#	np.savetxt(cov_dir+fname_corr_n, corr_betaj_n, header='corr_'+str(n))

cov_TS_galS, corr_TS_galS          = myanalysis.GetCovMatrixFromMaps(field1='TS', nsim=nsim, field2='galS', fname=fname_cov_TS_galS, fname_sims=fname_betaj_sims_TS_galS)
cov_TS_galS_mask, corr_TS_galS_mask          = myanalysis.GetCovMatrixFromMaps(field1='TS', nsim=nsim, field2='galS', fname=fname_cov_TS_galS_mask, mask = mask,fname_sims=fname_betaj_sims_TS_galS_mask)
#cov_galS_galS, corr_galS_galS           = myanalysis.GetCovMatrixFromMaps(field1='galS', nsim=nsim, field2='galS', fname=fname_cov_galS_galS, fname_sims=fname_betaj_sims_galS_galS)

print("...done...")

# <Beta_j>_MC
betaj_TS_galS_mean    = myanalysis.GetBetajMeanFromMaps('TS', nsim, field2='galS', fname_sims=fname_betaj_sims_TS_galS)
betaj_TS_galS_mask_mean     = myanalysis.GetBetajMeanFromMaps('TS', nsim, field2='galS', fname_sims=fname_betaj_sims_TS_galS_mask)
#betaj_galS_galS_mean      = myanalysis.GetBetajMeanFromMaps('kappaT', nsim, field2='deltaT', fname_sims=fname_betaj_sims_kappaT_deltaT)


# Some plots
print("...here come the plots...")

# Covariances
#fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(25,18))
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(38,23))   
fig.suptitle(r'$B = %1.2f $' %myanalysis.B + r'$N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

mask_ = np.tri(corr_TS_galS.shape[0],corr_TS_galS.shape[1],0)

#plt.subplot(131)
ax1.set_title(r'Corr $T^S\times gal^S$')
sns.color_palette("crest", as_cmap=True)
sns.heatmap(corr_TS_galS, annot=True, fmt='.2f', cmap  = 'viridis',mask=mask_, ax=ax1)

##plt.subplot(132)
ax2.set_title(r'Corr $T^S\times gal^S$ Masked')
sns.heatmap(corr_TS_galS_mask, annot=True, fmt='.2f',cmap  = 'viridis', mask=mask_, ax=ax2)
#
##plt.subplot(133)
#ax3.set_title(r'Corr $\delta^T \times \kappa^T$ Masked')
#sns.heatmap(corr_kappaT_deltaT_mask, annot=True, fmt='.2f', mask=mask_,ax=ax3)
fig.tight_layout()
plt.savefig(cov_dir+'corr_TS_galS_jmax'+str(jmax)+'_B = %1.2f ' %myanalysis.B +'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'_mask.png')


# Theory + Normalization Needlet power spectra

betatg    = need_theory.cl2betaj(jmax=jmax, cl=xcspectra.cltg)
beta_norm = need_theory.cl2betaj(jmax=jmax, cl=np.ones(xcspectra.cltg.size))
print(betatg)

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$B = %1.2f $' %myanalysis.B + r'$  N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.axhline(ls='--', color='k')
ax.errorbar(myanalysis.jvec-0.15, (betaj_TS_galS_mean -betatg)/betatg, yerr=np.sqrt(np.diag(cov_TS_galS))/(np.sqrt(nsim)*betatg), color='gold', fmt='o', capsize=0, label=r'$T^S \times gal^S$')

ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$(\langle \beta_j^{Tgal} \rangle - \beta_j^{Tgal, th})/\beta_j^{Tgal, th}$')
ax.set_ylim([-0.2,0.3])

fig.tight_layout()
plt.savefig(out_dir+'betaj_mean_T_gal_jmax'+str(jmax)+'_B = %1.2f ' %myanalysis.B+'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'_mask.png', bbox_inches='tight')

#plt.show()

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

# Sims mean spectrum

delta = need_theory.delta_beta_j(jmax, cltt = xcspectra.cltt, cltg = xcspectra.cltg, clgg = xcspectra.clg1g1)
#print(delta.shape)
fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)
#ax.plot(myanalysis.jvec[:4], betaj_TS_galS_mean[:4], 'o')
ax.errorbar(myanalysis.jvec-0.15, betaj_TS_galS_mean, yerr=delta/np.sqrt(nsim-1),fmt='o')
ax.set_ylabel(r'$\beta_j$')

ax.set_xlabel(r'j')
fig.tight_layout()
plt.savefig(out_dir+'betaj_mean_plot_nsim_'+str(jmax)+'_B = %1.2f ' %myanalysis.B+'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'_mask.png')

# Sims mean spectrum + one sims spectrum

beta_j_sim_200_S = betaj_sims_TS_galS[168,:]
beta_j_sim_200_S_mask = betaj_sims_TS_galS_mask[168,:]


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(38,23))

plt.suptitle(r'$B = %1.2f $' %myanalysis.B + r'$ ,~N_{side} =$'+str(simparams['nside']) + r' $,~N_{sim} = $'+str(nsim))

#ax = fig.add_subplot(1, 1, 1)
ax1.set_title(r'$B = %1.2f $' %myanalysis.B + r'$ ,~N_{side} =$'+str(simparams['nside']) + r' $,~N_{sim} = $'+str(nsim))#(r'$\beta_{j}T^S \times gal^S$')
ax1.errorbar(myanalysis.jvec-0.15, betaj_TS_galS_mean, yerr=delta/np.sqrt(nsim-1),fmt='o', label = 'betaj mean')
ax1.errorbar(myanalysis.jvec-0.15, beta_j_sim_200_S, yerr=delta, fmt='o', label = 'betaj sim')
ax1.set_ylabel(r'$\beta_j^{TG}$')
ax1.set_xlabel(r'j')
ax1.legend()

ax2.set_title(r'$B = %1.2f $' %myanalysis.B + r'$ ,~N_{side} =$'+str(simparams['nside']) + r' $,~N_{sim} = $'+str(nsim)+' Masked')#(r'$\beta_{j}T^S \times gal^S$')
ax2.errorbar(myanalysis.jvec-0.15, betaj_TS_galS_mask_mean, yerr=delta/np.sqrt(nsim-1),fmt='o', label = 'betaj mean')
ax2.errorbar(myanalysis.jvec-0.15, beta_j_sim_200_S_mask, yerr=delta, fmt='o', label = 'betaj sim')
ax2.set_ylabel(r'$\beta_j^{TG}$')
ax2.set_xlabel(r'j')
ax2.legend()

fig.tight_layout()
plt.savefig(out_dir+'betaj_mean_betaj_sim_plot_nsim_'+str(jmax)+'_B = %1.2f ' %myanalysis.B +'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'_mask.png')

## Chi - square test

#cov_inv = np.linalg.inv(cov_TS_galS)
#print(betaj_TS_galS_mean, betatg.shape,cov_inv.shape )

#print(np.matmul(cov_TS_galS,cov_inv))