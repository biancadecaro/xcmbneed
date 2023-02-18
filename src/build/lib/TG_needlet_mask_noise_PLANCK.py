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
plt.rcParams['axes.linewidth']  = 5.
plt.rcParams['axes.labelsize']  =30
plt.rcParams['xtick.labelsize'] =30
plt.rcParams['ytick.labelsize'] =30
plt.rcParams['xtick.major.size'] = 30
plt.rcParams['ytick.major.size'] = 30
plt.rcParams['xtick.minor.size'] = 30
plt.rcParams['ytick.minor.size'] = 30
plt.rcParams['legend.fontsize']  = 'large'
plt.rcParams['legend.frameon']  = False
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['axes.titlesize'] = 'large'
rcParams["errorbar.capsize"] = 15
#
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['font.size'] = 40
plt.rcParams['lines.linewidth']  = 5.
#plt.rcParams['backend'] = 'WX'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# embed()

# Parameters
simparams = {'nside'   : 256,
             'ngal'    : 5.76e5, #dovrebbe importare solo per lo shot noise (noise poissoniano)
 	     	 'ngal_dim': 'ster',
	     	 'pixwin'  : False}

# Paths
fname_xcspectra = 'spectra/CAMBSpectra_planck.dat'
sims_dir        = 'sims/Needlet/Planck/TGsims_'+str(simparams['nside'])+'_mask_noise_nonoisePlanck/'
out_dir         = 'output_needlet_TG/Planck/TG_'+str(simparams['nside'])+'_mask_noise_nonoisePlanck/'
path_inpainting = 'inpainting/inpainting.py'
cov_dir 		= 'covariance/Planck/covariance_TG'+str(simparams['nside'])+'_mask_noise_nonoisePlanck/'


jmax = 12
lmax = 782
nsim = 500
#B = 1.47

#jmax = round(np.log(lmax*1.00000000000000222045e+00)/np.log(B))
#lmax = round(B**(jmax+1))
#mylibc.debug_needlets()

mask = hp.read_map('mask/mask70_gal_nside='+str(simparams['nside'])+'.fits')#utils.GetGalMask(simparams['nside'], lat=20.)
fsky = np.mean(mask)

#nltt = np.loadtxt('noise/Noise_Planck2013_commander_2048.dat')
#
# 
# 
# print(fsky)

# Loading theory spectra
xcspectra = spectra.XCSpectraFile(fname_xcspectra,nltt = None, WantTG = True)#nltt=None,

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
fname_betaj_sims_TT_galT      = f'betaj_sims_TT_galT_jmax{jmax}_B_{myanalysis.B}_nside'+str(simparams['nside'])+'.dat'
fname_betaj_sims_TS_galS_mask      = f'betaj_sims_TS_galS_jmax{jmax}_B_{myanalysis.B}_nside'+str(simparams['nside'])+f'_fsky_{fsky}.dat'
fname_betaj_sims_TT_galT_mask      = f'betaj_sims_TT_galT_jmax{jmax}_B_{myanalysis.B}_nside'+str(simparams['nside'])+f'_fsky_{fsky}.dat'

#fname_betaj_sims_galS_galS      = 'betaj_sims_galS_galS.dat'

betaj_sims_TS_galS = myanalysis.GetBetajSimsFromMaps('TS', nsim, field2='galS', fname=fname_betaj_sims_TS_galS)
betaj_sims_TT_galT = myanalysis.GetBetajSimsFromMaps('TT', nsim, field2='galT', fname=fname_betaj_sims_TT_galT)
betaj_sims_TS_galS_mask = myanalysis.GetBetajSimsFromMaps('TS', nsim, field2='galS', mask= mask, fname=fname_betaj_sims_TS_galS_mask)
betaj_sims_TT_galT_mask = myanalysis.GetBetajSimsFromMaps('TT', nsim, field2='galT', mask= mask, fname=fname_betaj_sims_TT_galT_mask)

#betaj_sims_G_G = myanalysis.GetBetajSimsFromMaps('galS', nsim, field2='galS', fname=fname_betaj_sims_G_G)

# Covariances
print("...computing Cov Matrices...")
fname_cov_TS_galS      = f'cov_TS_galS_jmax{jmax}_B = %1.2f $' %myanalysis.B +'_nside'+str(simparams['nside'])+'.dat'
fname_cov_TT_galT      = f'cov_TT_galT_jmax{jmax}_B = %1.2f $' %myanalysis.B +'_nside'+str(simparams['nside'])+'.dat'
fname_cov_TS_galS_mask      = f'cov_TS_galS_jmax{jmax}_B = %1.2f $' %myanalysis.B +'_nside'+str(simparams['nside'])+'_mask.dat'
fname_cov_TT_galT_mask      = f'cov_TT_galT_jmax{jmax}_B = %1.2f $' %myanalysis.B +'_nside'+str(simparams['nside'])+'_mask.dat'
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
cov_TT_galT, corr_TT_galT          = myanalysis.GetCovMatrixFromMaps(field1='TT', nsim=nsim, field2='galT', fname=fname_cov_TT_galT, fname_sims=fname_betaj_sims_TT_galT)
cov_TS_galS_mask, corr_TS_galS_mask          = myanalysis.GetCovMatrixFromMaps(field1='TS', nsim=nsim, field2='galS', fname=fname_cov_TS_galS_mask, mask = mask,fname_sims=fname_betaj_sims_TS_galS_mask)
cov_TT_galT_mask, corr_TT_galT_mask          = myanalysis.GetCovMatrixFromMaps(field1='TT', nsim=nsim, field2='galT', fname=fname_cov_TT_galT_mask, mask = mask,fname_sims=fname_betaj_sims_TT_galT_mask)
#cov_galS_galS, corr_galS_galS           = myanalysis.GetCovMatrixFromMaps(field1='galS', nsim=nsim, field2='galS', fname=fname_cov_galS_galS, fname_sims=fname_betaj_sims_galS_galS)

print("...done...")

# <Beta_j>_MC
betaj_TS_galS_mean    = myanalysis.GetBetajMeanFromMaps('TS', nsim, field2='galS', fname_sims=fname_betaj_sims_TS_galS)
betaj_TT_galT_mean    = myanalysis.GetBetajMeanFromMaps('TT', nsim, field2='galT', fname_sims=fname_betaj_sims_TT_galT)
betaj_TS_galS_mask_mean     = myanalysis.GetBetajMeanFromMaps('TS', nsim, field2='galS', fname_sims=fname_betaj_sims_TS_galS_mask)
betaj_TT_galT_mask_mean     = myanalysis.GetBetajMeanFromMaps('TT', nsim, field2='galT', fname_sims=fname_betaj_sims_TT_galT_mask)
#betaj_galS_galS_mean      = myanalysis.GetBetajMeanFromMaps('kappaT', nsim, field2='deltaT', fname_sims=fname_betaj_sims_kappaT_deltaT)


# Some plots
print("...here come the plots...")

# Covariances
#fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(25,18))
fig, axs = plt.subplots(ncols=2, nrows=2,figsize=(38,23))   
fig.suptitle(r'$B = %1.2f $' %myanalysis.B + r'$N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

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
sns.heatmap(corr_TT_galT, annot=True, fmt='.2f',cmap  = 'viridis', mask=mask_, ax=axs[1,0])

axs[1,1].set_title(r'Corr $T^T\times gal^T$ Masked')
sns.heatmap(corr_TT_galT_mask, annot=True, fmt='.2f',cmap  = 'viridis', mask=mask_, ax=axs[1,1])


fig.tight_layout()
#plt.savefig(cov_dir+'corr_TS_galS_jmax'+str(jmax)+'_B = %1.2f ' %myanalysis.B +'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'_mask.png')
plt.savefig(cov_dir+'corr_T_gal_noise_mask_jmax'+str(jmax)+'_B = %1.2f ' %myanalysis.B +'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'_mask.png')

fig, axs = plt.subplots(ncols=2, nrows=1,figsize=(52,25))   
fig.suptitle(r'$B = %1.2f $' %myanalysis.B + r'$N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

mask_ = np.tri(corr_TS_galS.shape[0],corr_TS_galS.shape[1],0)

#plt.subplot(131)
axs[0].set_title(r'Corr $T^S\times gal^S$ Signal only')
sns.color_palette("crest", as_cmap=True)
sns.heatmap(corr_TS_galS, annot=True, fmt='.2f', cmap  = 'viridis',mask=mask_, ax=axs[0])

##plt.subplot(132)
axs[1].set_title(r'Corr $T^S\times gal^S$ Signal only Masked')
sns.heatmap(corr_TS_galS_mask, annot=True, fmt='.2f',cmap  = 'viridis', mask=mask_, ax=axs[1])
#
plt.tight_layout()
plt.savefig(cov_dir+'signal_only_corr_T_gal_noise_mask_jmax'+str(jmax)+'_B = %1.2f ' %myanalysis.B +'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'_mask.png')

fig, axs = plt.subplots(ncols=2, nrows=1,figsize=(52,25))   
fig.suptitle(r'$B = %1.2f $' %myanalysis.B + r'$N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

mask_ = np.tri(corr_TT_galT.shape[0],corr_TT_galT.shape[1],0)

#plt.subplot(131)
axs[0].set_title(r'Corr $T^T\times gal^T$ Signal + shot noise')
sns.color_palette("crest", as_cmap=True)
sns.heatmap(corr_TT_galT, annot=True, fmt='.2f', cmap  = 'viridis',mask=mask_, ax=axs[0])

##plt.subplot(132)
axs[1].set_title(r'Corr $T^T\times gal^T$ Signal + shot noise Masked')
sns.heatmap(corr_TT_galT_mask, annot=True, fmt='.2f',cmap  = 'viridis', mask=mask_, ax=axs[1])
#
plt.tight_layout()
plt.savefig(cov_dir+'total_corr_T_gal_noise_mask_jmax'+str(jmax)+'_B = %1.2f ' %myanalysis.B +'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'_mask.png')




# Theory + Normalization Needlet power spectra

betatg    = need_theory.cl2betaj(jmax=jmax, cl=xcspectra.cltg)
beta_norm = need_theory.cl2betaj(jmax=jmax, cl=np.ones(xcspectra.cltg.size))
#print(betatg)

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$B = %1.2f $' %myanalysis.B + r'$  N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.axhline(ls='--', color='k')
ax.errorbar(myanalysis.jvec-0.15, (betaj_TS_galS_mean -betatg)/betatg, yerr=np.sqrt(np.diag(cov_TS_galS))/(np.sqrt(nsim)*betatg), color='gold', fmt='o', capsize=0, label=r'$T^S \times gal^S$')
ax.errorbar(myanalysis.jvec-0.15, (betaj_TT_galT_mean -betatg)/betatg, yerr=np.sqrt(np.diag(cov_TT_galT))/(np.sqrt(nsim)*betatg), color='red', fmt='o', capsize=0, label=r'$T^T \times gal^T$')

ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$(\langle \beta_j^{Tgal} \rangle - \beta_j^{Tgal, th})/\beta_j^{Tgal, th}$')
ax.set_ylim([-0.2,0.3])

fig.tight_layout()
plt.savefig(out_dir+'betaj_mean_T_gal_noise_jmax'+str(jmax)+'_B = %1.2f ' %myanalysis.B+'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'_mask.png', bbox_inches='tight')

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

# Sims mean spectrum + one sims spectrum SIGNAL ONLY

num_sim = 23

beta_j_sim_200_S = betaj_sims_TS_galS[num_sim,:]
beta_j_sim_200_T = betaj_sims_TT_galT[num_sim,:]
beta_j_sim_200_S_mask = betaj_sims_TS_galS_mask[num_sim,:]
beta_j_sim_200_T_mask = betaj_sims_TT_galT_mask[num_sim,:]


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(50,25))

plt.suptitle(r'Signal-only, $B = %1.2f $' %myanalysis.B + r'$ ,~N_{side} =$'+str(simparams['nside']) + r' $,~N_{sim} = $'+str(nsim))

#ax = fig.add_subplot(1, 1, 1)
ax1.set_title(r'$\beta_{j}T^S \times gal^S$')
ax1.errorbar(myanalysis.jvec-0.15, betaj_TS_galS_mean, yerr=np.sqrt(np.diag(cov_TS_galS))/np.sqrt(nsim-1),fmt='o',ms=15, label = 'betaj mean')
ax1.errorbar(myanalysis.jvec-0.15, beta_j_sim_200_S, yerr=np.sqrt(np.diag(cov_TS_galS)), fmt='o',ms=15, label = 'betaj sim')
ax1.set_ylabel(r'$\beta_j^{TG}$')
ax1.set_xlabel(r'j')
ax1.legend()

ax2.set_title(r'$\beta_{j}T^S \times gal^S$ Masked')
ax2.errorbar(myanalysis.jvec-0.15, betaj_TS_galS_mask_mean, yerr=np.sqrt(np.diag(cov_TS_galS_mask))/np.sqrt(nsim-1),fmt='o',ms=15, label = 'betaj mean')
ax2.errorbar(myanalysis.jvec-0.15, beta_j_sim_200_S_mask, yerr=np.sqrt(np.diag(cov_TS_galS_mask)), fmt='o',ms=15, label = 'betaj sim')
ax2.set_ylabel(r'$\beta_j^{TG}$')
ax2.set_xlabel(r'j')
ax2.legend()

fig.tight_layout()
plt.savefig(out_dir+'signal-only_betaj_mean_betaj_sim_plot_nsim_'+str(jmax)+'_B = %1.2f ' %myanalysis.B +'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'_mask.png')

# Sims mean spectrum + one sims spectrum TOTAL SIGNAL

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(50,27))

plt.suptitle(r'Total signal, $B = %1.2f $' %myanalysis.B + r'$ ,~N_{side} =$'+str(simparams['nside']) + r' $,~N_{sim} = $'+str(nsim))

#ax = fig.add_subplot(1, 1, 1)
ax1.set_title(r'$\beta_{j}T^T \times gal^T$')
ax1.errorbar(myanalysis.jvec-0.15, betaj_TT_galT_mean, yerr=np.sqrt(np.diag(cov_TT_galT))/np.sqrt(nsim-1),fmt='o', label = 'betaj mean')
ax1.errorbar(myanalysis.jvec-0.15, beta_j_sim_200_T, yerr=np.sqrt(np.diag(cov_TT_galT)), fmt='o', label = 'betaj sim')
ax1.set_ylabel(r'$\beta_j^{TG}$')
ax1.set_xlabel(r'j')
ax1.legend()

ax2.set_title(r'$\beta_{j}T^T \times gal^T$ Masked')
ax2.errorbar(myanalysis.jvec-0.15, betaj_TT_galT_mask_mean, yerr=np.sqrt(np.diag(cov_TT_galT_mask))/np.sqrt(nsim-1),fmt='o', label = 'betaj mean')
ax2.errorbar(myanalysis.jvec-0.15, beta_j_sim_200_T_mask, yerr=np.sqrt(np.diag(cov_TT_galT_mask)), fmt='o', label = 'betaj sim')
ax2.set_ylabel(r'$\beta_j^{TG}$')
ax2.set_xlabel(r'j')
ax2.legend()

fig.tight_layout()
plt.savefig(out_dir+'total-signal_betaj_mean_betaj_sim_plot_nsim_'+str(jmax)+'_B = %1.2f ' %myanalysis.B +'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'_mask.png')


# One sims spectrum TOTAL SIGNAL +  SIGNAL ONLY

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(50,27), sharey = True)

plt.suptitle(r'$B = %1.2f $' %myanalysis.B + r'$ ,~N_{side} =$'+str(simparams['nside']) + r' $,~N_{sim} = $'+str(nsim))

#ax = fig.add_subplot(1, 1, 1)
ax1.set_title('No Masked')#(r'$\beta_{j}T^S \times gal^S$')
ax1.errorbar(myanalysis.jvec-0.15, beta_j_sim_200_S, yerr=np.sqrt(np.diag(cov_TS_galS)), fmt='o',ms=15,capthick=7, label = 'Signal only')
ax1.errorbar(myanalysis.jvec-0.15, beta_j_sim_200_T, yerr=np.sqrt(np.diag(cov_TT_galT)), fmt='o',ms=15,capthick=7, label = 'Signal + Shot Noise')

ax1.set_ylabel(r'$\beta_j^{TG}$')
ax1.set_xlabel(r'j')
ax1.legend()

ax2.set_title('Masked')#(r'$\beta_{j}T^T \times gal^T$ ')
ax2.errorbar(myanalysis.jvec-0.15, beta_j_sim_200_S_mask, yerr=np.sqrt(np.diag(cov_TS_galS_mask)), fmt='o',ms=15,capthick=7, label = 'Signal only')
ax2.errorbar(myanalysis.jvec-0.15, beta_j_sim_200_T_mask, yerr=np.sqrt(np.diag(cov_TT_galT_mask)), fmt='o',ms=15,capthick=7, label = 'Signal + Shot Noise')
ax2.set_ylabel(r'$\beta_j^{TG}$')
ax2.set_xlabel(r'j')
ax2.legend()

fig.tight_layout()
plt.savefig(out_dir+f'signal_noise_betaj_mean_betaj_sim_plot_jmax_{jmax}_B = %1.2f ' %myanalysis.B +f'_nsim_{nsim}_nside'+str(simparams['nside'])+'_mask.png')

# One sims spectrum TOTAL SIGNAL +  SIGNAL ONLY MASK

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(50,27), sharey= True)

plt.suptitle(r'$B = %1.2f $' %myanalysis.B + r'$ ,~N_{side} =$'+str(simparams['nside']) + r' $,~N_{sim} = $'+str(nsim))

#ax = fig.add_subplot(1, 1, 1)
ax1.set_title(r'$\beta_{j}~T^S \times gal^S$ Signal only')
ax1.errorbar(myanalysis.jvec-0.15, beta_j_sim_200_S, yerr=np.sqrt(np.diag(cov_TS_galS)), fmt='o',ms=15,capthick=7, label = 'No mask')
ax1.errorbar(myanalysis.jvec-0.15, beta_j_sim_200_S_mask, yerr=np.sqrt(np.diag(cov_TS_galS_mask)), fmt='o',ms=15,capthick=7, label = 'Mask')

ax1.set_ylabel(r'$\beta_j^{TG}$')
ax1.set_xlabel(r'j')
ax1.legend()

ax2.set_title(r'$\beta_{j}~T^T \times gal^T$ Signal + shot noise')
ax2.errorbar(myanalysis.jvec-0.15, beta_j_sim_200_T, yerr=np.sqrt(np.diag(cov_TT_galT)), fmt='o',ms=15,capthick=7, label = 'No mask')
ax2.errorbar(myanalysis.jvec-0.15, beta_j_sim_200_T_mask, yerr=np.sqrt(np.diag(cov_TT_galT_mask)), fmt='o',ms=15,capthick=7, label = 'Mask')
ax2.set_ylabel(r'$\beta_j^{TG}$')
ax2.set_xlabel(r'j')
ax2.legend()

fig.tight_layout()
plt.savefig(out_dir+f'mask_signal_noise_betaj_mean_betaj_sim_plot_jmax_{jmax}_B = %1.2f ' %myanalysis.B +f'_nsim_{nsim}_nside'+str(simparams['nside'])+'_mask.png')

## Chi - square test


#cov_inv = np.linalg.inv(cov_TS_galS)
#print(betaj_TS_galS_mean, betatg.shape,cov_inv.shape )

#print(np.matmul(cov_TS_galS,cov_inv))