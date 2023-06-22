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
simparams = {'nside'   : 128,
             'ngal'    : 5.76e5, #dovrebbe importare solo per lo shot noise (noise poissoniano)
 	     	 'ngal_dim': 'ster',
	     	 'pixwin'  : False}
nside = simparams['nside']
# Paths
fname_xcspectra = '/ehome/bdecaro/xcmbneed/src/spectra/inifiles/CAMBSpectra_planck_fiducial_lmin0_2050.dat'#'spectra/inifiles/CAMBSpectra_planck.dat' 
sims_dir        = f'sims/Needlet/Planck/TGsims_{nside}/'#planck_2_lmin0_prova_1/'
out_dir         = 'output_needlet_TG/Planck/TG_'+str(simparams['nside'])+'/'#planck_2_lmin0_prova_1/'
path_inpainting = 'inpainting/inpainting.py'
cov_dir 		= 'covariance/Planck/covariance_TG'+str(simparams['nside'])+'/'#planck_2_lmin0_prova_1/'
if not os.path.exists(cov_dir):
        os.makedirs(cov_dir)

jmax = 12
lmax = 512#782
nsim = 50
#B = 1.95

#jmax = round(np.log(lmax*1.00000000000000222045e+00)/np.log(B))
#lmax = round(B**(jmax+1))
#mylibc.debug_needlets()

#mask = utils.GetGalMask(simparams['nside'], lat=20.)
#fsky = np.mean(mask)

# Loading theory spectra
xcspectra = spectra.XCSpectraFile(fname_xcspectra, nltt = None,WantTG = True)

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
fname_betaj_sims_TS_galS      = 'betaj_sims_TS_galS_jmax'+str(jmax)+'_B'+str(myanalysis.B)+'_nside'+str(simparams['nside'])+'.dat'
#fname_betaj_sims_galS_galS      = 'betaj_sims_galS_galS.dat'

betaj_sims_TS_galS = myanalysis.GetBetajSimsFromMaps('TS', nsim, field2='galS', fname=fname_betaj_sims_TS_galS)
#betaj_sims_G_G = myanalysis.GetBetajSimsFromMaps('galS', nsim, field2='galS', fname=fname_betaj_sims_G_G)

# Covariances
print("...computing Cov Matrices...")
fname_cov_TS_galS      = 'cov_TS_galS_jmax'+str(jmax)+'_B'+str(myanalysis.B)+'_nside'+str(simparams['nside'])+'.dat'
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

cov_TS_galS, corr_TS_galS           = myanalysis.GetCovMatrixFromMaps(field1='TS', nsim=nsim, field2='galS', fname=fname_cov_TS_galS, fname_sims=fname_betaj_sims_TS_galS)
#cov_galS_galS, corr_galS_galS           = myanalysis.GetCovMatrixFromMaps(field1='galS', nsim=nsim, field2='galS', fname=fname_cov_galS_galS, fname_sims=fname_betaj_sims_galS_galS)

print("...done...")

# <Beta_j>_MC
betaj_TS_galS_mean      = myanalysis.GetBetajMeanFromMaps('TS', nsim, field2='galS', fname_sims=fname_betaj_sims_TS_galS)
#betaj_galS_galS_mean      = myanalysis.GetBetajMeanFromMaps('kappaT', nsim, field2='deltaT', fname_sims=fname_betaj_sims_kappaT_deltaT)


# Some plots
print("...here come the plots...")

# Covariances
#fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(25,18))
fig, ax1 = plt.subplots(1,1,figsize=(25,18))   
#fig.suptitle(r'$B = %1.2f $' %myanalysis.B + r'$N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

mask_ = np.tri(corr_TS_galS.shape[0],corr_TS_galS.shape[1],0)

#plt.subplot(131)
ax1.set_title(r'Corr $T^S\times gal^S$')
sns.color_palette("crest", as_cmap=True)
sns.heatmap(corr_TS_galS, annot=True, fmt='.2f', cmap  = 'viridis',mask=mask_, ax=ax1)

##plt.subplot(132)
#ax2.set_title(r'Corr $\delta^T \times \gal^T$')
#sns.heatmap(corr_kappaT_deltaT, annot=True, fmt='.2f', mask=mask_, ax=ax2)
#
##plt.subplot(133)
#ax3.set_title(r'Corr $\delta^T \times \kappa^T$ Masked')
#sns.heatmap(corr_kappaT_deltaT_mask, annot=True, fmt='.2f', mask=mask_,ax=ax3)
fig.tight_layout()
plt.savefig(cov_dir+'corr_TS_galS_jmax'+str(jmax)+'_B = %1.2f ' %myanalysis.B +'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'.png')


# Theory + Normalization Needlet power spectra

betatg    = need_theory.cl2betaj(jmax=jmax, cl=xcspectra.cltg)
beta_norm = need_theory.cl2betaj(jmax=jmax, cl=np.ones(xcspectra.cltg.size))

np.savetxt(out_dir+f'beta_TS_galS_theoretical_fiducial_B{myanalysis.B}.dat', betatg)

delta = need_theory.delta_beta_j(jmax, cltt = xcspectra.cltt, cltg = xcspectra.cltg, clgg = xcspectra.clg1g1)

fig = plt.figure(figsize=(27,20))

plt.suptitle('Beta from cl theory')

ax = fig.add_subplot(1, 1, 1)

ax.errorbar(myanalysis.jvec, betatg, yerr=delta/np.sqrt(nsim), color='firebrick', fmt='o', ms=10,capthick=5, label=r'$T^S \times gal^T$ theory')
ax.errorbar(myanalysis.jvec, betaj_TS_galS_mean, yerr=np.sqrt(np.diag(cov_TS_galS))/np.sqrt(nsim), color='seagreen', fmt='o',ms=10,capthick=5, label=r'$T^S \times gal^T$ from sim')

ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\beta_j^{Tgal}$')

fig.tight_layout()
plt.savefig(out_dir+f'betaj_theory_T_gal_noise_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')


fig = plt.figure(figsize=(25,15))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B + r'$~N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.axhline(ls='--', color='k')
ax.errorbar(myanalysis.jvec-0.15, (betaj_TS_galS_mean -betatg)/betatg, yerr=np.sqrt(np.diag(cov_TS_galS))/(np.sqrt(nsim)*betatg), fmt='o',ms=15, label=r'Variance from simulations')#, label=r'$T^S \times gal^S$, sim cov')
ax.errorbar(myanalysis.jvec-0.15, (betaj_TS_galS_mean -betatg)/betatg, yerr=delta/(np.sqrt(nsim)*betatg),  fmt='o',  ms=15,label=r'Variance from theory')
#print(np.sqrt(np.diag(cov_TS_galS))/(np.sqrt(nsim)*betatg) - delta/(np.sqrt(nsim)*betatg))

ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$(\langle \beta_j^{Tgal} \rangle - \beta_j^{Tgal, th})/\beta_j^{Tgal, th}$')
ax.set_ylim([-0.2,0.3])

fig.tight_layout()
plt.savefig(out_dir+'betaj_mean_T_gal_jmax'+str(jmax)+'_D = %1.2f ' %myanalysis.B+'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'.png', bbox_inches='tight')

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
ax.errorbar(myanalysis.jvec-0.15, betaj_TS_galS_mean, yerr=np.sqrt(np.diag(cov_TS_galS))/np.sqrt(nsim-1),fmt='o', label = 'errore covarianza')
#ax.errorbar(myanalysis.jvec-0.15, betaj_TS_galS_mean, yerr=delta/np.sqrt(nsim-1),fmt='o', label='errore teorico')

ax.set_ylabel(r'$\beta_j$')

ax.set_xlabel(r'j')
fig.tight_layout()
#plt.savefig(out_dir+'betaj_mean_plot_nsim_'+str(jmax)+'_B = %1.2f ' %myanalysis.B+'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'.png')
plt.savefig(out_dir+'betaj_mean_plot_nsim_'+str(jmax)+'_B = %1.2f ' %myanalysis.B+'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'.png')

# Sims mean spectrum + one sims spectrum

beta_j_sim_200_S = betaj_sims_TS_galS[9,:]

fig, ax1 = plt.subplots(1,1,figsize=(29,17))

#plt.suptitle(r'$B = %1.2f $' %myanalysis.B + r'$ ,~N_{side} =$'+str(simparams['nside']) + r' $,~N_{sim} = $'+str(nsim))

#ax = fig.add_subplot(1, 1, 1)
ax1.set_title(r'$D = %1.2f $' %myanalysis.B + r'$ ,~N_{side} =$'+str(simparams['nside']) + r' $,~N_{sim} = $'+str(nsim))#(r'$\beta_{j}T^S \times gal^S$')
ax1.errorbar(myanalysis.jvec-0.15, betaj_TS_galS_mean, yerr=np.sqrt(np.diag(cov_TS_galS))/np.sqrt(nsim-1),fmt='o', label = 'Mean from simulations')
ax1.errorbar(myanalysis.jvec-0.15, beta_j_sim_200_S, yerr=np.sqrt(np.diag(cov_TS_galS)), fmt='o', label = 'Single realization')
ax1.set_ylabel(r'$\beta_j^{TG}$')
ax1.set_xlabel(r'j')
ax1.legend(loc='upper left')

fig.tight_layout()
plt.savefig(out_dir+'betaj_mean_betaj_sim_plot_nsim_'+str(jmax)+'_D = %1.2f ' %myanalysis.B +'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'.png')

## Chi - square test

#cov_inv = np.linalg.inv(cov_TS_galS)
#print(betaj_TS_galS_mean, betatg.shape,cov_inv.shape )
#print(np.matmul(cov_TS_galS,cov_inv))