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
simparams = {'nside'   : 512,
             'ngal'    : 1.8e6, #5.76e5,
 	     	 'ngal_dim': 'ster',
	     	 'pixwin'  : False}

# Paths
fname_xcspectra = 'spectra/CAMBPietrobon.dat'
sims_dir        = 'sims/Needlet/PietrobonTGsims_'+str(simparams['nside'])+'/'
out_dir         = 'output_needlet_TG_Pietrobon/'
path_inpainting = 'inpainting/inpainting.py'
cov_dir 		= 'covariance_Pietrobon/' 

nsim = 500
jmax = 12
B = 1.5
lmax = np.floor(B**jmax).astype(int)

# Loading theory spectra
xcspectra = spectra.XCSpectraFile(fname_xcspectra, WantTG = True)

# Simulations class
simulations = sims.KGsimulations(xcspectra, sims_dir, simparams, WantTG = True)
simulations.Run(nsim, WantTG = True)

# Needlet Analysis
myanalysis = analysis.NeedAnalysis(jmax, lmax, out_dir, simulations)

need_theory = spectra.NeedletTheory(myanalysis.B)
print(myanalysis.B)

# Computing simulated Cls 
print("...computing Betajs for simulations...")
fname_betaj_sims_TS_galS      = 'betaj_sims_TS_galS_jmax'+str(jmax)+'_B = %1.2f ' %myanalysis.B +'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'.dat'
fname_betaj_sims_TS_galT      = 'betaj_sims_TS_galT_jmax'+str(jmax)+'_B = %1.2f ' %myanalysis.B +'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'.dat'

betaj_sims_TS_galS = myanalysis.GetBetajSimsFromMaps('TS', nsim, field2='galS', fname=fname_betaj_sims_TS_galS)
betaj_sims_TS_galT = myanalysis.GetBetajSimsFromMaps('TS', nsim, field2='galT', fname=fname_betaj_sims_TS_galT)


# Covariances
print("...computing Cov Matrices...")
fname_cov_TS_galS      = 'cov_TS_galS_jmax'+str(jmax)+'_B = %1.2f ' %myanalysis.B +'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'.dat'
fname_cov_TS_galT      = 'cov_TS_galT_jmax'+str(jmax)+'_B = %1.2f ' %myanalysis.B +'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'.dat'

cov_TS_galS, corr_TS_galS           = myanalysis.GetCovMatrixFromMaps(field1='TS', nsim=nsim, field2='galS', fname=fname_cov_TS_galS, fname_sims=fname_betaj_sims_TS_galS)
cov_TS_galT, corr_TS_galT           = myanalysis.GetCovMatrixFromMaps(field1='TS', nsim=nsim, field2='galT', fname=fname_cov_TS_galT, fname_sims=fname_betaj_sims_TS_galT)

print("...done...")

# <Beta_j>_MC
betaj_TS_galS_mean      = myanalysis.GetBetajMeanFromMaps('TS', nsim, field2='galS', fname_sims=fname_betaj_sims_TS_galS)
betaj_TS_galT_mean      = myanalysis.GetBetajMeanFromMaps('TS', nsim, field2='galT', fname_sims=fname_betaj_sims_TS_galT)


# Some plots
print("...here come the plots...")
# Covariances
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(25,18))   
mask_ = np.tri(corr_TS_galS.shape[0],corr_TS_galS.shape[1],0)

ax1.set_title(r'Corr $T^S\times gal^S$')
sns.heatmap(corr_TS_galS, annot=True, fmt='.2f', mask=mask_, ax=ax1)

ax2.set_title(r'Corr $T^S\times gal^T$')
sns.heatmap(corr_TS_galT, annot=True, fmt='.2f', mask=mask_, ax=ax2)


plt.savefig(cov_dir+'corr_TS_galS_galT_jmax'+str(jmax)+'_B = %1.2f ' %myanalysis.B +'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'.png')

# Theory + Normalization Needlet power spectra

betatg    = need_theory.cl2betaj(jmax=jmax, cl=xcspectra.cltg)
beta_norm = need_theory.cl2betaj(jmax=jmax, cl=np.ones(xcspectra.cltg.size))

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$B = %1.2f $' %myanalysis.B + r'$  N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.axhline(ls='--', color='k')
ax.errorbar(myanalysis.jvec-0.15, (betaj_TS_galS_mean -betatg)/betatg, yerr=np.sqrt(np.diag(cov_TS_galS))/(np.sqrt(nsim)*betatg), color='gold', fmt='o', capsize=0, label=r'$T^S \times gal^S$')
ax.errorbar(myanalysis.jvec-0.05, (betaj_TS_galT_mean -betatg)/betatg, yerr=np.sqrt(np.diag(cov_TS_galT))/(np.sqrt(nsim)*betatg), color='seagreen', fmt='o', capsize=0, label=r'$T^S \times gal^T$')

ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$(\langle \beta_j^{Tgal} \rangle - \beta_j^{Tgal, th})/\beta_j^{Tgal, th}$')
ax.set_ylim([-0.2,0.3])

plt.savefig(out_dir+'betaj_mean_T_gal_jmax'+str(jmax)+'_B = %1.2f ' %myanalysis.B +'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'.png', bbox_inches='tight')

#Beta_j mean beta_j sim 

beta_j_sim_200_S = betaj_sims_TS_galS[200,:]
beta_j_sim_200_T = betaj_sims_TS_galT[200,:]

delta = need_theory.delta_beta_j(jmax, cltt = xcspectra.cltt, cltg = xcspectra.cltg, clgg = xcspectra.clg1g1)

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(17,10))

plt.suptitle(r'$B = %1.2f $' %myanalysis.B + r'$  N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

#ax = fig.add_subplot(1, 1, 1)
ax1.set_title(r'$\beta_{j} T^S\times gal^S$')
ax1.errorbar(myanalysis.jvec-0.15, betaj_TS_galS_mean, yerr=delta/np.sqrt(nsim-1),fmt='o', label = 'betaj mean')
ax1.errorbar(myanalysis.jvec-0.15, beta_j_sim_200_S, yerr=delta, fmt='ro', label = 'betaj sim')
ax1.set_ylabel(r'$\beta_j$')
ax1.set_xlabel(r'j')
ax1.legend()

ax2.set_title(r'$\beta_{j} T^S\times gal^T$')
ax2.errorbar(myanalysis.jvec-0.15, betaj_TS_galT_mean, yerr=delta/np.sqrt(nsim-1),fmt='o', label = 'betaj mean')
ax2.errorbar(myanalysis.jvec-0.15, beta_j_sim_200_T, yerr=delta, fmt='ro', label = 'betaj sim')
ax2.set_ylabel(r'$\beta_j$')
ax2.set_xlabel(r'j')
ax2.legend()

plt.savefig(out_dir+'betaj_mean_betaj_sim_ST_plot_nsim_'+str(jmax)+'_B = %1.2f ' %myanalysis.B +'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'.png')

#Chi-squared

chi_squared_delta = np.sum((beta_j_sim_200_S[1:]-betaj_TS_galS_mean[1:])**2/delta[1:]**2)

#chi_squared = np.sum((beta_j_sim_200_S-betaj_TS_galS_mean)**2/np.diag(cov_TS_galS))
chi_squared = 0
cov_inv = np.linalg.inv(cov_TS_galS[1:,1:])
temp = np.zeros(len(cov_TS_galS[0])-1)
for i in range(len(cov_TS_galS[0])-1):
    for j in range(len(beta_j_sim_200_S)):
        temp[i] += cov_inv[i][j]*(beta_j_sim_200_S[j]-betaj_TS_galS_mean[j])
    chi_squared += (beta_j_sim_200_S[i]-betaj_TS_galS_mean[i]).T*temp[i]
print(chi_squared)

from scipy.stats import chi2

print('chi squared_delta=%1.2f'%chi_squared_delta, 'perc =' +str(chi2.cdf(chi_squared_delta, 13)))#'perc=%1.2f'%chi2.cdf(chi_squared_delta, 13))
print('chi squared=%1.2f'%chi_squared, 'perc=%1.2f'%chi2.cdf(chi_squared, 13))
print(np.diag(cov_TS_galS))

#chi squared_delta=33.11 perc =0.9983589504060464
#chi squared=19.05 perc=0.88

#Signal only : chi squared_delta=8.37 perc =0.18158391068244123 ; chi squared=11.44 perc=0.43
#Total : chi squared_delta=16.53 perc =0.7784048117779285;  chi squared=13.45 perc=0.59

