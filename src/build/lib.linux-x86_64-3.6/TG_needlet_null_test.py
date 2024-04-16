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
<<<<<<< HEAD

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
=======
sns.set()
sns.set(style = 'white')
sns.set_palette('husl')

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)

from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 

#plt.rcParams['axes.linewidth']  = 5.
plt.rcParams['axes.labelsize']  =18
plt.rcParams['xtick.labelsize'] =15
plt.rcParams['ytick.labelsize'] =15
#plt.rcParams['xtick.major.size'] = 20
#plt.rcParams['ytick.major.size'] = 20
#plt.rcParams['xtick.minor.size'] = 20
#plt.rcParams['ytick.minor.size'] = 20
plt.rcParams['legend.fontsize']  = 'large'
#plt.rcParams['legend.frameon']  = False
plt.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['axes.titlesize'] = '20'
rcParams["errorbar.capsize"] = 5
##
>>>>>>> euclid_implementation
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
<<<<<<< HEAD
plt.rcParams['font.size'] = 40
plt.rcParams['lines.linewidth']  = 5.
#plt.rcParams['backend'] = 'WX'
=======
plt.rcParams['font.size'] = 20
plt.rcParams['lines.linewidth']  = 3.
##plt.rcParams['backend'] = 'WX'
>>>>>>> euclid_implementation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# embed()

# Parameters
simparams = {'nside'   : 512,
             'ngal'    : 5.76e5,
 	     	 'ngal_dim': 'ster',
	     	 'pixwin'  : False}

nside = simparams['nside']

# Paths
fname_xcspectra = f'/ehome/bdecaro/xcmbneed/src/spectra/inifiles/CAMBNull_planck.dat'
sims_dir        = f'sims/Needlet/NullTGsims_{nside}_Planck/'
out_dir         = f'output_needlet_TG_Null/TGNull_{nside}_Planck/'
path_inpainting = f'inpainting/inpainting.py'
cov_dir 		= f'covariance_Null/TGNull_{nside}_Planck/' 
if not os.path.exists(cov_dir):
        os.makedirs(cov_dir)
nsim = 500
jmax = 12
lmax = 782

# Loading theory spectra
xcspectra = spectra.XCSpectraFile(fname_xcspectra, WantTG = True)

# Simulations class
simulations = sims.KGsimulations(xcspectra, sims_dir, simparams, WantTG = True)
simulations.Run(nsim, WantTG = True)

# Needlet Analysis
myanalysis = analysis.NeedAnalysis(jmax, lmax, out_dir, simulations)

need_theory = spectra.NeedletTheory(myanalysis.B)
B=myanalysis.B
print(myanalysis.B)

# Computing simulated Cls 
print("...computing Betajs for simulations...")
fname_betaj_sims_TS_galS      = f'betaj_sims_TS_galS_jmax{jmax}_B{B:1.2f}_nside{nside}.dat'

betaj_sims_TS_galS = myanalysis.GetBetajSimsFromMaps('TS', nsim, field2='galS', fname=fname_betaj_sims_TS_galS)

# Covariances
print("...computing Cov Matrices...")
fname_cov_TS_galS      = f'cov_TS_galS_jmax{jmax}_B{B}_nside{nside}.dat'

cov_TS_galS, corr_TS_galS           = myanalysis.GetCovMatrixFromMaps(field1='TS', nsim=nsim, field2='galS', fname=fname_cov_TS_galS, fname_sims=fname_betaj_sims_TS_galS)

print("...done...")

# <Beta_j>_MC
betaj_TS_galS_mean      = myanalysis.GetBetajMeanFromMaps('TS', nsim, field2='galS', fname_sims=fname_betaj_sims_TS_galS)

# Some plots
print("...here come the plots...")
# Covariances
fig, ax1 = plt.subplots(1,1,figsize=(25,18))   
mask_ = np.tri(corr_TS_galS.shape[0],corr_TS_galS.shape[1],0)
ax1.set_title(r'Corr $T^S\times gal^S$')
sns.heatmap(corr_TS_galS, annot=True, fmt='.2f', mask=mask_, ax=ax1)

plt.savefig(cov_dir+f'corr_TS_galS_jmax{jmax}_B{B:1.2f}_nsim{nsim}_nside{nside}.png')

# Theory + Normalization Needlet power spectra

betatg    = need_theory.cl2betaj(jmax=jmax, cl=xcspectra.cltg)
beta_norm = need_theory.cl2betaj(jmax=jmax, cl=np.ones(xcspectra.cltg.size))
print(betatg)

fig = plt.figure(figsize=(10,7))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B + r'$  N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.axhline(ls='--', color='k')
ax.errorbar(myanalysis.jvec-0.15, (betaj_TS_galS_mean -betatg)/betatg, yerr=np.sqrt(np.diag(cov_TS_galS))/(np.sqrt(nsim)*betatg), fmt='o', capsize=0, label=r'$T^S \times gal^S$')

ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$(\langle \beta_j^{Tgal} \rangle - \beta_j^{Tgal, th})/\beta_j^{Tgal, th}$')
ax.set_ylim([-0.2,0.3])
ax.yaxis.set_major_formatter(formatter) 

plt.savefig(out_dir+f'betaj_mean_T_gal_jmax{jmax}_B{{B:1.2f}}_nsim{nsim}_nside{nside}.png', bbox_inches='tight')

#Beta_j mean beta_j sim 

beta_j_one_sim = betaj_sims_TS_galS[96,:]
delta = need_theory.delta_beta_j(jmax, cltt = xcspectra.cltt, cltg = xcspectra.cltg, clgg = xcspectra.clg1g1)


std = np.zeros(len(myanalysis.jvec))
for j in range(len(myanalysis.jvec)):
    std[j]= np.sqrt(np.sum(betaj_sims_TS_galS[:,j]**2)/(nsim-1))

<<<<<<< HEAD
print(delta, np.sqrt(np.diag(cov_TS_galS)), std)
    
fig = plt.figure(figsize=(29,17))
plt.suptitle(r'$D = %1.2f $' %myanalysis.B + r'$ ,~N_{side} =$'+str(simparams['nside']) + r',$~N_{sim} = $'+str(nsim))
ax = fig.add_subplot(1, 1, 1)
ax.errorbar(myanalysis.jvec-0.15, betaj_TS_galS_mean, yerr=std,fmt='o', label = 'Mean simulation')
ax.errorbar(myanalysis.jvec-0.15, betatg, yerr=delta,fmt='o', label = 'Theory')
#ax.errorbar(myanalysis.jvec-0.15, beta_j_sim_400, yerr=delta, fmt='ro', label = 'betaj sim')np.sqrt(np.diag(cov_TS_galS))
ax.set_ylabel(r'$\beta_j$')
=======
print('delta=',delta, '\ndiag cov=',np.sqrt(np.diag(cov_TS_galS)))
    
fig = plt.figure(figsize=(17,10))
plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))
ax = fig.add_subplot(1, 1, 1)
ax.errorbar(myanalysis.jvec, betaj_TS_galS_mean, yerr=np.sqrt(np.diag(cov_TS_galS))/(np.sqrt(nsim)),fmt='o', label = 'Mean simulation')
ax.errorbar(myanalysis.jvec, betatg, yerr=delta/np.sqrt(nsim),fmt='o', label = 'Theory')
#ax.errorbar(myanalysis.jvec-0.15, beta_j_sim_400, yerr=delta, fmt='ro', label = 'betaj sim')np.sqrt(np.diag(cov_TS_galS))
ax.axhline(color='grey',ls='--')
ax.set_ylabel(r'$\beta^{\, \mathrm{TG}}_j$')
>>>>>>> euclid_implementation
ax.set_xlabel(r'j')
ax.yaxis.set_major_formatter(formatter) 
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+f'betaj_mean_betaj_sim_plot_jmax_{jmax}_B{B:1.2f}_nsim{nsim}_nside{nside}.png')
plt.savefig(f'plot_tesi/NULL_TEST_betaj_mean_jmax_{jmax}_B{B:1.2f}_nsim{nsim}_nside{nside}.pdf')

#Chi-squared

#chi_squared_delta = np.sum((beta_j_sim_400[1:]-betaj_TS_galS_mean[1:])**2/delta[1:]**2)
chi_squared_delta = 0#np.sum((beta_j_sim_400-beta_j_mean)**2/delta**2)
for j in range(1, jmax+1):
    #chi_squared_delta += (betaj_TS_galS_mean[j]-betatg[j])**2/np.diag(cov_TS_galS)[j]
    chi_squared_delta += (beta_j_one_sim[j]-betaj_TS_galS_mean[j])**2/delta[j]**2#delta[j]**2

from scipy.stats import chi2

print('chi squared_delta=%1.2f'%chi_squared_delta, 'perc=%1.2f'%chi2.cdf(chi_squared_delta, 12), 1-chi2.cdf(chi_squared_delta, 12))
print(xcspectra.cltg)
