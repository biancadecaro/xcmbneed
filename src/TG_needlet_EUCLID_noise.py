#!/usr/bin/env python
import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams, gridspec
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
plt.rcParams["errorbar.capsize"] = 15
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

jmax = 12
lmax = 256
nsim = 500


# Paths
fname_xcspectra = '/ehome/bdecaro/xcmbneed/src/spectra/inifiles/EUCLID_fiducial_lmin0.dat'
sims_dir        = f'/ehome/bdecaro/xcmbneed/src/sims/Euclid_sims_Marina/NSIDE{nside}/'
out_dir         = f'output_needlet_TG/EUCLID/Noise/TG_{nside}_nsim{nsim}_Marina/'
path_inpainting = 'inpainting/inpainting.py'
cov_dir 		= f'covariance/EUCLID/Noise/TG_{nside}_nsim{nsim}_Marina/'
if not os.path.exists(cov_dir):
        os.makedirs(cov_dir)

cl_theory = np.loadtxt('/ehome/bdecaro/xcmbneed/src/spectra/inifiles/EUCLID_fiducial_lmin0.dat')
ell_theory = cl_theory[0]
cl_theory_tt = cl_theory[1]
cl_theory_tg = cl_theory[2]
cl_theory_gg = cl_theory[3]

mask = hp.read_map(f'/ehome/bdecaro/xcmbneed/src/mask/EUCLID/mask_planck_comm_2018_x_euclid_fsky0p35_nside={nside}.fits')
fsky = np.mean(mask)

# Loading theory spectra
xcspectra = spectra.XCSpectraFile(fname_xcspectra,   WantTG = True)
xcspectra.cltg = cl_theory_tg
xcspectra.cltt = cl_theory_tt
xcspectra.clgg = cl_theory_gg

# Simulations class
simulations = sims.KGsimulations(xcspectra, sims_dir, simparams, WantTG = True)
simulations.Run(nsim, WantTG = True, EuclidSims=True)

# Needlet Analysis
myanalysis = analysis.NeedAnalysis(jmax, lmax, out_dir, simulations)

# Theory Needlet spectra
need_theory = spectra.NeedletTheory(myanalysis.B)

# Computing simulated Cls 
print("...computing Betajs for simulations...")
fname_betaj_sims_TS_galT      = f'betaj_sims_TS_galT_jmax{jmax}_B_{myanalysis.B:0.2f}_nside{nside}.dat'


betaj_sims_TS_galT = myanalysis.GetBetajSimsFromMaps('T', nsim, field2='g1noise', fname=fname_betaj_sims_TS_galT, EuclidSims=True)

# Covariances
print("...computing Cov Matrices...")
fname_cov_TS_galT      = f'cov_TS_galT_jmax{jmax}_B_{myanalysis.B:0.2f}_nside{nside}.dat'

cov_TS_galT, corr_TS_galT          = myanalysis.GetCovMatrixFromMaps(field1='T', nsim=nsim, field2='g1noise', fname=fname_cov_TS_galT, fname_sims=fname_betaj_sims_TS_galT)

print("...done...")

# <Beta_j>_MC
betaj_TS_galT_mean    = myanalysis.GetBetajMeanFromMaps('T', nsim, field2='g1noise', fname_sims=fname_betaj_sims_TS_galT)


# Beta_j sims

[num_sim_1, num_sim_2] = np.random.choice(np.arange(nsim),2 )
beta_j_sim_1_T = betaj_sims_TS_galT[num_sim_1,:]
beta_j_sim_2_T = betaj_sims_TS_galT[num_sim_2,:]

# Beta_j THEORY

betatg    = need_theory.cl2betaj(jmax=jmax, cl=cl_theory_tg)
beta_norm = need_theory.cl2betaj(jmax=jmax, cl=np.ones(xcspectra.cltg.size))
delta = need_theory.delta_beta_j(jmax=jmax, cltg=cl_theory_tg, cltt=cl_theory_tt, clgg=cl_theory_gg)

np.savetxt(out_dir+f'theory_beta_jmax{jmax}_D{myanalysis.B:0.2f}_lmax{lmax}.dat', betatg )
np.savetxt(out_dir+f'theory_variance_jmax{jmax}_D{myanalysis.B:0.2f}_lmax{lmax}.dat' , delta)

########################################################################################################################

# Some plots
print("...here come the plots...")

# Covariances
#fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(25,18))
fig, axs = plt.subplots(ncols=1, nrows=1,figsize=(25,14))   
fig.suptitle(r'$B = %1.2f $' %myanalysis.B + r'$N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

mask_ = np.tri(corr_TS_galT.shape[0],corr_TS_galT.shape[1],0)

#plt.subplot(131)
axs.set_title(r'Corr $T^T\times gal^T$ Signal + shot noise')
sns.color_palette("crest", as_cmap=True)
sns.heatmap(corr_TS_galT, annot=True, fmt='.2f', cmap  = 'viridis',mask=mask_, ax=axs)

plt.tight_layout()
plt.savefig(cov_dir+f'total_corr_T_gal_noise_mask_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}.png')


# Sims mean spectrum + one sims spectrum TOTAL SIGNAL

fig, ax1 = plt.subplots(1,1,figsize=(25,14))

plt.suptitle(r'Total signal, $D = %1.2f $' %myanalysis.B + r'$ ,~N_{side} =$'+str(simparams['nside']) + r' $,~N_{sim} = $'+str(nsim))

#ax = fig.add_subplot(1, 1, 1)
ax1.set_title(r'$\beta_{j}T^T \times gal^T$')
ax1.errorbar(myanalysis.jvec, betaj_TS_galT_mean, yerr=np.sqrt(np.diag(cov_TS_galT))/np.sqrt(nsim-1),fmt='o', label = 'betaj mean')
ax1.errorbar(myanalysis.jvec, beta_j_sim_1_T, yerr=np.sqrt(np.diag(cov_TS_galT)), fmt='o', label = 'betaj sim')
ax1.set_ylabel(r'$\beta_j^{TG}$')
ax1.set_xlabel(r'j')
ax1.legend()

fig.tight_layout()
plt.savefig(out_dir+f'total-signal_betaj_mean_betaj_sim_plot_jmax_{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}.png')

# Theory + Sims Needlet power spectra - nel caso di mask devo moltiplicare per fsky il cl


fig = plt.figure(figsize=(27,20))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B + r'$ ~N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)
ax.errorbar(myanalysis.jvec, betatg,  yerr=delta, color='firebrick', fmt='o', ms=10,capthick=5, label=r'theory')
ax.errorbar(myanalysis.jvec, beta_j_sim_1_T, yerr=np.sqrt(np.diag(cov_TS_galT)), color='seagreen', fmt='o',ms=10,capthick=5, label=r'sim')
ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\beta_j^{Tgal}$')
fig.tight_layout()
plt.savefig(out_dir+f'sim{num_sim_1}_betaj_theory_T_gal_noise_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')

fig = plt.figure(figsize=(27,20))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B + r'$ ~N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)
ax.errorbar(myanalysis.jvec, betatg,  yerr=delta, color='firebrick', fmt='o', ms=10,capthick=5, label=r'theory')
ax.errorbar(myanalysis.jvec, beta_j_sim_2_T, yerr=np.sqrt(np.diag(cov_TS_galT)), color='seagreen', fmt='o',ms=10,capthick=5, label=r'sim')
ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\beta_j^{Tgal}$')
fig.tight_layout()
plt.savefig(out_dir+f'sim{num_sim_2}_betaj_theory_T_gal_noise_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')

#print(f'ratio beta mean mask / beta theory = {betaj_TS_galT_mask_mean/betatg}')
#print(f'ratio beta mean mask / beta mean = {betaj_TS_galT_mask_mean/betaj_TS_galT_mean}')
#print(f'ratio beta mean  / beta theory = {betaj_TS_galT_mean/betatg}')


###Comparison variance FULL SKY


fig = plt.figure(figsize=(27,20))

gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], wspace=0)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

plt.suptitle(r'$D = %1.2f $' %myanalysis.B + r'$ ~N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

ax0.errorbar(myanalysis.jvec[1:jmax], betaj_TS_galT_mean[1:jmax], yerr=delta[1:jmax]/(np.sqrt(nsim)), color='firebrick', fmt='o', ms=10,capthick=5, label=r'Variance from theory')
ax0.errorbar(myanalysis.jvec[1:jmax], betaj_TS_galT_mean[1:jmax], yerr=np.sqrt(np.diag(cov_TS_galT)[1:jmax])/np.sqrt(nsim), color='seagreen', fmt='o',ms=10,capthick=5, label=r'Variance from sim')
difference = (betaj_TS_galT_mean -betatg)/(betatg)      
ax1.errorbar(myanalysis.jvec[1:jmax], difference[1:jmax],yerr=delta[1:jmax]/(betatg[1:jmax]*np.sqrt(nsim) ),color='firebrick', fmt='o',  ms=10,capthick=5, label=r'Variance from theory')
ax1.errorbar(myanalysis.jvec[1:jmax], difference[1:jmax], yerr=np.sqrt(np.diag(cov_TS_galT)[1:jmax])/(betatg[1:jmax]*np.sqrt(nsim) ), color='seagreen', fmt='o',ms=10,capthick=5, label=r'Variance from sim')
ax1.axhline(ls='--', color='k')
ax1.set_ylabel(r'$(\langle \beta_j^{Tgal} \rangle - \beta_j^{Tgal, th})/\beta_j^{Tgal, th}$')

ax0.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.set_xlabel(r'$j$')
ax0.set_ylabel(r'$\beta_j^{Tgal}$')

fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(out_dir+f'betaj_theory_T_gal_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}.png', bbox_inches='tight')


#### RELATIVE DIFFERENCE THEORY + SIMULATIONS FULL SKY

fig = plt.figure(figsize=(27,20))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B + r'$ ~N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.axhline(ls='--', color='k')
ax.errorbar(myanalysis.jvec[1:jmax], (betaj_TS_galT_mean[1:jmax] )/(betatg[1:jmax])-1, yerr=delta[1:jmax]/(np.sqrt(nsim)*betatg[1:jmax]), color='firebrick', fmt='o',  ms=10,capthick=5, label=r'Variance from theory')
ax.errorbar(myanalysis.jvec[1:jmax], (betaj_TS_galT_mean[1:jmax] )/(betatg[1:jmax])-1, yerr=np.sqrt(np.diag(cov_TS_galT)[1:jmax])/(np.sqrt(nsim)*betatg[1:jmax]), color='seagreen', fmt='o', ms=10,capthick=5, label=r'Variance from sim')

ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\langle \beta_j^{Tgal} \rangle/\beta_j^{Tgal, th}$-1')# - \beta_j^{Tgal, th}
#ax.set_ylim([-0.3,1.3])

fig.tight_layout()
plt.savefig(out_dir+f'betaj_mean_T_gal_noise_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}.png', bbox_inches='tight')


## One sims spectrum TOTAL SIGNAL +  SIGNAL ONLY MASK
#
#fig, ax2 = plt.subplots(1,1,figsize=(25,14), sharey= True)
#
#plt.suptitle(r'$D = %1.2f $' %myanalysis.B + r'$ ,~N_{side} =$'+str(simparams['nside']) + r' $,~N_{sim} = $'+str(nsim))
#
#
#ax2.set_title(r'$\beta_{j}~T^T \times gal^T$ Signal + shot noise')
#ax2.errorbar(myanalysis.jvec-0.15, beta_j_sim_1_T, yerr=np.sqrt(np.diag(cov_TS_galT)), fmt='o',ms=10,capthick=5, color='firebrick',label = 'No mask')
#ax2.set_ylabel(r'$\beta_j^{TG}$')
#ax2.set_xlabel(r'j')
#ax2.legend()
#
#fig.tight_layout()
#plt.savefig(out_dir+f'mask_signal_noise_betaj_mean_betaj_sim_plot_jmax_{jmax}_D = %1.2f ' %myanalysis.B +f'_nsim_{nsim}_nside'+str(simparams['nside'])+'_mask.png')

#darkorange
## PLOT STUPIDO
#clj=need_theory.cl_binned(jmax, cl=xcspectra.cltg )
#
#fig = plt.figure(figsize=(27,20))
#
#plt.suptitle(r'$D = %1.2f $' %myanalysis.B + r'$ ~N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))
#
#ax = fig.add_subplot(1, 1, 1)
#
#ax.plot(myanalysis.jvec[1:jmax],  betatg[1:jmax]/beta_norm[1:jmax] , color='firebrick', marker='o',  ms=10, label='beta norm')
#ax.plot(myanalysis.jvec[1:jmax],  clj[1:jmax] , color='seagreen', marker='o',  ms=10, label='cl')
#
#ax.legend(loc='best')
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#ax.set_xlabel(r'$j$')
#ax.set_ylabel(r'$\beta_j^{Tgal, th}/\beta_{j,norm}^{Tgal, th}$')# - \beta_j^{Tgal, th}
##ax.set_ylim([-0.3,1.3])
#fig.tight_layout()
#plt.savefig(out_dir+f'beta_TG_noise_norm_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}.png', bbox_inches='tight')
