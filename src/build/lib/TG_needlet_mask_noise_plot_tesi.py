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
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['font.size'] = 20
plt.rcParams['lines.linewidth']  = 3.
##plt.rcParams['backend'] = 'WX'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# embed()

#plt.rcParams['backend'] = 'WX'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# embed()
import os
path = os.path.abspath(spectra.__file__)
# Parameters
simparams = {'nside'   : 512,
             'ngal'    : 5.76e5, #dovrebbe importare solo per lo shot noise (noise poissoniano)
 	     	 'ngal_dim': 'ster',
	     	 'pixwin'  : False}
nside = simparams['nside']
# Paths
fname_xcspectra = '/ehome/bdecaro/xcmbneed/src/spectra/inifiles/CAMBSpectra_planck_fiducial_lmin0_2050.dat'#'spectra/inifiles/CAMBSpectra_planck.dat' 
sims_dir        = f'/ehome/bdecaro/xcmbneed/src/sims/Needlet/Planck/TGsims_{nside}/'
out_dir         = f'output_needlet_TG/Planck/Mask_noise/TG_{nside}_tesi_NO_fsky_approx/'
path_inpainting = 'inpainting/inpainting.py'
#cov_dir 		= f'covariance/Planck/Mask_noise/TG_{nside}_tesi/covariance_TG{nside}_tesi/'
#if not os.path.exists(cov_dir):
#        os.makedirs(cov_dir)

jmax = 12
lmax = 782
nsim = 500


mask = hp.read_map(f'/ehome/bdecaro/xcmbneed/src/mask/mask_planck_comm_2018_nside={nside}.fits')
fsky = np.mean(mask)
print(fsky)


# Loading theory spectra
xcspectra = spectra.XCSpectraFile(fname_xcspectra, WantTG = True)
Nll = np.ones(xcspectra.clg1g1.shape[0])/simparams['ngal']
# Simulations class
simulations = sims.KGsimulations(xcspectra, sims_dir, simparams, WantTG = True)
simulations.Run(nsim, WantTG = True)

# Needlet Analysis
myanalysis = analysis.NeedAnalysis(jmax, lmax, out_dir, simulations)

# Theory Needlet spectra
need_theory = spectra.NeedletTheory(myanalysis.B)
print(myanalysis.B)
B=myanalysis.B

# Computing simulated betajs
print("...computing Betajs for simulations...")
fname_betaj_sims_TS_galS      = f'betaj_sims_TS_galS_jmax{jmax}_B_{myanalysis.B}_nside{nside}.dat'
fname_betaj_sims_TS_galT      = f'betaj_sims_TS_galT_jmax{jmax}_B_{myanalysis.B}_nside{nside}.dat'
fname_betaj_sims_TS_galS_mask      = f'betaj_sims_TS_galS_jmax{jmax}_B_{myanalysis.B}_nside{nside}_fsky_{fsky}.dat'
fname_betaj_sims_TS_galT_mask      = f'betaj_sims_TS_galT_jmax{jmax}_B_{myanalysis.B}_nside{nside}_fsky_{fsky}.dat'


betaj_sims_TS_galS = myanalysis.GetBetajSimsFromMaps('TS', nsim, field2='galS', fname=fname_betaj_sims_TS_galS)
betaj_sims_TS_galT = myanalysis.GetBetajSimsFromMaps('TS', nsim, field2='galT', fname=fname_betaj_sims_TS_galT)
betaj_sims_TS_galS_mask = myanalysis.GetBetajSimsFromMaps('TS', nsim, field2='galS', mask= mask, fname=fname_betaj_sims_TS_galS_mask)
betaj_sims_TS_galT_mask = myanalysis.GetBetajSimsFromMaps('TS', nsim, field2='galT', mask= mask, fname=fname_betaj_sims_TS_galT_mask)


# Covariances
print("...computing Cov Matrices...")
fname_cov_TS_galS      = f'cov_TS_galS_jmax{jmax}_B = %1.2f $' %myanalysis.B +'_nside'+str(simparams['nside'])+'.dat'
fname_cov_TS_galT      = f'cov_TS_galT_jmax{jmax}_B = %1.2f $' %myanalysis.B +'_nside'+str(simparams['nside'])+'.dat'
fname_cov_TS_galS_mask      = f'cov_TS_galS_jmax{jmax}_B = %1.2f $' %myanalysis.B +'_nside'+str(simparams['nside'])+'_mask.dat'
fname_cov_TS_galT_mask      = f'cov_TS_galT_jmax{jmax}_B = %1.2f $' %myanalysis.B +'_nside'+str(simparams['nside'])+'_mask.dat'



cov_TS_galS, corr_TS_galS          = myanalysis.GetCovMatrixFromMaps(field1='TS', nsim=nsim, field2='galS', fname=fname_cov_TS_galS, fname_sims=fname_betaj_sims_TS_galS)
cov_TS_galT, corr_TS_galT          = myanalysis.GetCovMatrixFromMaps(field1='TS', nsim=nsim, field2='galT', fname=fname_cov_TS_galT, fname_sims=fname_betaj_sims_TS_galT)
cov_TS_galS_mask, corr_TS_galS_mask          = myanalysis.GetCovMatrixFromMaps(field1='TS', nsim=nsim, field2='galS', fname=fname_cov_TS_galS_mask, mask = mask,fname_sims=fname_betaj_sims_TS_galS_mask)
cov_TS_galT_mask, corr_TS_galT_mask          = myanalysis.GetCovMatrixFromMaps(field1='TS', nsim=nsim, field2='galT', fname=fname_cov_TS_galT_mask, mask = mask,fname_sims=fname_betaj_sims_TS_galT_mask)

print("...done...")

# <Beta_j>_MC
betaj_TS_galS_mean    = myanalysis.GetBetajMeanFromMaps('TS', nsim, field2='galS', fname_sims=fname_betaj_sims_TS_galS)
betaj_TS_galT_mean    = myanalysis.GetBetajMeanFromMaps('TS', nsim, field2='galT', fname_sims=fname_betaj_sims_TS_galT)
betaj_TS_galS_mask_mean     = myanalysis.GetBetajMeanFromMaps('TS', nsim, field2='galS', fname_sims=fname_betaj_sims_TS_galS_mask)
betaj_TS_galT_mask_mean     = myanalysis.GetBetajMeanFromMaps('TS', nsim, field2='galT', fname_sims=fname_betaj_sims_TS_galT_mask)


# Some plots
print("...here come the plots...")

# Covariances
#fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(25,18))
fig, axs = plt.subplots(ncols=2, nrows=2,figsize=(38,23)) 
#fig.suptitle(r'$B = %1.2f $' %myanalysis.B + r'$N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))
plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))
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
axs[1,0].set_title(r'Corr $T^S\times gal^T$')
sns.heatmap(corr_TS_galT, annot=True, fmt='.2f',cmap  = 'viridis', mask=mask_, ax=axs[1,0])

axs[1,1].set_title(r'Corr $T^S\times gal^T$ Masked')
sns.heatmap(corr_TS_galT_mask, annot=True, fmt='.2f',cmap  = 'viridis', mask=mask_, ax=axs[1,1])


fig.tight_layout()
#plt.savefig(cov_dir+'corr_TS_galS_jmax'+str(jmax)+'_B = %1.2f ' %myanalysis.B +'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'_mask.png')
plt.savefig(out_dir+'corr_T_gal_noise_mask_jmax'+str(jmax)+'_B = %1.2f ' %myanalysis.B +'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'_mask.png')

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
plt.savefig(out_dir+'signal_only_corr_T_gal_noise_mask_jmax'+str(jmax)+'_B = %1.2f ' %myanalysis.B +'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'_mask.png')

fig, axs = plt.subplots(ncols=2, nrows=1,figsize=(52,25))   
fig.suptitle(r'$B = %1.2f $' %myanalysis.B + r'$N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

mask_ = np.tri(corr_TS_galT.shape[0],corr_TS_galT.shape[1],0)

#plt.subplot(131)
axs[0].set_title(r'Corr $T^T\times gal^T$ Signal + shot noise')
sns.color_palette("crest", as_cmap=True)
sns.heatmap(corr_TS_galT, annot=True, fmt='.2f', cmap  = 'viridis',mask=mask_, ax=axs[0])

##plt.subplot(132)
axs[1].set_title(r'Corr $T^T\times gal^T$ Signal + shot noise Masked')
sns.heatmap(corr_TS_galT_mask, annot=True, fmt='.2f',cmap  = 'viridis', mask=mask_, ax=axs[1])
#
plt.tight_layout()
plt.savefig(out_dir+'total_corr_T_gal_noise_mask_jmax'+str(jmax)+'_B = %1.2f ' %myanalysis.B +'_nsim'+str(nsim)+'_nside'+str(simparams['nside'])+'_mask.png')

########################################################################################################################################################################################

# Theory + Normalization Needlet power spectra

betatg    = need_theory.cl2betaj(jmax=jmax, cl=xcspectra.cltg)
beta_norm = need_theory.cl2betaj(jmax=jmax, cl=np.ones(xcspectra.cltg.size))

np.savetxt(out_dir+f'beta_TS_galS_theoretical_fiducial_B{myanalysis.B}.dat', betatg)

delta = need_theory.delta_beta_j(jmax, cltt = xcspectra.cltt, cltg = xcspectra.cltg, clgg = xcspectra.clg1g1)
delta_noise = need_theory.delta_beta_j(jmax=jmax, cltg=xcspectra.cltg, cltt=xcspectra.cltt, clgg=xcspectra.clg1g1, noise_gal_l=Nll)

np.savetxt(out_dir+f'variance_beta_TS_galS_theoretical_fiducial_B{myanalysis.B}_noise.dat', delta_noise)

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.plot(myanalysis.jvec[1:jmax], betatg[1:jmax], label='Theory')
ax.errorbar(myanalysis.jvec[1:jmax], betaj_TS_galT_mask_mean[1:jmax], yerr=np.sqrt(np.diag(cov_TS_galT_mask)[1:jmax])/np.sqrt(nsim) ,color='#2b7bbc',fmt='o',ms=5,capthick=2, label='Error of the mean of the simulations')
ax.errorbar(myanalysis.jvec[1:jmax], betaj_TS_galT_mask_mean[1:jmax], yerr=np.sqrt(np.diag(cov_TS_galT_mask)[1:jmax]) ,color='grey',fmt='o',ms=0,capthick=2, label='Error of simulations')

ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\beta_j^{\mathrm{TG}}$')

fig.tight_layout()
plt.savefig(out_dir+f'betaj_theory_T_gal_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')
#plt.savefig(f'plot_tesi/PLANCK_VALIDATION_mask_noise_betaj_mean_theory_jmax_{jmax}_B{B:1.2f}_nsim{nsim}_nside{nside}.pdf')

####################################################################################################################################################################

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.axhline(ls='--', color='grey')
ax.errorbar(myanalysis.jvec[1:jmax], (betaj_TS_galT_mean[1:jmax] -betatg[1:jmax])/betatg[1:jmax], yerr=np.sqrt(np.diag(cov_TS_galT)[1:jmax])/(np.sqrt(nsim)*betatg[1:jmax]), fmt='o',ms=5, label=r'Full sky')#, label=r'$T^S \times gal^S$, sim cov')
ax.errorbar(myanalysis.jvec[1:jmax], (betaj_TS_galT_mask_mean[1:jmax] -betatg[1:jmax])/betatg[1:jmax], yerr=np.sqrt(np.diag(cov_TS_galT_mask)[1:jmax])/(np.sqrt(nsim)*betatg[1:jmax]),color='#6d7e3f',  fmt='o',  ms=5,label=r'Mask')
#print(np.sqrt(np.diag(cov_TS_galS))/(np.sqrt(nsim)*betatg) - delta/(np.sqrt(nsim)*betatg))

ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xlabel(r'$j$', fontsize=22)
ax.set_ylabel(r'$\frac{\langle \beta_j^{\mathrm{TG}} \rangle - \beta_j^{\mathrm{TG}\,, th}}{\beta_j^{\mathrm{TG}\,, th}}$', fontsize=22)
#ax.set_ylim([-0.2,0.3])

fig.tight_layout()
plt.savefig(out_dir+f'betaj_mean_T_gal_jmax{jmax}_D{B:1.2f}_nsim{nsim}_nside{nside}_full_sky_mask.png', bbox_inches='tight')
#plt.savefig(f'plot_tesi/PLANCK_VALIDATION_MASK_betaj_ratio_mean_theory_jmax_{jmax}_B{B:1.2f}_nsim{nsim}_nside{nside}.pdf')

###################################################################################################################################################################################

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.axhline(ls='--', color='grey')
ax.errorbar(myanalysis.jvec[1:jmax], (betaj_TS_galS_mean[1:jmax] -betatg[1:jmax])/betatg[1:jmax], yerr=np.sqrt(np.diag(cov_TS_galS)[1:jmax])/(np.sqrt(nsim)*betatg[1:jmax]), fmt='o',ms=5, label=r'No shot noise')#, label=r'$T^S \times gal^S$, sim cov')
ax.errorbar(myanalysis.jvec[1:jmax], (betaj_TS_galT_mean[1:jmax] -betatg[1:jmax])/betatg[1:jmax], yerr=np.sqrt(np.diag(cov_TS_galT)[1:jmax])/(np.sqrt(nsim)*betatg[1:jmax]),color='#6d7e3f',  fmt='o',  ms=5,label=r'Shot Noise')
#print(np.sqrt(np.diag(cov_TS_galS))/(np.sqrt(nsim)*betatg) - delta/(np.sqrt(nsim)*betatg))

ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xlabel(r'$j$', fontsize=22)
ax.set_ylabel(r'$\frac{\langle \beta_j^{\mathrm{TG}} \rangle - \beta_j^{\mathrm{TG}\,, th}}{\beta_j^{\mathrm{TG}\,, th}}$', fontsize=22)
#ax.set_ylim([-0.2,0.3])

fig.tight_layout()
plt.savefig(out_dir+f'betaj_mean_T_gal_jmax{jmax}_D{B:1.2f}_nsim{nsim}_nside{nside}_shot_noise.png', bbox_inches='tight')
#plt.savefig(f'plot_tesi/PLANCK_VALIDATION_NOISE_betaj_ratio_mean_theory_jmax_{jmax}_B{B:1.2f}_nsim{nsim}_nside{nside}.pdf')


#######################################################################################################################


# One sims spectrum TOTAL SIGNAL +  SIGNAL ONLY
num_sim =23
beta_j_one_sim_S = betaj_sims_TS_galS[num_sim,:]
beta_j_one_sim_T = betaj_sims_TS_galT[num_sim,:]
beta_j_one_sim_S_mask = betaj_sims_TS_galS_mask[num_sim,:]
beta_j_one_sim_T_mask = betaj_sims_TS_galT_mask[num_sim,:]

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(17,10), sharey = True)

plt.suptitle(r'$B = %1.2f $' %myanalysis.B + r'$ ,~N_{side} =$'+str(simparams['nside']) + r' $,~N_{sim} = $'+str(nsim))

#ax = fig.add_subplot(1, 1, 1)
ax1.set_title('No Masked')#(r'$\beta_{j}T^S \times gal^S$')
ax1.errorbar(myanalysis.jvec-0.15, beta_j_one_sim_S, yerr=np.sqrt(np.diag(cov_TS_galS)), fmt='o',ms=2,capthick=2, label = 'Signal only')
ax1.errorbar(myanalysis.jvec-0.15, beta_j_one_sim_T, yerr=np.sqrt(np.diag(cov_TS_galT)), fmt='o',ms=2,capthick=2, label = 'Signal + Shot Noise')

ax1.yaxis.set_major_formatter(formatter) 
ax1.set_ylabel(r'$\beta_j^{TG}$')
ax1.set_xlabel(r'j')
ax1.legend()

ax2.set_title('Masked')#(r'$\beta_{j}T^T \times gal^T$ ')
ax2.errorbar(myanalysis.jvec-0.15, beta_j_one_sim_S_mask, yerr=np.sqrt(np.diag(cov_TS_galS_mask)), fmt='o',ms=2,capthick=2, label = 'Signal only')
ax2.errorbar(myanalysis.jvec-0.15, beta_j_one_sim_T_mask, yerr=np.sqrt(np.diag(cov_TS_galT_mask)), fmt='o',ms=2,capthick=2, label = 'Signal + Shot Noise')
ax2.yaxis.set_major_formatter(formatter) 
ax2.set_ylabel(r'$\beta_j^{TG}$')
ax2.set_xlabel(r'j')
ax2.legend()

fig.tight_layout()
plt.savefig(out_dir+f'signal_noise_betaj_mean_betaj_sim_plot_jmax_{jmax}_B = %1.2f ' %myanalysis.B +f'_nsim_{nsim}_nside'+str(simparams['nside'])+'_mask.png')

#############################################

# PROVA MASTER
wl = hp.anafast(mask, lmax=lmax)
mll  = need_theory.get_Mll(wl, lmax=lmax)

###############
#MASTER NEEDLETS

gammaJ_tg = need_theory.gammaJ(xcspectra.cltg, wl, jmax, lmax)
delta_gammaj = need_theory.variance_gammaj(cltg=xcspectra.cltg,cltt=xcspectra.cltt, clgg=xcspectra.clg1g1, wl=wl, jmax=jmax, lmax=lmax, noise_gal_l=Nll)

np.savetxt(out_dir+f'gammaj_theory_B{myanalysis.B:1.2f}_jmax{jmax}_nside{nside}_lmax{lmax}_nsim{nsim}_fsky{fsky:1.2f}.dat', gammaJ_tg)
np.savetxt(out_dir+f'variance_gammaj_theory_B{myanalysis.B:1.2f}_jmax{jmax}_nside{nside}_lmax{lmax}_nsim{nsim}_fsky{fsky:1.2f}.dat', delta_gammaj)

##RELATIVE DIFFERENCE
fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim)+' MASK')

ax = fig.add_subplot(1, 1, 1)

ax.axhline(ls='--', color='grey')
ax.errorbar(myanalysis.jvec[1:jmax+1], (betaj_TS_galT_mask_mean[1:jmax+1]*fsky /gammaJ_tg[1:jmax+1]-1), yerr=np.sqrt(np.diag(delta_gammaj)[1:jmax+1])/(np.sqrt(nsim)*gammaJ_tg[1:jmax+1]),  fmt='o', ms=5, label=r'Error of the mean of the simulations')
ax.errorbar(myanalysis.jvec[1:jmax+1], (betaj_TS_galT_mask_mean[1:jmax+1]*fsky /gammaJ_tg[1:jmax+1]-1), yerr=np.sqrt(np.diag(cov_TS_galT_mask)[1:jmax+1]*fsky)/(np.sqrt(nsim)*gammaJ_tg[1:jmax+1]),color='#2b7bbc',  fmt='o', ms=5, label=r'Error of simulations')

ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\langle \tilde{\Gamma}_j^{TG} \rangle/\tilde{\Gamma}_j^{TG, th}$-1')# - \beta_j^{Tgal, th}
#ax.set_ylim([-0.3,1.3])

fig.tight_layout()
plt.savefig(out_dir+f'gammaj_mean_T_gal_noise_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')
#plt.savefig(out_dir+f'gammaj_mean_T_gal_noise_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.pdf', bbox_inches='tight')

### SPECTRUM  CUT SKY

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.plot(myanalysis.jvec[1:jmax+1], gammaJ_tg[1:jmax+1], label='Theory')
ax.errorbar(myanalysis.jvec[1:jmax+1], betaj_TS_galT_mask_mean[1:jmax+1]*fsky, yerr=np.sqrt(np.diag(cov_TS_galT_mask)[1:jmax+1]*fsky)/np.sqrt(nsim) ,color='#2b7bbc',fmt='o',ms=5,capthick=2, label='Error of the mean of the simulations')
ax.errorbar(myanalysis.jvec[1:jmax+1], betaj_TS_galT_mask_mean[1:jmax+1]*fsky, yerr=np.sqrt(np.diag(cov_TS_galT_mask)[1:jmax+1]*fsky) ,color='grey',fmt='o',ms=0,capthick=2, label='Error of simulations')

ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\tilde{\Gamma}^{\mathrm{\,TG}}_j$')

fig.tight_layout()
plt.savefig(out_dir+f'gammaj_theory_T_gal_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')
#plt.savefig(out_dir+f'gammaj_theory_T_gal_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.pdf', bbox_inches='tight')

diff_cov_mask = np.abs((np.sqrt(np.diag(cov_TS_galT_mask))-delta_noise))/delta_noise*100
diff_cov_mean_mask = np.mean(diff_cov_mask[1:jmax+1])
print(f'diff cov mask={diff_cov_mask}, diff cov mean={diff_cov_mean_mask}')

######################################################################################
############################## SIGNAL TO NOISE RATIO #################################
def S_2_N(beta, cov_matrix):

    cov_inv = np.linalg.inv(cov_matrix)
    s_n = 0
    temp = np.zeros(len(cov_matrix[0]))
    for i in range(len(cov_matrix[0])):
        for j in range(len(beta)):
            temp[i] += cov_inv[i][j]*beta[j]
        s_n += beta[i].T*temp[i]
    return np.sqrt(s_n)

def S_2_N_th(beta, variance):
    s_n = np.divide((beta)**2, variance)
    return np.sqrt(s_n)

s2n_theory=S_2_N_th(betatg, delta_noise**2)
s2n_mean_sim=S_2_N_th(betaj_TS_galT_mask_mean, delta_noise**2)


print(np.where(s2n_theory==s2n_theory.max()),np.where(betatg==betatg.max()) )
print(betaj_TS_galS_mean)

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.plot(myanalysis.jvec, s2n_theory, color='#2b7bbc',marker = 'o',ms=10, label='Theory')
ax.plot(myanalysis.jvec, s2n_mean_sim,marker = 'o',ms=10, label='Simulations')


plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
plt.legend()
ax.set_xlabel(r'$j$')
ax.set_ylabel('Signal-to-Noise ratio')

fig.tight_layout()
plt.savefig(out_dir+f'SNR_betaj_theory_T_gal_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')
plt.savefig(f'plot_tesi/Euclid/SNR_betaj_theory_jmax_{jmax}_B{B:1.2f}_nsim{nsim}_nside{nside}.pdf')
