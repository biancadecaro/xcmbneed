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
simparams = {'nside'   : 128,#512,
             'ngal'    : 5.76e5, #dovrebbe importare solo per lo shot noise (noise poissoniano)
 	     	 'ngal_dim': 'ster',
	     	 'pixwin'  : False}
nside = simparams['nside']
# Paths
fname_xcspectra = 'spectra/inifiles/CAMBSpectra_planck_fiducial_lmin0_2050.dat'#'spectra/inifiles/CAMBSpectra_planck.dat' 
sims_dir        = f'sims/Needlet/Planck/TGsims_{nside}/'#planck_2_lmin0_prova_1/'
out_dir         = f'output_needlet_TG/Planck/TG_{nside}/'#planck_2_lmin0_prova_1/'
path_inpainting = 'inpainting/inpainting.py'
cov_dir 		= f'covariance/Planck/covariance_TG{nside}/'#planck_2_lmin0_prova_1/'
if not os.path.exists(cov_dir):
        os.makedirs(cov_dir)

jmax = 12
lmax = 256#782
nsim = 1000#500
#B = 1.95

#jmax = round(np.log(lmax*1.00000000000000222045e+00)/np.log(B))
#lmax = round(B**(jmax+1))
#mylibc.debug_needlets()

#mask = utils.GetGalMask(simparams['nside'], lat=20.)
#fsky = np.mean(mask)

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
B=myanalysis.B

# Computing simulated Cls 
print("...computing Betajs for simulations...")
fname_betaj_sims_TS_galS      = f'betaj_sims_TS_galS_jmax{jmax}_B{B:1.2f}_nside{nside}.dat'
#fname_betaj_sims_galS_galS      = 'betaj_sims_galS_galS.dat'

betaj_sims_TS_galS = myanalysis.GetBetajSimsFromMaps('TS', nsim, field2='galS', fname=fname_betaj_sims_TS_galS)
#betaj_sims_G_G = myanalysis.GetBetajSimsFromMaps('galS', nsim, field2='galS', fname=fname_betaj_sims_G_G)

# Covariances
print("...computing Cov Matrices...")
fname_cov_TS_galS      = f'cov_TS_galS_jmax{jmax}_B{B:1.2f}_nside{nside}.dat'
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
fig, ax1 = plt.subplots(1,1,figsize=(10,10))   
#fig.suptitle(r'$B = %1.2f $' %myanalysis.B + r'$N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))
plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))


mask_ = np.tri(corr_TS_galS.shape[0],corr_TS_galS.shape[1],0)

#plt.subplot(131)
ax1.set_title(r'Correlation T$\times$G')
sns.color_palette("crest", as_cmap=True)
sns.heatmap(corr_TS_galS, annot=True, fmt='.2f', annot_kws={"size": 10},cmap  = 'crest', ax=ax1, mask=mask_)

##plt.subplot(132)
#ax2.set_title(r'Corr $\delta^T \times \gal^T$')
#sns.heatmap(corr_kappaT_deltaT, annot=True, fmt='.2f', mask=mask_, ax=ax2)
#
##plt.subplot(133)
#ax3.set_title(r'Corr $\delta^T \times \kappa^T$ Masked')
#sns.heatmap(corr_kappaT_deltaT_mask, annot=True, fmt='.2f', mask=mask_,ax=ax3)
fig.tight_layout()
#plt.savefig(out_dir+f'corr_TS_galS_jmax{jmax}_B{B:1.2f}_nsim{nsim}_nside{nside}.png')
#plt.savefig(f'plot_tesi/PLANCK_VALIDATION_fullsky_corr_TS_galS_jmax{jmax}_B{B:1.2f}_nsim{nsim}_nside{nside}.pdf')
plt.show()
# Theory + Normalization Needlet power spectra

betatg    = need_theory.cl2betaj(jmax=jmax, cl=xcspectra.cltg ,lmax=lmax)
beta_norm = need_theory.cl2betaj(jmax=jmax, cl=np.ones(xcspectra.cltg.size), lmax=lmax)

np.savetxt(out_dir+f'beta_TS_galS_theoretical_fiducial_B{myanalysis.B}.dat', betatg)

delta = need_theory.delta_beta_j(jmax=jmax, lmax=lmax,cltt = xcspectra.cltt, cltg = xcspectra.cltg, clgg = xcspectra.clg1g1)

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.plot(myanalysis.jvec[1:jmax+1], betatg[1:jmax+1], label='Theory')
ax.errorbar(myanalysis.jvec[1:jmax+1], betaj_TS_galS_mean[1:jmax+1], yerr=np.sqrt(np.diag(cov_TS_galS)[1:jmax+1])/np.sqrt(nsim) ,color='#2b7bbc',fmt='o',ms=5,capthick=2, label='Error of the mean of the simulations')
ax.errorbar(myanalysis.jvec[1:jmax+1], betaj_TS_galS_mean[1:jmax+1], yerr=np.sqrt(np.diag(cov_TS_galS)[1:jmax+1]) ,color='grey',fmt='o',ms=0,capthick=2, label='Error of simulations')

ax.legend(loc='best')
ax.set_xticks(myanalysis.jvec[1:jmax+1])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\beta_j^{\mathrm{TG}}$')

fig.tight_layout()
#plt.savefig(out_dir+f'betaj_theory_T_gal_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')
#plt.savefig(f'plot_tesi/PLANCK_VALIDATION_TEST_betaj_mean_theory_jmax_{jmax}_B{B:1.2f}_nsim{nsim}_nside{nside}.pdf')
plt.show()

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.axhline(ls='--', color='grey')
ax.errorbar(myanalysis.jvec[1:jmax+1], (betaj_TS_galS_mean[1:jmax+1] -betatg[1:jmax+1])/betatg[1:jmax+1], yerr=np.sqrt(np.diag(cov_TS_galS)[1:jmax+1])/(np.sqrt(nsim)*betatg[1:jmax+1]), fmt='o',ms=5, label=r'Variance on the mean from simulations')#, label=r'$T^S \times gal^S$, sim cov')
ax.errorbar(myanalysis.jvec[1:jmax+1], (betaj_TS_galS_mean[1:jmax+1] -betatg[1:jmax+1])/betatg[1:jmax+1], yerr=delta[1:jmax+1]/(np.sqrt(nsim)*betatg[1:jmax+1]),color='#6d7e3f',  fmt='o',  ms=5,label=r'Variance from theory')
#print(np.sqrt(np.diag(cov_TS_galS))/(np.sqrt(nsim)*betatg) - delta/(np.sqrt(nsim)*betatg))
print('perc diff covariance:',100*(betaj_TS_galS_mean[1:jmax] -betatg[1:jmax])/betatg[1:jmax])

ax.set_xticks(myanalysis.jvec[1:jmax+1])
ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xlabel(r'$j$', fontsize=22)
ax.set_ylabel(r'$\frac{\langle \beta_j^{\mathrm{TG}} \rangle - \beta_j^{\mathrm{TG}\,, th}}{\beta_j^{\mathrm{TG}\,, th}}$', fontsize=22)
ax.set_ylim([-0.2,0.3])

fig.tight_layout()
#plt.savefig(out_dir+f'betaj_mean_T_gal_jmax{jmax}_D{B:1.2f}_nsim{nsim}_nside{nside}.png', bbox_inches='tight')
#plt.savefig(f'plot_tesi/PLANCK_VALIDATION_betaj_ratio_mean_theory_jmax_{jmax}_B{B:1.2f}_nsim{nsim}_nside{nside}.pdf')
plt.show()

#Difference divided one sigma

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.axhline(ls='--', color='grey')
ax.plot(myanalysis.jvec[1:jmax+1], (betaj_TS_galS_mean[1:jmax+1] -betatg[1:jmax+1])/(delta[1:jmax+1]/np.sqrt(nsim)),'o', ms=10,color='#2b7bbc')#, label=r'$T^S \times gal^S$, sim cov')
#print(np.sqrt(np.diag(cov_TS_galS))/(np.sqrt(nsim)*betatg) - delta/(np.sqrt(nsim)*betatg))

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(myanalysis.jvec[1:jmax+1])
ax.set_xlabel(r'$j$', fontsize=22)
ax.set_ylabel(r'$\Delta \beta_j^{\mathrm{TG}} / \sigma $', fontsize=22)
ax.set_ylim([-2.0,2.0])

fig.tight_layout()
#plt.savefig(out_dir+f'diff_betaj_mean_theory_over_sigma_T_gal_jmax{jmax}_D{B:1.2f}_nsim{nsim}_nside{nside}.png', bbox_inches='tight')
#plt.savefig(f'plot_tesi/PLANCK_VALIDATION_diff_betaj_mean_theory_over_sigma_jmax_{jmax}_B{B:1.2f}_nsim{nsim}_nside{nside}.pdf')
plt.show()
######################################################################################
############################## SIGNAL TO NOISE RATIO #################################
def S_2_N(beta, cov_matrix):
    s_n = np.zeros(len(beta))
    cov_inv = np.linalg.inv(cov_matrix)
    temp = np.zeros(len(cov_matrix[0]))
    for i in range(len(cov_matrix[0])):
        for j in range(len(beta)):
            #s_n[i] += np.dot(beta[i], np.dot(cov_inv[i, j], beta[j]))
            temp[i] += cov_inv[i][j]*beta[j]
        s_n[i] = beta[i].T*temp[i]
    return s_n

def S_2_N_th(beta, variance):
    s_n = np.divide((beta)**2, variance)
    return s_n

def S_2_N_cum(s2n, jmax):
    s2n_cum = np.zeros(jmax.shape[0])
    for j,jj in enumerate(jmax):
        for ijj in range(1,jj):
            s2n_cum[j] +=s2n[ijj]
        s2n_cum[j]= np.sqrt(s2n_cum[j])      
    return s2n_cum

jvec = np.arange(0,jmax+1)

s2n_theory=S_2_N_th(betatg, delta**2)
s2n_mean_sim=S_2_N_th(betaj_TS_galS_mean, delta**2)

s2n_mean_sim_cov=S_2_N(betaj_TS_galS_mean, cov_TS_galS)
print(s2n_mean_sim_cov)

s2n_cum_theory = S_2_N_cum(s2n_theory, jvec)
s2n_cum_sim_cov = S_2_N_cum(s2n_mean_sim_cov, jvec)
s2n_cum_sim = S_2_N_cum(s2n_mean_sim, jvec)
#print(np.where(s2n_theory==s2n_theory.max()),np.where(betatg==betatg.max()) )
#print(betaj_TS_galS_mean)
print(f's2n_cum_theory={s2n_cum_theory}', f's2n_cum_sim={s2n_cum_sim}', f's2n_cum_sim_cov={s2n_cum_sim_cov}')

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.plot(myanalysis.jvec[1:jmax], s2n_theory[1:jmax], color='#2b7bbc',marker = 'o',ms=10, label='Theory')
ax.plot(myanalysis.jvec[1:jmax], s2n_mean_sim[1:jmax],marker = 'o',ms=10, label='Simulations')

ax.set_xticks(myanalysis.jvec[1:jmax])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
plt.legend()
ax.set_xlabel(r'$j$')
ax.set_ylabel('Signal-to-Noise ratio')

fig.tight_layout()
#plt.savefig(out_dir+f'SNR_betaj_theory_T_gal_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')
#plt.savefig(f'plot_tesi/PLANCK_VALIDATION_SNR_betaj_theory_jmax_{jmax}_B{B:1.2f}_nsim{nsim}_nside{nside}.pdf')
