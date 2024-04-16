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
import random


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
             'ngal'    : 354543085.80126834, #dovrebbe importare solo per lo shot noise (noise poissoniano)
 	     	 'ngal_dim': 'ster',
	     	 'pixwin'  : False}

nside = simparams['nside']

jmax = 12
lmax = 256
nsim = 1000
delta_ell =1


# Paths
fname_xcspectra = '/ehome/bdecaro/xcmbneed/src/spectra/inifiles/EUCLID_fiducial_lmin0.dat'
sims_dir        = f'/ehome/bdecaro/xcmbneed/src/sims/Euclid_sims_Bianca/TG_{nside}_noiseEuclid_nsim{nsim}/'
out_dir         = f'output_needlet_TG/EUCLID/Mask_noise/TG_{nside}_nsim{nsim}_Bianca_mask_noiseEuclid/'
path_inpainting = 'inpainting/inpainting.py'
cov_dir 		= f'covariance/EUCLID/Mask_noise/TG_{nside}_nsim{nsim}_Bianca_mask_noiseEuclid/'
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
xcspectra = spectra.XCSpectraFile(fname_xcspectra,  WantTG = True)

# Shot noise power spectrum
Nll = np.ones(cl_theory_gg.shape[0])/ simparams['ngal']

# Simulations class
simulations = sims.KGsimulations(xcspectra, sims_dir, simparams, WantTG = True)
simulations.Run(nsim, WantTG = True)

# Needlet Analysis
myanalysis = analysis.NeedAnalysis(jmax, lmax, out_dir, simulations)

# Cl Analysis

myanalysis_cl = analysis.HarmAnalysis(lmax, out_dir, simulations, lmin=2, delta_ell=delta_ell,mask=mask, pixwin=simparams['pixwin'], fsky_approx=True)
lbins = myanalysis_cl.ell_binned

# Theory Needlet spectra
need_theory = spectra.NeedletTheory(myanalysis.B)

# Computing simulated Betajs 
print("...computing Betajs for simulations...")
fname_betaj_sims_TS_galT_mask      = f'betaj_sims_TS_galT_jmax{jmax}_B_{myanalysis.B:0.2f}_nside{nside}_fsky{fsky:0.2f}.dat'


betaj_sims_TS_galT_mask = myanalysis.GetBetajSimsFromMaps('TS', nsim, field2='galT', mask=mask,fname=fname_betaj_sims_TS_galT_mask)


# Computing simulated Cls 
print("...computing Betajs for simulations...")
fname_cl_sims_TS_galT_mask      = f'cl_sims_TS_galT_lmax{lmax}_nside{nside}_fsky{fsky}.dat'

#cl_TS_galT_mask = myanalysis_cl.GetClSimsFromMaps(field1='TS', nsim=nsim, field2='galT', fname=fname_cl_sims_TS_galT_mask)*fsky #qua voglio ri-ottenere i cls biassati

cl_TS_galT_mask = np.zeros((nsim,lmax+1))

for n in range(nsim):
    fname_T =  sims_dir + "sim_" + ('%04d' % n) + "_TS_" + ('%04d' % nside) + ".fits"
    fname_gal = sims_dir + "sim_" + ('%04d' % n) + "_galT_" + ('%04d' % nside) + ".fits"
    mapT = hp.read_map(fname_T, verbose=False)
    mapgal = hp.read_map(fname_gal, verbose=False)
    mapT = hp.remove_dipole(mapT, verbose=False)
    mapgal = hp.remove_dipole(mapgal, verbose=False)

    cl_TS_galT_mask[n, :] =hp.anafast(map1=mapT*mask, map2=mapgal*mask, lmax=lmax)


# Covariances Betajs
print("...computing Cov Matrices...")
fname_cov_TS_galT_mask      = f'cov_TS_galT_jmax{jmax}_B_{myanalysis.B:0.2f}_nside{nside}_fsky{fsky:0.2f}.dat'

cov_TS_galT_mask, corr_TS_galT_mask          = myanalysis.GetCovMatrixFromMaps(field1='TS', nsim=nsim, field2='galT', mask=mask, fname=fname_cov_TS_galT_mask, fname_sims=fname_betaj_sims_TS_galT_mask)

#cov_TS_galT_mask = cov_TS_galT_mask*fsky**2 #qua voglio ri-ottenere i cls biassati

# Covariances Cls

fname_cov_cl_TS_galT_mask      = f'cov_TS_galT_lmax{lmax}_nside{nside}_fsky{fsky}.dat'

#cov_cl_TS_galT_mask, corr_cl_TS_galT_mask           = myanalysis_cl.GetCovMatrixFromMaps(field1='TS', nsim=nsim, field2='galT', fname=fname_cov_cl_TS_galT_mask, fname_sims=fname_cl_sims_TS_galT_mask)
cov_cl_TS_galT_mask = np.cov(cl_TS_galT_mask.T)#qua voglio ri-ottenere i cls biassati

print("...done...")

# <Beta_j>_MC
betaj_TS_galT_mask_mean    = myanalysis.GetBetajMeanFromMaps('TS', nsim, field2='galT', mask=mask, fname_sims=fname_betaj_sims_TS_galT_mask)

# <C_l>_MC
cl_TS_galT_mask_mean=np.mean(cl_TS_galT_mask, axis=0)

# Beta_j sims

[num_sim_1, num_sim_2] = np.random.choice(np.arange(nsim),2 )

beta_j_sim_1_T_mask = betaj_sims_TS_galT_mask[num_sim_1,:]
beta_j_sim_2_T_mask = betaj_sims_TS_galT_mask[num_sim_2,:]

# Beta_j THEORY

betatg    = need_theory.cl2betaj(jmax=jmax, cl=cl_theory_tg)
beta_norm = need_theory.cl2betaj(jmax=jmax, cl=np.ones(xcspectra.cltg.size))
delta = need_theory.delta_beta_j(jmax=jmax, cltg=cl_theory_tg, cltt=cl_theory_tt, clgg=cl_theory_gg, noise_gal_l=Nll)

#np.savetxt(out_dir+f'theory_beta_jmax{jmax}_D{myanalysis.B:0.2f}_lmax{lmax}.dat', betatg )
#np.savetxt(out_dir+f'theory_variance_jmax{jmax}_D{myanalysis.B:0.2f}_lmax{lmax}.dat' , delta)

########################################################################################################################

# Some plots
print("...here come the plots...")

# Covariances
#fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(25,18))
fig, axs = plt.subplots(ncols=1, nrows=1,figsize=(25,14))   
fig.suptitle(r'$B = %1.2f $' %myanalysis.B + r'$N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

mask_ = np.tri(corr_TS_galT_mask.shape[0],corr_TS_galT_mask.shape[1],0)

#plt.subplot(131)
axs.set_title(r'Corr $T^T\times gal^T$ Signal + shot noise MASKED')
sns.color_palette("crest", as_cmap=True)
sns.heatmap(corr_TS_galT_mask, annot=True, fmt='.2f', cmap  = 'viridis',mask=mask_, ax=axs)

plt.tight_layout()
plt.savefig(cov_dir+f'total_corr_T_gal_noise_mask_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.png')


# Sims mean spectrum + one sims spectrum TOTAL SIGNAL MASK

fig, ax1 = plt.subplots(1,1,figsize=(25,14))

plt.suptitle(r'Total signal, $D = %1.2f $' %myanalysis.B + r'$ ,~N_{side} =$'+str(simparams['nside']) + r' $,~N_{sim} = $'+str(nsim))

#ax = fig.add_subplot(1, 1, 1)
ax1.set_title(r'$\beta_{j}T^T \times gal^T$')
ax1.errorbar(myanalysis.jvec, betaj_TS_galT_mask_mean/fsky, yerr=np.sqrt(np.diag(cov_TS_galT_mask/fsky**2))/np.sqrt(nsim-1),fmt='o', label = 'betaj mean')
ax1.errorbar(myanalysis.jvec, beta_j_sim_1_T_mask/fsky, yerr=np.sqrt(np.diag(cov_TS_galT_mask/fsky**2)), fmt='o', label = 'betaj sim')
ax1.set_ylabel(r'$\beta_j^{TG}$')
ax1.set_xlabel(r'j')
ax1.legend()

fig.tight_layout()
plt.savefig(out_dir+f'total-signal_betaj_mean_betaj_sim_plot_nsim_{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.png')



## Theory + Sims Needlet power spectra - nel caso di mask devo moltiplicare per fsky il cl
##MASKEK
#
#fig = plt.figure(figsize=(27,20))
#
#plt.suptitle(r'$D = %1.2f $' %myanalysis.B + r'$ ~N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))
#
#ax = fig.add_subplot(1, 1, 1)
#ax.errorbar(myanalysis.jvec, betatg,  yerr=delta, color='firebrick', fmt='o', ms=10,capthick=5, label=r'theory')
#ax.errorbar(myanalysis.jvec, beta_j_sim_1_T_mask/fsky, yerr=np.sqrt(np.diag(cov_TS_galT_mask/fsky**2)), color='seagreen', fmt='o',ms=10,capthick=5, label=r'sim')
#ax.legend(loc='best')
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#ax.set_xlabel(r'$j$')
#ax.set_ylabel(r'$\beta_j^{Tgal}$')
#fig.tight_layout()
#plt.savefig(out_dir+f'sim{num_sim_1}_betaj_theory_T_gal_noise_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')
#
#fig = plt.figure(figsize=(27,20))
#
#plt.suptitle(r'$D = %1.2f $' %myanalysis.B + r'$ ~N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))
#
#ax = fig.add_subplot(1, 1, 1)
#ax.errorbar(myanalysis.jvec, betatg,  yerr=delta, color='firebrick', fmt='o', ms=10,capthick=5, label=r'theory')
#ax.errorbar(myanalysis.jvec, beta_j_sim_2_T_mask/fsky, yerr=np.sqrt(np.diag(cov_TS_galT_mask/fsky**2)), color='seagreen', fmt='o',ms=10,capthick=5, label=r'sim')
#ax.legend(loc='best')
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#ax.set_xlabel(r'$j$')
#ax.set_ylabel(r'$\beta_j^{Tgal}$')
#fig.tight_layout()
#plt.savefig(out_dir+f'sim{num_sim_2}_betaj_theory_T_gal_noise_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')

#print(f'ratio beta mean mask / beta theory = {betaj_TS_galT_mask_mean/betatg}')
#print(f'ratio beta mean mask / beta mean = {betaj_TS_galT_mask_mean/betaj_TS_galT_mean}')
#print(f'ratio beta mean  / beta theory = {betaj_TS_galT_mean/betatg}')



###Comparison variance MASKED SKY


fig = plt.figure(figsize=(27,20))

gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], wspace=0)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

plt.suptitle(r'$D = %1.2f $' %myanalysis.B + r'$ ~N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

ax0.errorbar(myanalysis.jvec[1:jmax], betaj_TS_galT_mask_mean[1:jmax]/fsky, yerr=delta[1:jmax]/(np.sqrt(nsim)), color='firebrick', fmt='o', ms=10,capthick=5, label=r'Variance from theory')
ax0.errorbar(myanalysis.jvec[1:jmax], betaj_TS_galT_mask_mean[1:jmax]/fsky, yerr=np.sqrt(np.diag(cov_TS_galT_mask/fsky**2)[1:jmax])/np.sqrt(nsim), color='seagreen', fmt='o',ms=10,capthick=5, label=r'Variance from sim')
difference = (betaj_TS_galT_mask_mean/fsky -betatg)/(betatg)      
ax1.errorbar(myanalysis.jvec[1:jmax], difference[1:jmax],yerr=delta[1:jmax]/(betatg[1:jmax]*np.sqrt(nsim) ),color='firebrick', fmt='o',  ms=10,capthick=5, label=r'Variance from theory')
ax1.errorbar(myanalysis.jvec[1:jmax], difference[1:jmax], yerr=np.sqrt(np.diag(cov_TS_galT_mask/fsky**2)[1:jmax])/(betatg[1:jmax]*np.sqrt(nsim) ), color='seagreen', fmt='o',ms=10,capthick=5, label=r'Variance from sim')
ax1.axhline(ls='--', color='k')
ax1.set_ylabel(r'$(\langle \beta_j^{Tgal} \rangle - \beta_j^{Tgal, th})/\beta_j^{Tgal, th}$')

ax0.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.set_xlabel(r'$j$')
ax0.set_ylabel(r'$\beta_j^{Tgal}$')

fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(out_dir+f'betaj_theory_T_gal_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')


#### RELATIVE DIFFERENCE THEORY + SIMULATIONS FULL SKY

fig = plt.figure(figsize=(27,20))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B + r'$ ~N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.axhline(ls='--', color='k')
ax.errorbar(myanalysis.jvec[1:jmax], (betaj_TS_galT_mask_mean[1:jmax]/fsky )/(betatg[1:jmax])-1, yerr=delta[1:jmax]/(np.sqrt(nsim)*betatg[1:jmax]), color='firebrick', fmt='o',  ms=10,capthick=5, label=r'Variance from theory')
ax.errorbar(myanalysis.jvec[1:jmax], (betaj_TS_galT_mask_mean[1:jmax]/fsky )/(betatg[1:jmax])-1, yerr=np.sqrt(np.diag(cov_TS_galT_mask/fsky**2)[1:jmax])/(np.sqrt(nsim)*betatg[1:jmax]), color='seagreen', fmt='o', ms=10,capthick=5, label=r'Variance from sim')

ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\langle \beta_j^{Tgal} \rangle/\beta_j^{Tgal, th}$-1')# - \beta_j^{Tgal, th}
#ax.set_ylim([-0.3,1.3])

fig.tight_layout()
plt.savefig(out_dir+f'betaj_mean_T_gal_noise_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')

######################################################################################################

# PROVA MASTER

def cov_cl(Mll, cltg,cltt, clgg, lmax, noise_gal_l=None):
        """
        Returns the Cov(Pseudo-C_\ell, Pseudo-C_\ell') 
        Notes
        -----
        Cov(Pseudo-C_\ell, Pseudo-C_\ell') .shape = (lmax+1, lmax+1)
        """
        if noise_gal_l is not None:
            clgg_tot = clgg+noise_gal_l
        else:
            clgg_tot = clgg

        #Mll  = self.get_Mll(wl, lmax=lmax)
        covll = np.zeros((lmax+1, lmax+1))
        for ell1 in range(lmax+1):
            for ell2 in range(lmax+1):
                #print(Mll[ell1,ell2]*cltg[ell1]*cltg[ell2])
                covll[ell1,ell2] = Mll[ell1,ell2]*(cltg[ell1]*cltg[ell2]+np.sqrt(cltt[ell1]*cltt[ell2]*clgg_tot[ell1]*clgg_tot[ell2]))/(2.*ell1+1)
        return covll

wl = hp.anafast(mask, lmax=lmax)
mll  = need_theory.get_Mll(wl, lmax=lmax)

#PCL
pcl  = np.dot(mll, cl_theory_tg[:lmax+1])
cov_ll = cov_cl(mll, cltg=cl_theory_tg, cltt=cl_theory_tt, clgg=cl_theory_gg, lmax=lmax, noise_gal_l=Nll )
ell = np.arange(lmax+1)

print(pcl.shape)

f, ax1 = plt.subplots(1,1, sharex=True, figsize=(27,20))

ax1.plot(ell,ell*(ell+1)/(2*np.pi)*fsky*cl_theory_tg[:lmax+1],marker='o',markersize= 3, color='firebrick',label=r'fsky*$C_{\ell}^{Th}$')
ax1.plot(ell,ell*(ell+1)/(2*np.pi)*pcl,  marker='o',markersize= 3,color='seagreen',label=r'$\tilde{C}_{\ell}$ Euclid Mask')
ax1.plot(ell[2:],ell[2:]*(ell[2:]+1)/(2*np.pi)*cl_TS_galT_mask_mean[2:], 'o',markersize= 10,color='goldenrod',label=r'$\tilde{C}_{\ell}$ from sims')
#ax1.set_yscale('log')
ax1.set_xticks(np.arange(lmax+1, step=20))
#ax1.set_xlim([0,20])
ax1.axvline(x=2, ls='--', color='k')
ax1.legend()
plt.savefig(out_dir+'master_pcl.png', bbox_inches='tight')

##################################

#MASTER NEEDLETS

gammaJ_tg = need_theory.gammaJ(cl_theory_tg, wl, jmax, lmax)
delta_gammaj = need_theory.variance_gammaj(cltg=cl_theory_tg,cltt=cl_theory_tt, clgg=cl_theory_gg, wl=wl, jmax=jmax, lmax=lmax, noise_gal_l=Nll)

##RELATIVE DIFFERENCE \beta_j
fig = plt.figure(figsize=(27,20))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B + r'$ ~N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim)+f' fsky={fsky:0.2f}')

ax = fig.add_subplot(1, 1, 1)
ax.set_title('Bianca sims')
ax.axhline(ls='--', color='k')
ax.errorbar(myanalysis.jvec[1:jmax], (betaj_TS_galT_mask_mean[1:jmax] /gammaJ_tg[1:jmax]-1)*100, yerr=100*np.sqrt(np.diag(delta_gammaj)[1:jmax])/(np.sqrt(nsim)*gammaJ_tg[1:jmax]), color='firebrick', fmt='o', ms=10,capthick=5, label=r'Variance from theory')
ax.errorbar(myanalysis.jvec[1:jmax], (betaj_TS_galT_mask_mean[1:jmax] /gammaJ_tg[1:jmax]-1)*100, yerr=100*np.sqrt(np.diag(cov_TS_galT_mask)[1:jmax])/(np.sqrt(nsim)*gammaJ_tg[1:jmax]), color='seagreen', fmt='o', ms=10,capthick=5, label=r'Variance from sim')
#ax.errorbar(myanalysis.jvec[1:jmax], (betaj_TS_galT_mask_mean[1:jmax] )/(betatg[1:jmax])-1, yerr=np.sqrt(np.diag(cov_TS_galT_mask)[1:jmax])/(betatg[1:jmax]), color='darkgrey', fmt='o', ms=10,capthick=5, label=r'Variance one sim')

ax.legend(loc='best')
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\langle \tilde{\beta}_j^{Tgal} \rangle/\tilde{\beta}_j^{Tgal, th}$-1 %')# - \beta_j^{Tgal, th}
ax.set_xlim([-1,12])
#ax.set_ylim([-0.3,1.3])

fig.tight_layout()
plt.savefig(out_dir+f'gammaj_mean_T_gal_noise_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')


##RELATIVE DIFFERENCE VARIANCE \beta_j
fig = plt.figure(figsize=(27,20))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B + r'$ ~N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim)+f' fsky={fsky:0.2f}')

ax = fig.add_subplot(1, 1, 1)

ax.set_title('Bianca sims')

ax.axhline(ls='--', color='k')
ax.plot(myanalysis.jvec[1:jmax], 100*(np.diag(cov_TS_galT_mask)[1:jmax]/np.diag(delta_gammaj)[1:jmax] -1),  'o', markersize=15, color='firebrick')

ax.legend(loc='best')
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\langle \sigma(\tilde{\beta}_j^{Tgal}) \rangle/(\sigma\tilde{\beta}_j^{Tgal, th})$-1 %')# - \beta_j^{Tgal, th}
ax.set_xlim([-1,12])
#ax.set_ylim([-0.3,1.3])

fig.tight_layout()
plt.savefig(out_dir+f'diff_percent_variance_gammaj_T_gal_noise_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')


### SPECTRUM + SCATTER PLOT

fig = plt.figure(figsize=(27,20))
  
plt.suptitle('Variance comparison - Cut sky')

gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], wspace=0)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

plt.suptitle(r'$D = %1.2f $' %myanalysis.B + r'$ ~N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim)+' MASK')

ax0.plot(myanalysis.jvec[1:jmax], gammaJ_tg[1:jmax],'k', label=r'$\Gamma_j}$')
ax0.errorbar(myanalysis.jvec[1:jmax], betaj_TS_galT_mask_mean[1:jmax], yerr=np.sqrt(np.diag(delta_gammaj)[1:jmax])/(np.sqrt(nsim)), color='firebrick', fmt='o',ms=10,capthick=5, label=r'Variance from theory')
ax0.errorbar(myanalysis.jvec[1:jmax], betaj_TS_galT_mask_mean[1:jmax], yerr=np.sqrt(np.diag(cov_TS_galT_mask)[1:jmax])/(np.sqrt(nsim)), color='seagreen', fmt='o',ms=10,capthick=5, label=r'Variance from sim')

difference = (betaj_TS_galT_mask_mean -gammaJ_tg)/(gammaJ_tg)      
#ax1.errorbar(myanalysis.jvec[1:jmax], difference[1:jmax],yerr=delta_noise[1:jmax]/(gammaJ_tg[1:jmax]/(4*np.pi)*np.sqrt(nsim) ),color='seagreen', fmt='o',  ms=10,capthick=5, label=r'Variance from theory')
ax1.errorbar(myanalysis.jvec[1:jmax], difference[1:jmax], yerr=np.sqrt(np.diag(delta_gammaj)[1:jmax])/(gammaJ_tg[1:jmax]*np.sqrt(nsim) ), color='firebrick', fmt='o',ms=10,capthick=5, label=r'Variance from theory')
ax1.errorbar(myanalysis.jvec[1:jmax], difference[1:jmax], yerr=np.sqrt(np.diag(cov_TS_galT_mask)[1:jmax])/(gammaJ_tg[1:jmax]*np.sqrt(nsim) ), color='seagreen', fmt='o',ms=10,capthick=5, label=r'Variance from sim')
ax1.axhline(ls='--', color='k')
ax1.set_ylabel(r'$(\langle \tilde{\beta}_j^{Tgal} \rangle - \tilde{\beta}_j^{Tgal, th})/\beta_j^{Tgal, th}$')

ax0.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.set_xlabel(r'$j$')
ax0.set_ylabel(r'$\tilde{\beta}_j^{Tgal}$')

fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(out_dir+f'gammaj_theory_T_gal_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')

#diff_cov_mask = np.abs((np.sqrt(np.diag(cov_TS_galT_mask))-delta_noise))/delta_noise*100
#diff_cov_mean_mask = np.mean(diff_cov_mask[1:jmax])
#print(f'diff cov mask={diff_cov_mask}, diff cov mean={diff_cov_mean_mask}')

##RELATIVE DIFFERENCE Cls
fig = plt.figure(figsize=(27,20))

plt.suptitle(r'$\ell_{max}$= '+ str(lmax) + r'$ ~N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim)+f' fsky={fsky:0.2f}')

ax = fig.add_subplot(1, 1, 1)
ax.set_title('Bianca sims')
ax.axhline(ls='--', color='k')
ax.errorbar(ell[2:], (cl_TS_galT_mask_mean[2:] /pcl[2:]-1)*100, yerr=100*np.sqrt(np.diag(cov_ll)[2:])/(np.sqrt(nsim)*pcl[2:]), color='firebrick', fmt='o', ms=10,capthick=5, label=r'Variance from theory')
ax.errorbar(ell[2:], (cl_TS_galT_mask_mean[2:] /pcl[2:]-1)*100, yerr=100*np.sqrt(np.diag(cov_cl_TS_galT_mask)[2:])/(np.sqrt(nsim)*pcl[2:]), color='seagreen', fmt='o', ms=10,capthick=5, label=r'Variance from sim')
#ax.errorbar(myanalysis.jvec[1:jmax], (betaj_TS_galT_mask_mean[1:jmax] )/(betatg[1:jmax])-1, yerr=np.sqrt(np.diag(cov_TS_galT_mask)[1:jmax])/(betatg[1:jmax]), color='darkgrey', fmt='o', ms=10,capthick=5, label=r'Variance one sim')

ax.legend(loc='best')
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\langle \tilde{C}_{\ell}^{Tgal} \rangle/\tilde{C}_{\ell}^{Tgal, th}$-1 %')# - \beta_j^{Tgal, th}
ax.set_xticks(np.arange(lmax+1, step=5))
ax.set_xlim([-1,40])
ax.set_ylim([-25,25])

fig.tight_layout()
plt.savefig(out_dir+f'pseudocl_mean_T_gal_noise_lmax{lmax}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')


##RELATIVE DIFFERENCE VARIANCE \C_ell
fig = plt.figure(figsize=(27,20))

plt.suptitle(r'$ ~N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim)+f' fsky={fsky:0.2f}')

ax = fig.add_subplot(1, 1, 1)

ax.set_title('Bianca sims')

ax.axhline(ls='--', color='k')
ax.plot(ell[2:], 100*(np.diag(cov_cl_TS_galT_mask)[2:]/np.diag(cov_ll)[2:] -1),  'o', markersize=15, color='firebrick')

ax.legend(loc='best')
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\langle \sigma(\tilde{C}_{\ell}^{Tgal}) \rangle/\sigma(\tilde{C}_{\ell}^{Tgal, th})$-1 %')# - \beta_j^{Tgal, th}
ax.set_xlim([-1,45])
#ax.set_ylim([-0.3,1.3])

fig.tight_layout()
plt.savefig(out_dir+f'diff_percent_variance_pseudo_T_gal_noise_lmax{lmax}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')