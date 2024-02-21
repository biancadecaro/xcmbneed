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
sns.set_palette('husl', n_colors=10)

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
simparams = {'nside'   : 128,
             'ngal'    : 5.76e5, #dovrebbe importare solo per lo shot noise (noise poissoniano)
             
 	     	 'ngal_dim': 'ster',
	     	 'pixwin'  : False}
nside = simparams['nside']
jmax = 12
lmax = 256
nsim = 500
nbins = 3
# Paths
fname_xcspectra = '/ehome/bdecaro/xcmbneed/src/spectra/inifiles/fiducial_EUCLID_tomo_nbins3.dat'#'spectra/inifiles/CAMBSpectra_planck.dat' 
sims_dir        = f'/ehome/bdecaro/xcmbneed/src/sims/Needlet/Euclid_mie/Tomography/TGsims_{nside}_nbins{nbins}/'
out_dir         = f'output_needlet_TG/Euclid_mie/Tomography/TG_{nside}_lmax{lmax}_nbins{nbins}_nsim{nsim}/'
path_inpainting = 'inpainting/inpainting.py'
#cov_dir 		= f'covariance/Planck/Mask_noise/TG_{nside}_tesi/covariance_TG{nside}_tesi/'
#if not os.path.exists(cov_dir):
#        os.makedirs(cov_dir)




mask = hp.read_map(f'/ehome/bdecaro/xcmbneed/src/mask/mask_planck_comm_2018_nside={nside}.fits')
mask[np.where(mask>=0.5 )]=1
mask[np.where(mask<0.5 )]=0
fsky = np.mean(mask)
print(fsky)

# Loading theory spectra
xcspectra = spectra.XCSpectraFile(fname_xcspectra, WantTG = True, nbins=3)
Nll = np.ones(xcspectra.clgg[0].shape)/simparams['ngal']

# Simulations class
simulations = sims.KGsimulations(xcspectra, sims_dir, simparams,  WantTG = True)
simulations.Run(nsim, WantTG = True,nbins=3)

# Needlet Analysis
myanalysis = analysis.NeedAnalysis(jmax, lmax, out_dir, simulations, nbins=3)

# Theory Needlet spectra
need_theory = spectra.NeedletTheory(myanalysis.B)

B=myanalysis.B
print(B)

# Computing simulated betajs
print("...computing Betajs for simulations...")
fname_betaj_sims_TS_galS = [f'betaj_sims_TS_galS_nbins{bin}_jmax{jmax}_B_{B}_nside{nside}.dat' for bin in range(nbins)]
fname_betaj_sims_TS_galS_mask = [f'betaj_sims_TS_galS_nbins{bin}_jmax{jmax}_B_{B}_nside{nside}_fsky_{fsky}.dat' for bin in range(nbins)]
betaj_sims_TS_galS = np.zeros((nbins,nsim, jmax+1))
betaj_sims_TS_galS_mask = np.zeros((nbins,nsim, jmax+1))
fname_cov_TS_galS = [f'cov_TS_galS_nbins{bin}_jmax{jmax}_B{B:0.2f}_nside{nside}.dat' for bin in range(nbins)]
fname_cov_TS_galS_mask = [f'cov_TS_galS_nbins{bin}_jmax{jmax}_B{B:0.2f}_nside{nside}_mask.dat' for bin in range(nbins)]
cov_TS_galS = np.zeros((nbins, jmax+1,jmax+1))
cov_TS_galS_mask = np.zeros((nbins, jmax+1,jmax+1))
corr_TS_galS = np.zeros((nbins, jmax+1,jmax+1))
corr_TS_galS_mask = np.zeros((nbins, jmax+1,jmax+1))
betaj_TS_galS_mean = np.zeros((nbins, jmax+1))
betaj_TS_galS_mask_mean = np.zeros((nbins, jmax+1))

fname_betaj_sims_TS_galT = [f'betaj_sims_TS_galT_nbins{bin}_jmax{jmax}_B_{B}_nside{nside}.dat' for bin in range(nbins)]
fname_betaj_sims_TS_galT_mask = [f'betaj_sims_TS_galT_nbins{bin}_jmax{jmax}_B_{B}_nside{nside}_fsky_{fsky}.dat' for bin in range(nbins)]
betaj_sims_TS_galT = np.zeros((nbins,nsim, jmax+1))
betaj_sims_TS_galT_mask = np.zeros((nbins,nsim, jmax+1))
fname_cov_TS_galT = [f'cov_TS_galT_nbins{bin}_jmax{jmax}_B{B:0.2f}_nside{nside}.dat' for bin in range(nbins)]
fname_cov_TS_galT_mask = [f'cov_TS_galT_nbins{bin}_jmax{jmax}_B{B:0.2f}_nside{nside}_mask.dat' for bin in range(nbins)]
cov_TS_galT = np.zeros((nbins, jmax+1,jmax+1))
cov_TS_galT_mask = np.zeros((nbins, jmax+1,jmax+1))
corr_TS_galT = np.zeros((nbins, jmax+1,jmax+1))
corr_TS_galT_mask = np.zeros((nbins, jmax+1,jmax+1))
betaj_TS_galT_mean = np.zeros((nbins, jmax+1))
betaj_TS_galT_mask_mean = np.zeros((nbins, jmax+1))



for bin in range(nbins):
    print(f'Bin={bin}')
    betaj_sims_TS_galS[bin] = myanalysis.GetBetajSimsFromMaps(f'TS{bin}', nsim, nbins,field2=f'galS{bin}', fname=fname_betaj_sims_TS_galS[bin], fsky_approx=True)
    betaj_sims_TS_galS_mask[bin] = myanalysis.GetBetajSimsFromMaps(f'TS{bin}', nsim, nbins,field2=f'galS{bin}', mask= mask, fname=fname_betaj_sims_TS_galS_mask[bin], fsky_approx=True)

    betaj_sims_TS_galT[bin] = myanalysis.GetBetajSimsFromMaps(f'TS{bin}', nsim, nbins,field2=f'galT{bin}', fname=fname_betaj_sims_TS_galT[bin], fsky_approx=True)
    betaj_sims_TS_galT_mask[bin] = myanalysis.GetBetajSimsFromMaps(f'TS{bin}', nsim, nbins, field2=f'galT{bin}', mask= mask, fname=fname_betaj_sims_TS_galT_mask[bin], fsky_approx=True)


    # Covariances
    print("...computing Cov Matrices...")
    cov_TS_galS[bin], corr_TS_galS[bin]         = myanalysis.GetCovMatrixFromMaps(field1=f'TS{bin}', nsim=nsim, nbins=nbins,field2=f'galS{bin}', fname=fname_cov_TS_galS[bin], fname_sims=fname_betaj_sims_TS_galS[bin])
    cov_TS_galS_mask[bin], corr_TS_galS_mask[bin]          = myanalysis.GetCovMatrixFromMaps(field1=f'TS{bin}', nsim=nsim,nbins=nbins, field2=f'galS{bin}', fname=fname_cov_TS_galS_mask[bin], mask = mask,fname_sims=fname_betaj_sims_TS_galS_mask[bin])

    cov_TS_galT[bin], corr_TS_galT[bin]         = myanalysis.GetCovMatrixFromMaps(field1=f'TS{bin}', nsim=nsim,nbins=nbins, field2=f'galT{bin}', fname=fname_cov_TS_galT[bin], fname_sims=fname_betaj_sims_TS_galT[bin])
    cov_TS_galT_mask[bin], corr_TS_galT_mask[bin]          = myanalysis.GetCovMatrixFromMaps(field1=f'TS{bin}', nsim=nsim,nbins=nbins, field2=f'galT{bin}', fname=fname_cov_TS_galT_mask[bin], mask = mask,fname_sims=fname_betaj_sims_TS_galT_mask[bin])


    print("...done...")

# <Beta_j>_MC
    betaj_TS_galS_mean[bin]    = myanalysis.GetBetajMeanFromMaps(f'TS{bin}', nsim, nbins,field2=f'galS{bin}', fname_sims=fname_betaj_sims_TS_galS[bin])
    betaj_TS_galS_mask_mean[bin]     = myanalysis.GetBetajMeanFromMaps(f'TS{bin}', nsim, nbins,field2=f'galS{bin}', fname_sims=fname_betaj_sims_TS_galS_mask[bin])
    
    betaj_TS_galT_mean[bin]    = myanalysis.GetBetajMeanFromMaps(f'TS{bin}', nsim, nbins,field2=f'galT{bin}', fname_sims=fname_betaj_sims_TS_galT[bin])
    betaj_TS_galT_mask_mean[bin]     = myanalysis.GetBetajMeanFromMaps(f'TS{bin}', nsim, nbins,field2=f'galT{bin}', fname_sims=fname_betaj_sims_TS_galT_mask[bin])


# Some plots
print("...here come the plots...")

# Covariances
#fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(25,18))
fig, axs = plt.subplots(ncols=2, nrows=2,figsize=(20,17)) 
#fig.suptitle(r'$B = %1.2f $' %B + r'$N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))
plt.suptitle(r'$D = %1.2f $' %B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))
mask_ = np.tri(corr_TS_galS[0].shape[0],corr_TS_galS.shape[1],0)

#plt.subplot(131)
axs[0,0].set_title(r'Corr $T^S\times gal^S$ Bin=1')
sns.color_palette("crest", as_cmap=True)
sns.heatmap(corr_TS_galS[0], annot=True, fmt='.2f', cmap  = 'viridis',mask=mask_, ax=axs[0,0])

##plt.subplot(132)
axs[0,1].set_title(r'Corr $T^S\times gal^S$ Bin=2')
sns.heatmap(corr_TS_galS[1], annot=True, fmt='.2f',cmap  = 'viridis', mask=mask_, ax=axs[0,1])
#
##plt.subplot(133)
#ax3.set_title(r'Corr $\delta^T \times \kappa^T$ Masked')
#sns.heatmap(corr_kappaT_deltaT_mask, annot=True, fmt='.2f', mask=mask_,ax=ax3)
axs[1,0].set_title(r'Corr $T^S\times gal^T$ Bin=3')
sns.heatmap(corr_TS_galS[2], annot=True, fmt='.2f',cmap  = 'viridis', mask=mask_, ax=axs[1,0])



fig.tight_layout()
plt.savefig(out_dir+f'corr_T_gal_noise_nbins{nbins}_jmax{jmax}_B{B:0.2f}_nsim{nsim}_nside{nside}.png')


########################################################################################################################################################################################

# Theory + Normalization Needlet power spectra
betatg = np.zeros((nbins,jmax+1))
delta = np.zeros((nbins,jmax+1))
for bin in range(nbins):
    betatg[bin]    = need_theory.cl2betaj(jmax=jmax, cl=xcspectra.cltg[bin])
    delta[bin] = need_theory.delta_beta_j(jmax, cltt = xcspectra.cltt, cltg = xcspectra.cltg[bin], clgg = xcspectra.clgg[bin,bin])

np.savetxt(out_dir+f'beta_TS_galS_theoretical_fiducial_B{B}_nbins{nbins}.dat', betatg)

#delta_noise = need_theory.delta_beta_j(jmax=jmax, cltg=xcspectra.cltg, cltt=xcspectra.cltt, clgg=xcspectra.clg1g1, noise_gal_l=Nll)

np.savetxt(out_dir+f'variance_beta_TS_galS_theoretical_fiducial_B{B}_nbins{nbins}.dat', delta)

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) +r', $f_{\mathrm{sky}}$ = %1.2f'%fsky+ r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))


ax = fig.add_subplot(1, 1, 1)
for bin in range(nbins):

    ax.plot(myanalysis.jvec[1:jmax], betatg[bin][1:jmax], label=f'Theory, Bin = {bin}')
    ax.errorbar(myanalysis.jvec[1:jmax], betaj_TS_galS_mean[bin][1:jmax], yerr=np.sqrt(np.diag(cov_TS_galS[bin])[1:jmax])/np.sqrt(nsim) ,fmt='o',ms=5,capthick=2, label=f'Error of the mean , Bin = {bin} No noise')

ax.set_xticks(myanalysis.jvec[1:jmax])
ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\beta_j^{\mathrm{TG}}$')

fig.tight_layout()
plt.savefig(out_dir+f'betaj_theory_T_gal_jmax{jmax}_D{B:0.2f}_nsim{nsim}_nside{nside}_nbins{nbins}.png', bbox_inches='tight')


####NOISE
fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) +r', $f_{\mathrm{sky}}$ = %1.2f'%fsky+ r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

color = sns.color_palette("husl").as_hex()
ax = fig.add_subplot(1, 1, 1)
for bin, c in zip(range(nbins), color):

    ax.plot(myanalysis.jvec[1:jmax], betatg[bin][1:jmax], color=c,label=f'Theory, Bin = {bin}')
    ax.errorbar(myanalysis.jvec[1:jmax], betaj_TS_galT_mean[bin][1:jmax], yerr=np.sqrt(np.diag(cov_TS_galT[bin])[1:jmax])/np.sqrt(nsim) ,color=c,fmt='o',ms=5,capthick=2, label=f'Error of the mean , Bin = {bin} Shot noise')

ax.set_xticks(myanalysis.jvec[1:jmax])
ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\beta_j^{\mathrm{TG}}$')

fig.tight_layout()
plt.savefig(out_dir+f'betaj_theory_NOISE_T_gal_jmax{jmax}_D{B:0.2f}_nsim{nsim}_nside{nside}_nbins{nbins}.png', bbox_inches='tight')






####################################################################################################################################################################

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) +r', $f_{\mathrm{sky}}$ = %1.2f'%fsky+ r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.axhline(ls='--', color='grey')
for bin in range(nbins):
    #ax.errorbar(myanalysis.jvec[1:jmax], (betaj_TS_galS_mean[bin][1:jmax] -betatg[bin][1:jmax])/betatg[bin][1:jmax], yerr=np.sqrt(np.diag(cov_TS_galS[bin])[1:jmax])/(np.sqrt(nsim)*betatg[bin][1:jmax]),  fmt='o',  ms=5,label=f'Bin = {bin} No Noise')
    ax.errorbar(myanalysis.jvec[1:jmax], (betaj_TS_galT_mean[bin][1:jmax] -betatg[bin][1:jmax])/betatg[bin][1:jmax], yerr=np.sqrt(np.diag(cov_TS_galT[bin])[1:jmax])/(np.sqrt(nsim)*betatg[bin][1:jmax]),  fmt='o',  ms=5,label=f'Bin = {bin}, Shot Noise')

ax.legend(loc='best')
ax.set_xticks(myanalysis.jvec[1:jmax])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xlabel(r'$j$', fontsize=22)
ax.set_ylabel(r'$\frac{\langle \beta_j^{\mathrm{TG}} \rangle - \beta_j^{\mathrm{TG}\,, th}}{\beta_j^{\mathrm{TG}\,, th}}$', fontsize=22)
#ax.set_ylim([-0.2,0.3])

fig.tight_layout()
plt.savefig(out_dir+f'betaj_mean_T_gal_jmax{jmax}_D{B:1.2f}_nsim{nsim}_nside{nside}_nbins{nbins}.png', bbox_inches='tight')


#######################################################################################################################
#Difference divided one sigma

#fig = plt.figure(figsize=(17,10))
#
#plt.suptitle(r'$D = %1.2f $' %B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) +r', $f_{\mathrm{sky}}$ = %1.2f'%fsky+ r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))
#
#
#ax = fig.add_subplot(1, 1, 1)
#
#ax.axhline(ls='--', color='grey')
##ax.plot(myanalysis.jvec[1:jmax], (betaj_TS_galT_mask_mean[1:jmax] -betatg[1:jmax])/(delta_noise[1:jmax]/np.sqrt(nsim)),'o', ms=10, label= 'Shot Noise')#, label=r'$T^S \times gal^S$, sim cov')
#ax.plot(myanalysis.jvec[1:jmax], (betaj_TS_galS_mask_mean[1:jmax] -betatg[1:jmax])/(delta[1:jmax]/np.sqrt(nsim)),'o', ms=10,color='#2b7bbc')#, label=r'$T^S \times gal^S$, sim cov')
#
##print(np.sqrt(np.diag(cov_TS_galS))/(np.sqrt(nsim)*betatg) - delta/(np.sqrt(nsim)*betatg))
##plt.legend(loc='best')
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#ax.yaxis.set_major_formatter(formatter) 
#ax.set_xticks(myanalysis.jvec[1:jmax])
#ax.set_xlabel(r'$j$', fontsize=22)
#ax.set_ylabel(r'$\Delta \beta_j^{\mathrm{TG}} / \sigma $', fontsize=22)
#ax.set_ylim([-3,3])
#
#fig.tight_layout()
#plt.savefig(out_dir+f'diff_betaj_mean_theory_over_sigma_T_gal_jmax{jmax}_D{B:1.2f}_nsim{nsim}_nside{nside}_MASK.png', bbox_inches='tight')
#

####################################################################################
############################# DIFF COVARIANCES #####################################
#fig = plt.figure(figsize=(17,10))
#
#plt.suptitle(r'$D = %1.2f $' %B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))
#
#ax = fig.add_subplot(1, 1, 1)
#
##ax.plot(myanalysis.jvec[1:jmax+1], gammaJ_tg[1:jmax+1], label='Theory')
#ax.plot(myanalysis.jvec[1:jmax], 100*(np.sqrt(np.diag(cov_TS_galT_mask)[1:jmax])/delta_noise[1:jmax]-1) , 'o',ms=10,color='#2b7bbc')
#
##ax.legend(loc='best')
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#ax.yaxis.set_major_formatter(formatter) 
#ax.set_xlabel(r'$j$')
#ax.set_ylabel(r'% $(\Delta \beta)^2_{\mathrm{sims}}/(\Delta \beta)^2_{\mathrm{analytic}}$ - 1')
#
#fig.tight_layout()
#plt.savefig(out_dir+f'diff_cov_betaj_theory_T_gal_jmax{jmax}_D{B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')

######################################################################################
############################## SIGNAL TO NOISE RATIO #################################

#beta_vec = np.array([betaj_sims_TS_galS[bin] for bin in range(nbins)])
#print(cov_TS_galS.shape)
#cov_mat_tot = np.zeros((cov_TS_galS.shape, cov_TS_galS.shape))
#for bin in range(nbins):
#    cov_mat_tot[bin, bin] = cov_TS_galS[bin]
#print(cov_mat_tot)

delta_beta_tomo = need_theory.delta_beta_j_tomo(nbins,jmax,lmax, cltt = xcspectra.cltt, cltg = xcspectra.cltg, clgg = xcspectra.clgg)
#delta_beta_tomo_noise = need_theory.delta_beta_j_tomo(nbins,jmax,lmax, cltt = xcspectra.cltt, cltg = xcspectra.cltg, clgg = xcspectra.clgg, noise_gal_l=Nll)


ell_cl = np.arange(2050+1)
factor = ell_cl*(ell_cl+1)/(2*np.pi)
fig = plt.figure(figsize=(17,10))
ax = fig.src/TG_needlet_EUCLID_mask_nbins3.pyadd_subplot(1, 1, 1)
for b in range(nbins):
    for bb in range(nbins):
        #if bb==b: continue
        ax.plot(ell_cl, xcspectra.clgg[b,bb][ell_cl], label=f'G{b+1}G{bb+1}')

ax.legend(ncol=3)
plt.savefig(out_dir+'clgg_bins.png')
print(matplotlib.__version__)

fig , ax= plt.subplots(1,1)
cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
sns.heatmap(delta_beta_tomo.sum(axis=3).sum(axis=2), cmap=cmap, ax=ax)#, xticklabels=[1,2,3], yticklabels=[1,2,3])
ax.invert_yaxis()
ax.set_xlabel('redshift bin')
ax.set_ylabel('redshift bin')
ax.set_title('Covariance - Tomography')
plt.tight_layout()
fig.savefig(out_dir+'covariance_tomography.png')

fig, ax=plt.subplots(1,1)
ax.set_title('Covariance')
plt1=ax.imshow(delta_beta_tomo[0,0,:,:], norm='log')
plt.colorbar(plt1, ax=ax)
ax.invert_yaxis()
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\ell$')
plt.tight_layout()
fig.savefig(out_dir+'covariance_redshift0_ell.png')

icov=np.zeros((nbins, nbins, jmax+1,jmax+1))
for ij in range(jmax+1):
        for iij in range(jmax+1):
                icov[:, :, ij, iij] = np.linalg.inv(delta_beta_tomo[:, :, ij,iij])

def S_2_N(beta, icov):
    nj  = beta.shape[1]
    s2n = np.zeros(nj)
    for ij in range(nj):
        for iij in range(nj):
            s2n[ij] += np.dot(beta[:, ij], np.dot(icov[:, :, ij, iij], beta[:, iij]))
    return s2n

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
#
s2n=0
s2n_noise =0
#for bin in range(nbins):
#    s2n+=S_2_N(betaj_TS_galS_mean[bin], cov_TS_galS[bin])
#    s2n_noise+=S_2_N(betaj_TS_galT_mean[bin], cov_TS_galT[bin])
s2n = S_2_N(betaj_TS_galS_mean, delta_gamma_tomo)
s2n_noise = S_2_N(betaj_TS_galT_mean, delta_gamma_tomo_noise)

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) +r', $n_{\mathrm{bins}}$ = %d'%nbins+ r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.plot(myanalysis.jvec, s2n,marker = 'o',ms=10, label='No noise')
ax.plot(myanalysis.jvec, s2n_noise, color='#2b7bbc',marker = 'o',ms=10, label='Shot Noise')



ax.set_xticks(myanalysis.jvec[1:jmax])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
plt.legend()
ax.set_xlabel(r'$j$')
ax.set_ylabel('Signal-to-Noise ratio')

fig.tight_layout()
plt.savefig(out_dir+f'SNR_betaj_T_gal_jmax{jmax}_D{B:0.2f}_nsim{nsim}_nside{nside}_nbins{nbins}.png', bbox_inches='tight')
#
#CUMULATIVE
#
def fl_j(j_m):
    
    l_j = np.zeros(j_m+1, dtype=int)
    
    for j in range(j_m+1):
            lmin = np.floor(myanalysis.B**(j-1))
            lmax = np.floor(myanalysis.B**(j+1))
            ell  = np.arange(lmin, lmax+1, dtype=int)
            l_j[j] = int(ell[int(np.ceil((len(ell))/2))])
    return l_j
beta_T_galT_one_bin_sim = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000_nuova_mask/betaj_sims_TS_galT_jmax12_B_1.59_nside128.dat')
beta_T_galT_one_bin_mean = np.mean(beta_T_galT_one_bin_sim, axis=0)
cov_T_galT_one_bin = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000_nuova_mask/cov_TS_galT_jmax12_B_1.59_nside128.dat')
s2n_onebin = S_2_N(beta_T_galT_one_bin_mean, cov_T_galT_one_bin)

jvec = np.arange(0,jmax+1)
jmax_vec = myanalysis.jvec
lmax_vec=fl_j(jmax)
s2n_cum= S_2_N_cum(s2n, jvec)
s2n_cum_noise= S_2_N_cum(s2n_noise, jvec)
s2n_cum_one_bin= S_2_N_cum(s2n_onebin, jvec)

print(f's2n_cum_no_tomo={s2n_cum_one_bin[-1]}, s2n_cum_tomo={s2n_cum_noise[-1]}')

fig = plt.figure(figsize=(10,11))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

#ax.plot(lmax_vec, s2n_cum, label='No Noise')
ax.plot(lmax_vec, s2n_cum_one_bin, label='No tomo')
ax.plot(lmax_vec, s2n_cum_noise,color='#2b7bbc', label='Tomo')
ax.set_xscale('log')
ax.set_xlim(left=3, right=250)
#ax.set_ylim(top=4.)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 

ax.set_xlabel(r'$\ell_{\mathrm{max}}$')
ax.set_ylabel('Cumulative Signal-to-Noise ratio')
ax.legend()

fig.tight_layout()

plt.savefig(out_dir+f'SNR_cumulative_betaj_theory_T_gal_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_nbins{nbins}.png', bbox_inches='tight') #questa
