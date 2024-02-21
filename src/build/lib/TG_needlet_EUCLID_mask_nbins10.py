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
             'ngal'    :  35454308.580126834, #dovrebbe importare solo per lo shot noise (noise poissoniano)
             
 	     	 'ngal_dim': 'ster',
	     	 'pixwin'  : False}
nside = simparams['nside']
jmax = 12
lmax = 256
nsim = 1000
nbins = 10
# Paths
fname_xcspectra = '/ehome/bdecaro/xcmbneed/src/spectra/inifiles/euclid_fiducials_tomography_lmin0.txt'#'spectra/inifiles/CAMBSpectra_planck.dat' 
sims_dir        = f'/ehome/bdecaro/xcmbneed/src/sims/Euclid_sims_Marina/NSIDE128_TOMOGRAPHY/'
out_dir         = f'output_needlet_TG/EUCLID/Tomography/TG_{nside}_lmax{lmax}_nbins{nbins}_nsim{nsim}/'
out_dir_prova         = f'output_needlet_TG/EUCLID/Tomography/TG_{nside}_lmax{lmax}_nbins{nbins}_nsim{nsim}/prova_plot/'
path_inpainting = 'inpainting/inpainting.py'
cov_dir 		= out_dir+f'covariance/'
if not os.path.exists(cov_dir):
        os.makedirs(cov_dir)
if not os.path.exists(out_dir_prova):
        os.makedirs(out_dir_prova)





mask = hp.read_map(f'/ehome/bdecaro/xcmbneed/src/mask/EUCLID/mask_planck_comm_2018_x_euclid_fsky0p35_nside={nside}.fits')
mask[np.where(mask>=0.5 )]=1
mask[np.where(mask<0.5 )]=0
fsky = np.mean(mask)
print(fsky)
wl = hp.anafast(mask, lmax=lmax)

# Loading theory spectra
xcspectra = spectra.XCSpectraFile(fname_xcspectra, WantTG = True, nbins=10)

Nll = np.ones_like(xcspectra.clgg)/simparams['ngal']

ell_cl = np.arange(lmax+1)
factor = ell_cl*(ell_cl+1)/(2*np.pi)
fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)
for b in range(nbins):
        #if bb==b: continue
    ax.plot(ell_cl, xcspectra.clgg[b,b][ell_cl]/(xcspectra.clgg[b,b][ell_cl]+Nll[b,b][ell_cl])-1, label=f'G{b+1}G{b+1}')

ax.legend(ncol=3)
plt.savefig(out_dir+'noise.png')

# Simulations class
simulations = sims.KGsimulations(xcspectra, sims_dir, simparams,  WantTG = True)
simulations.Run(nsim, WantTG = True,EuclidSims=True,nbins=10)

# Needlet Analysis
myanalysis = analysis.NeedAnalysis(jmax, lmax, out_dir, simulations, nbins=10, EuclidSims=True)

# Theory Needlet spectra
need_theory = spectra.NeedletTheory(myanalysis.B)

B=myanalysis.B
print(B)

# Computing simulated betajs
print("...computing Betajs for simulations...")
fname_betaj_sims_TS_galT = [f'betaj_sims_TS_galT_nbins{bin}_jmax{jmax}_B_{B}_nside{nside}.dat' for bin in range(nbins)]
fname_betaj_sims_TS_galT_mask = [f'betaj_sims_TS_galT_nbins{bin}_jmax{jmax}_B_{B}_nside{nside}_fsky_{fsky}.dat' for bin in range(nbins)]
betaj_sims_TS_galT = np.zeros((nbins,nsim, jmax+1))
betaj_sims_TS_galT_mask = np.zeros((nbins,nsim, jmax+1))
fname_cov_TS_galT = [f'cov_TS_galT_nbins{bin}_jmax{jmax}_B{B:0.2f}_nside{nside}.dat' for bin in range(nbins) ]
fname_cov_TS_galT_mask = [f'cov_TS_galT_nbins{bin}_jmax{jmax}_B{B:0.2f}_nside{nside}_fsky_{fsky}.dat' for bin in range(nbins) ]
cov_TS_galT = np.zeros((nbins,nbins, (jmax+1),(jmax+1)))
cov_TS_galT_mask = np.zeros((nbins,nbins, (jmax+1),(jmax+1)))
corr_TS_galT = np.zeros((nbins,nbins, 2*(jmax+1),2*(jmax+1)))
corr_TS_galT_mask = np.zeros((nbins,nbins, 2*(jmax+1),2*(jmax+1)))
betaj_TS_galT_mean = np.zeros((nbins, jmax+1))
betaj_TS_galT_mask_mean = np.zeros((nbins, jmax+1))


#funzione per covariance
def covarfn(a, b):
    cov = np.zeros((a.shape[1], a.shape[1]))
    av_a = a.mean(axis=0)
    av_b=b.mean(axis=0)
    for j in range(0, cov.shape[0]):
        for jj in range(0, cov.shape[0]):
            for i in range(0, a.shape[0]):
                cov[j,jj] += (a[i,j] - av_a[j]) * (b[i,jj] - av_b[jj])
    return (cov / (a.shape[0]-1))

for bin in range(nbins):
    print(f'Bin={bin}')
    betaj_sims_TS_galT[bin] = myanalysis.GetBetajSimsFromMaps(f'T', nsim, nbins,field2=f'g{bin+1}', fname=fname_betaj_sims_TS_galT[bin])
    betaj_sims_TS_galT_mask[bin] = myanalysis.GetBetajSimsFromMaps(f'T', nsim, nbins,field2=f'g{bin+1}', mask= mask, fname=fname_betaj_sims_TS_galT_mask[bin])
    # <Beta_j>_MC
    betaj_TS_galT_mean[bin]    = myanalysis.GetBetajMeanFromMaps(f'T', nsim, nbins,field2=f'g{bin+1}', fname_sims=fname_betaj_sims_TS_galT[bin])
    betaj_TS_galT_mask_mean[bin]     = myanalysis.GetBetajMeanFromMaps(f'T', nsim, nbins,field2=f'g{bin+1}', fname_sims=fname_betaj_sims_TS_galT_mask[bin])


# Covariances
print("...computing Cov Matrices...")
for bin in range(nbins):
        #fname_cov_TS_galT[bin] = f'cov_TS_galT_nbins{bin}_jmax{jmax}_B{B:0.2f}_nside{nside}.dat' 
        #fname_cov_TS_galT_mask[bin] = f'cov_TS_galT_nbins{bin}_jmax{jmax}_B{B:0.2f}_nside{nside}_fsky_{fsky}.dat' 
        #cov_TS_galT[b], corr_TS_galT[b]         = myanalysis.GetCovMatrixFromMaps(field1=f'T', nsim=nsim, nbins=nbins,field2=f'g{b+1}',fname=fname_cov_TS_galT[b], fname_sims=fname_betaj_sims_TS_galT[b])
        #cov_TS_galT_mask[b], corr_TS_galT_mask[b]          = myanalysis.GetCovMatrixFromMaps(field1=f'T', nsim=nsim,nbins=nbins, field2=f'g{b+1}',fname=fname_cov_TS_galT_mask[b], mask = mask,fname_sims=fname_betaj_sims_TS_galT_mask[b])

    for bbin in range(nbins):
        cov_TS_galT[bin,bbin] = covarfn(betaj_sims_TS_galT[bin], betaj_sims_TS_galT[bbin])
        cov_TS_galT_mask[bin,bbin] = covarfn(betaj_sims_TS_galT_mask[bin], betaj_sims_TS_galT_mask[bbin])
        #print(cov_TS_galT_mask[bin,bbin])
        #print(betaj_sims_TS_galT[bin].T.shape, betaj_sims_TS_galT[bbin].T.shape)
        #cov_TS_galT[bin,bbin] = np.cov(m=betaj_sims_TS_galT[bin].T, y= betaj_sims_TS_galT[bbin].T )
        #cov_TS_galT_mask[bin,bbin] = np.cov(m=betaj_sims_TS_galT_mask[bin].T,y= betaj_sims_TS_galT_mask[bbin].T )
        print("...saving Covariance Matrix to output " + cov_dir + f'cov_TS_galT_nbins{bin}x{bbin}_jmax{jmax}_B{B:0.2f}_nside{nside}.dat' + "...")
        np.savetxt(cov_dir+f'cov_TS_galT_nbins{bin}x{bbin}_jmax{jmax}_B{B:0.2f}_nside{nside}.dat',  cov_TS_galT[bin,bbin], header='Covariance matrix <beta_j1 beta_j2>')
        print("...saving Covariance Matrix to output " + cov_dir + f'cov_TS_galT_nbins{bin}x{bbin}_jmax{jmax}_B{B:0.2f}_nside{nside}_fsky_{fsky}.dat' + "...")
        np.savetxt(cov_dir+f'cov_TS_galT_nbins{bin}x{bbin}_jmax{jmax}_B{B:0.2f}_nside{nside}_fsky_{fsky}.dat',  cov_TS_galT_mask[bin,bbin], header='Covariance matrix <beta_j1 beta_j2>')
        corr_TS_galT[bin,bbin]=np.corrcoef(betaj_sims_TS_galT[bin].T, betaj_sims_TS_galT[bbin].T )
        corr_TS_galT_mask[bin,bbin]=np.corrcoef(betaj_sims_TS_galT_mask[bin].T, betaj_sims_TS_galT_mask[bbin].T )
print("...done...")

    


# Some plots
print("...here come the plots...")

# Covariances
#fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(25,18))
fig, axs = plt.subplots(ncols=2, nrows=2,figsize=(20,17)) 
#fig.suptitle(r'$B = %1.2f $' %B + r'$N_{side} =$'+str(simparams['nside']) + r' $N_{sim} = $'+str(nsim))
plt.suptitle(r'$D = %1.2f $' %B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~n_{\mathrm{bins}} =$'+str(nbins) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))
mask_ = np.tri(corr_TS_galT[0,0].shape[0],corr_TS_galT[0,0].shape[1],0)

#plt.subplot(131)
axs[0,0].set_title(r'Corr $T^S\times gal^S$ Bin=1,1')
sns.color_palette("crest", as_cmap=True)
sns.heatmap(corr_TS_galT[0,0], annot=True, fmt='.2f',  mask=mask_,cmap  = 'viridis', ax=axs[0,0])

##plt.subplot(132)
axs[0,1].set_title(r'Corr $T^S\times gal^S$ Bin=1,2')
sns.heatmap(corr_TS_galT[0,1], annot=True, fmt='.2f',cmap  = 'viridis',  mask=mask_, ax=axs[0,1])
#
##plt.subplot(133)
#ax3.set_title(r'Corr $\delta^T \times \kappa^T$ Masked')
#sns.heatmap(corr_kappaT_deltaT_mask, annot=True, fmt='.2f', mask=mask_,ax=ax3)
axs[1,0].set_title(r'Corr $T^S\times gal^T$ Bin=1,3')
sns.heatmap(corr_TS_galT[0,2], annot=True, fmt='.2f',cmap  = 'viridis', mask=mask_, ax=axs[1,0])

fig.tight_layout()
plt.savefig(cov_dir+f'corr_T_gal_noise_nbins{nbins}_jmax{jmax}_B{B:0.2f}_nsim{nsim}_nside{nside}.png')

fig, ax=plt.subplots(2,2)
plt.suptitle('Covariance from simualtion')
cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)

norm=matplotlib.colors.Normalize(vmin = cov_TS_galT_mask[0,0,:,:].min(), vmax = cov_TS_galT_mask[0,0,:,:].max(), clip = False)
plt1=ax[0,0].imshow(cov_TS_galT_mask[0,0,:,:], norm=norm, cmap=cmap)
plt2=ax[0,1].imshow(cov_TS_galT_mask[0,1,:,:], norm=norm, cmap=cmap)
plt3=ax[1,0].imshow(cov_TS_galT_mask[1,0,:,:], norm=norm, cmap=cmap)
plt4=ax[1,1].imshow(cov_TS_galT_mask[1,1,:,:], norm=norm, cmap=cmap)

plt.colorbar(plt1, ax=ax)
ax[0,0].invert_yaxis()

ax[0,1].invert_yaxis()

ax[1,0].invert_yaxis()
ax[1,0].set_ylabel('j')
ax[1,0].set_xlabel('j')

ax[1,1].invert_yaxis()

#plt.tight_layout()
fig.savefig(cov_dir+'covariance_sims_redshift0_j.png')


########################################################################################################################################################################################

# Theory + Normalization Needlet power spectra
gammatg = np.zeros((nbins,jmax+1))
delta_gamma_tomo = need_theory.variance_gammaj_tomo(nbins=nbins,jmax=jmax,lmax=lmax, cltt = xcspectra.cltt, cltg = xcspectra.cltg, clgg = xcspectra.clgg, wl=wl, noise_gal_l=simparams['ngal'])
for bin in range(nbins):
    gammatg[bin]    = need_theory.gammaJ(jmax=jmax,lmax=lmax, wl=wl, cl=xcspectra.cltg[bin])
    #print(xcspectra.cltg[bin].shape)
    #for ibin in range(nbins):
    #   
    #    variance_gamma[bin,ibin] = need_theory.variance_gammaj(jmax=jmax, lmax=lmax, wl=wl,cltt = xcspectra.cltt, cltg = xcspectra.cltg[bin], clgg = xcspectra.clgg[bin,ibin], noise_gal_l=Nll)

np.savetxt(out_dir+f'beta_TS_galT_theoretical_fiducial_B{B}_nbins{nbins}.dat', gammatg)


#np.savetxt(out_dir+f'variance_beta_TS_galT_theoretical_fiducial_B{B}_nbins{nbins}.dat', delta_gamma_tomo.sum(axis=1))


####  NOISE
fig = plt.figure(figsize=(17,10))
plt.suptitle(r'$D = %1.2f $' %B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~n_{\mathrm{bins}} =$'+str(nbins) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

color = sns.color_palette("husl",n_colors=10).as_hex()
ax = fig.add_subplot(1, 1, 1)
for bin, c in zip(range(nbins), color):

    ax.plot(myanalysis.jvec[1:jmax], gammatg[bin][1:jmax], color=c,label=f'Theory, Bin = {bin}')
    ax.errorbar(myanalysis.jvec[1:jmax], betaj_TS_galT_mask_mean[bin][1:jmax], yerr=np.sqrt(np.diag(cov_TS_galT_mask[bin,bin])[1:jmax])/np.sqrt(nsim) ,color=c,fmt='o',ms=5,capthick=2)

ax.set_xticks(myanalysis.jvec[1:jmax])
ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\Gamma_j^{\mathrm{TG}}$')

fig.tight_layout()
plt.savefig(out_dir_prova+f'betaj_theory_NOISE_T_gal_jmax{jmax}_D{B:0.2f}_nsim{nsim}_nside{nside}_nbins{nbins}.png', bbox_inches='tight')






####################################################################################################################################################################

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~n_{\mathrm{bins}} =$'+str(nbins) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.axhline(ls='--', color='grey')
for bin in range(1):
    #ax.errorbar(myanalysis.jvec[1:jmax], (betaj_TS_galT_mean[bin][1:jmax] -betatg[bin][1:jmax])/betatg[bin][1:jmax], yerr=np.sqrt(np.diag(cov_TS_galT[bin])[1:jmax])/(np.sqrt(nsim)*betatg[bin][1:jmax]),  fmt='o',  ms=5,label=f'Bin = {bin} No Noise')
    ax.errorbar(myanalysis.jvec[1:jmax], (betaj_TS_galT_mask_mean[bin][1:jmax] -gammatg[bin][1:jmax])/gammatg[bin][1:jmax], yerr=np.sqrt(np.diag(cov_TS_galT_mask[bin,bin])[1:jmax])/(np.sqrt(nsim)*gammatg[bin][1:jmax]),  fmt='o',  ms=5,label=f'Bin = {bin}, Shot Noise, variance from sim')
    ax.errorbar(myanalysis.jvec[1:jmax], (betaj_TS_galT_mask_mean[bin][1:jmax] -gammatg[bin][1:jmax])/gammatg[bin][1:jmax], yerr=np.sqrt(np.diag(delta_gamma_tomo[bin,bin])[1:jmax])/(np.sqrt(nsim)*gammatg[bin][1:jmax]),  fmt='o',  ms=5,label=f'Bin = {bin}, Shot Noise, analytical variance')

ax.legend(loc='best')
ax.set_xticks(myanalysis.jvec[1:jmax])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xlabel(r'$j$', fontsize=22)
ax.set_ylabel(r'$\frac{\langle \Gamma_j^{\mathrm{TG}} \rangle - \Gamma_j^{\mathrm{TG}\,, th}}{\Gamma_j^{\mathrm{TG}\,, th}}$', fontsize=22)
#ax.set_ylim([-0.2,0.3])

fig.tight_layout()
plt.savefig(out_dir_prova+f'betaj_mean_T_gal_jmax{jmax}_D{B:1.2f}_nsim{nsim}_nside{nside}_nbins{nbins}.png', bbox_inches='tight')


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
#ax.plot(myanalysis.jvec[1:jmax], (betaj_TS_galT_mask_mean[1:jmax] -betatg[1:jmax])/(delta[1:jmax]/np.sqrt(nsim)),'o', ms=10,color='#2b7bbc')#, label=r'$T^S \times gal^S$, sim cov')
#
##print(np.sqrt(np.diag(cov_TS_galT))/(np.sqrt(nsim)*betatg) - delta/(np.sqrt(nsim)*betatg))
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
fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)
for b in range(nbins):
    ax.plot(myanalysis.jvec[1:], (np.sqrt(np.diag(cov_TS_galT_mask[b,b])[1:jmax+1])/np.sqrt(np.diag(delta_gamma_tomo[b,b])[1:jmax+1])-1)*100 ,'o',ms=10, label=f'Bin={b}x{b}')
ax.axhline(ls='--', c='k', linewidth=0.8)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(myanalysis.jvec[1:])
ax.legend(ncol=2)
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'% $\sigma_{\mathrm{sims}}/\sigma_{\mathrm{analytic}}$ - 1')

fig.tight_layout()
plt.savefig(out_dir_prova+f'diff_cov_betaj_theory_T_gal_jmax{jmax}_D{B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')

#### fuoei diagonale per bin i bin i

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)
for b in range(nbins):
    ax.plot(myanalysis.jvec[1:jmax], (np.sqrt(np.diag(cov_TS_galT_mask[b,b], k=-1)[1:jmax])/np.sqrt(np.diag(delta_gamma_tomo[b,b], k=-1)[1:jmax])-1)*100 ,'o',ms=10, label=f'Bin={b}')
ax.axhline(ls='--', c='k', linewidth=0.8)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(myanalysis.jvec[1:])
ax.legend(ncol=2)
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'% $Cov(superdiag)_{\mathrm{sims}}/Cov(superdiag)_{\mathrm{analytic}}$ - 1')

fig.tight_layout()
plt.savefig(out_dir_prova+f'diff_cov_out_diag_betaj_theory_T_gal_jmax{jmax}_D{B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')


#diff redshift b redshift b+1

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)
for b in range(nbins-1):
    ax.plot(myanalysis.jvec[1:jmax+1], (np.sqrt(np.diag(cov_TS_galT_mask[b,b+1])[1:jmax+1])/np.sqrt(np.diag(delta_gamma_tomo[b,b+1])[1:jmax+1])-1)*100 ,'o',ms=10, label=f'Bin={b}x{b+1}')
ax.axhline(ls='--', c='k', linewidth=0.8)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(myanalysis.jvec[1:])
ax.legend(ncol=2)
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'% $Cov(diag)_{\mathrm{sims}}/Cov(diag)_{\mathrm{analytic}}$ - 1')

fig.tight_layout()
plt.savefig(out_dir_prova+f'diff_cov_redshift_out_diag+1_betaj_theory_T_gal_jmax{jmax}_D{B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')



#diff redshift b redshift b+2

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)
for b in range(nbins-2):
    ax.plot(myanalysis.jvec, (np.sqrt(np.diag(cov_TS_galT_mask[b,b+2]))/np.sqrt(np.diag(delta_gamma_tomo[b,b+2]))-1)*100 ,'o',ms=10, label=f'Bin={b}x{b+2}')
    print(f'bin={b,b+2}:{np.diag(delta_gamma_tomo[b,b+2])}')

ax.axhline(ls='--', c='k', linewidth=0.8)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(myanalysis.jvec[1:])
ax.legend(ncol=2)
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'% $Cov(diag)_{\mathrm{sims}}/Cov(diag)_{\mathrm{analytic}}$ - 1')

fig.tight_layout()
plt.savefig(out_dir_prova+f'diff_cov_redshift_out_diag+2_betaj_theory_T_gal_jmax{jmax}_D{B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')



################################################################################################


ell_cl = np.arange(lmax+1)
factor = ell_cl*(ell_cl+1)/(2*np.pi)
fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)
for b in range(nbins):
    for bb in range(nbins):
        #if bb==b: continue
        ax.plot(ell_cl, xcspectra.clgg[b,bb][ell_cl], label=f'G{b+1}G{bb+1}')

ax.legend(ncol=3)
plt.savefig(out_dir+'clgg_bins.png')


fig , ax= plt.subplots(1,1)
cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
sns.heatmap(delta_gamma_tomo.sum(axis=3).sum(axis=2), cmap=cmap, ax=ax)#, xticklabels=[1,2,3], yticklabels=[1,2,3])
ax.invert_yaxis()
ax.set_xlabel('redshift bin')
ax.set_ylabel('redshift bin')
ax.set_title('Covariance - Tomography')
plt.tight_layout()
fig.savefig(out_dir_prova+'covariance_tomography_analytic.png')

fig, ax=plt.subplots(1,1)
ax.set_title('Covariance')
norm=matplotlib.colors.Normalize(vmin = np.min(delta_gamma_tomo[0,1,:,:]), vmax = np.max(delta_gamma_tomo[0,1,:,:]), clip = False)
plt1=ax.imshow(delta_gamma_tomo[0,1,:,:], norm=norm, cmap=cmap)
plt.colorbar(plt1, ax=ax)
ax.invert_yaxis()
ax.set_xlabel(r'j')
ax.set_ylabel(r'j')
plt.tight_layout()
fig.savefig(out_dir_prova+'covariance_redshift01_j_analytic.png')

icov=np.zeros((nbins, nbins, jmax+1,jmax+1))
icov_theory=np.zeros((nbins, nbins, jmax+1,jmax+1))
#for ij in range(jmax+1):
#    for iij in range(jmax+1):
#        icov[:,:,ij, iij] = np.linalg.inv(cov_TS_galT_mask[:, :, ij, iij])
#        icov_theory[:,:,ij, iij] = np.linalg.inv(delta_gamma_tomo[:, :, ij, iij])


for b in range(nbins):
    for bb in range(nbins):
        icov[b,bb,1:,1:] = np.linalg.inv(cov_TS_galT_mask[b, bb, 1:, 1:])
        icov_theory[b,bb,1:,1:] = np.linalg.inv(delta_gamma_tomo[b, bb, 1:, 1:])


fig, ax=plt.subplots(1,1)
ax.set_title('Inverse covariance, j=1')
norm=matplotlib.colors.Normalize(vmin = np.min(icov[:,:,1,1]), vmax = np.max(icov[:,:,1,1]), clip = False)
plt1=ax.imshow(icov[:,:,1,1], norm=norm, cmap=cmap)
plt.colorbar(plt1, ax=ax)
ax.invert_yaxis()
ax.set_xlabel(r'redshift')
ax.set_ylabel(r'redshift')
plt.tight_layout()
fig.savefig(out_dir_prova+'icov_j1_redshift.png')


fig, ax=plt.subplots(1,1)
ax.set_title('Inversexcovariance, j=0')
norm=matplotlib.colors.Normalize(vmin = -1, vmax = 1, clip = False)
plt1=ax.imshow(np.matmul(icov[:,:,0,0],cov_TS_galT_mask[:, :, 0, 0]),  cmap=cmap)
plt.colorbar(plt1, ax=ax)
ax.invert_yaxis()
ax.set_xlabel(r'redshift')
ax.set_ylabel(r'redshift')
plt.tight_layout()
fig.savefig(out_dir_prova+'icov_prova.png')

def S_2_N(beta, icov):
    nj  = beta.shape[1]
    s2n=np.zeros(nj)
    temp = np.zeros((nj, nj, nbins, nbins))
    temp1 = np.zeros((nj, nj, nbins))
    temp2 = np.zeros((nj, nj))
    for ij in range(1,nj):
        for iij in range(1,nj):
            for ibin in range(nbins):
                for iibin in range(nbins):
                    temp[ij, iij, ibin, iibin] = beta[ibin, ij]*beta[iibin, iij]*icov[ibin, iibin, ij, iij ]
                    temp1[ij, iij, ibin] +=temp[ij, iij, ibin, iibin]
                temp2[ij, iij] +=temp1[ij, iij, ibin]
            s2n[ij]+=temp2[ij,iij]
    #nj  = beta.shape[1]
    #s2n = np.zeros(nj)
    #for ij in range(nj):
    #    for iij in range(nj):
    #        s2n[ij] += np.dot(beta[:, ij], np.dot(icov[:, :, ij, iij], beta[:, iij]))
    return s2n

######################################################################################
############################## SIGNAL TO NOISE RATIO #################################


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

def S_2_N_j_tomo(beta, cov):
   nj  = beta.shape[1]
   nbins=beta.shape[0]
   s2n = np.zeros((nj, nbins))
   temp = np.zeros((nj, nj,nbins))
   for ibins in range(nbins):
       for ij in range(1,nj):
            s2n[ij, ibins] =beta[ibins,ij]**2/cov[ibins,ibins,ij, ij]
   return s2n   

s2n = S_2_N(betaj_TS_galT_mask_mean, icov)

s2n_theory = S_2_N(gammatg, icov_theory)


fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~n_{\mathrm{bins}} =$'+str(nbins) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

#ax.plot(myanalysis.jvec[:jmax+1], s2n_theory, color='#2b7bbc',marker = 'o',ms=10, label='Theory')
ax.plot(myanalysis.jvec[:jmax+1], s2n, marker = 'o',ms=10,label='Sims')

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
plt.legend()
ax.set_xlabel(r'$j$')
ax.set_ylabel('Signal-to-Noise ratio')

fig.tight_layout()
plt.savefig(out_dir_prova+f'SNR_betaj_theory_T_gal_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_nbins{nbins}.png', bbox_inches='tight') #questa


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
#s2n_onebin = S_2_N(beta_T_galT_one_bin_mean, cov_T_galT_one_bin)

jvec = np.arange(0,jmax+1)
jmax_vec = myanalysis.jvec
lmax_vec=fl_j(jmax)
s2n_cum= S_2_N_cum(s2n, jvec)
s2n_cum_theory= S_2_N_cum(s2n_theory, jvec)

#s2n_cum_one_bin= S_2_N_cum(s2n_onebin, jvec)

print(f's2n_cum_tomo={s2n_cum[-1]},s2n_cum_theory={s2n_cum_theory[-1]}') #s2n_cum_no_tomo={s2n_cum_one_bin[-1]}, 

fig = plt.figure(figsize=(10,11))

plt.suptitle(r'$D = %1.2f $' %B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~n_{\mathrm{bins}} =$'+str(nbins) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

#ax.plot(lmax_vec, s2n_cum, label='No Noise')
#ax.plot(lmax_vec, s2n_cum_one_bin, label='No tomo')
ax.plot(lmax_vec, s2n_cum,label='Sims')
#ax.plot(lmax_vec, s2n_cum_theory,color='#2b7bbc', label='Theory')
ax.set_xscale('log')
ax.set_xlim(left=3, right=250)
#ax.set_ylim(top=4.)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 

ax.set_xlabel(r'$\ell_{\mathrm{max}}$')
ax.set_ylabel('Cumulative Signal-to-Noise ratio')
ax.legend()

fig.tight_layout()

plt.savefig(out_dir_prova+f'SNR_cumulative_betaj_theory_T_gal_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_nbins{nbins}.png', bbox_inches='tight') #questa

s2n_tomo = S_2_N_j_tomo(betaj_TS_galT_mask_mean,delta_gamma_tomo )
s2n_tomo_cum = np.zeros((len(jvec),nbins))
for ibins in range(nbins):
    s2n_tomo_cum[:,ibins] = S_2_N_cum(s2n_tomo[:,ibins], jvec)


fig = plt.figure(figsize=(10,11))

plt.suptitle(r'$D = %1.2f $' %B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~n_{\mathrm{bins}} =$'+str(nbins) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)
for ibins in range(nbins):
    plt.plot(lmax_vec,s2n_tomo_cum[:,ibins], label=r'$G^{%d}\times G^{%d}$'%(ibins+1,ibins+1))
plt.xscale('log')
plt.xlim(left=2)
plt.xlabel(r'$\ell_{\mathrm{max}}$')
plt.ylabel(r'S/N')
plt.tight_layout()
plt.legend(ncol=2)
plt.savefig(out_dir+f'SNR_cumulative_tomo_betaj_theory_T_gal_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_nbins{nbins}.png', bbox_inches='tight') #questa
