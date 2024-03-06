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
out_dir_prova         = f'output_needlet_TG/EUCLID/Tomography/TG_{nside}_lmax{lmax}_nbins{nbins}_nsim{nsim}_nuovo/'
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


for bin in range(nbins):
    print(f'Bin={bin}')
    betaj_sims_TS_galT[bin] = myanalysis.GetBetajSimsFromMaps(f'T', nsim, nbins,field2=f'g{bin+1}', fname=fname_betaj_sims_TS_galT[bin])
    betaj_sims_TS_galT_mask[bin] = myanalysis.GetBetajSimsFromMaps(f'T', nsim, nbins,field2=f'g{bin+1}', mask= mask, fname=fname_betaj_sims_TS_galT_mask[bin])
    # <Beta_j>_MC
    betaj_TS_galT_mean[bin]    = myanalysis.GetBetajMeanFromMaps(f'T', nsim, nbins,field2=f'g{bin+1}', fname_sims=fname_betaj_sims_TS_galT[bin])
    betaj_TS_galT_mask_mean[bin]     = myanalysis.GetBetajMeanFromMaps(f'T', nsim, nbins,field2=f'g{bin+1}', fname_sims=fname_betaj_sims_TS_galT_mask[bin])


# Covariances
print("...computing Cov Matrices...")
betaj_sims_TS_galT_reshaped = betaj_sims_TS_galT.reshape((nsim,nbins*(jmax+1)))
betaj_sims_TS_galT_mask_reshaped = betaj_sims_TS_galT_mask.reshape((nsim,nbins*(jmax+1)))

betaj_TS_galT_mean_reshaped = betaj_sims_TS_galT_reshaped.mean(axis=0)
betaj_TS_galT_mask_mean_reshaped = betaj_sims_TS_galT_mask_reshaped.mean(axis=0)

cov_betaj_sims_TS_galT_full = np.cov(betaj_sims_TS_galT_reshaped.T)
cov_betaj_sims_TS_galT_mask_full = np.cov(betaj_sims_TS_galT_mask_reshaped.T)

corr_betaj_sims_TS_galT_full = np.corrcoef(betaj_sims_TS_galT_reshaped.T)
corr_betaj_sims_TS_galT_mask_full = np.corrcoef(betaj_sims_TS_galT_mask_reshaped.T)

print("...done...")

# Some plots
print("...here come the plots...")

# Covariances

fig,ax = plt.subplots(1,1,figsize=(5, 5))
plt.title('Cov from sims')
plt1=ax.imshow(corr_betaj_sims_TS_galT_mask_full, cmap='crest', norm='log')#, vmin=-0.1, vmax=0.1)
ax.invert_yaxis()
plt.colorbar(plt1, ax=ax)
plt.show()
fig.savefig(cov_dir+'correlation_sims_full.png')


########################################################################################################################################################################################

# Theory + Normalization Needlet power spectra
gammatg = np.zeros((nbins,jmax+1))
delta_gamma_tomo = need_theory.variance_gammaj_tomo_abs(nbins=nbins,jmax=jmax,lmax=lmax, cltt = xcspectra.cltt, cltg = xcspectra.cltg, clgg = xcspectra.clgg, wl=wl, noise_gal_l=simparams['ngal'])
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

############################# DIFF COVARIANCES #####################################
fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)
for b in range(nbins):
    ax.plot(myanalysis.jvec[1:], (np.sqrt(np.diag(cov_betaj_sims_TS_galT_mask_full[b*(lmax+1):(b+1)*(lmax+1), b*(lmax+1):(b+1)*(lmax+1)])[1:jmax+1])/np.sqrt(np.diag(delta_gamma_tomo[b,b])[1:jmax+1])-1)*100 ,'o',ms=10, label=f'Bin={b}x{b}')
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
    ax.plot(myanalysis.jvec[1:jmax], (np.sqrt(np.diag(cov_betaj_sims_TS_galT_mask_full[b*(lmax+1):(b+1)*(lmax+1), b*(lmax+1):(b+1)*(lmax+1)], k=-1)[1:jmax])/np.sqrt(np.diag(delta_gamma_tomo[b,b], k=-1)[1:jmax])-1)*100 ,'o',ms=10, label=f'Bin={b}')
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
    ax.plot(myanalysis.jvec[1:jmax+1], (np.sqrt(np.diag(cov_betaj_sims_TS_galT_mask_full[b*(lmax+1):(b+1)*(lmax+1), (b+1)*(lmax+1):(b+2)*(lmax+1)])[1:jmax+1])/np.sqrt(np.diag(delta_gamma_tomo[b,b+1])[1:jmax+1])-1)*100 ,'o',ms=10, label=f'Bin={b}x{b+1}')
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
    ax.plot(myanalysis.jvec, (np.sqrt(np.diag(cov_TS_galT_mask[b*(lmax+1):(b+1)*(lmax+1), (b+2)*(lmax+1):(b+3)*(lmax+1)]))/np.sqrt(np.diag(delta_gamma_tomo[b,b+2]))-1)*100 ,'o',ms=10, label=f'Bin={b}x{b+2}')
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
delta_gamma_tomo_reshaped = delta_gamma_tomo.reshape((nbins*(jmax+1), nbins*(jmax+1)))

fig , ax= plt.subplots(1,1)
cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
sns.heatmap(delta_gamma_tomo_reshaped, cmap=cmap, ax=ax)#, xticklabels=[1,2,3], yticklabels=[1,2,3])
ax.invert_yaxis()
ax.set_xlabel('j')
ax.set_ylabel('redshift bin')
ax.set_title('Covariance - Tomography')
plt.tight_layout()
fig.savefig(out_dir_prova+'covariance_tomography_analytic.png')
