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
out_dir_prova         = f'output_needlet_TG/EUCLID/Tomography/TG_{nside}_lmax{lmax}_nbins{nbins}_nsim{nsim}/prova_covariance/'
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
lmax_negative = []

for ibin in range(nbins):
    for iibin in range(ibin,nbins):
        l_negative_l = []
        for l in xcspectra.ell:
            if xcspectra.clgg[ibin, iibin][l]<0:
                l_negative_l.append(l)
            else:
                l_negative_l.append(0)
        lmax_negative.append(max(l_negative_l))
fig = plt.figure(figsize=(17,10))
for ibin in range(nbins):
    for iibin in range(ibin,nbins):
        if True in (xcspectra.clgg[ibin, iibin][l] <0 for l in xcspectra.ell):
            plt.plot(xcspectra.ell[:max(lmax_negative) ], xcspectra.clgg[ibin, iibin][:max(lmax_negative)], color=sns.color_palette('husl',nbins)[iibin], label=r'G$^{%d}\times$G$^{%d}$'%(ibin+1,iibin+1))


plt.xlim([0,max(lmax_negative) ])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C^{\mathrm{TG}}_{\ell}$')
#plt.ylabel(r'$\dfrac{\ell(\ell+1)}{2\pi}C^{\mathrm{TG}}_{\ell}$')
plt.legend(ncol=5)
plt.savefig(out_dir_prova+f'clgg_negative.png')

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
#cov_TS_galT = np.zeros((nbins,nbins, (jmax+1),(jmax+1)))
cov_TS_galT_mask = np.zeros((nbins,nbins, (jmax+1),(jmax+1)))
#corr_TS_galT = np.zeros((nbins,nbins, 2*(jmax+1),2*(jmax+1)))
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
for bin in range(nbins):
    for bbin in range(nbins):
        #cov_TS_galT[bin,bbin] = np.loadtxt(cov_dir+f'cov_TS_galT_nbins{bin}x{bbin}_jmax{jmax}_B{B:0.2f}_nside{nside}.dat')
        cov_TS_galT_mask[bin,bbin] = np.loadtxt(cov_dir+f'cov_TS_galT_nbins{bin}x{bbin}_jmax{jmax}_B{B:0.2f}_nside{nside}_fsky_{fsky}.dat')
        
        #corr_TS_galT[bin,bbin]=np.corrcoef(betaj_sims_TS_galT[bin].T, betaj_sims_TS_galT[bbin].T )
        corr_TS_galT_mask[bin,bbin]=np.corrcoef(betaj_sims_TS_galT_mask[bin].T, betaj_sims_TS_galT_mask[bbin].T )
print("...done...")

# Theory + Normalization Needlet power spectra
gammatg = np.zeros((nbins,jmax+1))
delta_gamma_tomo = need_theory.variance_gammaj_tomo(nbins=nbins,jmax=jmax,lmax=lmax, cltt = xcspectra.cltt, cltg = xcspectra.cltg, clgg = xcspectra.clgg, wl=wl, noise_gal_l=simparams['ngal'])
for bin in range(nbins):
    gammatg[bin]    = need_theory.gammaJ(jmax=jmax,lmax=lmax, wl=wl, cl=xcspectra.cltg[bin])

####################################################################################
############################# DIFF COVARIANCES #####################################
fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)
for b in range(nbins):
    ax.plot(myanalysis.jvec[1:], (np.sqrt(np.diag(cov_TS_galT_mask[b,b])[1:])/np.sqrt(np.diag(delta_gamma_tomo[b,b])[1:])-1)*100 ,'o',ms=10, label=f'Bin={b}x{b}')
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
    ax.plot(myanalysis.jvec[1:], (np.sqrt(np.diag(cov_TS_galT_mask[b,b+2])[1:])/np.sqrt(np.diag(delta_gamma_tomo[b,b+2])[1:])-1)*100 ,'o',ms=10, label=f'Bin={b}x{b+2}')
    print(f'bin={b,b+2}:{np.diag(delta_gamma_tomo[b,b+2])[1:]}')

ax.axhline(ls='--', c='k', linewidth=0.8)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(myanalysis.jvec[1:])
ax.legend(ncol=2)
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'% $Cov(diag)_{\mathrm{sims}}/Cov(diag)_{\mathrm{analytic}}$ - 1')

fig.tight_layout()
plt.savefig(out_dir_prova+f'diff_cov_redshift_out_diag+2_betaj_theory_T_gal_jmax{jmax}_D{B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')


####################################################################################################
############################# DIFF COVARIANCES CLGG NEGATIVE =0 ####################################
delta_gamma_tomo0 = need_theory.variance_gammaj_tomo_0(nbins=nbins,jmax=jmax,lmax=lmax, cltt = xcspectra.cltt, cltg = xcspectra.cltg, clgg = xcspectra.clgg, wl=wl, noise_gal_l=simparams['ngal'])


#diff redshift b redshift b+1

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)
for b in range(nbins-1):
    ax.plot(myanalysis.jvec[1:jmax+1], (np.sqrt(np.diag(cov_TS_galT_mask[b,b+1])[1:jmax+1])/np.sqrt(np.diag(delta_gamma_tomo0[b,b+1])[1:jmax+1])-1)*100 ,'o',ms=10, label=f'Bin={b}x{b+1}')
ax.axhline(ls='--', c='k', linewidth=0.8)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(myanalysis.jvec[1:])
ax.legend(ncol=2)
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'% $Cov(diag)_{\mathrm{sims}}/Cov(diag)_{\mathrm{analytic}}$ - 1')

fig.tight_layout()
plt.savefig(out_dir_prova+f'NEGATIVE0_diff_cov_redshift_out_diag+1_betaj_theory_T_gal_jmax{jmax}_D{B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')



#diff redshift b redshift b+2

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)
for b in range(nbins-2):
    ax.plot(myanalysis.jvec[1:], (np.sqrt(np.diag(cov_TS_galT_mask[b,b+2])[1:])/np.sqrt(np.diag(delta_gamma_tomo0[b,b+2])[1:])-1)*100 ,'o',ms=10, label=f'Bin={b}x{b+2}')
    print(f'bin={b,b+2}:{np.diag(cov_TS_galT_mask[b,b+2])[1:]}')

ax.axhline(ls='--', c='k', linewidth=0.8)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(myanalysis.jvec[1:])
ax.legend(ncol=2)
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'% $Cov(diag)_{\mathrm{sims}}/Cov(diag)_{\mathrm{analytic}}$ - 1')

fig.tight_layout()
plt.savefig(out_dir_prova+f'NEGATIVE0_diff_cov_redshift_out_diag+2_betaj_theory_T_gal_jmax{jmax}_D{B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')


#######################################################################################################
############################# DIFF COVARIANCES CLGG NEGATIVE = abs ####################################
delta_gamma_tomo_abs = need_theory.variance_gammaj_tomo_abs(nbins=nbins,jmax=jmax,lmax=lmax, cltt = xcspectra.cltt, cltg = xcspectra.cltg, clgg = xcspectra.clgg, wl=wl, noise_gal_l=simparams['ngal'])


#diff redshift b redshift b+1

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)
for b in range(nbins-1):
    ax.plot(myanalysis.jvec[1:jmax+1], (np.sqrt(np.diag(cov_TS_galT_mask[b,b+1])[1:jmax+1])/np.sqrt(np.diag(delta_gamma_tomo_abs[b,b+1])[1:jmax+1])-1)*100 ,'o',ms=10, label=f'Bin={b}x{b+1}')
ax.axhline(ls='--', c='k', linewidth=0.8)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(myanalysis.jvec[1:])
ax.legend(ncol=2)
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'% $Cov(diag)_{\mathrm{sims}}/Cov(diag)_{\mathrm{analytic}}$ - 1')

fig.tight_layout()
plt.savefig(out_dir_prova+f'NEGATIVE_ABS_diff_cov_redshift_out_diag+1_betaj_theory_T_gal_jmax{jmax}_D{B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')



#diff redshift b redshift b+2

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)
for b in range(nbins-2):
    ax.plot(myanalysis.jvec[1:], (np.sqrt(np.diag(cov_TS_galT_mask[b,b+2])[1:])/np.sqrt(np.diag(delta_gamma_tomo_abs[b,b+2])[1:])-1)*100 ,'o',ms=10, label=f'Bin={b}x{b+2}')
    print(f'bin={b,b+2}:{np.diag(cov_TS_galT_mask[b,b+2])[1:]}')

ax.axhline(ls='--', c='k', linewidth=0.8)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(myanalysis.jvec[1:])
ax.legend(ncol=2)
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'% $Cov(diag)_{\mathrm{sims}}/Cov(diag)_{\mathrm{analytic}}$ - 1')

fig.tight_layout()
plt.savefig(out_dir_prova+f'NEGATIVE_ABS_diff_cov_redshift_out_diag+2_betaj_theory_T_gal_jmax{jmax}_D{B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')


########## SIGNAL TO NOISE #########
#print(np.where(cov_TS_galT_mask==0))

def S_2_N_j(gammaj, icov):
    nj  = gammaj.shape[1]
    s2n = np.zeros(nj)
    print(nj, gammaj.shape)
    for ij in range(nj):
        for iij in range(nj):
            s2n[ij] += np.dot(gammaj[:, ij], np.dot(icov[:, :, ij, iij], gammaj[:, iij]))
    return s2n

def S_2_N_cum(s2n, jmax):
    s2n_cum = np.zeros(jmax.shape[0])
    for j,jj in enumerate(jmax):
        for ijj in range(1,jj):
            s2n_cum[j] +=s2n[ijj]
        s2n_cum[j]= np.sqrt(s2n_cum[j])      
    return s2n_cum

def fl_j(j_m):
    
    l_j = np.zeros(j_m+1, dtype=int)
    
    for j in range(j_m+1):
            lmin = np.floor(myanalysis.B**(j-1))
            lmax = np.floor(myanalysis.B**(j+1))
            ell  = np.arange(lmin, lmax+1, dtype=int)
            l_j[j] = int(ell[int(np.ceil((len(ell))/2))])
    return l_j

icov_sim=np.zeros((nbins, nbins, jmax+1,jmax+1))

for ij in range(jmax+1):
        for iij in range(jmax+1):
            icov_sim[:, :, ij, iij] = np.linalg.inv(cov_TS_galT_mask[:, :, ij,iij])

lmax_vec=fl_j(jmax)
s2n_j_sim = S_2_N_j(betaj_TS_galT_mask_mean, icov_sim)
#s2n_cum_sim = S_2_N_cum(s2n_j_sim, myanalysis.jvec)

#print(cov_TS_galT_mask, s2n_j_sim)
