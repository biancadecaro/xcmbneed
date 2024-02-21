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
fname_xcspectra = '/ehome/bdecaro/xcmbneed/src/spectra/inifiles/euclid_fiducials_tomography_lmin0.txt' 
sims_dir        = f'/ehome/bdecaro/xcmbneed/src/sims/Euclid_sims_Marina/NSIDE128_TOMOGRAPHY/'
out_dir         = f'output_cls_TG/EUCLID/Tomography/TG_{nside}_lmax{lmax}_nbins{nbins}_nsim{nsim}/'
path_inpainting = 'inpainting/inpainting.py'
cov_dir 		= out_dir+f'covariance/'
if not os.path.exists(cov_dir):
        os.makedirs(cov_dir)


mask = hp.read_map(f'/ehome/bdecaro/xcmbneed/src/mask/EUCLID/mask_planck_comm_2018_x_euclid_fsky0p35_nside={nside}.fits')
mask[np.where(mask>=0.5 )]=1
mask[np.where(mask<0.5 )]=0
bad_v = np.where(mask==0)
fsky = np.mean(mask)
print(fsky)
wl = hp.anafast(mask, lmax=lmax)

# Loading theory spectra
xcspectra = spectra.XCSpectraFile(fname_xcspectra, WantTG = True, nbins=10)
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
plt.savefig(out_dir+f'clgg_negative.png')

# Simulations class
simulations = sims.KGsimulations(xcspectra, sims_dir, simparams,  WantTG = True)
simulations.Run(nsim, WantTG = True,EuclidSims=True,nbins=10)

# Needlet Analysis
myanalysis = analysis.NeedAnalysis(jmax, lmax, out_dir, simulations, nbins=10, EuclidSims=True)

# Theory Needlet spectra
need_theory = spectra.NeedletTheory(myanalysis.B)


#cl_TT_mask_sims = np.zeros((nsim,lmax+1))
#cl_TG_mask_sims = np.zeros((nsim,nbins,lmax+1))
#cl_GG_mask_sims = np.zeros((nsim,nbins,lmax+1))
##mask = hp.read_map('/ehome/bdecaro/xcmbneed/src/mask/EUCLID/mask_planck_comm_2018_x_euclid_fsky0p35_nside=128.fits', verbose=False)
#for n in range(nsim):
#    fname_T = sims_dir + f'map_nbin10_NSIDE{nside}_lmax{lmax}_T_{n+1:05d}.fits' #sims_dir + "sim_" + ('%04d' % n) + "_galT_" + ('%04d' % nside) + ".fits"
#    mapT_mask = hp.read_map(fname_T, verbose=False)
#    mapT_mask[bad_v]=hp.UNSEEN
#    #cl_TT_mask_sims[n, :] =hp.anafast(map1=mapT_mask, map2=mapT_mask, lmax=lmax)
#
#    for bin in range(nbins):
#        fname_gal = sims_dir + f'map_nbin10_NSIDE{nside}_lmax{lmax}_g{bin+1}_{n+1:05d}_noise.fits' # sims_dir + "sim_" + ('%04d' % n) + "_TS_" + ('%04d' % nside) + ".fits"
#        mapgal_mask = hp.read_map(fname_gal, verbose=False)
#        mapgal_mask[bad_v]=hp.UNSEEN
#        #cl_GG_mask_sims[n,bin, :] =hp.anafast(map1=mapgal_mask, map2=mapgal_mask, lmax=lmax)
#        cl_TG_mask_sims[n,bin,:] =hp.anafast(map1=mapT_mask, map2=mapgal_mask, lmax=lmax)  # al posto di zero della maschera mettere badvalue
#
#    #cl_GG_sims[n, :] =hp.anafast(map1=mapgal, map2=mapgal, lmax=lmax)
#    print(f'Num sim={n+1}')


filename_TT_mask = out_dir+f'cls_TT_anafast_nside{nside}_lmax{lmax}_Euclidnoise_nbins10_nsim{nsim}_fsky{fsky:0.2f}.dat'#lmin{lmin}_marina.dat'
filename_GG_mask = out_dir+f'cls_galnoisegalnoise_anafast_nside{nside}_lmax{lmax}_Euclidnoise_nbins10_nsim{nsim}_fsky{fsky:0.2f}.dat'#lmin{lmin}_marina.dat'
filename_TG_mask = out_dir+f'cls_Tgalnoise_anafast_nside{nside}_lmax{lmax}_Euclidnoise_nbins10_nsim{nsim}_fsky{fsky:0.2f}.dat'#lmin{lmin}_marina.dat'

#np.savetxt(filename_TT_mask, cl_TT_mask_sims)
#np.savetxt(filename_GG_mask, cl_GG_mask_sims)
import pickle
#with open(filename_TG_mask+'.pkl', 'wb') as f:
#        pickle.dump(cl_TG_mask_sims, f)
#
with open(filename_TG_mask+'.pkl', 'rb') as f:
        cl_TG_mask_sims = pickle.load(f)


cl_TG_mask_mean = cl_TG_mask_sims.mean(axis=0)


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

fname_cov_TS_galT_mask = [f'cov_TS_galT_nbins{bin}_lmax{lmax}_nside{nside}_fsky_{fsky}.dat' for bin in range(nbins) ]
cov_TS_galT_mask = np.zeros((nbins,nbins, (lmax+1),(lmax+1)))
# Covariances
print("...computing Cov Matrices...")
for bin in range(nbins):
    for bbin in range(nbins):
        #cov_TS_galT_mask[bin,bbin] = covarfn(cl_TG_mask_sims[bin], cl_TG_mask_sims[bbin])
        #print("...saving Covariance Matrix to output " + cov_dir + f'cov_TS_galT_nbins{bin}x{bbin}_lmax{lmax}_nside{nside}.dat' + "...")
        #np.savetxt(cov_dir+f'cov_TS_galT_nbins{bin}x{bbin}_lmax{lmax}_nside{nside}.dat',  cov_TS_galT_mask[bin,bbin], header='Covariance matrix <beta_j1 beta_j2>')
        cov_TS_galT_mask[bin,bbin]=np.loadtxt(cov_dir+f'cov_TS_galT_nbins{bin}x{bbin}_lmax{lmax}_nside{nside}.dat')
print("...done...")


# Theory + Normalization Needlet power spectra
def cov_pseudo_cl_tomo(cltg,cltt, clgg, wl, lmax, noise_gal_l=None):
    """
    Returns the Cov(Pseudo-C_\ell, Pseudo-C_\ell') 
    Notes
    -----
    Cov(Pseudo-C_\ell, Pseudo-C_\ell') .shape = (lmax+1, lmax+1)
    """
    noise_vec = np.zeros_like(clgg)
        #print(noise_vec.shape)
    if noise_gal_l is not None:
        #clgg_tot = clgg+noise_gal_l
        noise = 1./noise_gal_l
        for i in range(nbins):
            noise_vec[i,i,:] = noise


    Mll  = need_theory.get_Mll(wl, lmax=lmax)
    covll = np.zeros((nbins, nbins,lmax+1,lmax+1))
    for ibin in range(nbins):
        for iibin in range(nbins):
            for ell1 in range(lmax+1):
                for ell2 in range(lmax+1):
                    covll[ibin,iibin,ell1,ell2] = Mll[ell1,ell2]*(np.sqrt(cltg[ibin,ell1]*cltg[ibin,ell2]*cltg[iibin,ell1]*cltg[iibin,ell2])+np.sqrt(cltt[ell1]*cltt[ell2]*(clgg[ibin,iibin,ell1]+noise_vec[ibin,iibin,ell1])*(clgg[ibin,iibin,ell2]+noise_vec[ibin,iibin,ell2])))/(2.*ell1+1)
                    #if np.isnan(covll[ibin,iibin,ell1,ell2]): 
                    #    print(f'ibin={ibin}, iibin={iibin}, ell1={ell1}, ell2={ell2}')
    return covll

Mll  = need_theory.get_Mll(wl,lmax)
pseudo_cl = np.array([np.dot(Mll, xcspectra.cltg[b][:lmax+1]) for b in range(nbins)])
cov_pseudo_cl = cov_pseudo_cl_tomo(cltg=xcspectra.cltg,cltt=xcspectra.cltt, clgg=xcspectra.clgg,wl= wl, lmax=lmax, noise_gal_l=simparams['ngal'])


# Some plots
print("...here come the plots...")

ell_vec = np.arange(lmax+1)

####### Relative diff spectrum

fig = plt.figure(figsize=(17,10))

plt.suptitle( r'$\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~n_{\mathrm{bins}} =$'+str(nbins) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.axhline(ls='--', color='grey')
for bin in range(1):
    #ax.errorbar(myanalysis.jvec[1:jmax], (betaj_TS_galT_mean[bin][1:jmax] -betatg[bin][1:jmax])/betatg[bin][1:jmax], yerr=np.sqrt(np.diag(cov_TS_galT[bin])[1:jmax])/(np.sqrt(nsim)*betatg[bin][1:jmax]),  fmt='o',  ms=5,label=f'Bin = {bin} No Noise')
    ax.errorbar(ell_vec[2:], (cl_TG_mask_mean[bin][2:] -pseudo_cl[bin][2:])/pseudo_cl[bin][2:], yerr=np.sqrt(np.diag(cov_TS_galT_mask[bin,bin])[2:])/(np.sqrt(nsim)*pseudo_cl[bin][2:]),  fmt='o',  ms=5,label=f'Bin = {bin}, Shot Noise, variance from sim')
    ax.errorbar(ell_vec[2:], (cl_TG_mask_mean[bin][2:] -pseudo_cl[bin][2:])/pseudo_cl[bin][2:], yerr=np.sqrt(np.diag(cov_pseudo_cl[bin,bin])[2:])/(np.sqrt(nsim)*pseudo_cl[bin][2:]),  fmt='o',  ms=5,label=f'Bin = {bin}, Shot Noise, analytical variance')

ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xlabel(r'$j$', fontsize=22)
ax.set_ylabel(r'$\frac{\langle \Gamma_j^{\mathrm{TG}} \rangle - \Gamma_j^{\mathrm{TG}\,, th}}{\Gamma_j^{\mathrm{TG}\,, th}}$', fontsize=22)
#ax.set_ylim([-0.2,0.3])

fig.tight_layout()
plt.savefig(out_dir+f'cl_mean_T_gal_lmax{lmax}_nsim{nsim}_nside{nside}_nbins{nbins}.png', bbox_inches='tight')


#### Relative diff covariance

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)
for b in range(nbins):
    ax.plot(ell_vec[2:], (np.sqrt(np.diag(cov_TS_galT_mask[b,b])[2:])/np.sqrt(np.diag(cov_pseudo_cl[b,b])[2:])-1)*100 ,'o',ms=10, label=f'Bin={b}x{b}')
ax.axhline(ls='--', c='k', linewidth=0.8)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.legend(ncol=2)
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'% $\sigma_{\mathrm{sims}}/\sigma_{\mathrm{analytic}}$ - 1')

fig.tight_layout()
plt.savefig(out_dir+f'diff_cov_cl_theory_T_gal_lmax{lmax}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')

#### fuoei diagonale per bin i bin i

fig = plt.figure(figsize=(17,10))

plt.suptitle( r'$\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)
for b in range(nbins):
    ax.plot(ell_vec[2:lmax], (np.sqrt(np.diag(cov_TS_galT_mask[b,b], k=-1)[2:])/np.sqrt(np.diag(cov_pseudo_cl[b,b], k=-1)[2:])-1)*100 ,'o',ms=10, label=f'Bin={b}')
ax.axhline(ls='--', c='k', linewidth=0.8)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.legend(ncol=2)
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'% $Cov(superdiag)_{\mathrm{sims}}/Cov(superdiag)_{\mathrm{analytic}}$ - 1')

fig.tight_layout()
plt.savefig(out_dir+f'diff_cov_out_diag_cl_theory_T_gal_lmax{lmax}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')


#diff redshift b redshift b+1

fig = plt.figure(figsize=(17,10))

plt.suptitle( r'$\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)
for b in range(nbins-1):
    ax.plot(ell_vec[2:], (np.sqrt(np.diag(cov_TS_galT_mask[b,b+1])[2:])/np.sqrt(np.diag(cov_pseudo_cl[b,b+1])[2:])-1)*100 ,'o',ms=10, label=f'Bin={b}x{b+1}')
    print(np.sqrt(np.diag(cov_TS_galT_mask[b,b+1])[2:]))

ax.axhline(ls='--', c='k', linewidth=0.8)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.legend(ncol=2)
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'% $Cov(diag)_{\mathrm{sims}}/Cov(diag)_{\mathrm{analytic}}$ - 1')

fig.tight_layout()
plt.savefig(out_dir+f'diff_cov_redshift_out_diag+1_cl_theory_T_gal_lmax{lmax}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')



#diff redshift b redshift b+2

fig = plt.figure(figsize=(17,10))

plt.suptitle( r'$\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)
for b in range(nbins-2):
    ax.plot(ell_vec[2:], (np.sqrt(np.diag(cov_TS_galT_mask[b,b+2])[2:])/np.sqrt(np.diag(cov_pseudo_cl[b,b+2])[2:])-1)*100 ,'o',ms=10, label=f'Bin={b}x{b+2}')
    #print(np.sqrt(np.diag(cov_pseudo_cl[b,b+2])[2:]))

ax.axhline(ls='--', c='k', linewidth=0.8)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
#ax.set_xticks(ell_vec)
ax.legend(ncol=2)
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'% $Cov(diag)_{\mathrm{sims}}/Cov(diag)_{\mathrm{analytic}}$ - 1')

fig.tight_layout()
plt.savefig(out_dir+f'diff_cov_redshift_out_diag+2_betaj_theory_T_gal_lmax{lmax}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')

