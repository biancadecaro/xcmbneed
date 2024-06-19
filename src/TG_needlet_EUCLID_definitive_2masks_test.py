#!/usr/bin/env python
import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams, gridspec
import healpy as hp
from astropy.io import fits
import argparse, os, sys, warnings, glob
import cython_mylibc as mylibc
import analysis, utils, spectra, sims
from IPython import embed
import seaborn as sns

sns.set()
sns.set(style = 'white')


import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)

from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 


# Parameters
simparams = {'nside'   : 128,
             'ngal'    : 35454308.580126834, #dovrebbe importare solo per lo shot noise (noise poissoniano)
 	     	 'ngal_dim': 'ster',
	     	 'pixwin'  : False}

nside = simparams['nside']

lmax = 256
nsim = 1000
jmax= 12

# Paths
fname_xcspectra = 'spectra/inifiles/EUCLID_fiducial_lmin0.dat'
sims_dir        = f'sims/Euclid_sims_Marina/NSIDE{nside}/'
out_dir         = f'output_needlet_TG/EUCLID/Mask_noise/TG_{nside}_nsim{nsim}_2masks_test/'
path_inpainting = 'inpainting/inpainting.py'


cl_theory = np.loadtxt('spectra/inifiles/EUCLID_fiducial_lmin0.dat')
ell_theory = cl_theory[0]
cl_theory_tt = cl_theory[1]
cl_theory_tg = cl_theory[2]
cl_theory_gg = cl_theory[3]



mask_euxpl = hp.read_map(f'mask/EUCLID/mask_planck_comm_2018_x_euclid_binary_fsky0.359375_nside={nside}.fits')

fsky_comb = np.mean(mask_euxpl)
print(f'fsky={fsky_comb}')

Nll = np.ones(cl_theory_gg.shape[0])/simparams['ngal']

# Loading theory spectra
xcspectra = spectra.XCSpectraFile(fname_xcspectra,   WantTG = True)

# Simulations class
simulations = sims.KGsimulations(xcspectra, sims_dir, simparams, WantTG = True)
simulations.Run(nsim, WantTG = True, EuclidSims=True)

# Needlet Analysis
myanalysis = analysis.NeedAnalysis(jmax, lmax, out_dir, simulations)
B=myanalysis.B
jvec = myanalysis.jvec
out_dir_plot    = out_dir+f'plot_D{B:1.2f}/'
if not os.path.exists(out_dir_plot):
        os.makedirs(out_dir_plot)

# Theory Needlet theory and windows functions
need_theory = spectra.NeedletTheory(myanalysis.B)
b_need = need_theory.get_bneed(jmax, lmax)

fig, ax1  = plt.subplots(1,1,figsize=(7,5)) 
plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax))

for i in range(b_need.shape[0]):
    ax1.plot(b_need[i]*b_need[i], label = 'j='+str(i) )
ax1.set_xscale('log')
ax1.set_xlabel(r'$\ell$')
ax1.set_ylabel(r'$w^{2}(\frac{\ell}{D^{j}})$')
ax1.legend(loc='right')
plt.tight_layout()

ell_binning=need_theory.ell_binning(jmax, lmax)
fig = plt.figure()
plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax))
ax = fig.add_subplot(1, 1, 1)
for i in range(0,jmax+1):
    ell_range = ell_binning[i][ell_binning[i]!=0]
    plt.plot(ell_range, i*ell_range/ell_range, label= f'j={i}')
    plt.text(ell_range[0], i, r'$\ell_{min}=%d,\,\ell_{max}=%d$'%(ell_range[0],ell_range[-1]))

ax.set_xlabel(r'$\ell$')
ax.legend(loc='right', ncol=2)
plt.tight_layout()


wl_plxeu = hp.anafast(mask_euxpl, lmax=lmax)
Mll_plxeu  = need_theory.get_Mll(wl_plxeu, lmax=lmax)

print(f'fsky_comb:{fsky_comb}, sum Mll {np.sum(Mll_plxeu[45])}')

gammaJ_tg = need_theory.gammaJ(cl_theory_tg, Mll_plxeu, lmax)
delta_gammaj = need_theory.variance_gammaj(cltg=cl_theory_tg,cltt=cl_theory_tt, clgg=cl_theory_gg, Mll_1x2=Mll_plxeu, Mll=Mll_plxeu, jmax=jmax, lmax=lmax, noise_gal_l=Nll)

# Computing simulated Betaj 
print("...computing Betajs for simulations...")
fname_gammaj_sims_TS_galT_mask = f'gamma_sims_TS_galT_jmax{jmax}_B_{myanalysis.B:0.2f}_nside{nside}_fsky{fsky_comb:0.2f}.dat'

gammaj_sims_TS_galT_mask  = myanalysis.GetBetajSimsFromMaps('T', nsim, field2='g1noise', mask1=mask_euxpl, mask2=mask_euxpl, fname=fname_gammaj_sims_TS_galT_mask, fsky_approx=False,EuclidSims=True)

# Covariances
print("...computing Cov Matrices...")
fname_cov_TS_galT_mask       = f'cov_TS_galT_jmax{jmax}_B_{myanalysis.B:0.2f}_nside{nside}_fsky{fsky_comb:0.2f}.dat'

cov_TS_galT_mask, corr_TS_galT_mask     = myanalysis.GetCovMatrixFromMaps(field1='T', nsim=nsim, field2='g1noise', mask1=mask_euxpl, mask2=mask_euxpl, fname=fname_cov_TS_galT_mask, fname_sims=fname_gammaj_sims_TS_galT_mask)

print("...done...")

# <Gamma_j>_MC
gammaj_TS_galT_mask_mean    = myanalysis.GetBetajMeanFromMaps('T', nsim, field2='g1noise', mask1=mask_euxpl, mask2=mask_euxpl, fname_sims=fname_gammaj_sims_TS_galT_mask)

##########################################################################################
# Some plots

print("...here come the plots...")

# Covariances

fig, axs = plt.subplots(1,1,figsize=(10,7))   
plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

mask_ = np.tri(corr_TS_galT_mask.shape[0],corr_TS_galT_mask.shape[1],0)

axs.set_title(r'Corr $T^T\times gal^T$ Signal + shot noise Masked')
sns.heatmap(corr_TS_galT_mask, annot=True, fmt='.2f',cmap  = 'crest', mask=mask_, ax=axs)
#
plt.tight_layout()
#plt.show()

### GAMMAJ TG 

fig = plt.figure()

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.plot(jvec[1:], gammaJ_tg[1:], label='Theory')
ax.errorbar(jvec[1:], gammaj_TS_galT_mask_mean[1:], yerr=np.sqrt(np.diag(cov_TS_galT_mask)[1:])/np.sqrt(nsim) ,fmt='o',ms=3, label='Error of the mean of the simulations')
ax.errorbar(jvec[1:], gammaj_TS_galT_mask_mean[1:], yerr=np.sqrt(np.diag(cov_TS_galT_mask)[1:]) ,color='grey',fmt='o',ms=0, label='Error of simulations')

ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(myanalysis.jvec[1:])
ax.set_xticklabels(myanalysis.jvec[1:])
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\tilde{\Gamma}^{\mathrm{\,TG}}_j$')

fig.tight_layout()
plt.savefig(out_dir_plot+f'gammaJ_D{B:1.2f}.png')
#plt.show()

##RELATIVE DIFFERENCE
fig = plt.figure()

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim)+' MASK')

ax = fig.add_subplot(1, 1, 1)

ax.axhline(ls='--', color='grey')

ax.errorbar(jvec[1:], (gammaj_TS_galT_mask_mean[1:] /gammaJ_tg[1:]-1)*100, yerr=100*np.sqrt(np.diag(delta_gammaj)[1:])/(np.sqrt(nsim)*gammaJ_tg[1:]),  fmt='o', ms=0, label='Variance of the mean from theory')
ax.errorbar(jvec[1:], (gammaj_TS_galT_mask_mean[1:] /gammaJ_tg[1:]-1)*100, yerr=100*np.sqrt(np.diag(cov_TS_galT_mask)[1:])/(np.sqrt(nsim)*gammaJ_tg[1:]),  ms=3,fmt='o',  label='Variance of the mean from simulations')

ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(jvec[1:])
ax.set_xticklabels(jvec[1:])
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\% \langle \tilde{\Gamma}_j^{TG} \rangle/\tilde{\Gamma}_j^{TG, th}$-1')# - \beta_j^{Tgal, th}
#ax.set_ylim([-0.3,1.3])

fig.tight_layout()
plt.savefig(out_dir_plot+f'relative_diff_gammaJ_D{B:1.2f}.png')
#plt.show()

####################################################################################
############################# DIFF COVARIANCES #####################################
##NEEDLETS
fig = plt.figure()

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~f_{sky} = %0.2f$'%fsky_comb)

ax = fig.add_subplot(1, 1, 1)

ax.plot(jvec[1:], (np.diag(cov_TS_galT_mask)[1:]/np.diag(delta_gammaj)[1:]-1)*100 ,'o')

ax.axhline(ls='--', color='grey')

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter)
ax.set_xticks(jvec[1:]) 
ax.set_xticklabels(jvec[1:])
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'% $(\Delta \Gamma)^2_{\mathrm{sims}}/(\Delta \Gamma)^2_{\mathrm{analytic}}$ - 1')
fig.tight_layout()
plt.savefig(out_dir_plot+f'relative_diff_diag_cov_D{B:1.2f}.png')
#plt.show()

################################################################################
###PSEUDO

def cov_pseudo_cl(cltg,cltt, clgg,Mll,  Mll_1x2, lmax, noise_gal_l=None):
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
    ell= np.arange(lmax+1)
    covll = np.zeros((ell.shape[0],ell.shape[0]))
    for l,ell1 in enumerate(ell):
        for ll,ell2 in enumerate(ell):
            covll[l,ll] = (Mll_1x2[l,ll]*(cltg[l]*cltg[ll])+Mll[l,ll]*(np.sqrt(cltt[l]*cltt[ll]*clgg_tot[l]*clgg_tot[ll])))/(2.*ell1+1)
    return covll

def cov_cl(cltg,cltt, clgg, lmax,lmin, fsky=1.,noise_gal_l=None):
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
    ell= np.arange(lmin, lmax+1)
    covll = np.zeros(( ell.shape[0], ell.shape[0]))
    for l,ell1 in enumerate(ell):
        for ll,ell2 in enumerate(ell):
            if l!=ll: covll[l,ll]=0
            else:
                covll[l,ll] = (cltg[lmin:][l]*cltg[lmin:][ll]+np.sqrt(cltt[lmin:][l]*cltt[lmin:][ll]*clgg_tot[lmin:][l]*clgg_tot[lmin:][ll]))/(fsky*(2.*ell1+1))
    return covll

cls_tg = np.loadtxt('cls_from_maps/EUCLID/Euclid_combined_mask_binary/cls_Tgalnoise_anafast_nside128_lmax256_Euclidnoise_Marina_nsim1000_fsky0.36.dat')
cls_tg_mean=np.mean(cls_tg, axis=0)

pcl = np.dot(Mll_plxeu,cl_theory_tg[:lmax+1]) 
cl_recovered = np.dot(np.linalg.inv(Mll_plxeu[2:,2:]), cls_tg_mean[2:]) 


cov_pcl_sim = np.cov(cls_tg.T)

cov_pcl= cov_pseudo_cl(cltg=cl_theory_tg,cltt=cl_theory_tt, clgg=cl_theory_gg, Mll=Mll_plxeu,  Mll_1x2=Mll_plxeu, lmax=lmax,noise_gal_l=Nll)
cov_cls = cov_cl(cltg=cl_theory_tg,cltt=cl_theory_tt, clgg=cl_theory_gg, lmax=lmax,lmin=2,noise_gal_l=Nll)

ell = np.arange(lmax+1)
factor = ell*(ell+1)/(2*np.pi)


fig = plt.figure()
plt.suptitle(r'Mask PCL $ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~f_{sky} = %.2f$'%fsky_comb)
ax = fig.add_subplot(1, 1, 1)
ax.plot(ell, factor*pcl, label='PCL')
ax.plot(ell, factor*cls_tg_mean,'+' ,label='Mean of simulations')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel('PCL')
plt.legend()
#plt.show()


fig = plt.figure()
plt.suptitle(r'CL recovered $ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~f_{sky} = %.2f$'%fsky_comb)
ax = fig.add_subplot(1, 1, 1)
ax.plot(ell[2:], factor[2:]*cl_theory_tg[2:lmax+1], label='Theory')
ax.plot(ell[2:], factor[2:]*cl_recovered,'+' ,label='Cl recovered')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi}C^{\rm TG}_{\ell}$')
plt.legend()
#plt.show()

fig = plt.figure(figsize=(7,4))
plt.suptitle(r'Mask PCL $ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~f_{sky} = %.2f$'%fsky_comb)
ax = fig.add_subplot(1, 1, 1)
ax.errorbar(ell[2:],  (cls_tg_mean[2:]/pcl[2:]-1)*100, yerr=100*np.sqrt(np.diag(cov_pcl)[2:])/(np.sqrt(nsim)*pcl[2:]),   fmt='o', ms=0, label='Variance of the mean from theory')
ax.errorbar(ell[2:],  (cls_tg_mean[2:]/pcl[2:]-1)*100, yerr=100*np.sqrt(np.diag(cov_pcl_sim)[2:])/(np.sqrt(nsim)*pcl[2:]),   ms=3,fmt='o',  label='Variance of the mean from simulations')

ax.axhline(ls='--', color='grey')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel('% relative diff ')
plt.legend()
#plt.show()

fig = plt.figure()#figsize=(10,7))

plt.suptitle(r'Mask PCL $ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~f_{sky} = %.2f$'%fsky_comb)

ax = fig.add_subplot(1, 1, 1)

ax.plot(ell[2:], (np.diag(cov_pcl_sim)[2:]/np.diag(cov_pcl)[2:]-1)*100 ,'o')

ax.axhline(ls='--', color='grey')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(np.arange(2, lmax+1,10))
ax.set_xticklabels(np.arange(2, lmax+1,10),rotation=40)
ax.set_xlabel(r'$\ell$')
ax.set_ylabel('% diag(sim cov)/diag(analyt cov)-1')
fig.tight_layout()
#plt.show()

#########################################################################
######################### SIGNAL - TO - NOISE ###########################

def fl_j(j_m):
    ell_binning=need_theory.ell_binning(jmax, lmax)
    l_j = np.zeros(j_m+1, dtype=int)
    
    for j in range(1,j_m+1):
            ell_range = ell_binning[j][ell_binning[j]!=0]
            if ell_range.shape[0] == 1:
                l_j[j] = ell_range
            else:
                l_j[j] = int(ell_range[int(np.ceil((len(ell_range))/2))])
    return l_j

def S_2_N(beta, cov_matrix):
    s_n = np.zeros(len(beta))
    cov_inv = np.linalg.inv(cov_matrix)
    temp = np.zeros(len(cov_matrix[0]))
    for i in range(len(cov_matrix[0])):
        for j in range(len(beta)):
            temp[i] += cov_inv[i][j]*beta[j]
        s_n[i] = beta[i].T*temp[i]
    return s_n

def S_2_N_th(beta, variance):
    s_n = np.divide((beta)**2, variance)
    return s_n

def S_2_N_cum(s2n, jmax):
    s2n_cum = np.zeros(jmax.shape[0])
    for j,jj in enumerate(jmax):
        for ijj in range(jj):
            s2n_cum[j] +=s2n[ijj]
        s2n_cum[j]= np.sqrt(s2n_cum[j])      
    return s2n_cum

def S_2_N_ell(cltg, cov):
    icov= np.linalg.inv(cov)
    nell  = cltg.shape[0]
    s2n = np.zeros(nell)
    for il in range(nell):
        for iil in range(nell):
            s2n[il] += np.dot(cltg[il], np.dot(icov[il, iil], cltg[iil]))
    return s2n

def S_2_N_cum_ell(s2n, lmax):
    s2n_cum = np.zeros(lmax.shape[0])
    for l,ell in enumerate(lmax):
        for ill in range(ell):
            s2n_cum[l] += s2n[ill]
        s2n_cum[l]= np.sqrt(s2n_cum[l])      
    return s2n_cum

lmax_vec=fl_j(jmax)
lmax_vec_cl = np.arange(start=2,stop=256,dtype=int)

s2n_mean_sim=S_2_N(gammaj_TS_galT_mask_mean[1:jmax+1], delta_gammaj[1:jmax+1,1:jmax+1])

s2n_cum = S_2_N_cum(s2n_mean_sim, myanalysis.jvec)
#s2n_mean_sim_cl=S_2_N_ell(cls_tg_mean[2:], cov_pcl[2:,2:])
#s2n_cum_cl = S_2_N_cum_ell(s2n_mean_sim_cl,lmax_vec_cl)

s2n_mean_sim_cl=S_2_N_ell(cl_recovered, cov_cls)

s2n_cum_cl = S_2_N_cum_ell(s2n_mean_sim_cl,lmax_vec_cl)
print(s2n_cum_cl)
#s2n_mean_sim_cl=S_2_N_ell(cl_recovered, cov_cls[2:,2:])
#print(s2n_mean_sim_cl)
#s2n_cum_cl = S_2_N_cum_ell(s2n_mean_sim_cl,lmax_vec_cl)

print(f's2n_cum_lmax_need:{s2n_cum[-1]:0.2f}, s2n_cum_lmax_pcl:{s2n_cum_cl[-1]:0.2f}')

fig = plt.figure(figsize=(6,5))
plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)
ax.plot(lmax_vec, s2n_cum, label='Needlets')
ax.plot(lmax_vec_cl, s2n_cum_cl, label= 'PCL')
ax.set_xscale('log')
ax.set_xlim(left=3, right=250)
ax.set_ylim(top=4.)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 

ax.set_xlabel(r'$\ell_{\mathrm{max}}$')
ax.set_ylabel('Cumulative Signal-to-Noise ratio')
ax.legend()

fig.tight_layout()
plt.show()