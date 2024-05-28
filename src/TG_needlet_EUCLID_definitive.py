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

sns.set()
sns.set(style = 'white')
sns.set_palette('husl', n_colors=8)

#plt.style.use("dark_background")
import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)

from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 


#plt.rcParams['axes.linewidth']  = 5.
plt.rcParams['axes.labelsize']  =10
plt.rcParams['xtick.labelsize'] =7
plt.rcParams['ytick.labelsize'] =7
plt.rcParams['legend.fontsize']  = 'medium'
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['axes.titlesize'] = '10'
plt.rcParams["errorbar.capsize"] = 2
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['font.size'] = 10
plt.rcParams['lines.linewidth']  = 2.
plt.rcParams['lines.markersize'] = 5.
plt.rcParams['xtick.labelsize']=10
plt.rcParams['ytick.labelsize']=10
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# embed()

# Parameters
simparams = {'nside'   : 128,
             'ngal'    : 35454308.580126834, #dovrebbe importare solo per lo shot noise (noise poissoniano)
 	     	 'ngal_dim': 'ster',
	     	 'pixwin'  : False}

nside = simparams['nside']

lmax = 256
nsim = 1000
jmax= 12#5

# Paths
fname_xcspectra = 'spectra/inifiles/EUCLID_fiducial_lmin0.dat'
sims_dir        = f'sims/Euclid_sims_Marina/NSIDE{nside}/'
out_dir         = f'output_needlet_TG/EUCLID/Mask_noise/TG_{nside}_nsim{nsim}_nuova_mask/'
path_inpainting = 'inpainting/inpainting.py'
#cov_dir 		= f'covariance/EUCLID/Mask_noise/TG_{nside}_nsim{nsim}_4/'
#if not os.path.exists(cov_dir):
#        os.makedirs(cov_dir)

cl_theory = np.loadtxt('spectra/inifiles/EUCLID_fiducial_lmin0.dat')
ell_theory = cl_theory[0]
cl_theory_tt = cl_theory[1]
cl_theory_tg = cl_theory[2]
cl_theory_gg = cl_theory[3]

mask = hp.read_map(f'mask/EUCLID/mask_planck_comm_2018_x_euclid_fsky0p35_nside={nside}.fits')
fsky = np.mean(mask)
print(f'fsky={fsky}')

Nll = np.ones(cl_theory_gg.shape[0])/simparams['ngal']

# Loading theory spectra
xcspectra = spectra.XCSpectraFile(fname_xcspectra,   WantTG = True)

# Simulations class
simulations = sims.KGsimulations(xcspectra, sims_dir, simparams, WantTG = True)
simulations.Run(nsim, WantTG = True, EuclidSims=True)

# Needlet Analysis
myanalysis = analysis.NeedAnalysis(jmax, lmax, out_dir, simulations)
B=myanalysis.B
out_dir_plot    = out_dir+f'plot_D{B:1.2f}/'
if not os.path.exists(out_dir_plot):
        os.makedirs(out_dir_plot)

# Theory Needlet theory and windows functions
need_theory = spectra.NeedletTheory(myanalysis.B)

b2_D = need_theory.get_bneed(jmax, lmax)**2#, mergej)
#filename_D = f'b_need/bneed_lmax256_jmax{jmax}_B{B:1.2f}.dat'
#b2_D = np.loadtxt(filename_D)

print(b2_D.shape)

fig, ax1  = plt.subplots(1,1,figsize=(7,5)) 
plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax))

for i in range(b2_D.shape[0]):
    ax1.plot(b2_D[i], label = 'j='+str(i) )
    print(b2_D[i])
ax1.set_xscale('log')
#ax1.set_xlim(-1,10)
ax1.set_xlabel(r'$\ell$')
ax1.set_ylabel(r'$w^{2}(\frac{\ell}{D^{j}})$')
ax1.legend(loc='right')
plt.tight_layout()
plt.savefig(out_dir_plot+f'b2_D{B:1.2f}.png')

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
plt.savefig(out_dir_plot+f'ell_binning_D{B:1.2f}.png')
plt.show()


# Computing simulated Cls 
print("...computing Betajs for simulations...")
fname_betaj_sims_TS_galT      = f'betaj_sims_TS_galT_jmax{jmax}_B_{myanalysis.B:0.2f}_nside{nside}.dat'
fname_betaj_sims_TS_galT_mask = f'betaj_sims_TS_galT_jmax{jmax}_B_{myanalysis.B:0.2f}_nside{nside}_fsky{fsky:0.2f}.dat'

betaj_sims_TS_galT       = myanalysis.GetBetajSimsFromMaps('T', nsim, field2='g1noise', fname=fname_betaj_sims_TS_galT, fsky_approx=False,EuclidSims=True)
betaj_sims_TS_galT_mask  = myanalysis.GetBetajSimsFromMaps('T', nsim, field2='g1noise', mask=mask, fname=fname_betaj_sims_TS_galT_mask, fsky_approx=False,EuclidSims=True)

# Covariances
print("...computing Cov Matrices...")
fname_cov_TS_galT            = f'cov_TS_galT_jmax{jmax}_B_{myanalysis.B:0.2f}_nside{nside}.dat'
fname_cov_TS_galT_mask       = f'cov_TS_galT_jmax{jmax}_B_{myanalysis.B:0.2f}_nside{nside}_fsky{fsky:0.2f}.dat'

cov_TS_galT, corr_TS_galT               = myanalysis.GetCovMatrixFromMaps(field1='T', nsim=nsim, field2='g1noise', fname=fname_cov_TS_galT, fname_sims=fname_betaj_sims_TS_galT)
cov_TS_galT_mask, corr_TS_galT_mask     = myanalysis.GetCovMatrixFromMaps(field1='T', nsim=nsim, field2='g1noise', mask=mask,fname=fname_cov_TS_galT_mask, fname_sims=fname_betaj_sims_TS_galT_mask)

print("...done...")

# <Beta_j>_MC
betaj_TS_galT_mean         = myanalysis.GetBetajMeanFromMaps('T', nsim, field2='g1noise', fname_sims=fname_betaj_sims_TS_galT)
betaj_TS_galT_mask_mean    = myanalysis.GetBetajMeanFromMaps('T', nsim, field2='g1noise', mask=mask, fname_sims=fname_betaj_sims_TS_galT_mask)



##########################################################################################
# Some plots
print("...here come the plots...")

# Covariances

fig, axs = plt.subplots(ncols=2, nrows=1,figsize=(15,7))   
plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))


mask_ = np.tri(corr_TS_galT.shape[0],corr_TS_galT.shape[1],0)

#plt.subplot(131)
axs[0].set_title(r'Corr $T^T\times gal^T$ Signal + shot noise')
sns.color_palette("crest", as_cmap=True)
sns.heatmap(corr_TS_galT, annot=True, fmt='.2f', cmap  = 'crest',mask=mask_, ax=axs[0])

##plt.subplot(132)
axs[1].set_title(r'Corr $T^T\times gal^T$ Signal + shot noise Masked')
sns.heatmap(corr_TS_galT_mask, annot=True, fmt='.2f',cmap  = 'crest', mask=mask_, ax=axs[1])
#
plt.tight_layout()
plt.show()

#############################################

# PROVA MASTER
wl = hp.anafast(mask, lmax=lmax)
Mll  = need_theory.get_Mll(wl, lmax=lmax)

###############
#MASTER NEEDLETS

gammaJ_tg = need_theory.gammaJ(cl_theory_tg, wl, jmax, lmax)
delta_gammaj = need_theory.variance_gammaj(cltg=cl_theory_tg,cltt=cl_theory_tt, clgg=cl_theory_gg, wl=wl, jmax=jmax, lmax=lmax, noise_gal_l=Nll)

print(f'gamma_j = {gammaJ_tg}')

np.savetxt(out_dir+f'gammaj_theory_B{myanalysis.B:1.2f}_jmax{jmax}_nside{nside}_lmax{lmax}_nsim{nsim}_fsky{fsky:1.2f}.dat', gammaJ_tg)
np.savetxt(out_dir+f'variance_gammaj_theory_B{myanalysis.B:1.2f}_jmax{jmax}_nside{nside}_lmax{lmax}_nsim{nsim}_fsky{fsky:1.2f}.dat', delta_gammaj)

##RELATIVE DIFFERENCE
fig = plt.figure()#figsize=(10,7))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim)+' MASK')

ax = fig.add_subplot(1, 1, 1)

ax.axhline(ls='--', color='grey')

ax.errorbar(myanalysis.jvec[0:jmax+1], (betaj_TS_galT_mask_mean[0:jmax+1] /gammaJ_tg[0:jmax+1]-1), yerr=np.sqrt(np.diag(delta_gammaj)[0:jmax+1])/(np.sqrt(nsim)*gammaJ_tg[0:jmax+1]),  fmt='o', ms=0, label=r'Variance of the mean from theory')
ax.errorbar(myanalysis.jvec[0:jmax+1], (betaj_TS_galT_mask_mean[0:jmax+1] /gammaJ_tg[0:jmax+1]-1), yerr=np.sqrt(np.diag(cov_TS_galT_mask)[0:jmax+1])/(np.sqrt(nsim)*gammaJ_tg[0:jmax+1]),color='#2b7bbc',  ms=3,fmt='o',  label=r'Variance of the mean from simulations')

ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(myanalysis.jvec[1:])
ax.set_xticklabels(myanalysis.jvec[1:])
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\langle \tilde{\Gamma}_j^{TG} \rangle/\tilde{\Gamma}_j^{TG, th}$-1')# - \beta_j^{Tgal, th}
#ax.set_ylim([-0.3,1.3])

fig.tight_layout()
plt.savefig(out_dir_plot+f'relative_diff_gammaJ_D{B:1.2f}.png')
plt.show()

### SPECTRUM  CUT SKY

fig = plt.figure()#figsize=(10,7))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.plot(myanalysis.jvec[0:jmax+1], gammaJ_tg[0:jmax+1], label='Theory')
ax.errorbar(myanalysis.jvec[0:jmax+1], betaj_TS_galT_mask_mean[0:jmax+1], yerr=np.sqrt(np.diag(cov_TS_galT_mask)[0:jmax+1])/np.sqrt(nsim) ,color='#2b7bbc',fmt='o',ms=3, label='Error of the mean of the simulations')
ax.errorbar(myanalysis.jvec[0:jmax+1], betaj_TS_galT_mask_mean[0:jmax+1], yerr=np.sqrt(np.diag(cov_TS_galT_mask)[0:jmax+1]) ,color='grey',fmt='o',ms=0, label='Error of simulations')

ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(myanalysis.jvec[1:])
ax.set_xticklabels(myanalysis.jvec[1:])
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\tilde{\Gamma}^{\mathrm{\,TG}}_j$')

fig.tight_layout()
plt.savefig(out_dir_plot+f'gammaJ_D{B:1.2f}.png')
plt.show()
diff_cov_mask = 100*np.abs(np.diag(cov_TS_galT_mask)[1:]-np.diag(delta_gammaj)[1:])/np.diag(delta_gammaj)[1:]
diff_cov_mean_mask = np.mean(diff_cov_mask)
print(f'diff cov mask={diff_cov_mask}, diff cov mean={diff_cov_mean_mask}')

####################################################################################
############################# DIFF COVARIANCES #####################################
##NEEDLETS
fig = plt.figure()#figsize=(10,7))

plt.suptitle(r'EuclidxPlanck MASK NEEDLETS $D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~f_{sky} = %0.2f$'%fsky)

ax = fig.add_subplot(1, 1, 1)

ax.plot(myanalysis.jvec[0:jmax+1], (np.diag(cov_TS_galT_mask)[0:jmax+1]-np.diag(delta_gammaj)[0:jmax+1])/np.diag(delta_gammaj)[0:jmax+1]*100 ,'o',color='#2b7bbc')#, label='MASK')
ax.axhline(ls='--', color='grey')

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter)
ax.set_xticks(myanalysis.jvec[0:jmax+1]) 
ax.set_xticklabels(myanalysis.jvec[0:jmax+1])
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'% $(\Delta \Gamma)^2_{\mathrm{sims}}/(\Delta \Gamma)^2_{\mathrm{analytic}}$ - 1')

fig.tight_layout()
plt.savefig(out_dir_plot+f'relative_diff_diag_cov_D{B:1.2f}.png')
plt.show()

###PSEUDO

def cov_pseudo_cl(cltg,cltt, clgg, wl, lmax, noise_gal_l=None):
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
    Mll  = need_theory.get_Mll(wl,lmax)
    ell= np.arange(lmax+1)
    covll = np.zeros((ell.shape[0],ell.shape[0]))
    for l,ell1 in enumerate(ell):
        for ll,ell2 in enumerate(ell):
            covll[l,ll] = Mll[l,ll]*(cltg[l]*cltg[ll]+np.sqrt(cltt[l]*cltt[ll]*clgg_tot[l]*clgg_tot[ll]))/(2.*ell1+1)
    return covll

cls_tg = np.loadtxt('cls_from_maps/EUCLID/Euclid_combined_mask/cls_Tgalnoise_anafast_nside128_lmax256_Euclidnoise_Marina_nsim1000_fsky0.36.dat')

cov_pcl_sim = np.cov(cls_tg.T)

cov_pcl= cov_pseudo_cl(cltg=cl_theory_tg,cltt=cl_theory_tt, clgg=cl_theory_gg, wl=wl, lmax=lmax,noise_gal_l=Nll)

fig = plt.figure()#figsize=(10,7))

plt.suptitle(r'EuclidxPlanck Mask PCL $ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~f_{sky} = %.2f$'%fsky)

ax = fig.add_subplot(1, 1, 1)

ax.plot(np.arange(2, lmax+1), ((np.diag(cov_pcl_sim)[2:]-np.diag(cov_pcl)[2:])/np.diag(cov_pcl)[2:])*100 ,'o',color='#2b7bbc')#, label='MASK')
ax.axhline(ls='--', color='grey')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(np.arange(2, lmax+1,10))
ax.set_xticklabels(np.arange(2, lmax+1,10),rotation=40)
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'% diag(sim cov)/diag(analyt cov)-1')
plt.savefig(out_dir_plot+f'relative_diff_diag_cov_PCL.png')
fig.tight_layout()
plt.show()

################################ DIFF VARIANCE #####################################


fig = plt.figure()#figsize=(10,7))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.plot(myanalysis.jvec[0:jmax+1], (np.sqrt(np.diag(cov_TS_galT_mask)[0:jmax+1])/np.sqrt(np.diag(delta_gammaj)[0:jmax+1])-1)*100 ,'o',color='#2b7bbc')#, label='MASK')
ax.axhline(ls='--', color='grey')

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(myanalysis.jvec[0:jmax+1])
ax.set_xticklabels(myanalysis.jvec[0:jmax+1])
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'% $\sigma_{\mathrm{sims}}/\sigma_{\mathrm{analytic}}$ - 1')

fig.tight_layout()
plt.savefig(out_dir_plot+f'relative_diff_variance_D{B:1.2f}.png')
plt.show()

#######################################################################################################################
#Difference divided one sigma

fig = plt.figure()#figsize=(10,7))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) +r', $f_{\mathrm{sky}}$ = %1.2f'%fsky+ r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))


ax = fig.add_subplot(1, 1, 1)

ax.axhline(ls='--', color='grey')
ax.plot(myanalysis.jvec[0:jmax+1], (betaj_TS_galT_mask_mean[0:jmax+1] -gammaJ_tg[0:jmax+1])/(np.sqrt(np.diag(delta_gammaj)[0:jmax+1])/np.sqrt(nsim)),'o', color='#2b7bbc')#, label=r'$T^S \times gal^S$, sim cov')

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(myanalysis.jvec[0:jmax+1])
ax.set_xticklabels(myanalysis.jvec[0:jmax+1])
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\Delta \Gamma_j^{\mathrm{TG}} / \sigma $')
ax.set_ylim([-3,3])

fig.tight_layout()
plt.savefig(out_dir_plot+f'diff_gammaJ_over_sigma_D{B:1.2f}.png')
plt.show()


#################################
####### SIGNAL TO NOISE ##########

cls_tg_mean = np.mean(cls_tg, axis=0)
cls_recovered = np.dot(np.linalg.inv(Mll[2:,2:]),cls_tg_mean[2:] ) 


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
        for ill in range(2,ell-1):
            s2n_cum[l] += s2n[ill]
        s2n_cum[l]= np.sqrt(s2n_cum[l])      
    return s2n_cum

def fl_j(j_m):
    ell_binning=need_theory.ell_binning(jmax, lmax)
    l_j = np.zeros(j_m+1, dtype=int)
    
    for j in range(j_m+1):
            ell_range = ell_binning[j][ell_binning[j]!=0]
            #lmin = np.floor(myanalysis.B**(j-1))
            #lmax = np.floor(myanalysis.B**(j+1))
            #ell  = np.arange(lmin, lmax+1, dtype=int)
            l_j[j] = int(ell_range[int(np.ceil((len(ell_range))/2))])#int(ell[int(np.ceil((len(ell))/2))])
    return l_j

lmax_vec=fl_j(jmax)
lmax_vec_cl = np.arange(start=2,stop=256,dtype=int)

s2n_theory_gamma=S_2_N(gammaJ_tg[0:jmax+1], delta_gammaj[0:jmax+1,0:jmax+1])
s2n_mean_sim=S_2_N(betaj_TS_galT_mask_mean[0:jmax+1], delta_gammaj[0:jmax+1,0:jmax+1])
s2n_cum = S_2_N_cum(s2n_mean_sim, myanalysis.jvec)
s2n_mean_sim_cl=S_2_N_ell(cls_tg_mean[2:], cov_pcl[2:,2:])
s2n_cum_cl = S_2_N_cum_ell(s2n_mean_sim_cl,lmax_vec_cl)

print(f's2n_cum_lmax_need:{s2n_cum[-1]},2n_cum_lmax_pcl:{s2n_cum_cl[-1]}')

fig = plt.figure(figsize=(10,11))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)
print(s2n_cum, lmax_vec)
ax.plot(lmax_vec, s2n_cum, label='Needlets')
ax.plot(lmax_vec_cl, s2n_cum_cl, color='#2b7bbc', label= 'PCL')
ax.set_xscale('log')
ax.set_xlim(left=3, right=250)
ax.set_ylim(top=4.)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 

ax.set_xlabel(r'$\ell_{\mathrm{max}}$')
ax.set_ylabel('Cumulative Signal-to-Noise ratio')
ax.legend()

fig.tight_layout()
plt.savefig(out_dir_plot+f's2n_cum_B{B:1.2f}_jmax{jmax}_lmax{lmax}_nside{nside}.png')
plt.show()