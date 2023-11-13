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


# Paths
fname_xcspectra = '/ehome/bdecaro/xcmbneed/src/spectra/inifiles/EUCLID_fiducial_lmin0.dat'
sims_dir        = f'/ehome/bdecaro/xcmbneed/src/sims/Euclid_sims_Marina/NSIDE{nside}/'
out_dir         = f'output_needlet_TG/EUCLID/Mask_noise/TG_{nside}_nsim{nsim}_3/'
path_inpainting = 'inpainting/inpainting.py'
cov_dir 		= f'covariance/EUCLID/Mask_noise/TG_{nside}_nsim{nsim}_3/'
if not os.path.exists(cov_dir):
        os.makedirs(cov_dir)

cl_theory = np.loadtxt('/ehome/bdecaro/xcmbneed/src/spectra/inifiles/EUCLID_fiducial_lmin0.dat')
ell_theory = cl_theory[0]
cl_theory_tt = cl_theory[1]
cl_theory_tg = cl_theory[2]
cl_theory_gg = cl_theory[3]

mask = hp.read_map(f'/ehome/bdecaro/xcmbneed/src/mask/EUCLID/mask_planck_comm_2018_x_euclid_fsky0p35_nside={nside}.fits')
fsky = np.mean(mask)
print(f'fsky={fsky}')
print(cl_theory_gg.shape[0])
## Map gal noise

#Nll, ngal =utils.GetNlgg(simparams['ngal'], dim=simparams['ngal_dim'],lmax=cl_theory_gg.shape[0], return_ngal=True)
#print(ngal)
Nll = np.ones(cl_theory_gg.shape[0])/simparams['ngal']

# Loading theory spectra
xcspectra = spectra.XCSpectraFile(fname_xcspectra,   WantTG = True)

# Simulations class
simulations = sims.KGsimulations(xcspectra, sims_dir, simparams, WantTG = True)
simulations.Run(nsim, WantTG = True, EuclidSims=True)

# Needlet Analysis
myanalysis = analysis.NeedAnalysis(jmax, lmax, out_dir, simulations)

# Theory Needlet spectra
need_theory = spectra.NeedletTheory(myanalysis.B)

# Computing simulated Cls 
print("...computing Betajs for simulations...")
fname_betaj_sims_TS_galT      = f'betaj_sims_TS_galT_jmax{jmax}_B_{myanalysis.B:0.2f}_nside{nside}.dat'
fname_betaj_sims_TS_galT_mask = f'betaj_sims_TS_galT_jmax{jmax}_B_{myanalysis.B:0.2f}_nside{nside}_fsky{fsky:0.2f}.dat'

betaj_sims_TS_galT       = myanalysis.GetBetajSimsFromMaps('T', nsim, field2='g1noise', fname=fname_betaj_sims_TS_galT, EuclidSims=True)
betaj_sims_TS_galT_mask  = myanalysis.GetBetajSimsFromMaps('T', nsim, field2='g1noise', mask=mask, fname=fname_betaj_sims_TS_galT_mask, EuclidSims=True)

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


# Beta_j sims

[num_sim_1, num_sim_2] = np.random.choice(np.arange(nsim),2 )
beta_j_sim_1_T = betaj_sims_TS_galT[num_sim_1,:]
beta_j_sim_2_T = betaj_sims_TS_galT[num_sim_2,:]

beta_j_sim_1_T_mask = betaj_sims_TS_galT_mask[num_sim_1,:]
beta_j_sim_2_T_mask = betaj_sims_TS_galT_mask[num_sim_2,:]

# Beta_j THEORY

betatg    = need_theory.cl2betaj(jmax=jmax, cl=cl_theory_tg)
beta_norm = need_theory.cl2betaj(jmax=jmax, cl=np.ones(xcspectra.cltg.size))
delta = need_theory.delta_beta_j(jmax=jmax, cltg=cl_theory_tg, cltt=cl_theory_tt, clgg=cl_theory_gg)
delta_noise = need_theory.delta_beta_j(jmax=jmax, cltg=cl_theory_tg, cltt=cl_theory_tt, clgg=cl_theory_gg, noise_gal_l=Nll)
print(f'diff delta delta noise ={delta_noise-delta}')

diff_cov = np.abs((np.sqrt(np.diag(cov_TS_galT))-delta))/delta*100
diff_cov_mean = np.mean(diff_cov[1:jmax+1])
print(f'diff cov={diff_cov}, diff cov mean={diff_cov_mean}')



#np.savetxt(out_dir+f'theory_beta_jmax{jmax}_D{myanalysis.B:0.2f}_lmax{lmax}.dat', betatg )
#np.savetxt(out_dir+f'theory_variance_jmax{jmax}_D{myanalysis.B:0.2f}_lmax{lmax}.dat' , delta)

########################################################################################################################

# Some plots
print("...here come the plots...")

# Covariances
#fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(25,18))
fig, axs = plt.subplots(ncols=2, nrows=1,figsize=(22,10))   
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
plt.savefig(f'plot_tesi/Euclid/total_corr_T_gal_noise_mask_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}.png')


# Sims mean spectrum + one sims spectrum TOTAL SIGNAL

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(22,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))


#ax = fig.add_subplot(1, 1, 1)
ax1.set_title(r'$\beta_{j}T^T \times gal^T$')
ax1.errorbar(myanalysis.jvec, betaj_TS_galT_mean, yerr=np.sqrt(np.diag(cov_TS_galT))/np.sqrt(nsim-1),fmt='o',  ms=10,capthick=5, label = 'Mean')
ax1.errorbar(myanalysis.jvec, beta_j_sim_2_T, yerr=np.sqrt(np.diag(cov_TS_galT)),  fmt='o', ms=10,capthick=5, label = 'One Sim')
ax1.set_ylabel(r'$\beta_j^{TG}$')
ax1.set_xlabel(r'j')
ax1.legend()

ax2.set_title(r'$\beta_{j}T^T \times gal^T$ Masked')
ax2.errorbar(myanalysis.jvec, betaj_TS_galT_mask_mean, yerr=np.sqrt(np.diag(cov_TS_galT_mask))/np.sqrt(nsim-1), fmt='o', ms=10,capthick=5, label = 'Mean')
ax2.errorbar(myanalysis.jvec, beta_j_sim_2_T_mask, yerr=np.sqrt(np.diag(cov_TS_galT_mask)), fmt='o', ms=10,capthick=5, label = 'One Sim')
ax2.set_ylabel(r'$\beta_j^{TG}$')
ax2.set_xlabel(r'j')
ax2.legend()

fig.tight_layout()
#plt.savefig(f'plot_tesi/Euclid/total-signal_betaj_mean_betaj_sim_plot_jmax_{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}.png')



############## CUT SKY ######################


fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.plot(myanalysis.jvec[1:jmax+1], betatg[1:jmax+1], label='Theory')
ax.errorbar(myanalysis.jvec[1:jmax+1], betaj_TS_galT_mask_mean[1:jmax+1]/fsky, yerr=np.sqrt(np.diag(cov_TS_galT_mask)[1:jmax+1])/np.sqrt(nsim) ,color='#2b7bbc',fmt='o',ms=5,capthick=2, label=r'$\langle \hat{\beta}^{\, \mathrm{TG}}_j \rangle$')
ax.errorbar(myanalysis.jvec[1:jmax+1], betaj_TS_galT_mask_mean[1:jmax+1]/fsky, yerr=np.sqrt(np.diag(cov_TS_galT_mask)[1:jmax+1]) ,color='grey',fmt='o',ms=0,capthick=2)

ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\hat{\beta}_j^{\mathrm{TG}}$')

fig.tight_layout()
plt.savefig(f'plot_tesi/Euclid/betaj_theory_T_gal_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')
plt.savefig(f'plot_tesi/Euclid/betaj_theory_T_gal_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.pdf', bbox_inches='tight')



#### RELATIVE DIFFERENCE THEORY + SIMULATIONS MASK SKY

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim)+' MASK')


ax = fig.add_subplot(1, 1, 1)

ax.axhline(ls='--', color='grey')
ax.errorbar(myanalysis.jvec[1:jmax+1], (betaj_TS_galT_mask_mean[1:jmax+1]/fsky )/(betatg[1:jmax+1])-1, yerr=delta_noise[1:jmax+1]/(np.sqrt(nsim)*betatg[1:jmax+1]),  fmt='o',  ms=5, label=r'Error of the mean of the simulations')
ax.errorbar(myanalysis.jvec[1:jmax+1], (betaj_TS_galT_mask_mean[1:jmax+1]/fsky )/(betatg[1:jmax+1])-1, yerr=np.sqrt(np.diag(cov_TS_galT_mask/fsky**2)[1:jmax+1])/(np.sqrt(nsim)*betatg[1:jmax+1]),color='#2b7bbc',fmt='o', ms=5, label=r'Error of the simulations')
#ax.errorbar(myanalysis.jvec[1:jmax+1], (betaj_TS_galT_mask_mean[1:jmax+1] )/(betatg[1:jmax+1])-1, yerr=np.sqrt(np.diag(cov_TS_galT_mask)[1:jmax+1])/(betatg[1:jmax+1]), color='darkgrey', fmt='o', ms=10,capthick=5, label=r'Variance one sim')

ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\langle \hat{\beta}_j^{\mathrm{TG}} \rangle/\beta_j^{\mathrm{TG,\,th}}$-1')# - \beta_j^{Tgal, th}
#ax.set_ylim([-0.3,1.3])

fig.tight_layout()
plt.savefig(f'plot_tesi/Euclid/betaj_mean_T_gal_noise_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')
plt.savefig(f'plot_tesi/Euclid/betaj_mean_T_gal_noise_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.pdf', bbox_inches='tight')


#############################################

# PROVA MASTER
wl = hp.anafast(mask, lmax=lmax)
mll  = need_theory.get_Mll(wl, lmax=lmax)

###############
#MASTER NEEDLETS

gammaJ_tg = need_theory.gammaJ(cl_theory_tg, wl, jmax, lmax)
delta_gammaj = need_theory.variance_gammaj(cltg=cl_theory_tg,cltt=cl_theory_tt, clgg=cl_theory_gg, wl=wl, jmax=jmax, lmax=lmax, noise_gal_l=Nll)

np.savetxt(out_dir+f'gammaj_theory_B{myanalysis.B:1.2f}_jmax{jmax}_nside{nside}_lmax{lmax}_nsim{nsim}_fsky{fsky:1.2f}.dat', gammaJ_tg)
np.savetxt(out_dir+f'variance_gammaj_theory_B{myanalysis.B:1.2f}_jmax{jmax}_nside{nside}_lmax{lmax}_nsim{nsim}_fsky{fsky:1.2f}.dat', delta_gammaj)

##RELATIVE DIFFERENCE
fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim)+' MASK')

ax = fig.add_subplot(1, 1, 1)

ax.axhline(ls='--', color='grey')
ax.errorbar(myanalysis.jvec[1:jmax+1], (betaj_TS_galT_mask_mean[1:jmax+1] /gammaJ_tg[1:jmax+1]-1), yerr=np.sqrt(np.diag(delta_gammaj)[1:jmax+1])/(np.sqrt(nsim)*gammaJ_tg[1:jmax+1]),  fmt='o', ms=5, label=r'Error of the mean of the simulations')
ax.errorbar(myanalysis.jvec[1:jmax+1], (betaj_TS_galT_mask_mean[1:jmax+1] /gammaJ_tg[1:jmax+1]-1), yerr=np.sqrt(np.diag(cov_TS_galT_mask)[1:jmax+1])/(np.sqrt(nsim)*gammaJ_tg[1:jmax+1]),color='#2b7bbc',  fmt='o', ms=5, label=r'Error of simulations')

ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\langle \tilde{\Gamma}_j^{TG} \rangle/\tilde{\Gamma}_j^{TG, th}$-1')# - \beta_j^{Tgal, th}
#ax.set_ylim([-0.3,1.3])

fig.tight_layout()
plt.savefig(f'plot_tesi/Euclid/gammaj_mean_T_gal_noise_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')
plt.savefig(f'plot_tesi/Euclid/gammaj_mean_T_gal_noise_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.pdf', bbox_inches='tight')

### SPECTRUM  CUT SKY

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.plot(myanalysis.jvec[1:jmax+1], gammaJ_tg[1:jmax+1], label='Theory')
ax.errorbar(myanalysis.jvec[1:jmax+1], betaj_TS_galT_mask_mean[1:jmax+1], yerr=np.sqrt(np.diag(cov_TS_galT_mask)[1:jmax+1])/np.sqrt(nsim) ,color='#2b7bbc',fmt='o',ms=5,capthick=2, label='Error of the mean of the simulations')
ax.errorbar(myanalysis.jvec[1:jmax+1], betaj_TS_galT_mask_mean[1:jmax+1], yerr=np.sqrt(np.diag(cov_TS_galT_mask)[1:jmax+1]) ,color='grey',fmt='o',ms=0,capthick=2, label='Error of simulations')

ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\tilde{\Gamma}^{\mathrm{\,TG}}_j$')

fig.tight_layout()
plt.savefig(f'plot_tesi/Euclid/gammaj_theory_T_gal_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')
plt.savefig(f'plot_tesi/Euclid/gammaj_theory_T_gal_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.pdf', bbox_inches='tight')

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

s2n_theory=S_2_N_th(betatg[1:jmax+1], delta_noise[1:jmax+1]**2)
s2n_mean_sim=S_2_N_th(betaj_TS_galT_mask_mean[1:jmax+1], np.diag(delta_gammaj)[1:jmax+1])


print(np.where(s2n_theory==s2n_theory.max()),np.where(betatg==betatg.max()) )

fig = plt.figure(figsize=(17,10))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.plot(myanalysis.jvec[1:jmax+1], s2n_theory, color='#2b7bbc',marker = 'o',ms=10, label='Expected')
ax.plot(myanalysis.jvec[1:jmax+1], s2n_mean_sim,marker = 'o',ms=10, label='From simulations')
ax.set_ylim(top=3.8)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
plt.legend()
ax.set_xlabel(r'$j$')
ax.set_ylabel('Signal-to-Noise ratio')

fig.tight_layout()
plt.savefig(f'plot_tesi/Euclid/SNR_betaj_theory_T_gal_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.png', bbox_inches='tight')
plt.savefig(f'plot_tesi/Euclid/SNR_betaj_theory_T_gal_jmax{jmax}_D{myanalysis.B:0.2f}_nsim{nsim}_nside{nside}_mask.pdf', bbox_inches='tight')

print(betatg)