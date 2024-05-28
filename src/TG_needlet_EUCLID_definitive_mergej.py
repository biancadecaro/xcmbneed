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
out_dir         = f'output_needlet_TG/EUCLID/Mask_noise/TG_{nside}_nsim{nsim}_mergej/'
path_inpainting = 'inpainting/inpainting.py'


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

mergej=[1,2]
myanalysis = analysis.NeedAnalysis(jmax, lmax, out_dir, simulations, mergej=mergej)
B=myanalysis.B
out_dir_plot    = out_dir+f'plot_D{B:1.2f}_mergej_{len(mergej)}/'
if not os.path.exists(out_dir_plot):
        os.makedirs(out_dir_plot)
jvec = myanalysis.jvec

# Theory Needlet theory and windows functions
need_theory = spectra.NeedletTheory(myanalysis.B)
wl = hp.anafast(mask, lmax=lmax)
Mll  = need_theory.get_Mll(wl, lmax=lmax)

b_need = need_theory.get_bneed(jmax, lmax, mergej)


fig, ax1  = plt.subplots(1,1) 
plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax))

for i in range(b_need.shape[0]):
    ax1.plot(b_need[i]*b_need[i], label = 'j='+str(i) )
ax1.set_xscale('log')
#ax1.set_xlim(-1,10)
ax1.set_xlabel(r'$\ell$')
ax1.set_ylabel(r'$w^{2}(\frac{\ell}{D^{j}})$')
ax1.legend(loc='right')
plt.savefig(out_dir_plot+f'b2_D{B:1.2f}.png')
plt.tight_layout()

ell_binning=need_theory.ell_binning(jmax, lmax)
fig = plt.figure()
plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax))
ax = fig.add_subplot(1, 1, 1)
for i in range(0,jvec.shape[0]):
    ell_range = ell_binning[i][ell_binning[i]!=0]
    plt.plot(ell_range, i*ell_range/ell_range, label= f'j={i}')
    plt.text(ell_range[0], i, r'$\ell_{min}=%d,\,\ell_{max}=%d$'%(ell_range[0],ell_range[-1]))

ax.set_xlabel(r'$\ell$')
ax.legend(loc='right', ncol=2)
plt.tight_layout()
plt.savefig(out_dir_plot+f'ell_binning_D{B:1.2f}.png')
#plt.show()
gammaJ_tg = need_theory.gammaJ(cl_theory_tg, wl, jmax, lmax)
delta_gammaj = need_theory.variance_gammaj(cltg=cl_theory_tg,cltt=cl_theory_tt, clgg=cl_theory_gg, wl=wl, jmax=jmax, lmax=lmax, noise_gal_l=Nll)


# Computing simulated Gammaj 
print("...computing Betajs for simulations...")
fname_gammaj_sims_TS_galT_mask = f'betaj_sims_TS_galT_mergej_{len(mergej)}_jmax{jmax}_B_{myanalysis.B:0.2f}_nside{nside}_fsky{fsky:0.2f}.dat'

gammaj_sims_TS_galT_mask  = myanalysis.GetBetajSimsFromMaps('T', nsim, field2='g1noise', mask=mask, fname=fname_gammaj_sims_TS_galT_mask, fsky_approx=False,EuclidSims=True)

# Covariances
print("...computing Cov Matrices...")
fname_cov_TS_galT_mask       = f'cov_TS_galT_mergej_{len(mergej)}_jmax{jmax}_B_{myanalysis.B:0.2f}_nside{nside}_fsky{fsky:0.2f}.dat'

cov_TS_galT_mask, corr_TS_galT_mask     = myanalysis.GetCovMatrixFromMaps(field1='T', nsim=nsim, field2='g1noise', mask=mask,fname=fname_cov_TS_galT_mask, fname_sims=fname_gammaj_sims_TS_galT_mask)

print("...done...")

# <Gamma_j>_MC
gammaj_TS_galT_mask_mean    = myanalysis.GetBetajMeanFromMaps('T', nsim, field2='g1noise', mask=mask, fname_sims=fname_gammaj_sims_TS_galT_mask)

# Covariances
print("...computing Cov Matrices...")
fname_cov_TS_galT_mask       = f'cov_TS_galT_jmax{jmax}_B_{myanalysis.B:0.2f}_nside{nside}_fsky{fsky:0.2f}.dat'

cov_TS_galT_mask, corr_TS_galT_mask     = myanalysis.GetCovMatrixFromMaps(field1='T', nsim=nsim, field2='g1noise', mask=mask,fname=fname_cov_TS_galT_mask, fname_sims=fname_gammaj_sims_TS_galT_mask)

print("...done...")

# <Gamma_j>_MC
gammaj_TS_galT_mask_mean    = myanalysis.GetBetajMeanFromMaps('T', nsim, field2='g1noise', mask=mask, fname_sims=fname_gammaj_sims_TS_galT_mask)

##########################################################################################
# Some plots
print("...here come the plots...")

### GAMMAJ TG 

fig = plt.figure()#figsize=(10,7))

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
plt.show()

##RELATIVE DIFFERENCE
fig = plt.figure()

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~N_{\mathrm{sim}} = $'+str(nsim)+' MASK')

ax = fig.add_subplot(1, 1, 1)

ax.axhline(ls='--', color='grey')

ax.errorbar(jvec[1:], (gammaj_TS_galT_mask_mean[1:] /gammaJ_tg[1:]-1), yerr=np.sqrt(np.diag(delta_gammaj)[1:])/(np.sqrt(nsim)*gammaJ_tg[1:]),  fmt='o', ms=0, label=r'Variance of the mean from theory')
ax.errorbar(jvec[1:], (gammaj_TS_galT_mask_mean[1:] /gammaJ_tg[1:]-1), yerr=np.sqrt(np.diag(cov_TS_galT_mask)[1:])/(np.sqrt(nsim)*gammaJ_tg[1:]),  ms=3,fmt='o',  label=r'Variance of the mean from simulations')

ax.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(jvec[1:])
ax.set_xticklabels(jvec[1:])
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\langle \tilde{\Gamma}_j^{TG} \rangle/\tilde{\Gamma}_j^{TG, th}$-1')# - \beta_j^{Tgal, th}
#ax.set_ylim([-0.3,1.3])

fig.tight_layout()
plt.savefig(out_dir_plot+f'relative_diff_gammaJ_D{B:1.2f}.png')
plt.show()

####################################################################################
############################# DIFF COVARIANCES #####################################
##NEEDLETS
fig = plt.figure()#figsize=(10,7))

plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~f_{sky} = %0.2f$'%fsky)

ax = fig.add_subplot(1, 1, 1)

ax.plot(jvec[1:], (np.diag(cov_TS_galT_mask)[1:]-np.diag(delta_gammaj)[1:])/np.diag(delta_gammaj)[1:]*100 ,'o',)
ax.axhline(ls='--', color='grey')

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter)
ax.set_xticks(jvec[1:]) 
ax.set_xticklabels(jvec[1:])
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'% $(\Delta \Gamma)^2_{\mathrm{sims}}/(\Delta \Gamma)^2_{\mathrm{analytic}}$ - 1')

fig.tight_layout()
plt.savefig(out_dir_plot+f'relative_diff_diag_cov_D{B:1.2f}.png')
plt.show()

