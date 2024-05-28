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
jmax= 5

# Paths
fname_xcspectra = 'spectra/inifiles/EUCLID_fiducial_lmin0.dat'
sims_dir        = f'sims/Euclid_sims_Marina/NSIDE{nside}/'
out_dir         = f'output_needlet_TG/EUCLID/Mask_noise/TG_{nside}_nsim{nsim}_mergej/'
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

mergej=[1,2,3]
myanalysis = analysis.NeedAnalysis(jmax, lmax, out_dir, simulations, mergej=mergej)
B=myanalysis.B
out_dir_plot    = out_dir+f'plot_D{B:1.2f}/'
if not os.path.exists(out_dir_plot):
        os.makedirs(out_dir_plot)

# Theory Needlet theory and windows functions
need_theory = spectra.NeedletTheory(myanalysis.B)


#filename_D = f'b_need/merge_bneed_lmax256_jmax{jmax}_B{B:1.2f}.dat'
#b2_D = np.loadtxt(filename_D)
#
#print(b2_D.shape)
#
#fig, ax1  = plt.subplots(1,1,figsize=(7,5)) 
#plt.suptitle(r'$D = %1.2f $' %myanalysis.B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax))
#
#for i in range(b2_D.shape[0]):
#    ax1.plot(b2_D[i], label = 'j='+str(i) )
#    print(b2_D[i])
#ax1.set_xscale('log')
##ax1.set_xlim(-1,10)
#ax1.set_xlabel(r'$\ell$')
#ax1.set_ylabel(r'$w^{2}(\frac{\ell}{D^{j}})$')
#ax1.legend(loc='right')
#plt.tight_layout()
#plt.savefig(out_dir_plot+f'b2_D{B:1.2f}.png')

# Computing simulated Cls 
print("...computing Betajs for simulations...")
fname_betaj_sims_TS_galT      = f'betaj_sims_TS_galT_jmax{jmax}_B_{myanalysis.B:0.2f}_nside{nside}.dat'
fname_betaj_sims_TS_galT_mask = f'betaj_sims_TS_galT_jmax{jmax}_B_{myanalysis.B:0.2f}_nside{nside}_fsky{fsky:0.2f}.dat'

betaj_sims_TS_galT       = myanalysis.GetBetajSimsFromMaps('T', nsim, field2='g1noise', fname=fname_betaj_sims_TS_galT, fsky_approx=False,EuclidSims=True)
betaj_sims_TS_galT_mask  = myanalysis.GetBetajSimsFromMaps('T', nsim, field2='g1noise', mask=mask, fname=fname_betaj_sims_TS_galT_mask, fsky_approx=False,EuclidSims=True)

# Covariances
#print("...computing Cov Matrices...")
#fname_cov_TS_galT            = f'cov_TS_galT_jmax{jmax}_B_{myanalysis.B:0.2f}_nside{nside}.dat'
#fname_cov_TS_galT_mask       = f'cov_TS_galT_jmax{jmax}_B_{myanalysis.B:0.2f}_nside{nside}_fsky{fsky:0.2f}.dat'
#
#cov_TS_galT, corr_TS_galT               = myanalysis.GetCovMatrixFromMaps(field1='T', nsim=nsim, field2='g1noise', fname=fname_cov_TS_galT, fname_sims=fname_betaj_sims_TS_galT)
#cov_TS_galT_mask, corr_TS_galT_mask     = myanalysis.GetCovMatrixFromMaps(field1='T', nsim=nsim, field2='g1noise', mask=mask,fname=fname_cov_TS_galT_mask, fname_sims=fname_betaj_sims_TS_galT_mask)
#
#print("...done...")
#
## <Beta_j>_MC
#betaj_TS_galT_mean         = myanalysis.GetBetajMeanFromMaps('T', nsim, field2='g1noise', fname_sims=fname_betaj_sims_TS_galT)
#betaj_TS_galT_mask_mean    = myanalysis.GetBetajMeanFromMaps('T', nsim, field2='g1noise', mask=mask, fname_sims=fname_betaj_sims_TS_galT_mask)
