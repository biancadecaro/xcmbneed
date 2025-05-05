#!/usr/bin/env python
import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams, gridspec
import healpy as hp
from astropy.io import fits
import os
import cython_mylibc as mylibc
import analysis, spectra, sims
from IPython import embed
import seaborn as sns
sns.set_theme()
sns.set_theme(style = 'white')
pal = sns.color_palette("crest", n_colors=13)
pal_cmap = sns.color_palette("crest", n_colors=13, as_cmap=True)

sns.set_theme(style = 'white')
plt.rcParams['lines.linewidth']  = 2.
plt.rcParams['axes.labelsize']  =14
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['savefig.dpi']=300

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)

from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 

###########################################################################################################
################################################################################
###PSEUDO

def cov_pseudo_cl(cltg,cltt, clgg,Mll,  Mll_1x2, lmax, fsky=1.,noise_gal_l=None):
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
	for ell1 in ell:
		for ell2 in ell:
			covll[ell1,ell2] = (Mll_1x2[ell1,ell2]*(cltg[ell1]*cltg[ell2])+Mll[ell1,ell2]*(np.sqrt(cltt[ell1]*cltt[ell2]*clgg_tot[ell1]*clgg_tot[ell2])))/(fsky*(2.*ell1+1))
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


#################################################################################################################################
# Parameters
simparams = {'nside'   : 128,
			 'ngal'    : 354543085.80126834,#35454308.580126834, #dovrebbe importare solo per lo shot noise (noise poissoniano)
 		 	 'ngal_dim': 'ster',
		 	 'pixwin'  : False}

nside = simparams['nside']

lmax = 256
nsim = 1000
jmax= 12
jvec=np.arange(jmax+1)
# Paths
fname_xcspectra = 'spectra/inifiles/EUCLID_fiducial_lmin0.dat'
sims_dir        = f'sims/Euclid_sims_Marina/NSIDE{nside}/'
out_dir         = f'output_needlet_TG/EUCLID/Mask_noise/TG_{nside}_nsim{nsim}_2masks_1/'
path_inpainting = 'inpainting/inpainting.py'


cl_theory = np.loadtxt('spectra/inifiles/EUCLID_fiducial_lmin0.dat')
ell_theory = cl_theory[0]
cl_theory_tt = cl_theory[1]
cl_theory_tg = cl_theory[2]
cl_theory_gg = cl_theory[3]

#####################################################################
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
########################################################################################################
mask_eu = hp.read_map(f'mask/EUCLID/mask_rsd2022g-wide-footprint-year-6-equ-order-13-moc_ns0{nside}_G_filled_2deg2.fits')
mask_pl = hp.read_map(f'mask/mask_temp_ns{nside}.fits')

fsky_eu = np.mean(mask_eu)
fsky_pl = np.mean(mask_pl)

mask_comb = mask_pl*mask_eu

fsky_comb = np.mean(mask_comb)
print(f'fsky_pl = {fsky_pl}, fsky_eu={fsky_eu}, fsky={fsky_comb}')

del mask_eu; del mask_comb; del mask_pl

Nll = np.ones(cl_theory_gg.shape[0])/simparams['ngal']

with fits.open('mask/EUCLID/kern_RSD2022G_T_G_TT.fits') as hdul:
    Mll_cross = hdul[0].data

Mll_pl_eu  = np.loadtxt(f'mask/EUCLID/kernel_Euclid_Planck_TTGG_lmax{lmax}.dat')
Mll_comb  = np.loadtxt(f'mask/EUCLID/kernel_Euclid_Planck_TGTG_lmax{lmax}.dat')

gammaj_TS_galT_mask = np.loadtxt(out_dir+f'gamma_sims_TS_galT_jmax12_B_1.59_nside128_fsky0.36.dat') 
cov_TS_galT_mask = np.loadtxt(out_dir+f'cov_TS_galT_jmax12_B_1.59_nside128_fsky0.36.dat') 
gammaj_TS_galT_mask_mean = np.mean(gammaj_TS_galT_mask, axis=0)

gammaJ_tg = np.loadtxt(out_dir+'gammaj_tg_jmax12_lmax256.dat')
delta_gammaj = np.loadtxt(out_dir+'covariance_gammaj_tg_jmax12_lmax256.dat')

###################################
# RELATIVE DIFFERENCE ESTIMATES 
#fig = plt.figure()
#
#ax = fig.add_subplot(1, 1, 1)
#
#ax.axhline(ls='--', color='grey')
#
#ax.errorbar(jvec[1:], (gammaj_TS_galT_mask_mean[1:] /gammaJ_tg[1:]-1)*100, yerr=100*np.sqrt(np.diag(delta_gammaj)[1:])/(np.sqrt(nsim)*gammaJ_tg[1:]),  fmt='o', ms=0, label='Variance of the mean from theory')
#ax.errorbar(jvec[1:], (gammaj_TS_galT_mask_mean[1:] /gammaJ_tg[1:]-1)*100, yerr=100*np.sqrt(np.diag(cov_TS_galT_mask)[1:])/(np.sqrt(nsim)*gammaJ_tg[1:]),  ms=3,fmt='o',  label='Variance of the mean from simulations')
#
#ax.legend(loc='best')
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#ax.yaxis.set_major_formatter(formatter) 
#ax.set_xticks(jvec[1:])
#ax.set_xticklabels(jvec[1:])
#ax.set_xlabel(r'$j$')
#ax.set_ylabel(r'$\% \langle \tilde{\Gamma}_j^{TG} \rangle/\tilde{\Gamma}_j^{TG, th}$-1')
#
#fig.tight_layout()



###########################################################################################################
cls_tg = np.loadtxt('cls_from_maps/EUCLID/Euclid_Planck_masks/cls_Tgalnoise_anafast_nside128_lmax256_Euclidnoise_Marina_nsim1000_fsky0.36.dat')
cls_tg_mean=np.mean(cls_tg, axis=0)

pcl = np.dot(Mll_pl_eu,cl_theory_tg[:lmax+1]) 
cl_recovered = np.dot(np.linalg.inv(Mll_pl_eu[2:,2:]), cls_tg_mean[2:]) 


cov_pcl_sim = np.cov(cls_tg.T)

cov_pcl= cov_pseudo_cl(cltg=cl_theory_tg,cltt=cl_theory_tt, clgg=cl_theory_gg, Mll=Mll_cross,  Mll_1x2=Mll_comb, lmax=lmax,noise_gal_l=Nll)
cov_cls = cov_cl(cltg=cl_theory_tg,cltt=cl_theory_tt, clgg=cl_theory_gg, lmax=lmax,lmin=2,fsky=0.36,noise_gal_l=Nll)

#################################################################################################################
ell = np.arange(lmax+1)
factor = ell*(ell+1)/(2*np.pi)

#fig = plt.figure()
#plt.suptitle(r'CL recovered $ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~f_{sky} = %.2f$'%fsky_comb)
#ax = fig.add_subplot(1, 1, 1)
#ax.plot(ell[2:], factor[2:]*cl_theory_tg[2:lmax+1], label='Theory')
#ax.plot(ell[2:], factor[2:]*cl_recovered,'+' ,label='Cl recovered')
#ax.set_xlabel(r'$\ell$')
#ax.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi}C^{\rm TG}_{\ell}$')
#plt.legend()
#
#
#fig = plt.figure(figsize=(7,4))
#plt.suptitle(r'Mask PCL $ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(simparams['nside']) + r',$~f_{sky} = %.2f$'%fsky_comb)
#ax = fig.add_subplot(1, 1, 1)
#ax.errorbar(ell[2:],  (cls_tg_mean[2:]/pcl[2:]-1)*100, yerr=100*np.sqrt(np.diag(cov_pcl)[2:])/(np.sqrt(nsim)*pcl[2:]),   fmt='o', ms=0, label='Variance of the mean from theory')
#ax.errorbar(ell[2:],  (cls_tg_mean[2:]/pcl[2:]-1)*100, yerr=100*np.sqrt(np.diag(cov_pcl_sim)[2:])/(np.sqrt(nsim)*pcl[2:]),   ms=3,fmt='o',  label='Variance of the mean from simulations')
#
#ax.axhline(ls='--', color='grey')
#ax.set_xlabel(r'$\ell$')
#ax.set_ylabel('% relative diff ')
#plt.legend()

#########################################################################
######################### SIGNAL - TO - NOISE ###########################

def fl_j(j_m):
	ell_binning=need_theory.ell_binning(jmax, lmax)
	l_j = np.zeros(j_m+1, dtype=int)
	
	for j in range(1,l_j.shape[0]):
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

############################################################################

lmax_vec=fl_j(jmax)
lmax_vec_cl = np.arange(start=2,stop=256,dtype=int)

s2n_mean_sim=S_2_N(gammaj_TS_galT_mask_mean[1:jmax+1], delta_gammaj[1:jmax+1,1:jmax+1])
s2n_cum = S_2_N_cum(s2n_mean_sim, jvec)

s2n_mean_sim_cl=S_2_N_ell(cl_recovered, cov_cls)
s2n_cum_cl = S_2_N_cum_ell(s2n_mean_sim_cl,lmax_vec_cl)


print(f's2n_cum_lmax_need:{s2n_cum[-1]:0.2f}, s2n_cum_lmax_pcl:{s2n_cum_cl[-1]:0.2f}')

fig = plt.figure(figsize=(6,5))

ax = fig.add_subplot(1, 1, 1)
ax.plot(lmax_vec, s2n_cum, label='Needlets')
ax.plot(lmax_vec_cl, s2n_cum_cl, label= 'PCL')
ax.set_xscale('log')
ax.set_xlim(left=3, right=200)
ax.set_ylim(bottom=0.5,top=4.25)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 

ax.set_xlabel(r'$\ell_{\mathrm{max}}$')
ax.set_ylabel('Cumulative Signal-to-Noise ratio')
ax.legend()

fig.tight_layout()



plt.show()