import numpy as np
import matplotlib.pyplot as plt
import spectra, utils
import healpy as hp
import cython_mylibc as mylibc
from IPython import embed
from matplotlib import rc, rcParams
import seaborn as sns
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
import master_needlets


# Matplotlib defaults ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#rc('text',usetex=True)
#rc('font',**{'family':'serif','serif':['Computer Modern']})
plt.rcParams['axes.linewidth']  = 5.
plt.rcParams['axes.labelsize']  =30
plt.rcParams['xtick.labelsize'] =30
plt.rcParams['ytick.labelsize'] =30
plt.rcParams['xtick.major.size'] = 30
plt.rcParams['ytick.major.size'] = 30
plt.rcParams['xtick.minor.size'] = 30
plt.rcParams['ytick.minor.size'] = 30
plt.rcParams['legend.fontsize']  = 'large'
plt.rcParams['legend.frameon']  = False
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['axes.titlesize'] = 'large'
plt.rcParams["errorbar.capsize"] = 15
#
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['font.size'] = 40
plt.rcParams['lines.linewidth']  = 5.
#plt.rcParams['backend'] = 'WX'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# embed()



fname_xcspectra = '/ehome/bdecaro/xcmbneed/src/spectra/inifiles/EUCLID_fiducial_lmin0.dat'
simparams = {'nside'   : 128, 
             'ngal'    : 5.76e5, 
             'ngal_dim': 'ster',
             'pixwin'  : False}
nside = simparams['nside']

jmax = 12
lmax = 256
nsim = 100
B    = mylibc.mylibpy_jmax_lmax2B(jmax, lmax)
jvec = np.arange(jmax+1)
lbmin = 1
print(f'B={B:0.2f}')

# Loading mask
mask  = mask = hp.read_map(f'/ehome/bdecaro/xcmbneed/src/mask/mask_planck_comm_2018_nside={nside}.fits', verbose=False)

mask2 = hp.read_map('/ehome/bdecaro/xcmbneed/src/mask/EUCLID/mask_rsd2022g-wide-footprint-year-6-equ-order-13-moc_ns0064_G_filled_2deg2.fits', verbose=False)
wl    = hp.anafast(mask, lmax=lmax)
wl2   = hp.anafast(mask2, lmax=lmax)

mask_EP = hp.read_map(f'/ehome/bdecaro/xcmbneed/src/mask/EUCLID/mask_planck_comm_2018_x_euclid_fsky0p35_nside={nside}.fits')
fsky_EP = np.mean(mask_EP)
wl_EP = hp.anafast(mask_EP, lmax=lmax)

# Reading input spectra
xcspectra = spectra.XCSpectraFile(fname_xcspectra,   WantTG = True)
cl_theory = np.loadtxt('/ehome/bdecaro/xcmbneed/src/spectra/inifiles/EUCLID_fiducial_lmin0.dat')
ell_theory = cl_theory[0]
cl_theory_tt = cl_theory[1]
cl_theory_tg = cl_theory[2]
cl_theory_gg = cl_theory[3]
# Theory Needlet spectra
need_theory = spectra.NeedletTheory(B)

# Theory + Normalization Needlet power spectra
betatg    = need_theory.cl2betaj(jmax=jmax, cl=cl_theory_tg)
betatt    = need_theory.cl2betaj(jmax=jmax, cl=cl_theory_tt)
beta_norm = need_theory.cl2betaj(jmax=jmax, cl=np.ones(cl_theory_tg.size))

mll  = need_theory.get_Mll(wl, lmax=lmax)
mll2 = need_theory.get_Mll(wl2, lmax=lmax)
mll_EP = need_theory.get_Mll(wl_EP, lmax=lmax)
# plt.imshow(mll, interpolation='nearest')
# plt.show()
# for l in [2,10,50,100,200]:
# 	plt.plot(np.log10(mll[l,:]), label=r'$\ell=%d$'%l)
# plt.legend()
# plt.xlabel(r'$\ell$')
# plt.ylabel(r'$M_{\ell\ell}$')
# # plt.yscale('log')
# plt.show()

# Harmonic pseudo-Cl ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pcl  = np.dot(mll, cl_theory_tg[:lmax+1])
pcl2 = np.dot(mll2, cl_theory_tg[:lmax+1])
pcl_EP= np.dot(mll_EP, cl_theory_tg[:lmax+1])

f, (ax1,ax2,ax3) = plt.subplots(3, sharex=True, figsize=(10,8))

ax1.plot(cl_theory_tg[:lmax+1], label=r'$C_{\ell}$')
ax1.plot(pcl,  label=r'$\tilde{C}_{\ell}$ Planck Mask')
ax1.plot(pcl2, label=r'$\tilde{C}_{\ell}$ Euclid Mask')
ax1.set_yscale('log')
ax1.set_xlim([2,400])
ax1.legend()
ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))

ax2.plot(pcl/cl_theory_tg[:lmax+1], label=r'$\tilde{C}_{\ell}/C_{\ell}$ Planck')
ax2.axhline(np.mean(mask), ls='--', color='k', label=r'$f^{Planck}_{\rm sky}$')
ax2.legend(loc='best')
ax2.set_xlim([2,400])
ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))

ax3.plot(pcl2/cl_theory_tg[:lmax+1], label=r'$\tilde{C}_{\ell}/C_{\ell}$ Euclid')
ax3.axhline(np.mean(mask2), ls=':', color='k', label=r'$f^{Euclid}_{\rm sky}$')

ax3.legend(loc='best')
ax3.set_xlim([2,400])
ax3.set_xlabel(r'$\ell$', size=25)

f.subplots_adjust(hspace=0)

plt.savefig('master_prova/master_cl.png', bbox_inches='tight')
#plt.show()

# Loading betaj sims ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
betaj_sims_T_gal1noise         = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000_3/betaj_sims_TS_galT_jmax12_B_1.59_nside128.dat')
betaj_sims_T_gal1noise_EPmask  = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000_3/betaj_sims_TS_galT_jmax12_B_1.59_nside128_fsky0.36.dat')

cov_sims_T_gal1noise      = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000_3/cov_TS_galT_jmax12_B_1.59_nside128.dat')
cov_sims_T_gal1noise_mask = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000_3/cov_TS_galT_jmax12_B_1.59_nside128_fsky0.36.dat')

betaj_mean_T_gal1noise        = np.mean(betaj_sims_T_gal1noise, axis=0)
betaj_mean_T_gal1noise_EPmask = np.mean(betaj_sims_T_gal1noise_EPmask, axis=0)

gammaJ_tg = need_theory.gammaJ(cl_theory_tg, wl_EP, jmax, lmax)
gammaJ_tt = need_theory.gammaJ(cl_theory_tt, wl_EP, jmax, lmax)

f, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(27,20))

ax1.errorbar(jvec, betaj_mean_T_gal1noise_EPmask/beta_norm, yerr=np.diag(cov_sims_T_gal1noise_mask/nsim)**.5/beta_norm, color='tomato', fmt='o', capsize=0, label=r'$\hat{\Gamma}_j^{\kappa\kappa}$ H-ATLAS Mask')
ax1.plot(gammaJ_tg/beta_norm, label=r'$\frac{\Gamma_j}{4\pi}$')
ax1.set_xlabel(r'$j$', size=25)
ax1.legend(loc='best')
ax2.axhline(ls='--', color='k')
ax2.errorbar(jvec, (betaj_mean_T_gal1noise_EPmask-gammaJ_tg)/(gammaJ_tg),   yerr=np.sqrt(np.diag(cov_sims_T_gal1noise_mask))/(np.sqrt(nsim)*gammaJ_tg), color='firebrick', fmt='o', capsize=0,label=r'$\kappa^S \times \kappa^S$ H-ATLAS Mask')
ax2.set_xlabel(r'$j$')
ax2.set_ylabel(r'$4\pi\frac{\langle \hat{\Gamma_j} \rangle}{\Gamma_j}-1$')
ax2.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='upper'))
# ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
#ax2.set_xlim([0.5, 11.5])
#ax2.set_ylim([-0.03, 0.1])

f.subplots_adjust(hspace=0)
# plt.show()
plt.savefig('master_prova/master_gammaj.png', bbox_inches='tight')


#### BACKWARD MODELLING
master_fsky = master_needlets.Master_needlets(B=B,lmin=0, lmax=lmax, jmax=jmax, mask=mask_EP, fsky_approx=True)
#fsky approx
betaj_fsky = np.zeros_like(betaj_sims_T_gal1noise_EPmask)

for n in range(betaj_sims_T_gal1noise_EPmask.shape[0]):
    betaj_fsky[n] = master_fsky.get_spectra(betaj_sims_T_gal1noise_EPmask[n])
    #print(betaj_sims_T_gal1noise_EPmask[n]/betaj_fsky[n])
cov_fsky = np.cov(betaj_fsky.T)
betaj_mean_fsky = np.mean(betaj_fsky, axis = 0 )

master = master_needlets.Master_needlets(B=B,lmin=0, lmax=lmax, jmax=jmax, mask=mask_EP, fsky_approx=False)
betaj_master = np.zeros_like(betaj_sims_T_gal1noise_EPmask[:, 1:])

for n in range(betaj_sims_T_gal1noise_EPmask.shape[0]):
    betaj_master[n] = master.get_spectra(betaj_sims_T_gal1noise_EPmask[n, 1:])
    #print(betaj_sims_T_gal1noise_EPmask[n]/betaj_fsky[n])

cov_master = np.cov(betaj_master.T)
betaj_mean_master = np.mean(betaj_master, axis = 0 ) #master.get_spectra(np.dot(master.P_jl,pcl_EP))   


##RELATIVE DIFFERENCE
fig = plt.figure(figsize=(27,20))


ax = fig.add_subplot(1, 1, 1)

ax.axhline(ls='--', color='k')
#ax.errorbar(myanalysis.jvec[1:jmax], (betaj_TS_galT_mask_mean[1:jmax] )/(gammaJ_tg[1:jmax]/(4*np.pi))-1, yerr=delta_noise[1:jmax]/(np.sqrt(nsim)*gammaJ_tg[1:jmax]/(4*np.pi)), color='firebrick', fmt='o',  ms=10,capthick=5, label=r'Variance from theory')
ax.errorbar(jvec[1:jmax], (betaj_mean_fsky[1:jmax] /betatg[1:jmax]-1), yerr=np.sqrt(np.diag(cov_sims_T_gal1noise_mask)[1:jmax])/(np.sqrt(nsim)*betatg[1:jmax]), color='firebrick', fmt='o', ms=10,capthick=5, label=r'fsky approx')
ax.errorbar(jvec[1:jmax], (betaj_mean_master[:jmax-1] /betatg[1:jmax]-1), yerr=np.sqrt(np.diag(cov_master)[:jmax-1])/(np.sqrt(nsim)*betatg[1:jmax]), color='seagreen', fmt='o', ms=10,capthick=5, label=r'Master')
#ax.errorbar(myanalysis.jvec[1:jmax], (betaj_TS_galT_mask_mean[1:jmax] )/(betatg[1:jmax])-1, yerr=np.sqrt(np.diag(cov_TS_galT_mask)[1:jmax])/(betatg[1:jmax]), color='darkgrey', fmt='o', ms=10,capthick=5, label=r'Variance one sim')

ax.legend(loc='best')
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\langle \tilde{\beta}_j^{Tgal} \rangle/\tilde{\beta}_j^{Tgal, th}$-1')# - \beta_j^{Tgal, th}
ax.set_xlim([2.1,12])
ax.set_ylim([-2,2])

fig.tight_layout()

plt.savefig('master_prova/confronto_fsky_master.png', bbox_inches='tight')



