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


sns.set(rc={"figure.figsize": (8, 6)})
plt.clf()
sns.set_style("ticks", {'figure.facecolor': 'grey'})


#rc('text',usetex=True)
#rc('font',**{'family':'serif','serif':['Computer Modern']})
plt.rcParams['axes.linewidth']  = 2.
plt.rcParams['axes.labelsize']  = 22
plt.rcParams['axes.titlesize']  = 22
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['xtick.major.size'] = 7
plt.rcParams['ytick.major.size'] = 7
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['legend.fontsize']  = 18
plt.rcParams['legend.frameon']  = False

plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1



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
betaj_sims_T_gal1noise         = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000/betaj_sims_TS_galT_jmax12_B_1.59_nside128.dat')
betaj_sims_T_gal1noise_EPmask  = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000/betaj_sims_TS_galT_jmax12_B_1.59_nside128_fsky0.36.dat')

cov_sims_T_gal1noise      = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000/cov_TS_galT_jmax12_B_1.59_nside128.dat')
cov_sims_T_gal1noise_mask = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000/cov_TS_galT_jmax12_B_1.59_nside128_fsky0.36.dat')

betaj_mean_T_gal1noise        = np.mean(betaj_sims_T_gal1noise, axis=0)
betaj_mean_T_gal1noise_EPmask = np.mean(betaj_sims_T_gal1noise_EPmask, axis=0)

gammaJ_tg = need_theory.gammaJ(cl_theory_tg, wl_EP, jmax, lmax)
gammaJ_tt = need_theory.gammaJ(cl_theory_tt, wl_EP, jmax, lmax)

f, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(27,20))

ax1.errorbar(jvec, betaj_mean_T_gal1noise_EPmask/beta_norm, yerr=np.diag(cov_sims_T_gal1noise_mask/nsim)**.5/beta_norm, color='tomato', fmt='o', capsize=0, label=r'$\hat{\Gamma}_j^{\kappa\kappa}$ H-ATLAS Mask')
ax1.plot(gammaJ_tg/beta_norm/(4*np.pi), label=r'$\frac{\Gamma_j}{4\pi}$')
ax1.set_xlabel(r'$j$', size=25)
ax1.legend(loc='best')
ax2.axhline(ls='--', color='k')
ax2.errorbar(jvec, (betaj_mean_T_gal1noise_EPmask-gammaJ_tg/(4*np.pi))/(gammaJ_tg/(4*np.pi)),   yerr=np.sqrt(np.diag(cov_sims_T_gal1noise_mask))/(np.sqrt(nsim)*gammaJ_tg/(4*np.pi)), color='firebrick', fmt='o', capsize=0,label=r'$\kappa^S \times \kappa^S$ H-ATLAS Mask')
ax2.set_xlabel(r'$j$')
ax2.set_ylabel(r'$4\pi\frac{\langle \hat{\Gamma_j} \rangle}{\Gamma_j}-1$')
ax2.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='upper'))
# ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
#ax2.set_xlim([0.5, 11.5])
#ax2.set_ylim([-0.03, 0.1])

f.subplots_adjust(hspace=0)
# plt.show()
plt.savefig('master_prova/master_beta_4pi.png', bbox_inches='tight')

# plt.plot(gammaJ_kk/beta_norm, label=r'$\Gamma^{\kappa\kappa}_j$')
# plt.plot(betakk/beta_norm, label=r'$\beta^{\kappa\kappa}_j$')
# plt.errorbar(jvec, betaj_mean_kappaS_kappaS/beta_norm, yerr=np.diag(np.cov(betaj_sims_kappaS_kappaS.T)/nsim)**.5/beta_norm, color='royalblue', fmt='o', capsize=0, label=r'$\hat{\beta}_j^{\kappa\kappa}$ Full sky')
# plt.errorbar(jvec, betaj_mean_kappaS_kappaS_hermask/beta_norm, yerr=np.diag(np.cov(betaj_sims_kappaS_kappaS_hermask.T)/nsim)**.5/beta_norm, color='tomato', fmt='o', capsize=0, label=r'$\hat{\Gamma}_j^{\kappa\kappa}$ H-ATLAS Mask')
# plt.errorbar(jvec, betaj_mean_kappaS_kappaS_hermask/beta_norm/np.mean(mask), yerr=np.diag(np.cov(betaj_sims_kappaS_kappaS_hermask.T/np.mean(mask))/nsim)**.5/beta_norm, color='orange', fmt='o', capsize=0, label=r'$\hat{\Gamma}_j^{\kappa\kappa}/f_{\rm sky}$ H-ATLAS Mask')
# plt.xlim([0, jmax+2])
# plt.xlabel(r'$j$', size=25)
# plt.ylim([0,4e-7])
# plt.legend()
# plt.savefig('master_betaj.pdf', bbox_inches='tight')
# plt.show()



Z_j = (betaj_sims_T_gal1noise_EPmask - gammaJ_tg)/need_theory.sigmaJ(cl_theory_tg, wl, jmax, lmax)

Z_j_mean  = np.mean(Z_j, axis=0)
Z_j_sigma = np.var(Z_j, axis=0)**5

Z_ = (betaj_sims_T_gal1noise_EPmask - gammaJ_tg)/(need_theory.sigmaJ(cl_theory_tg, wl_EP, jmax, lmax)/np.sqrt(nsim))

for i in np.random.randint(1, high=nsim-1, size=nsim):
    plt.plot(jvec, Z_j[i,:], color='grey', alpha=0.3)
plt.plot(jvec, Z_j[0,:], color='grey', alpha=0.3, label='Sims')
plt.plot(jvec, Z_j_mean, lw=2, color='r', label='EUCLID mask')
# plt.errorbar(jvec, Z_, yerr=Z_j_sigma/nsim**.5, color='tomato', fmt='o', capsize=0, label=r'$\kappa^S \times \kappa^S$ H-ATLAS mask')
# plt.errorbar(jvec, Z_j_mean, yerr=Z_j_sigma/nsim**.5, color='royalblue', fmt='o', capsize=0, label=r'$\kappa^S \times \kappa^S$ H-ATLAS mask')
plt.axhline(ls='--', color='k')
plt.xlabel(r'$j$', size=25)
plt.xlim([0, jmax+1])
plt.ylabel(r'$\langle Z_j \rangle$', size=25)
plt.legend()
plt.savefig('master_prova/master_Zj.png', bbox_inches='tight')
#plt.show()

plt.rcParams['axes.linewidth']  = 2.
plt.rcParams['axes.labelsize']  = 20
plt.rcParams['axes.titlesize']  = 20
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

for j in range(1,jmax+1):
    plt.subplot(3,4,j)
    # hist(Z_j[:,j], bins='blocks', histtype='step', label=r'$j=$%d' %j)
    plt.hist(Z_j[:,j], bins=30, histtype='step', label=r'$j=$%d' %j)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    if j > 8:
        plt.xlabel(r'$Z_j$')
    plt.legend()
plt.savefig('master_prova/master_Zj_hist.png', bbox_inches='tight')
#plt.show()

# embed()
# plt.subplot(211)
# plt.plot(gammaJ/beta_norm, label=r'$\Gamma_j$')
# plt.plot(betakg/beta_norm, label=r'$\beta_j$')
# plt.legend()
# plt.subplot(212)
# plt.plot(gammaJ/betakg)
# plt.axhline(np.mean(mask), ls='--', color='k', label=r'$f_{\rm sky}$')
# plt.legend()
# plt.show()

# plt.plot(betaj_mean/beta_norm, label='Sims')
# plt.plot(betaj_mean/beta_norm*np.mean(mask), label='Sims x fsky')
# plt.plot(betakg/beta_norm, label=r'$\beta_j$')
# plt.plot(gammaJ/beta_norm, label=r'$\Gamma_j$')
# plt.legend()
# plt.show()

#PROVARE A FARE IL RESHAPE DI QUELLA COSA

def Mll_binning(B, jmax, lmax, mll):
        """
        Returns the binning scheme in  multipole space
        """
        assert(np.floor(B**(jmax)) <= mll.shape[0]-1) 
        #ellj = []
        mjj=np.zeros((jmax,jmax))
        for j in range(jmax):
            lmin = np.floor(B**(j-1))
            lmax = np.floor(B**(j))
            ell  = np.arange(lmin, lmax+1, dtype=int)
            print(j, ell)
            for j1 in range(jmax):
                lmin1 = np.floor(B**(j1-1))
                lmax1 = np.floor(B**(j1+1))
                ell1  = np.arange(lmin1, lmax1+1, dtype=int)
                print(j1, ell1)
                mjj[j][j1]=[mll[:][l] for l in ell1]
        return mjj
        
def ell_binning(B, jmax, lmax, ell):
        """
        Returns the binning scheme in  multipole space
        """
        assert(np.floor(B**(jmax+1)) <= ell.size-1) 
        ellj = []
        for j in range(jmax+1):
            lmin = np.floor(B**(j-1))
            lmax = np.floor(B**(j+1))
            ell1  = np.arange(lmin, lmax+1, dtype=int)
            ellj.append(ell1)
        return np.asarray(ellj, dtype=object)

ell = ell_binning(B, jmax, lmax, ell_theory)
print(ell)
Mll_j = Mll_binning(B, jmax, lmax, mll)


print(mll[:13][:13], Mll_j)


