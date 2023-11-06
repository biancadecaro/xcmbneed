import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import master
from tqdm import tqdm
from IPython import embed
import seaborn as sns

from matplotlib import rc, rcParams
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter

arcmin2rad = 0.000290888208666

def SetPlotStyle():
   rc('text',usetex=True)
   rc('font',**{'family':'serif','serif':['Computer Modern']})
   plt.rcParams['axes.linewidth']  = 3.
   plt.rcParams['axes.labelsize']  = 28
   plt.rcParams['axes.titlesize']  = 22
   plt.rcParams['xtick.labelsize'] = 20
   plt.rcParams['ytick.labelsize'] = 18
   plt.rcParams['xtick.major.size'] = 7
   plt.rcParams['ytick.major.size'] = 7
   plt.rcParams['xtick.minor.size'] = 3
   plt.rcParams['ytick.minor.size'] = 3
   plt.rcParams['legend.fontsize']  = 22
   plt.rcParams['legend.frameon']  = False

   plt.rcParams['xtick.major.width'] = 1
   plt.rcParams['ytick.major.width'] = 1
   plt.rcParams['xtick.minor.width'] = 1
   plt.rcParams['ytick.minor.width'] = 1
   # plt.clf()
   sns.set(rc('font',**{'family':'serif','serif':['Computer Modern']}))
   sns.set_style("ticks", {'figure.facecolor': 'grey'})

def GetGalMask(nside, lat=None, fsky=None, nest=False):
    """
    Returns a symmetric Galactic Mask in Healpix format at nside resolution.
    Pixels with latitude < |lat| deg are set to 0.
    Otherwise you input the fsky and evaluates the required latitude.
    """
    if lat is None:
        if fsky is not None:
            lat = np.rad2deg(np.arcsin(1. - fsky))
        else:
            raise ValueError("Missing lat or fsky !")

    mask      = np.zeros(hp.nside2npix(nside))
    theta_cut = np.deg2rad(90. - lat)
    mask[np.where((hp.pix2ang(nside, np.arange(mask.size), nest=nest))[0] >= (np.pi - theta_cut))] = 1.
    mask[np.where((hp.pix2ang(nside, np.arange(mask.size), nest=nest))[0] <= theta_cut)] = 1.

    return mask

def GetStuff(cl):
    cl = np.asarray(cl)
    cl_mean = np.mean(cl, axis=0)
    cl_cov  = np.cov(cl.T)
    cl_corr = np.corrcoef(cl.T)
    cl_err  = np.sqrt(np.diag(cl_cov))

    return cl, cl_mean, cl_cov, cl_corr, cl_err

SetPlotStyle() 

# Pars
nside = 1024
beam  = 10.   # arcmin
lmin  = 2
lmax  = 2000
d_ell = 100
nsim  = 300
noise = 40. # \muK*arcmin
flat  = lambda l: l*(l+1)/2./np.pi

Bin = master.Binner(lmin=lmin, lmax=lmax, delta_ell=d_ell, flat=flat)
lb  = Bin.lb 

# Loading CMB spectra
l, cltt = np.loadtxt('CMB_spectra.dat', unpack=True)
cltt   *= (2*np.pi)/(l*(l+1))
cltt    = np.nan_to_num(cltt)
clttb   = Bin.bin_spectra(cltt)
nltt    = (noise * np.radians(1./60.))**2 * 1e-12

# Mask
mask_20 = GetGalMask(nside, fsky=0.2)
mask_01 = GetGalMask(nside, fsky=0.01)

# Initialize Master classes
# MST   = master.Master(mask_20, lmin=lmin, lmax=lmax, delta_ell=d_ell, flat=flat, pixwin=False, fwhm_smooth=None, MASTER=True)
# NOMST = master.Master(mask_20, lmin=lmin, lmax=lmax, delta_ell=d_ell, flat=flat, pixwin=False, fwhm_smooth=None, MASTER=False)

clMST   = []
clNOMST = []

# # Creating maps
# for i in tqdm(xrange(nsim)): 
#     mapTT = hp.synfast(cltt, nside, pixwin=False, verbose=False) 
#     mapNN = hp.synfast(nltt*np.ones(cltt.size), nside, pixwin=False, verbose=False)

#     # Extract spectra
#     clMST.append(MST.get_spectra(mapTT+mapNN, nl=np.ones(cltt.size)*nltt, pseudo=False))
#     clNOMST.append(NOMST.get_spectra(mapTT+mapNN, nl=np.ones(cltt.size)*nltt, pseudo=False))

clMST   = np.loadtxt('clMST.dat')
clNOMST = np.loadtxt('clNOMST.dat')

clMST, cl_meanMST, cl_covMST, cl_corrMST, cl_errMST = GetStuff(clMST)
clNOMST, cl_meanNOMST, cl_covNOMST, cl_corrNOMST, cl_errNOMST = GetStuff(clNOMST)

# plots
f, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(10,8))

# plt.suptitle(r'$  N_{\rm sim} = %d$ ' %nsim + r'$  N_{\rm side} = %d$ ' %nside + r'$\Delta\ell = %d$ ' %d_ell +' - Signal Only - ' + r'$f_{\rm sky} = 0.01$', size=20)
plt.suptitle('Nsim = %d ' %nsim + 'Nside = %d' %nside + 'DL = %d ' %d_ell + ' - DT = %2f muK - '%noise + ' fsky = %f '%np.mean(mask_20), size=20)

ax1.plot(cltt*l*(l+1)/2/np.pi, color='black', label='Theory')
ax1.plot(nltt*l*(l+1)/2/np.pi, ':', color='grey', label='Noise')
ax1.errorbar(lb-5, cl_meanMST,   yerr=cl_errMST/nsim**.5, fmt='o',color='royalblue', capsize=0, label=r'MASTER')# + r'$\chi^2/\nu$ = %4.2f' %(chi2_SS/kgb[lbmin:].size) )
ax1.errorbar(lb+5, cl_meanNOMST, yerr=cl_errNOMST/nsim**.5, color='tomato', fmt='d', capsize=0, label=r'$f_{\rm sky}$')# + r'$\chi^2/\nu$ = %4.2f' %(chi2_SS_galmask/kgb[lbmin:].size) )
ax1.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.set_ylabel(r'$ \hat{C}_{\ell}^{TT}$')
ax1.set_xlim([2, lmax+10])
# ax1.set_ylim([0,6.7e-7])
ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))

ax2.axhline(ls='--', color='k')
ax2.errorbar(lb-5, (cl_meanMST-clttb)/clttb, yerr=cl_errMST/nsim**.5/(clttb), color='royalblue', fmt='o', capsize=0)#, label=r'Full-sky')
ax2.errorbar(lb+5, (cl_meanNOMST-clttb)/clttb,   yerr=cl_errNOMST/nsim**.5/(clttb), color='tomato', fmt='d', capsize=0)#,label=r'Gal Mask')
ax2.set_ylabel(r'$\frac{\langle \hat{C}_{\ell}^{TT} \rangle}{C_{\ell}^{TT, th}}-1$')
# ax2.set_ylim([-0.3,0.3])
# ax2.set_ylim([-0.1,0.15])
ax2.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='upper'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

f.subplots_adjust(hspace=0)
# plt.show()
plt.savefig('TT_spectra_test_fsky001_srini.pdf', bbox_inches='tight')

plt.subplot(121); plt.imshow(cl_corrNOMST, interpolation='nearest', vmin=-1, vmax=1, cmap='RdBu'); plt.title(r'$f_{\rm sky}$')
plt.subplot(122); plt.imshow(cl_corrMST, interpolation='nearest', vmin=-1, vmax=1, cmap='RdBu'); plt.title(r'MASTER')
plt.tight_layout()
plt.savefig('TT_spectra_test_fsky001_cov_srini.pdf', bbox_inches='tight')

embed()




