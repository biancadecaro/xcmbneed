import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import master

from matplotlib import rc, rcParams
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter

arcmin2rad = 0.000290888208666

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

# Pars
nside = 512
beam  = 10.   # arcmin
lmin  = 2
lmax  = 1000
d_ell = 50
flat  = lambda l: l*(l+1)/2./np.pi

Bin = master.Binner(lmin=lmin, lmax=lmax, delta_ell=d_ell, flat=flat)
lb  = Bin.lb 

# Loading CMB spectra
l, cltt = np.loadtxt('CMB_spectra.dat', unpack=True)
cltt   *= (2*np.pi)/(l*(l+1))
cltt    = np.nan_to_num(cltt)
clttb   = Bin.bin_spectra(cltt)

# Creating maps
TT_nopix_nobeam = hp.synfast(cltt, nside, pixwin=False)
TT_pix_nobeam   = hp.synfast(cltt, nside, pixwin=True)
TT_nopix_beam   = hp.synfast(cltt, nside, pixwin=False, fwhm=beam*arcmin2rad)
TT_pix_beam     = hp.synfast(cltt, nside, pixwin=True, fwhm=beam*arcmin2rad)

# Mask
mask_20 = GetGalMask(nside, fsky=0.2)
mask_01 = GetGalMask(nside, fsky=0.01)

# Initialize Master classes
M_nopix_nobeam_20 = master.Master(mask_20, lmin=lmin, lmax=lmax, delta_ell=d_ell, flat=flat, pixwin=False, fwhm_smooth=None, MASTER=True)
M_pix_nobeam_20   = master.Master(mask_20, lmin=lmin, lmax=lmax, delta_ell=d_ell, flat=flat, pixwin=True,  fwhm_smooth=None, MASTER=True)
M_nopix_beam_20   = master.Master(mask_20, lmin=lmin, lmax=lmax, delta_ell=d_ell, flat=flat, pixwin=False, fwhm_smooth=beam, MASTER=True)
M_pix_beam_20     = master.Master(mask_20, lmin=lmin, lmax=lmax, delta_ell=d_ell, flat=flat, pixwin=True,  fwhm_smooth=beam, MASTER=True)

M_nopix_nobeam_01 = master.Master(mask_01, lmin=lmin, lmax=lmax, delta_ell=d_ell, flat=flat, pixwin=False, fwhm_smooth=None, MASTER=True)
M_pix_nobeam_01   = master.Master(mask_01, lmin=lmin, lmax=lmax, delta_ell=d_ell, flat=flat, pixwin=True,  fwhm_smooth=None, MASTER=True)
M_nopix_beam_01   = master.Master(mask_01, lmin=lmin, lmax=lmax, delta_ell=d_ell, flat=flat, pixwin=False, fwhm_smooth=beam, MASTER=True)
M_pix_beam_01     = master.Master(mask_01, lmin=lmin, lmax=lmax, delta_ell=d_ell, flat=flat, pixwin=True,  fwhm_smooth=beam, MASTER=True)

# Extract spectra
cltt_nopix_nobeam20, err_cltt_nopix_nobeam20 = M_nopix_nobeam_20.get_spectra(TT_nopix_nobeam, analytic_errors=True)
cltt_pix_nobeam20,   err_cltt_pix_nobeam20   = M_pix_nobeam_20.get_spectra(TT_pix_nobeam, analytic_errors=True)
cltt_nopix_beam20,   err_cltt_nopix_beam20   = M_nopix_beam_20.get_spectra(TT_nopix_beam, analytic_errors=True)
cltt_pix_beam20,     err_cltt_pix_beam20     = M_pix_beam_20.get_spectra(TT_pix_beam, analytic_errors=True)

cltt_nopix_nobeam01, err_cltt_nopix_nobeam01 = M_nopix_nobeam_01.get_spectra(TT_nopix_nobeam, analytic_errors=True)
cltt_pix_nobeam01,   err_cltt_pix_nobeam01   = M_pix_nobeam_01.get_spectra(TT_pix_nobeam, analytic_errors=True)
cltt_nopix_beam01,   err_cltt_nopix_beam01   = M_nopix_beam_01.get_spectra(TT_nopix_beam, analytic_errors=True)
cltt_pix_beam01,     err_cltt_pix_beam01     = M_pix_beam_01.get_spectra(TT_pix_beam, analytic_errors=True)

# plots
f, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(10,8))

plt.suptitle(r'$  N_{\rm side} = %d$ ' %nside + r'$\Delta\ell = %d$ ' %d_ell +' - Signal Only - ' + r'$f_{\rm sky} = 0.2$', size=25)

ax1.plot(cltt*l*(l+1)/2/np.pi, color='royalblue', label='Theory')
ax1.errorbar(lb-7.5, cltt_nopix_nobeam20, yerr=err_cltt_nopix_nobeam20, fmt='o',color='seagreen', capsize=0, label=r'No pix, No beam')# + r'$\chi^2/\nu$ = %4.2f' %(chi2_SS/kgb[lbmin:].size) )
ax1.errorbar(lb-2.5, cltt_pix_nobeam20,   yerr=err_cltt_pix_nobeam20, color='orange', fmt='x', capsize=0, label=r'Pix, No beam ')# + r'$\chi^2/\nu$ = %4.2f' %(chi2_SS_galmask/kgb[lbmin:].size) )
ax1.errorbar(lb+2.5, cltt_nopix_beam20,   yerr=err_cltt_nopix_beam20, color='purple', fmt='d', capsize=0, label=r'No pix, beam = 10 arcmin ')# + r'$\chi^2/\nu$ = %4.2f' %(chi2_SS_hermask/kgb[lbmin:].size))
ax1.errorbar(lb+7.5, cltt_pix_beam20,     yerr=err_cltt_pix_beam20, color='tomato', fmt='^', capsize=0, label=r'Pix, beam = 10 arcmin ')# + r'$\chi^2/\nu$ = %4.2f' %(chi2_SS_hermask_MST/kgb[lbmin:].size))
ax1.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.set_ylabel(r'$ \hat{C}_{\ell}^{TT}$')
ax1.set_xlim([2, lmax+10])
# ax1.set_ylim([0,6.7e-7])
ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))

ax2.axhline(ls='--', color='k')
ax2.errorbar(lb-7.5, (cltt_nopix_nobeam20-clttb)/clttb, yerr=err_cltt_nopix_nobeam20/(clttb), color='seagreen', fmt='o', capsize=0)#, label=r'Full-sky')
ax2.errorbar(lb-2.5, (cltt_pix_nobeam20-clttb)/clttb,   yerr=err_cltt_pix_nobeam20/(clttb), color='orange', fmt='x', capsize=0)#,label=r'Gal Mask')
ax2.errorbar(lb+2.5, (cltt_nopix_beam20-clttb)/clttb,   yerr=err_cltt_nopix_beam20/(clttb), color='purple', fmt='d', capsize=0)#,label=r'H-ATLAS Mask')
ax2.errorbar(lb+7.5, (cltt_pix_beam20-clttb)/clttb,     yerr=err_cltt_pix_beam20/(clttb), color='tomato', fmt='^', capsize=0)#,label=r'H-ATLAS Mask MASTER')
ax2.set_xlabel(r'$\ell$')
ax2.set_ylabel(r'$\frac{\langle \hat{C}_{\ell}^{TT} \rangle}{C_{\ell}^{TT, th}}-1$')
# ax2.set_ylim([-0.3,0.3])
# ax2.set_ylim([-0.1,0.15])
ax2.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='upper'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

f.subplots_adjust(hspace=0)
# plt.show()
plt.savefig('TT_spectra_test_fsky02.pdf', bbox_inches='tight')


f, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(10,8))

plt.suptitle(r'$  N_{\rm side} = %d$ ' %nside + r'$\Delta\ell = %d$ ' %d_ell +' - Signal Only - ' + r'$f_{\rm sky} = 0.01$', size=25)

ax1.plot(cltt*l*(l+1)/2/np.pi, color='royalblue', label='Theory')
ax1.errorbar(lb-7.5, cltt_nopix_nobeam01, yerr=err_cltt_nopix_nobeam01, fmt='o',color='seagreen', capsize=0, label=r'No pix, No beam')# + r'$\chi^2/\nu$ = %4.2f' %(chi2_SS/kgb[lbmin:].size) )
ax1.errorbar(lb-2.5, cltt_pix_nobeam01,   yerr=err_cltt_pix_nobeam01, color='orange', fmt='x', capsize=0, label=r'Pix, No beam ')# + r'$\chi^2/\nu$ = %4.2f' %(chi2_SS_galmask/kgb[lbmin:].size) )
ax1.errorbar(lb+2.5, cltt_nopix_beam01,   yerr=err_cltt_nopix_beam01, color='purple', fmt='d', capsize=0, label=r'No pix, beam = 10 arcmin ')# + r'$\chi^2/\nu$ = %4.2f' %(chi2_SS_hermask/kgb[lbmin:].size))
ax1.errorbar(lb+7.5, cltt_pix_beam01,     yerr=err_cltt_pix_beam01, color='tomato', fmt='^', capsize=0, label=r'Pix, beam = 10 arcmin ')# + r'$\chi^2/\nu$ = %4.2f' %(chi2_SS_hermask_MST/kgb[lbmin:].size))
ax1.legend(loc='best')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.set_ylabel(r'$ \hat{C}_{\ell}^{TT}$')
ax1.set_xlim([2, lmax+10])
# ax1.set_ylim([0,6.7e-7])
ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))

ax2.axhline(ls='--', color='k')
ax2.errorbar(lb-7.5, (cltt_nopix_nobeam01-clttb)/clttb, yerr=err_cltt_nopix_nobeam01/(clttb), color='seagreen', fmt='o', capsize=0)#, label=r'Full-sky')
ax2.errorbar(lb-2.5, (cltt_pix_nobeam01-clttb)/clttb,   yerr=err_cltt_pix_nobeam01/(clttb), color='orange', fmt='x', capsize=0)#,label=r'Gal Mask')
ax2.errorbar(lb+2.5, (cltt_nopix_beam01-clttb)/clttb,   yerr=err_cltt_nopix_beam01/(clttb), color='purple', fmt='d', capsize=0)#,label=r'H-ATLAS Mask')
ax2.errorbar(lb+7.5, (cltt_pix_beam01-clttb)/clttb,     yerr=err_cltt_pix_beam01/(clttb), color='tomato', fmt='^', capsize=0)#,label=r'H-ATLAS Mask MASTER')
ax2.set_xlabel(r'$\ell$')
ax2.set_ylabel(r'$\frac{\langle \hat{C}_{\ell}^{TT} \rangle}{C_{\ell}^{TT, th}}-1$')
# ax2.set_ylim([-0.3,0.3])
# ax2.set_ylim([-0.1,0.15])
ax2.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='upper'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

f.subplots_adjust(hspace=0)
# plt.show()
plt.savefig('TT_spectra_test_fsky001.pdf', bbox_inches='tight')


