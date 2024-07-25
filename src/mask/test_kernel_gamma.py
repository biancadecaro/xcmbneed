import numpy as np 
import healpy as hp
from astropy.io import fits
import matplotlib.pyplot as plt
import seaborn as sns
from mll import mll
import cython_mylibc as pippo
sns.set_theme(style = 'white')

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)
from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 

#######################################################################
def get_Mll(wl, lmax=None):
        """
        Returns the Coupling Matrix M_ll from l = 0 (Hivon et al. 2002)

        Notes
        -----
        M_ll.shape = (lmax+1, lmax+1)
        """
        #sif lmax == None:
        #s    lmax = wl.size-1
        #sassert(lmax <= wl.size-1)
        return np.float64(mll.get_mll(wl[:lmax+1], lmax))

def gammaJ(b_values, cl, Mll_1x2,lmax):
        """
        Returns the \Gamma_j vector from Domenico's notes

        Notes
        -----
        gamma_lj.shape = (lmax+1, jmax+1)
        """
        #Mll1  = self.get_Mll(wl, lmax=lmax)
        ell  = np.arange(0, lmax+1, dtype=np.int)
        bjl  = b_values**2
        return (bjl*(2*ell+1.)*np.dot(Mll_1x2, cl[:lmax+1])).sum(axis=1)/(4*np.pi)

def variance_gammaj(b_values,cltg,cltt, clgg, Mll, Mll_1x2,noise_gal_l=None):
        """
        Returns the Cov(\Gamma_j, \Gamma_j') 
        Notes
        -----
        Cov(gamma)_jj'.shape = (jmax+1, jmax+1)
        """
        if noise_gal_l is not None:
            clgg_tot = clgg+noise_gal_l
        else:
            clgg_tot = clgg

        ell  = np.arange(Mll.shape[0], dtype=int)
        bjl  = b_values**2*(2*ell+1.)

        covll = np.zeros((ell.shape[0],ell.shape[0]))
        for ell1 in range(ell.shape[0]):
            for ell2 in range(ell.shape[0]):
                covll[ell1,ell2] = (Mll_1x2[ell1,ell2]*(cltg[ell1]*cltg[ell2])+Mll[ell1,ell2]*(np.sqrt(cltt[ell1]*cltt[ell2]*clgg_tot[ell1]*clgg_tot[ell2])))/(2.*ell1+1)
        delta_gammaj = np.dot(bjl, np.dot(covll, bjl.T))
        return delta_gammaj/(4*np.pi)**2

def variance_gammaj_new(b_values,cltg,cltt, clgg, Mll, Mll_1x2, noise_gal_l=None):
        """
        Returns the Cov(\Gamma_j, \Gamma_j') 
        Notes
        -----
        Cov(gamma)_jj'.shape = (jmax+1, jmax+1)
        """
        if noise_gal_l is not None:
            clgg_tot = clgg+noise_gal_l
        else:
            clgg_tot = clgg
        lmax = 256
        ell  = np.arange(2, lmax+1, dtype=int)
        bjl  = b_values**2*(2*ell+1.)

        covll = np.zeros((ell.shape[0],ell.shape[0]))
        for ell1 in range(ell.shape[0]):
            for ell2 in range(ell.shape[0]):
                print()
                covll[ell1,ell2] = (Mll_1x2[ell1,ell2]*(cltg[ell1]*cltg[ell2])+Mll[ell1,ell2]*(np.sqrt(cltt[ell1]*cltt[ell2]*clgg_tot[ell1]*clgg_tot[ell2])))/(2.*ell1+1)
        delta_gammaj = np.dot(bjl, np.dot(covll, bjl.T))
        return delta_gammaj/(4*np.pi)**2

def ell_binning(b_values, lmax):
    """
    Returns the binning scheme in  multipole space
    """
    #assert(np.floor(self.B**(jmax+1)) <= ell.size-1) 
    
    bjl  = b_values**2
    ell  = np.arange(lmax+1)*np.ones((bjl.shape[0], lmax+1))
    ellj =np.zeros((bjl.shape[0], lmax+1))
    for j in range(bjl.shape[0]):
        bjl[j,:][bjl[j,:]!=0] = 1.
        ellj[j,:] = ell[j,:]*bjl[j,:]
    return ellj 
########################################################################

nside=128
lmax=256 
jmax=12
jvec = np.arange(jmax+1)
B = pippo.mylibpy_jmax_lmax2B(jmax, lmax)
b_values = pippo.mylibpy_needlets_std_init_b_values(B, jmax, lmax)

fig, ax1  = plt.subplots(1,1) 
#plt.suptitle(r'$D = %1.2f $' %B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax))

for i in range(1,jmax):
    ax1.plot(b_values[i]*b_values[i], label = 'j='+str(i) )
ax1.set_xscale('log')
ax1.set_xlim([0.40, 350 ])
ax1.set_xlabel(r'$\ell$')
ax1.set_ylabel(r'$w^{2}(\frac{\ell}{D^{j}})$')
ax1.legend(loc='upper left', fontsize=11)
plt.tight_layout()
plt.show()

ell_bin=ell_binning(b_values, lmax)
fig = plt.figure()
plt.suptitle(r'$D = %1.2f $' %B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax))
ax = fig.add_subplot(1, 1, 1)
for i in range(0,jmax+1):
    ell_range = ell_bin[i][ell_bin[i]!=0]
    plt.plot(ell_range, i*ell_range/ell_range, label= f'j={i}')
    plt.text(ell_range[0], i, r'$\ell_{min}=%d,\,\ell_{max}=%d$'%(ell_range[0],ell_range[-1]))

ax.set_xlabel(r'$\ell$')
ax.legend(loc='right', ncol=2)
plt.tight_layout()

##########################################################################################################

mask_eu = hp.read_map(f'EUCLID/mask_rsd2022g-wide-footprint-year-6-equ-order-13-moc_ns0{nside}_G_filled_2deg2.fits')
mask_pl = hp.read_map(f'mask_temp_ns{nside}.fits')
fsky_eu = np.mean(mask_eu)
fsky_pl = np.mean(mask_pl)

mask_comb = mask_pl*mask_eu
fsky_comb = np.mean(mask_comb)

#fig = plt.figure()
#fig.add_subplot(311) 
#hp.mollview(mask_eu, cmap = 'crest', title=f'Euclid mask, fsky={fsky_eu:0.3f}', hold=True)
#fig.add_subplot(312) 
#hp.mollview(mask_pl, cmap = 'crest', title=f'Planck mask, fsky={fsky_pl:0.3f}', hold=True)
#fig.add_subplot(313) 
#hp.mollview(mask_comb, cmap = 'crest', title=f'Combined mask, fsky:{fsky_comb:0.3f}', hold=True)
#plt.show()

print(f'fsky_pl = {fsky_pl:0.3f}, fsky_eu={fsky_eu:0.3f}, fsky={fsky_comb:0.3f}')

wl_pl_eu_spe = hp.anafast(map1=mask_pl, map2=mask_eu, lmax=2*lmax) # stima dello spettro
wl_comb = hp.anafast(map1=mask_comb, map2=mask_comb, lmax=2*lmax)
wl_plpl_eueu = hp.anafast(map1=mask_pl*mask_pl, map2=mask_eu*mask_eu, lmax=2*lmax) # stima dello spettro
wl_comb = hp.anafast(map1=mask_comb, map2=mask_comb, lmax=2*lmax)
Mll_pl_eu_spe  = get_Mll(wl_pl_eu_spe, lmax=2*lmax)
Mll_plpl_eueu  = get_Mll(wl_plpl_eueu, lmax=2*lmax)
Mll_comb  = get_Mll(wl_comb, lmax=2*lmax)

Mll_pl_eu_spe= Mll_pl_eu_spe[:lmax+1, :lmax+1]
Mll_plpl_eueu = Mll_plpl_eueu[:lmax+1, :lmax+1]
Mll_comb = Mll_comb[:lmax+1, :lmax+1]

with fits.open('EUCLID/kern_RSD2022G_T_G_TT.fits') as hdul:
    Mll_cross_Marina = hdul[0].data

########################################################################################################

cl_theory = np.loadtxt('../spectra/inifiles/EUCLID_fiducial_lmin0.dat')
ell_theory = cl_theory[0]
cl_theory_tt = cl_theory[1]
cl_theory_tg = cl_theory[2]
cl_theory_gg = cl_theory[3]


Nll = np.ones(cl_theory_gg.shape[0])/354543085.80126834

gammaj_sims = np.loadtxt('../output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000_2masks/gamma_sims_TS_galT_jmax12_B_1.59_nside128_fsky0.36.dat')
#gammaj_sims_no_mask = np.loadtxt('../output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000_2masks_remove_dipole/gamma_sims_TS_galT_jmax12_B_1.59_nside128.dat')

nsim = gammaj_sims.shape[0]

gammaj_mean = np.mean(gammaj_sims, axis=0)
#gammaj_mean_no_mask = np.mean(gammaj_sims_no_mask, axis=0)

cov_gammaj_sims = np.cov(gammaj_sims.T)
#cov_gammaj_sims_no_mask = np.cov(gammaj_sims_no_mask.T)

gammaj = gammaJ(b_values, cl_theory_tg, Mll_pl_eu_spe, lmax)
gammaj_vecchio = gammaJ(b_values, cl_theory_tg, Mll_cross_Marina, lmax)

cov_gammaj = variance_gammaj(b_values=b_values,cltg=cl_theory_tg,cltt=cl_theory_tt, clgg=cl_theory_gg, Mll=Mll_plpl_eueu, Mll_1x2=Mll_comb, noise_gal_l=Nll)
cov_gammaj_vecchia = variance_gammaj(b_values=b_values,cltg=cl_theory_tg,cltt=cl_theory_tt, clgg=cl_theory_gg, Mll=Mll_plpl_eueu, Mll_1x2=Mll_cross_Marina,noise_gal_l=Nll)

#####################################################################
############ Estimation of the spectrum ###############################

fig = plt.figure()
plt.suptitle('mean of the Gammaj from sims')
ax = fig.add_subplot(1, 1, 1)
ax.plot(jvec[1:jmax], gammaj[1:jmax], label='Theory, Mll PlxEu')
ax.plot(jvec[1:jmax], gammaj_vecchio[1:jmax],'x', label='Theory, Mll Marina')
ax.plot(jvec[1:jmax], gammaj_mean[1:jmax],'+' ,label='Mean of the sims, Mll plxeu')
ax.set_xlabel('j')
ax.set_ylabel(r'$\Gamma_{j}$')
plt.legend()

###################################################################
#################### COVARIANCE#####################################

fig = plt.figure()
plt.suptitle(' Cov from simulations')
ax = fig.add_subplot(1, 1, 1)

ax.plot(jvec[1:jmax], (np.diag(cov_gammaj_sims)[1:jmax]/np.diag(cov_gammaj)[1:jmax]-1)*100 ,'o', label='Mll comb')
ax.plot(jvec[1:jmax], (np.diag(cov_gammaj_sims)[1:jmax]/np.diag(cov_gammaj_vecchia)[1:jmax]-1)*100 ,'o', label='Mll Marina')

ax.axhline(ls='--', color='grey')
ax.set_ylim([-30,30])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(jvec)
ax.set_xticklabels(jvec)
plt.legend()
ax.set_xlabel('j')
ax.set_ylabel('% diag(sim cov)/diag(analyt cov)-1')
fig.tight_layout()
##########################################################
fig = plt.figure()
plt.suptitle('% Relative diff covariance')
ax = fig.add_subplot(1, 1, 1)

ax.plot(jvec[1:jmax], (np.diag(cov_gammaj_sims)[1:jmax]/np.diag(cov_gammaj)[1:jmax]-1)*100 ,'o')

ax.axhline(ls='--', color='grey')
ax.set_ylim([-30,30])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(jvec)
ax.set_xticklabels(jvec)
ax.set_xlabel('j')
ax.set_ylabel('% diag(sim cov)/diag(analyt cov)-1')
fig.tight_layout()



################################
# ok quindi io qua sto vedendo che nel caso in cui la Mll sia l'identità torno al caso senza maschera, 
# quindi non è un problema di implementazione
#Mll_ones = np.eye(N=lmax+1)
#cov_gammaj_ones = variance_gammaj(b_values,cl_theory_tg,cl_theory_tt, cl_theory_gg, Mll_ones, Mll_ones,noise_gal_l=Nll)
#
#fig = plt.figure()
#plt.suptitle(' Cov from simulations no mask (Mll ones)')
#ax = fig.add_subplot(1, 1, 1)
#
#ax.plot(jvec[1:jmax], (np.diag(cov_gammaj_sims_no_mask)[1:jmax]/np.diag(cov_gammaj_ones)[1:jmax]-1)*100 ,'o')
#
#ax.axhline(ls='--', color='grey')
#ax.set_ylim([-30,30])
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#ax.yaxis.set_major_formatter(formatter) 
#ax.set_xticks(jvec)
#ax.set_xticklabels(jvec)
#ax.set_xlabel(r'j')
#ax.set_ylabel('% diag(sim cov)/diag(analyt cov)-1')
#fig.tight_layout()



#############################################################################################
fig = plt.figure()
plt.suptitle('Diff GammaJ over sigma')
ax = fig.add_subplot(1, 1, 1)

ax.plot(jvec[1:jmax], (gammaj_mean[1:jmax]-gammaj[1:jmax])/(np.sqrt(np.diag(cov_gammaj_sims)[1:jmax])/(np.sqrt(nsim))) ,'o')
#ax.plot(ell[2:], (np.diag(cov_cl_sim)/np.diag(cov_cl_dec_vecchio)-1)*100 ,'o', label= 'Nll vecchio')

ax.axhline(ls='--', color='grey')
#ax.set_ylim([-4,4])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\Delta \Gamma_{j}/\sigma$')
fig.tight_layout()


########################################################################
##################### ULTIMO TEST PER LA Mll ###########################

Mll_pl_eu_nuova =  Mll_plpl_eueu[2:,2:]
Mll_comb_nuova =  Mll_comb[2:,2:]
b_values_new=b_values[1:,2:]

cov_gammaj_nuova = variance_gammaj_new(b_values=b_values_new,cltg=cl_theory_tg[2:],cltt=cl_theory_tt[2:], clgg=cl_theory_gg[2:], Mll=Mll_pl_eu_nuova, Mll_1x2=Mll_comb_nuova,noise_gal_l=Nll[2:])


cov_gammaj_sims = cov_gammaj_sims[1:,1:]
print(cov_gammaj_nuova.shape, cov_gammaj_sims.shape)

fig = plt.figure()
plt.suptitle(' Cov from simulations NUOVA')
ax = fig.add_subplot(1, 1, 1)

ax.plot(jvec[1:], np.diag(cov_gammaj_sims) ,'o', label='Sims')
ax.plot(jvec[1:], np.diag(cov_gammaj_nuova) ,'o', label='Mll from lmin=2')
ax.plot(jvec[1:], np.diag(cov_gammaj)[1:] ,'o', label='Mll from lmin=0')

#ax.axhline(ls='--', color='grey')
#ax.set_ylim([-30,30])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(jvec)
ax.set_xticklabels(jvec)
plt.legend()
ax.set_xlabel('j')
ax.set_ylabel('% diag(sim cov)/diag(analyt cov)-1')
fig.tight_layout()




plt.show()