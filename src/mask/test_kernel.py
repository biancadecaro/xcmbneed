import numpy as np 
import healpy as hp
from astropy.io import fits
import matplotlib.pyplot as plt
import seaborn as sns
from mll import mll
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
        #if lmax == None:
        #    lmax = wl.size-1
        #assert(lmax <= wl.size-1)
        return np.float64(mll.get_mll(wl, lmax))

def cov_pseudo_cl(cltg,cltt, clgg,Mll,  Mll_1x2, lmax, noise_gal_l=None):
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
    for l,ell1 in enumerate(ell):
        for ll,ell2 in enumerate(ell):
            covll[l,ll] = (Mll_1x2[l,ll]*(cltg[l]*cltg[ll])+Mll[l,ll]*(np.sqrt(cltt[l]*cltt[ll]*clgg_tot[l]*clgg_tot[ll])))/(2.*ell1+1)
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
########################################################################

nside=128
lmax=256

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

wl_pl_eu = hp.anafast(map1=mask_pl, map2=mask_eu, lmax=2*lmax) # stima dello spettro
wl_comb = hp.anafast(map1=mask_comb, map2=mask_comb, lmax=2*lmax)
Mll_pl_eu  = get_Mll(wl_pl_eu, lmax=lmax)
Mll_comb  = get_Mll(wl_comb, lmax=lmax)

#Mll_pl_eu  = Mll_pl_eu[:lmax+1, :lmax+1]
#Mll_comb  = Mll_comb[:lmax+1, :lmax+1]

with fits.open('EUCLID/kern_RSD2022G_T_G_TT.fits') as hdul:
    Mll_cross_Marina = hdul[0].data
#fig=plt.figure()
#plt.suptitle('Mll cross Marina')
#plt.imshow(Mll_cross_Marina, cmap='crest')
#plt.colorbar()
#
#fig=plt.figure()
#plt.suptitle('Mll cross mio - Mll cross Marina')
#plt.imshow(Mll_pl_eu[2:,2:]/Mll_cross_Marina[2:,2:]-1, cmap='crest')
#plt.colorbar()
##plt.show()


#print(np.mean(Mll_pl_eu[2:,2:]/Mll_cross_Marina[2:,2:]-1))

#quindi quella di Marina è fatta da w^T e w^G e va nella stima e nel davanti al termine con ClTT ClGG nella matrice di covrianza

###########################################################
ell = np.arange(lmax+1)

#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)
#ax.plot(ell[2:], (np.diag(Mll_pl_eu)[2:]/np.diag(Mll_cross_Marina)[2:]-1),'o')
#ax.axhline(ls='--', color='grey')
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#ax.yaxis.set_major_formatter(formatter)
#ax.set_xlabel(r'$j$')
#ax.set_ylabel('% Mll(TT GG) Bianca / Mll(TT GG) Marina - 1')
#fig.tight_layout()
#plt.show()

########################################################################################################

cl_theory = np.loadtxt('../spectra/inifiles/EUCLID_fiducial_lmin0.dat')
ell_theory = cl_theory[0]
cl_theory_tt = cl_theory[1]
cl_theory_tg = cl_theory[2]
cl_theory_gg = cl_theory[3]


Nll = np.ones(cl_theory_gg.shape[0])/354543085.80126834
#Nll_vecchio = np.ones(cl_theory_gg.shape[0])/35454308.580126834


pcl_tg_sim = np.loadtxt('../cls_from_maps/EUCLID/Euclid_Planck_masks/cls_Tgalnoise_anafast_nside128_lmax256_Euclidnoise_Marina_nsim1000_fsky0.36.dat')[:,:lmax+1]
cl_tg_sim_no_mask = np.loadtxt('../cls_from_maps/EUCLID/Euclid_Planck_masks/cls_Tgalnoise_anafast_nside128_lmax256_Euclidnoise_Marina_nsim1000.dat')[:,:lmax+1]

nsim = pcl_tg_sim.shape[0]

pcl_tg_mean=np.mean(pcl_tg_sim, axis=0)
cl_tg_sim = np.array([np.matmul(np.linalg.inv(Mll_pl_eu[2:,2:]), pcl_tg_sim[i,2:]) for i in range(pcl_tg_sim.shape[0]) ])

cl_recovered = np.dot(np.linalg.inv(Mll_pl_eu[2:,2:]), pcl_tg_mean[2:]) 
pcl = np.dot(Mll_pl_eu,cl_theory_tg[:lmax+1]) 

cl_tg_mean = np.mean(cl_tg_sim, axis=0)
cl_tg_mean_no_mask = np.mean(cl_tg_sim_no_mask, axis=0)


cov_pcl_sim = np.cov(pcl_tg_sim.T)
cov_cl_sim = np.cov(cl_tg_sim.T)
cov_cl_sim_no_mask = np.cov(cl_tg_sim_no_mask.T)
print(cov_cl_sim.shape)

cov_pcl= cov_pseudo_cl(cltg=cl_theory_tg,cltt=cl_theory_tt, clgg=cl_theory_gg, Mll=Mll_pl_eu,  Mll_1x2=Mll_comb, lmax=lmax,noise_gal_l=Nll)
cov_cls = cov_cl(cltg=cl_theory_tg,cltt=cl_theory_tt, clgg=cl_theory_gg, lmax=lmax,lmin=2,noise_gal_l=Nll)

#cov_pcl_vecchio= cov_pseudo_cl(cltg=cl_theory_tg,cltt=cl_theory_tt, clgg=cl_theory_gg, Mll=Mll_cross_Marina,  Mll_1x2=Mll_comb, lmax=lmax,noise_gal_l=Nll_vecchio)
#cov_cls_vecchio = cov_cl(cltg=cl_theory_tg,cltt=cl_theory_tt, clgg=cl_theory_gg, lmax=lmax,lmin=2,noise_gal_l=Nll_vecchio)


##################################################################
factor = ell*(ell+1)/(2.*np.pi)

fig = plt.figure()
plt.suptitle('mean of the deconvolved Pseudo Cl from sims')
ax = fig.add_subplot(1, 1, 1)
ax.plot(ell[2:], factor[2:]*cl_theory_tg[2:lmax+1], label='Theory')
ax.plot(ell[2:], factor[2:]*cl_tg_mean,'+' ,label='Mean of the deconvolved cl')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi}C^{\rm TG}_{\ell}$')
plt.legend()
#
#
fig = plt.figure()
plt.suptitle('Cl no mask')
ax = fig.add_subplot(1, 1, 1)
ax.plot(ell[2:], factor[2:]*cl_theory_tg[2:lmax+1], label='Theory')
ax.plot(ell[2:], factor[2:]*cl_tg_mean_no_mask[2:],'+' ,label='Sims')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi}C^{\rm TG}_{\ell}$')
plt.legend()


fig = plt.figure()#figsize=(10,7))
plt.suptitle(' Cov of PCl')
ax = fig.add_subplot(1, 1, 1)

ax.plot(ell[2:], (np.diag(cov_pcl_sim)[2:]/np.diag(cov_pcl)[2:]-1)*100 ,'o')
#ax.plot(ell[2:], (np.diag(cov_cl_sim)/np.diag(cov_cl_dec_vecchio)-1)*100 ,'o', label= 'Nll vecchio')
print( (np.diag(cov_pcl_sim)[2:]/np.diag(cov_pcl)[2:]-1)*100)
ax.axhline(ls='--', color='grey')
#ax.set_ylim([-30,30])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(np.arange(2, lmax+1,10))
ax.set_xticklabels(np.arange(2, lmax+1,10),rotation=40)
ax.set_xlabel(r'$\ell$')
ax.set_ylabel('% diag(sim cov)/diag(analyt cov)-1')
fig.tight_layout()


# secondo me devo mettere i pcl stimati

cov_cl_dec = np.matmul(np.linalg.inv(Mll_pl_eu[2:,2:]), np.matmul(cov_pcl[2:,2:], np.linalg.inv(Mll_pl_eu.T[2:,2:])) )
#cov_cl_dec_vecchio = np.matmul(np.linalg.inv(Mll_cross_Marina[2:,2:]), np.matmul(cov_pcl_vecchio[2:,2:], np.linalg.inv(Mll_cross_Marina[2:,2:])) )


fig = plt.figure()
plt.suptitle('Deconvolved Cov from the sims')
ax = fig.add_subplot(1, 1, 1)

ax.plot(ell[2:], (np.diag(cov_cl_sim)/np.diag(cov_cl_dec)-1)*100 ,'o')
#ax.plot(ell[2:], (np.diag(cov_cl_sim)/np.diag(cov_cl_dec_vecchio)-1)*100 ,'o', label= 'Nll vecchio')

ax.axhline(ls='--', color='grey')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(np.arange(2, lmax+1,10))
ax.set_xticklabels(np.arange(2, lmax+1,10),rotation=40)
ax.set_xlabel(r'$\ell$')
ax.set_ylabel('% diag(sim cov)/diag(analyt cov)-1')
fig.tight_layout()

plt.show()
################################
# ok quindi io qua sto vedendo che nel caso in cui la Mll sia l'identità torno al caso senza maschera, 
# quindi non è un problema di implementazione
Mll_ones = np.eye(N=lmax+1)

cov_pcl_ones = cov_pseudo_cl(cltg=cl_theory_tg,cltt=cl_theory_tt, clgg=cl_theory_gg, Mll=Mll_ones,  Mll_1x2=Mll_ones, lmax=lmax,noise_gal_l=Nll)
cov_cl_dec_ones = np.matmul(np.linalg.inv(Mll_ones[2:,2:]), np.matmul(cov_pcl_ones[2:,2:], np.linalg.inv(Mll_ones.T[2:,2:])) )

fig = plt.figure()

plt.suptitle('Diff Cov no mask')
ax = fig.add_subplot(1, 1, 1)

ax.plot(ell[2:], (np.diag(cov_cl_sim_no_mask)[2:]/np.diag(cov_cls)-1)*100 ,'o', label='No mask')
ax.plot(ell[2:], (np.diag(cov_cl_sim_no_mask)[2:]/np.diag(cov_cl_dec_ones)-1)*100 ,'o', label='Mll ones')

ax.axhline(ls='--', color='grey')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(np.arange(2, lmax+1,10))
ax.set_xticklabels(np.arange(2, lmax+1,10),rotation=40)
ax.set_xlabel(r'$\ell$')
ax.set_ylabel('% diag(sim cov)/diag(analyt cov)-1')
plt.legend()
fig.tight_layout()

####################################

print(np.sqrt(cl_tg_sim.shape[0]))

fig = plt.figure()
plt.suptitle('Diff recovered Cl over sigma')
ax = fig.add_subplot(1, 1, 1)

ax.plot(ell[2:], (cl_recovered-cl_theory_tg[2:lmax+1])/(np.sqrt(np.diag(cov_cl_dec))/(np.sqrt(nsim))) ,'o')
#ax.plot(ell[2:], (np.diag(cov_cl_sim)/np.diag(cov_cl_dec_vecchio)-1)*100 ,'o', label= 'Nll vecchio')

ax.axhline(ls='--', color='grey')
ax.set_ylim([-4,4])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.set_major_formatter(formatter) 
ax.set_xticks(np.arange(2, lmax+1,10))
ax.set_xticklabels(np.arange(2, lmax+1,10),rotation=40)
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\Delta C_{\ell}^{\rm TG}/\sigma$')
fig.tight_layout()


















plt.show()



