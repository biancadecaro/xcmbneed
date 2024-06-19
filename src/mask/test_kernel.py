import numpy as np 
import healpy as hp
from astropy.io import fits
import matplotlib.pyplot as plt
import seaborn as sns
from mll import mll
sns.set()
sns.set(style = 'white')

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
        if lmax == None:
            lmax = wl.size-1
        assert(lmax <= wl.size-1)
        return np.float64(mll.get_mll(wl[:lmax+1], lmax))
########################################################################

nside=128
lmax=256 

mask_eu = hp.read_map(f'EUCLID/mask_rsd2022g-wide-footprint-year-6-equ-order-13-moc_ns0{nside}_G_filled_2deg2.fits')
mask_pl = hp.read_map(f'mask_planck_comm_2018_nside={nside}.fits')
mask_pl[np.where(mask_pl>=0.5 )]=1
mask_pl[np.where(mask_pl<0.5 )]=0
fsky_eu = np.mean(mask_eu)
fsky_pl = np.mean(mask_pl)

mask_comb = mask_pl*mask_eu
mask_comb[np.where(mask_comb>=0.5 )]=1
mask_comb[np.where(mask_comb<0.5 )]=0
fsky_comb = np.mean(mask_comb)

fig = plt.figure()
fig.add_subplot(311) 
hp.mollview(mask_eu, cmap = 'crest', title=f'Euclid mask, fsky={fsky_eu:0.3f}', hold=True)
fig.add_subplot(312) 
hp.mollview(mask_pl, cmap = 'crest', title=f'Planck mask, fsky={fsky_pl:0.3f}', hold=True)
fig.add_subplot(313) 
hp.mollview(mask_comb, cmap = 'crest', title=f'Combined mask, fsky:{fsky_comb:0.3f}', hold=True)
plt.show()

print(f'fsky_pl = {fsky_pl:0.3f}, fsky_eu={fsky_eu:0.3f}, fsky={fsky_comb:0.3f}')

wl_pl_eu = hp.anafast(map1=mask_pl, map2=mask_eu, lmax=lmax)
wl_comb = hp.anafast(map1=mask_comb, map2=mask_comb, lmax=lmax)
Mll_pl_eu  = get_Mll(wl_pl_eu, lmax=lmax)
Mll_comb  = get_Mll(wl_comb, lmax=lmax)

with fits.open('EUCLID/kern_RSD2022G_T_G_TT.fits') as hdul:
    Mll_cross_Marina = hdul[0].data
fig=plt.figure()
plt.suptitle('Mll cross Marina')
plt.imshow(Mll_cross_Marina, cmap='crest')
plt.colorbar()

fig=plt.figure()
plt.suptitle('Mll cross')
plt.imshow(Mll_pl_eu-Mll_cross_Marina, cmap='crest')
plt.colorbar()
plt.show()
