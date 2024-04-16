import numpy as np
import matplotlib.pyplot as plt
from mll import mll
import healpy as hp
import seaborn as sns

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

lmax=256
nside=128
mask = hp.read_map(f'mask/EUCLID/mask_planck_comm_2018_x_euclid_fsky0p35_nside={nside}.fits')

wl = hp.anafast(mask, lmax=lmax)
Mll  = get_Mll(wl, lmax=lmax)

print(Mll.shape)

fig, ax= plt.subplots(1,1)
cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
ax.imshow(Mll, cmap=cmap)
ax.set_xticks(np.arange(2, lmax+1,10))
ax.set_xticklabels(np.arange(2, lmax+1,10),rotation=40)
ax.set_yticks(np.arange(2, lmax+1,10))
ax.set_yticklabels(np.arange(2, lmax+1,10),rotation=40)
plt.tight_layout()
plt.show()