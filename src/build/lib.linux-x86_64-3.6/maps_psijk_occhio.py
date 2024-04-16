import healpy as hp
import cython_mylibc as mylibc
import analysis, utils, spectra, sims
import matplotlib.pyplot as plt
import numpy as np

nside = 512
npix=hp.nside2npix(nside)

m=np.zeros(npix)

pix = hp.ang2pix(nside=nside,theta=np.pi / 2, phi=0)

m[pix]=1

hp.write_map('map_0_pix_1.fits',m,nest=False, overwrite=True)
m1 = hp.read_map('map_0_pix_1.fits')


lmax = 782
B=1.95#mylibc.mylibpy_jmax_lmax2B(jmax, lmax)
jmax=round(np.log(lmax*1.00000000000000222045e+00)/np.log(B))



betajk1 = mylibc.mylibpy_needlets_f2betajk_healpix_harmonic(m1, B, jmax, lmax)
np.savetxt('betajk1_B=1.95.txt',betajk1 )

print(betajk1.shape, npix, B)