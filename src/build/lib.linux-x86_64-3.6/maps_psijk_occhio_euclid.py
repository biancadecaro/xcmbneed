import healpy as hp
import cython_mylibc as mylibc
import analysis, utils, spectra, sims
import matplotlib.pyplot as plt
import numpy as np

nside = 128
npix=hp.nside2npix(nside)

mT=np.zeros((100, npix))
mg=np.zeros((100, npix))
for n in range(100):
    fnameT=f'/ehome/bdecaro/xcmbneed/src/sims/Euclid_sims_Marina/NSIDE128/map_nbin1_NSIDE128_lmax256_{(n+1):05d}_T.fits'
    fnameg=f'/ehome/bdecaro/xcmbneed/src/sims/Euclid_sims_Marina/NSIDE128/map_nbin1_NSIDE128_lmax256_{(n+1):05d}_g1noise.fits'
    mT[n] = hp.read_map(fnameT)
    mg[n] = hp.read_map(fnameg)


lmax = 256
jmax=12
B=mylibc.mylibpy_jmax_lmax2B(jmax, lmax)

betajkT=np.zeros((100, jmax+1, npix))
betajkg=np.zeros((100, jmax+1, npix))
for n in range(100):
    betajkT[n] = mylibc.mylibpy_needlets_f2betajk_healpix_harmonic(mT[n], B, jmax, lmax)
    betajkg[n] = mylibc.mylibpy_needlets_f2betajk_healpix_harmonic(mg[n], B, jmax, lmax)

print((betajkT*betajkg).shape, npix, B)

#np.savetxt(f'betajk1_B={B:0.2f}_TG_euclid.txt',betajkT*betajkg )

