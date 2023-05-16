import numpy as np
import healpy as hp

nsim = 500
nside = 512
npix= hp.nside2npix(nside)
lmax=800
lmin=0

sims_dir = f'sims/Needlet/Planck/TGsims_{nside}_planck_2_lmin{lmin}/'
cl_TT_sims = np.zeros((nsim,lmax+1))
cl_TG_sims = np.zeros((nsim,lmax+1))
cl_GG_sims = np.zeros((nsim,lmax+1))

for n in range(nsim):
    fname_T = sims_dir + "/sim_" + ('%04d' % n) + "_TS_" + ('%04d' % nside) + ".fits"
    fname_gal = sims_dir + "/sim_" + ('%04d' % n) + "_galS_" + ('%04d' % nside) + ".fits"
    mapT = hp.read_map(fname_T, verbose=False)
    mapgal = hp.read_map(fname_gal, verbose=False)
    mapT = hp.remove_dipole(mapT, verbose=False)
    mapgal = hp.remove_dipole(mapgal, verbose=False)
    
    cl_TT_sims[n, :] =hp.anafast(map1=mapT, map2=mapT, lmax=lmax)
    cl_TG_sims[n, :] =hp.anafast(map1=mapT, map2=mapgal, lmax=lmax)
    cl_GG_sims[n, :] =hp.anafast(map1=mapgal, map2=mapgal, lmax=lmax)

filename_TT= f'cls_TT_anafast_nside{nside}_lmax{lmax}_lmin{lmin}.dat'
filename_TG = f'cls_Tgal_anafast_nside{nside}_lmax{lmax}_lmin{lmin}.dat'
filename_GG = f'cls_galgal_anafast_nside{nside}_lmax{lmax}_lmin{lmin}.dat'

np.savetxt(filename_TT, cl_TT_sims)
np.savetxt(filename_TG, cl_TG_sims)
np.savetxt(filename_GG, cl_GG_sims)


