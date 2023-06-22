import numpy as np
import healpy as hp

nsim = 1000
nside = 128
npix= hp.nside2npix(nside)
lmax=256
lmin=2

sims_dir = f'/ehome/bdecaro/xcmbneed/src/sims/Euclid_sims_Marina/NSIDE{nside}/'#f'sims/Needlet/Planck/TGsims_{nside}_planck_2_lmin{lmin}/'
cl_TT_sims = np.zeros((nsim,lmax+1))
cl_TG_sims = np.zeros((nsim,lmax+1))
cl_TG_mask_sims = np.zeros((nsim,lmax+1))
cl_GG_sims = np.zeros((nsim,lmax+1))
mask = hp.read_map('/ehome/bdecaro/xcmbneed/src/mask/EUCLID/mask_planck_comm_2018_x_euclid_fsky0p35_nside=128.fits', verbose=False)
fsky = np.mean(mask)
for n in range(nsim):
    fname_T = sims_dir + f'map_nbin1_NSIDE{nside}_lmax{lmax}_{n+1:05d}_g1noise.fits' # sims_dir + "sim_" + ('%04d' % n) + "_TS_" + ('%04d' % nside) + ".fits"
    fname_gal = sims_dir + f'map_nbin1_NSIDE{nside}_lmax{lmax}_{n+1:05d}_T.fits' #sims_dir + "sim_" + ('%04d' % n) + "_galT_" + ('%04d' % nside) + ".fits"
    mapT = hp.read_map(fname_T, verbose=False)
    mapgal = hp.read_map(fname_gal, verbose=False)
    mapT = hp.remove_dipole(mapT, verbose=False)
    mapgal = hp.remove_dipole(mapgal, verbose=False)
    
    #cl_TT_sims[n, :] =hp.anafast(map1=mapT, map2=mapT, lmax=lmax)
    #cl_TG_sims[n, :] =hp.anafast(map1=mapT, map2=mapgal, lmax=lmax)
    cl_TG_mask_sims[n, :] =hp.anafast(map1=mapT*mask, map2=mapgal*mask, lmax=lmax)
    #cl_GG_sims[n, :] =hp.anafast(map1=mapgal, map2=mapgal, lmax=lmax)

print(cl_TT_sims.shape)

filename_TT= f'EUCLID/cls_TT_anafast_nside{nside}_lmax{lmax}_Euclidnoise_Marina_nsim{nsim}.dat'#lmin{lmin}_marina.dat'
filename_TG = f'EUCLID/cls_Tgalnoise_anafast_nside{nside}_lmax{lmax}_Euclidnoise_Marina_nsim{nsim}.dat'#lmin{lmin}_marina.dat'
filename_TG_mask = f'EUCLID/cls_Tgalnoise_anafast_nside{nside}_lmax{lmax}_Euclidnoise_Marina_nsim{nsim}_fsky{fsky:0.2f}.dat'#lmin{lmin}_marina.dat'
filename_GG = f'EUCLID/cls_galgalnoise_anafast_nside{nside}_lmax{lmax}_Euclidnoise_Marina_nsim{nsim}.dat'#lmin{lmin}_marina.dat'

#np.savetxt(filename_TT, cl_TT_sims)
#np.savetxt(filename_TG, cl_TG_sims)
#np.savetxt(filename_GG, cl_GG_sims)
np.savetxt(filename_TG_mask, cl_TG_mask_sims)

