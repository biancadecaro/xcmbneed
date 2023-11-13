import numpy as np
import healpy as hp

nsim = 1000
nside = 128
npix= hp.nside2npix(nside)
lmax=256
lmin=2

sims_dir = f'/ehome/bdecaro/xcmbneed/src/sims/Needlet/Planck/Mask_noise/TGsims_{nside}_noise_Euclid_tesi/'
cl_TT_mask_sims = np.zeros((nsim,lmax+1))
cl_TT_sims = np.zeros((nsim,lmax+1))
cl_GG_sims = np.zeros((nsim,lmax+1))
cl_TG_sims = np.zeros((nsim,lmax+1))
cl_TG_mask_sims = np.zeros((nsim,lmax+1))
cl_GG_mask_sims = np.zeros((nsim,lmax+1))
#mask = hp.read_map('/ehome/bdecaro/xcmbneed/src/mask/EUCLID/mask_planck_comm_2018_x_euclid_fsky0p35_nside=128.fits', verbose=False)
mask = hp.read_map(f'/ehome/bdecaro/xcmbneed/src/mask/mask_planck_comm_2018_nside={nside}.fits')
mask[np.where(mask>=0.5 )]=1
mask[np.where(mask<0.5 )]=0
fsky = np.mean(mask)
print(fsky)
for n in range(nsim):
    fname_T = sims_dir + "sim_" + ('%04d' % n) + "_TS_" + ('%04d' % nside) + ".fits"
    fname_gal = sims_dir + "sim_" + ('%04d' % n) + "_galT_" + ('%04d' % nside) + ".fits"
    mapT = hp.read_map(fname_T, verbose=False)
    mapgal = hp.read_map(fname_gal, verbose=False)
    mapT_mask = hp.read_map(fname_T, verbose=False)
    mapgal_mask = hp.read_map(fname_gal, verbose=False)
    bad_v = np.where(mask==0)
    mapT_mask[bad_v]=hp.UNSEEN
    mapgal_mask[bad_v]=hp.UNSEEN
    #mapT = hp.remove_dipole(mapT, verbose=False)
    #mapgal = hp.remove_dipole(mapgal, verbose=False)
    #mapT_mask = hp.remove_dipole(mapT_mask, verbose=False)
    #mapgal_mask = hp.remove_dipole(mapgal_mask, verbose=False)
    #mapT_mask[bad_v]=0.0
    #mapgal_mask[bad_v]=0.0
    ##print(bad_v[0].shape, hp.nside2npix(nside)-np.sum(mask))
    #print(bad_v[0].shape, hp.nside2npix(nside)-np.sum(mask))
    
    cl_TT_sims[n, :] =hp.anafast(map1=mapT, map2=mapT, lmax=lmax)
    cl_GG_sims[n, :] =hp.anafast(map1=mapgal, map2=mapgal, lmax=lmax)
    cl_TT_mask_sims[n, :] =hp.anafast(map1=mapT_mask, map2=mapT_mask, lmax=lmax)
    cl_GG_mask_sims[n, :] =hp.anafast(map1=mapgal_mask, map2=mapgal_mask, lmax=lmax)
    cl_TG_sims[n, :] =hp.anafast(map1=mapT, map2=mapgal, lmax=lmax)  # al posto di zero della maschera mettere badvalue
    cl_TG_mask_sims[n, :] =hp.anafast(map1=mapT_mask, map2=mapgal_mask, lmax=lmax)  # al posto di zero della maschera mettere badvalue

    #cl_GG_sims[n, :] =hp.anafast(map1=mapgal, map2=mapgal, lmax=lmax)
    print(f'Num sim={n+1}')

#filename_TT= f'EUCLID/cls_TT_anafast_nside{nside}_lmax{lmax}_Euclidnoise_Marina_nsim{nsim}_nuova_mask.dat'#lmin{lmin}_marina.dat'
#filename_TG = f'EUCLID/cls_Tgalnoise_anafast_nside{nside}_lmax{lmax}_Euclidnoise_Marina_nsim{nsim}_nuova_mask.dat'#lmin{lmin}_marina.dat'
filename_TT = f'PLANCK/cls_TT_anafast_nside{nside}_lmax{lmax}_Euclidnoise_nsim{nsim}.dat'#lmin{lmin}_marina.dat'
filename_GG = f'PLANCK/cls_galnoisegalnoise_anafast_nside{nside}_lmax{lmax}_Euclidnoise_nsim{nsim}.dat'#lmin{lmin}_marina.dat'

filename_TT_mask = f'PLANCK/cls_TT_anafast_nside{nside}_lmax{lmax}_Euclidnoise_nsim{nsim}_fsky{fsky:0.2f}.dat'#lmin{lmin}_marina.dat'
filename_GG_mask = f'PLANCK/cls_galnoisegalnoise_anafast_nside{nside}_lmax{lmax}_Euclidnoise_nsim{nsim}_fsky{fsky:0.2f}.dat'#lmin{lmin}_marina.dat'

filename_TG = f'PLANCK/cls_Tgalnoise_anafast_nside{nside}_lmax{lmax}_Euclidnoise_nsim{nsim}.dat'#lmin{lmin}_marina.dat'
filename_TG_mask = f'PLANCK/cls_Tgalnoise_anafast_nside{nside}_lmax{lmax}_Euclidnoise_nsim{nsim}_fsky{fsky:0.2f}.dat'#lmin{lmin}_marina.dat'
#filename_GG = f'EUCLID/cls_galgalnoise_anafast_nside{nside}_lmax{lmax}_Euclidnoise_Marina_nsim{nsim}_nuova_mask.dat'#lmin{lmin}_marina.dat'
filename_map_GG = f'PLANCK/cls_galgalnoise_map_nside{nside}_lmax{lmax}_Euclidnoise_nsim{nsim}.fits'
filename_map_GG_mask = f'PLANCK/cls_galgalnoise_map_nside{nside}_lmax{lmax}_Euclidnoise_nsim{nsim}_fsky{fsky:0.2f}.fits'
#np.savetxt(filename_TT, cl_TT_sims)
#np.savetxt(filename_TG, cl_TG_sims)
#np.savetxt(filename_GG, cl_GG_sims)
hp.write_map(filename_map_GG,mapgal, overwrite=True)
hp.write_map(filename_map_GG_mask,mapgal_mask, overwrite=True)

np.savetxt(filename_TT, cl_TT_sims)
np.savetxt(filename_GG, cl_GG_sims)
np.savetxt(filename_TT_mask, cl_TT_mask_sims)
np.savetxt(filename_GG_mask, cl_GG_mask_sims)
np.savetxt(filename_TG_mask, cl_TG_mask_sims)
np.savetxt(filename_TG, cl_TG_sims)

