import numpy as np
import healpy as hp

nsim = 1000
nside = 128
npix= hp.nside2npix(nside)
lmax=256

sims_dir = f'/ehome/bdecaro/xcmbneed/src/sims/Euclid_sims_Bianca/TG_{nside}_noiseEuclid_nsim{nsim}/'
cl_TT_sims = np.zeros((nsim,lmax+1))
cl_TG_sims = np.zeros((nsim,lmax+1))
cl_TG_mask_sims = np.zeros((nsim,lmax+1))
cl_TG_noise_sims = np.zeros((nsim,lmax+1))
cl_TG_noise_mask_sims = np.zeros((nsim,lmax+1))
cl_GG_sims = np.zeros((nsim,lmax+1))

mask = hp.read_map(f'/ehome/bdecaro/xcmbneed/src/mask/EUCLID/mask_planck_comm_2018_x_euclid_fsky0p35_nside={nside}.fits')
fsky = np.mean(mask)
for n in range(nsim):
    fname_T = sims_dir + f'sim_{n:04d}_TS_{nside:04d}.fits'
    fname_gal = sims_dir + f'sim_{n:04d}_galS_{nside:04d}.fits'  
    fname_galT = sims_dir + f'sim_{n:04d}_galT_{nside:04d}.fits'  
    mapT = hp.read_map(fname_T, verbose=False)
    mapgal = hp.read_map(fname_gal, verbose=False)
    mapgalT = hp.read_map(fname_galT, verbose=False)
    mapT_mask= hp.read_map(fname_T, verbose=False)
    mapgal_mask = hp.read_map(fname_gal, verbose=False)
    mapgalT_mask = hp.read_map(fname_galT, verbose=False)
    bad_v = np.where(mask==0)
    mapT_mask[bad_v]=hp.UNSEEN
    mapgal_mask[bad_v]=hp.UNSEEN
    mapgalT_mask[bad_v]=hp.UNSEEN
    mapT = hp.remove_dipole(mapT, verbose=False)
    mapgal = hp.remove_dipole(mapgal, verbose=False)
    mapT_mask = hp.remove_dipole(mapT_mask, verbose=False)
    mapgal_mask = hp.remove_dipole(mapgal_mask, verbose=False)
    mapT_mask[bad_v]=0.0
    mapgal_mask[bad_v]=0.0
    #cl_TT_sims[n, :] =hp.anafast(map1=mapT, map2=mapT, lmax=lmax)
    cl_TG_sims[n, :] =hp.anafast(map1=mapT, map2=mapgal, lmax=lmax)
    cl_TG_mask_sims[n, :] =hp.anafast(map1=mapT_mask, map2=mapgal_mask, lmax=lmax)  

    cl_TG_noise_sims[n, :] =hp.anafast(map1=mapT, map2=mapgalT, lmax=lmax)
    cl_TG_noise_mask_sims[n, :] =hp.anafast(map1=mapT_mask, map2=mapgalT_mask, lmax=lmax)  

    #cl_GG_sims[n, :] =hp.anafast(map1=mapgal, map2=mapgal, lmax=lmax)
    print(f'Num sim={n+1}')


filename_TG_mask_noise = f'mie_sim_Euclid/cls_Tgalnoise_anafast_nside{nside}_lmax{lmax}_mie_sim_Euclid_nsim{nsim}_fsky{fsky:0.2f}_noise_remove_dipole.dat'
filename_TG_noise = f'mie_sim_Euclid/cls_Tgalnoise_anafast_nside{nside}_lmax{lmax}_mie_sim_Euclid_nsim{nsim}_noise_remove_dipole.dat'

filename_TG_mask = f'mie_sim_Euclid/cls_Tgalnoise_anafast_nside{nside}_lmax{lmax}_mie_sim_Euclid_nsim{nsim}_fsky{fsky:0.2f}_remove_dipole.dat'
filename_TG = f'mie_sim_Euclid/cls_Tgalnoise_anafast_nside{nside}_lmax{lmax}_mie_sim_Euclid_nsim{nsim}_remove_dipole.dat'

filename_map_GG = f'mie_sim_Euclid/cls_galgalnoise_map_nside{nside}_lmax{lmax}_nsim{nsim}_fsky{fsky:0.2f}_remove_dipole.fits'

hp.write_map(filename_map_GG,mapgal_mask, overwrite=True)
np.savetxt(filename_TG, cl_TG_sims)
np.savetxt(filename_TG_mask, cl_TG_mask_sims)
np.savetxt(filename_TG_noise, cl_TG_noise_sims)
np.savetxt(filename_TG_mask_noise, cl_TG_noise_mask_sims)
