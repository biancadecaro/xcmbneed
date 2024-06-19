import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nsim = 1000
nside = 128
npix= hp.nside2npix(nside)
lmax=256
lmin=2

sims_dir = f'../sims/Euclid_sims_Marina/NSIDE{nside}/'#f'sims/Needlet/Planck/TGsims_{nside}_planck_2_lmin{lmin}/'
cl_TT_mask_sims = np.zeros((nsim,lmax+1))
cl_TT_sims = np.zeros((nsim,lmax+1))
cl_GG_sims = np.zeros((nsim,lmax+1))
cl_TG_sims = np.zeros((nsim,lmax+1))
cl_TG_mask_sims = np.zeros((nsim,lmax+1))
cl_GG_mask_sims = np.zeros((nsim,lmax+1))

mask_eu = hp.read_map(f'../mask/EUCLID/mask_rsd2022g-wide-footprint-year-6-equ-order-13-moc_ns0{nside}_G_filled_2deg2.fits')
mask_pl = hp.read_map(f'../mask/mask_planck_comm_2018_nside={nside}.fits')
fsky_eu = np.mean(mask_eu)
fsky_pl = np.mean(mask_pl)
fsky = np.mean(mask_pl*mask_eu)
print(f'fsky_pl = {fsky_pl}, fsky_eu={fsky_eu}, fsky={fsky}')

fname_T = sims_dir + f'map_nbin1_NSIDE{nside}_lmax{lmax}_00678_T.fits' #sims_dir + "sim_" + ('%04d' % n) + "_galT_" + ('%04d' % nside) + ".fits"
fname_gal = sims_dir + f'map_nbin1_NSIDE{nside}_lmax{lmax}_00678_g1noise.fits' # sims_dir + "sim_" + ('%04d' % n) + "_TS_" + ('%04d' % nside) + ".fits"
mapT = hp.read_map(fname_T, verbose=False)
mapgal = hp.read_map(fname_gal, verbose=False)
mapT_mask = hp.read_map(fname_T, verbose=False)
mapgal_mask = hp.read_map(fname_gal, verbose=False)
bad_v_pl = np.where(mask_pl==0)
bad_v_eu = np.where(mask_eu==0)
mapT_mask[bad_v_pl]=hp.UNSEEN
mapgal_mask[bad_v_eu]=hp.UNSEEN

hp.mollview(mapT_mask,cmap= 'viridis', title= 'T maps-masked')
hp.mollview(mapgal_mask,cmap= 'viridis', title= 'G maps-masked')

plt.show()

for n in range(nsim):
    fname_gal = sims_dir + f'map_nbin1_NSIDE{nside}_lmax{lmax}_{n+1:05d}_g1noise.fits' # sims_dir + "sim_" + ('%04d' % n) + "_TS_" + ('%04d' % nside) + ".fits"
    fname_T = sims_dir + f'map_nbin1_NSIDE{nside}_lmax{lmax}_{n+1:05d}_T.fits' #sims_dir + "sim_" + ('%04d' % n) + "_galT_" + ('%04d' % nside) + ".fits"
    mapT = hp.read_map(fname_T, verbose=False)
    mapgal = hp.read_map(fname_gal, verbose=False)
    mapT_mask = hp.read_map(fname_T, verbose=False)
    mapgal_mask = hp.read_map(fname_gal, verbose=False)
    bad_v_pl = np.where(mask_pl==0)
    bad_v_eu = np.where(mask_eu==0)
    mapT_mask[bad_v_pl]=hp.UNSEEN
    mapgal_mask[bad_v_eu]=hp.UNSEEN

    cl_TT_sims[n, :] =hp.anafast(map1=mapT, map2=mapT, lmax=lmax)
    cl_GG_sims[n, :] =hp.anafast(map1=mapgal, map2=mapgal, lmax=lmax)
    cl_TT_mask_sims[n, :] =hp.anafast(map1=mapT_mask, map2=mapT_mask, lmax=lmax)
    cl_GG_mask_sims[n, :] =hp.anafast(map1=mapgal_mask, map2=mapgal_mask, lmax=lmax)
    cl_TG_sims[n, :] =hp.anafast(map1=mapT, map2=mapgal, lmax=lmax)  # al posto di zero della maschera mettere badvalue
    cl_TG_mask_sims[n, :] =hp.anafast(map1=mapT_mask, map2=mapgal_mask, lmax=lmax)  # al posto di zero della maschera mettere badvalue

    print(f'Num sim={n+1}')

filename_TT_mask = f'EUCLID/Euclid_Planck_masks/cls_TT_anafast_nside{nside}_lmax{lmax}_Euclidnoise_Marina_nsim{nsim}_fsky{fsky:0.2f}.dat'#lmin{lmin}_marina.dat'
filename_GG_mask = f'EUCLID/Euclid_Planck_masks/cls_galnoisegalnoise_anafast_nside{nside}_lmax{lmax}_Euclidnoise_Marina_nsim{nsim}_fsky{fsky:0.2f}.dat'#lmin{lmin}_marina.dat'

filename_TG_mask = f'EUCLID/Euclid_Planck_masks/cls_Tgalnoise_anafast_nside{nside}_lmax{lmax}_Euclidnoise_Marina_nsim{nsim}_fsky{fsky:0.2f}.dat'#lmin{lmin}_marina.dat'
filename_map_GG_mask = f'EUCLID/Euclid_Planck_masks/cls_galgalnoise_map_nside{nside}_lmax{lmax}_Euclidnoise_Marina_nsim{nsim}_fsky{fsky:0.2f}.fits'

hp.write_map(filename_map_GG_mask,mapgal_mask, overwrite=True)

np.savetxt(filename_TT_mask, cl_TT_mask_sims)
np.savetxt(filename_GG_mask, cl_GG_mask_sims)
np.savetxt(filename_TG_mask, cl_TG_mask_sims)
