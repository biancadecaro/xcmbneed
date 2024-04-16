import numpy as np
import re
import os
from glob import glob
import shutil

path_sims_in = f'/ehome/bdecaro/xcmbneed/src/sims/Euclid_sims_Marina copy/NSIDE128/'

nside=re.findall(r'\d+', path_sims_in)[0]

listdir_in = os.listdir(path_sims_in)

lmax = re.findall(r'\d+', str(listdir_in[0]))[2]

field_g_old = 'g1'
field_T= 'T'

listdir_out_s = os.listdir(path_sims_in)
nsim = 1000
listdir_out = sorted(listdir_out_s)
#print(listdir_out)
#print(listdir_out[1000:2000])

#if field_T in str(listdir_out):
#    for n in range(1, nsim+1):
#        name_old_T = f'map_nbin1_NSIDE{nside}_lmax{lmax}_T_{n:05d}.fits'#f'sim_{u:04d}_TT_{nside.zfill(4)}.fits'map_nbin1_NSIDE128_lmax256_T_00102
#        name_new_T = f'map_nbin1_NSIDE{nside}_lmax{lmax}_{n:05d}_T.fits'
#        print(name_new_T, name_old_T)
#        os.rename(path_sims_in+name_old_T, path_sims_in+name_new_T)
if field_g_old in str(listdir_out):
    for n1 in range(1, nsim+1):
        name_old_gal = f'map_nbin1_NSIDE{nside}_lmax{lmax}_g1_{n1:05d}_noise.fits'
        name_new_gal = f'map_nbin1_NSIDE{nside}_lmax{lmax}_{n1:05d}_g1noise.fits'
        print(name_new_gal, name_old_gal)
        os.rename(path_sims_in+name_old_gal, path_sims_in+name_new_gal)

#for n in range(1, nsim+1):
#    print(n)
#    if field_T in str(listdir_out[n]):
#        name_old_T = f'map_nbin1_NSIDE{nside}_lmax{lmax}_T_{n:04d}.fits'#f'sim_{u:04d}_TT_{nside.zfill(4)}.fits'map_nbin1_NSIDE128_lmax256_T_00102
#        name_new_T = f'map_nbin1_NSIDE{nside}_lmax{lmax}__{n:04d}_T.fits'
#        print(name_new_T, name_old_T)
#        os.rename(name_old_T, name_new_T)
#        #np.savetxt(path_sims_out+name_new_T, listdir_out[n])
#    elif field_g_old in str(listdir_out[n]):
#        name_old_gal = f'map_nbin1_NSIDE{nside}_lmax{lmax}_g1_{n:04d}_noise.fits'
#        name_new_gal = f'map_nbin1_NSIDE{nside}_lmax{lmax}_{n:04d}_g1noise.fits'
#        print(name_new_gal, name_old_gal)
#        os.rename(name_old_gal, name_new_gal)
#        #np.savetxt(path_sims_out+name_new_gal, listdir_out[n])
    
#
#
#