import numpy as np
import re
import os
from glob import glob
import shutil

path_sims_in = f'/edata/d2/cmbxc_repo/EUCLID/MAPS/NSIDE128/'

nside = re.findall(r'\d+', path_sims_in)[1]

listdir_in = os.listdir(path_sims_in)

lmax = re.findall(r'\d+', str(listdir_in[0]))[2]

field_g_old = 'g1'
field_T= 'T'

#fnames = glob(path_sims_in+f'map_nbin1_NSIDE{nside}_lmax{lmax}*.fits')
#print(fnames[0])
#
#listdir_out = [np.load(str(f)) for f in fnames]
#print(listdir_out[0])

path_sims_out = f'/ehome/bdecaro/xcmbneed/src/sims/Euclid_sims/TG_sims_{nside}/' 
#if not os.path.exists(path_sims_out):
#        os.makedirs(path_sims_out)

#shutil.copy(path_sims_in+f'map_nbin1_NSIDE128_lmax{lmax}_g1_00001_noise.fits', path_sims_out+f'map_nbin1_NSIDE128_lmax{lmax}_g1_00001_noise.fits') 
#shutil.copytree(path_sims_in,path_sims_out)
listdir_out_s = os.listdir(path_sims_out)

listdir_out = sorted(listdir_out_s)

#print(listdir_out[1000:2000])

nu = np.arange(0,len(listdir_out)/2, dtype=int)
num = np.vstack((nu, nu)).flatten()


for n,u in enumerate(num):
    print(n,u)
    if field_T in str(listdir_out[n]):
        name_old_T = f'sim_{u:04d}_TT_{nside.zfill(4)}.fits'
        name_new_T = f'sim_{u:04d}_TS_{nside.zfill(4)}.fits'
        print(name_new_T, name_old_T)
        os.rename(path_sims_out+name_old_T, path_sims_out+name_new_T)
        #np.savetxt(path_sims_out+name_new_T, listdir_out[n])
    #elif field_g_old in str(listdir_out[n]):
    #    name_old_gal = f'map_nbin1_NSIDE{nside}_lmax{lmax}_{field_g_old}_{(u+1):05d}_noise.fits'
    #    name_new_gal = f'sim_{u:04d}_galT_{nside.zfill(4)}.fits'
    #    print(name_new_gal, name_old_gal)
    #    os.rename(path_sims_out+name_old_gal, path_sims_out+name_new_gal)
    #    #np.savetxt(path_sims_out+name_new_gal, listdir_out[n])
    
#
#
#