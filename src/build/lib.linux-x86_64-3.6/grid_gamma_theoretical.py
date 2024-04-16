import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cython_mylibc as mylibc
import analysis, utils, spectra, sims
import grid_beta_module as grid
import cython_mylibc as pippo
import os
import healpy as hp

OmL = np.linspace(0.0,0.95,30)

simparams = {'nside'   : 512,
             'ngal'    : 5.76e5, #dovrebbe importare solo per lo shot noise (noise poissoniano)
 	     	 'ngal_dim': 'ster',
	     	 'pixwin'  : False}
nside = simparams['nside']
jmax = 12
lmax = 782
#nsim = 1
#B = 1.95
B = pippo.mylibpy_jmax_lmax2B(jmax, lmax)
#jmax = round(np.log(lmax*1.00000000000000222045e+00)/np.log(B))
mask = hp.read_map(f'/ehome/bdecaro/xcmbneed/src/mask/mask_planck_comm_2018_nside={nside}.fits')
wl = hp.anafast(mask, lmax=lmax)

fname_xcspectra = []
sim_dir         = []
out_dir         = []
cov_dir         = []
for om in OmL:
    if not os.path.exists(f'output_needlet_TG_OmL/Grid_spectra_{len(OmL)}_Gammaj_Planck/TGsims_theoretical_OmL{om}/'):
        os.makedirs(f'output_needlet_TG_OmL/Grid_spectra_{len(OmL)}_Gammaj_Planck/TGsims_theoretical_OmL{om}/')
    fname_xcspectra.append(f'spectra/Grid_spectra_{len(OmL)}_planck_1/CAMBSpectra_OmL{om}_lmin0.dat')#.replace('.', '') np.chararray(len(OmL))Grid_spectra_{len(OmL)}s
    out_dir.append(f'output_needlet_TG_OmL/Grid_spectra_{len(OmL)}_Gammaj_Planck/TGsims_theoretical_OmL{om}/' )#np.chararray(len(OmL))


dir= {'fname_xcspectra':fname_xcspectra, 'out_dir':out_dir}


#dir = grid.Initialize_dir(OmL, simparams['nside'])
print(dir['fname_xcspectra'])

gammaj_grid = grid.Compute_gamma_grid_theoretical(OmL,wl, dir, B,lmax, jmax)




