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


nside = 128
jmax = 12
lmax = 256
#nsim = 1
#B = 1.95
B = pippo.mylibpy_jmax_lmax2B(jmax, lmax)
#jmax = round(np.log(lmax*1.00000000000000222045e+00)/np.log(B))
mask = hp.read_map(f'/ehome/bdecaro/xcmbneed/src/mask/EUCLID/mask_planck_comm_2018_x_euclid_fsky0p35_nside={nside}.fits')
wl = hp.anafast(mask, lmax=lmax)

fname_xcspectra = []
sim_dir         = []
out_dir         = []
cov_dir         = []
for om in OmL:
    if not os.path.exists(f'output_needlet_TG_OmL/Grid_spectra_{len(OmL)}_Gammaj_Euclid/TGsims_theoretical_OmL{om}/'):
        os.makedirs(f'output_needlet_TG_OmL/Grid_spectra_{len(OmL)}_Gammaj_Euclid/TGsims_theoretical_OmL{om}/')
    fname_xcspectra.append(f'/ehome/bdecaro/xcmbneed/src/spectra/Grid_spectra_{len(OmL)}_EUCLID/EUCLID_cl_OmL{om}.dat')#.replace('.', '') np.chararray(len(OmL))Grid_spectra_{len(OmL)}s
    out_dir.append(f'output_needlet_TG_OmL/Grid_spectra_{len(OmL)}_Gammaj_Euclid/TGsims_theoretical_OmL{om}/' )#np.chararray(len(OmL))


dir= {'fname_xcspectra':fname_xcspectra, 'out_dir':out_dir}


#dir = grid.Initialize_dir(OmL, simparams['nside'])
print(dir['fname_xcspectra'])

gammaj_grid = np.zeros((len(OmL), jmax+1))

need_theory=spectra.NeedletTheory(B)
        # Needlet Analysis
        #myanalysis.append(analysis.NeedAnalysis(jmax, lmax, dir['out_dir'][xc], simulations[xc]))
for xc in range(len(OmL)):
    spectra_tg = np.loadtxt(fname_xcspectra[xc])[1]
    gammaj_grid[xc]    = need_theory.gammaJ(spectra_tg, wl, jmax, lmax)
    
for i in range(len(OmL)):
    np.savetxt(dir['out_dir'][i]+f'beta_TS_galS_theoretical_OmL{OmL[i]}_B{B}.dat', gammaj_grid[i])



