import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cython_mylibc as mylibc
import analysis, utils, spectra, sims
import grid_beta_module as grid
import cython_mylibc as pippo
import os

OmL = np.linspace(0.0,0.95,30)

simparams = {'nside'   : 256,
             'ngal'    : 5.76e5, #dovrebbe importare solo per lo shot noise (noise poissoniano)
 	     	 'ngal_dim': 'ster',
	     	 'pixwin'  : False}

jmax = 12
lmax = 782
#nsim = 1
#B = 1.95
B = pippo.mylibpy_jmax_lmax2B(jmax, lmax)
#jmax = round(np.log(lmax*1.00000000000000222045e+00)/np.log(B))

fname_xcspectra = []
sim_dir         = []
out_dir         = []
cov_dir         = []
for om in OmL:
    if not os.path.exists(f'output_needlet_TG_OmL/Grid_spectra_{len(OmL)}_planck_2_lmin0/TGsims_theoretical_OmL{om}/'):
        os.makedirs(f'output_needlet_TG_OmL/Grid_spectra_{len(OmL)}_planck_2_lmin0/TGsims_theoretical_OmL{om}/')
    fname_xcspectra.append(f'spectra/Grid_spectra_{len(OmL)}_planck_1/CAMBSpectra_OmL{om}_lmin0.dat')#.replace('.', '') np.chararray(len(OmL))Grid_spectra_{len(OmL)}s
    out_dir.append(f'output_needlet_TG_OmL/Grid_spectra_{len(OmL)}_planck_2_lmin0/TGsims_theoretical_OmL{om}/' )#np.chararray(len(OmL))

dir_spectra_fid=f'spectra/Grid_spectra_{len(OmL)}_planck_1/CAMBSpectra_OmL_fiducial_lmin0.dat'
dir_out_fid = f'output_needlet_TG_OmL/Grid_spectra_{len(OmL)}_planck_2_lmin0/'
dir= {'fname_xcspectra':fname_xcspectra, 'out_dir':out_dir}


#dir = grid.Initialize_dir(OmL, simparams['nside'])
print(dir['fname_xcspectra'])

betaj_TS_galS_grid, delta__TS_galS_grid = grid.Compute_beta_grid_theoretical(OmL, dir, B,lmax, jmax)


spectra_fid = spectra.XCSpectraFile(clfname= dir_spectra_fid,  WantTG = True)  
need_theory=spectra.NeedletTheory(B)

betatg_fid   = need_theory.cl2betaj(jmax=jmax, cl=spectra_fid.cltg)
delta_fid = need_theory.delta_beta_j(jmax, cltt = spectra_fid.cltt, cltg = spectra_fid.cltg, clgg = spectra_fid.clg1g1)
#
np.savetxt(dir_out_fid+f'beta_TS_galS_theoretical_OmL_fiducial_B{B}.dat', betatg_fid)
np.savetxt(dir_out_fid+f'variance_TS_galS_theoretical_OmL_fiducial_B{B}.dat', delta_fid)


#grid.Make_plot(betaj_sims_TS_galS_grid, OmL, jmax, sims_analysis, dir, simparams['nside'])


