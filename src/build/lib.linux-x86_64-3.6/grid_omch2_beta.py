import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cython_mylibc as mylibc
import analysis, utils, spectra, sims
import grid_beta_module as grid

omch2 = np.linspace(0.0,0.6736**2, 30)

simparams = {'nside'   : 512,
             'ngal'    : 5.76e5, #dovrebbe importare solo per lo shot noise (noise poissoniano)
 	     	 'ngal_dim': 'ster',
	     	 'pixwin'  : False}

jmax = 12
lmax = 782
#nsim = 1
#B = 1.95

#jmax = round(np.log(lmax*1.00000000000000222045e+00)/np.log(B))

dir = grid.Initialize_dir(omch2, simparams['nside'])
print(dir['fname_xcspectra'])

sims_analysis = grid.Analysis_sims_grid(omch2, dir, simparams, lmax, jmax )

betaj_sims_TS_galS_grid = grid.Compute_beta_grid(omch2, jmax, sims_analysis, simparams['nside'])

grid.Make_plot(betaj_sims_TS_galS_grid, omch2, jmax, sims_analysis, dir, simparams['nside'])


