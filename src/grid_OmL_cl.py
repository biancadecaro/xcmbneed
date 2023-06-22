import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cython_mylibc as mylibc
import analysis, utils, spectra, sims
import grid_beta_module as grid
import os

OmL = np.linspace(0.0,0.95,30)

simparams = {'nside'   : 512,
             'ngal'    : 5.76e5, #dovrebbe importare solo per lo shot noise (noise poissoniano)
 	     	 'ngal_dim': 'ster',
	     	 'pixwin'  : False}


lmax = 800
delta_ell =50

fname_xcspectra = []
sim_dir         = []
out_dir         = []
cov_dir         = []
for om in OmL:
    #if not os.path.exists('output_cl_TG_OmL/Grid_spectra_'+str(len(OmL))+'_planck_lmin0/TGsims_'+str(simparams['nside'])+'_OmL'+str(om)+'/' ):
    #    os.makedirs('output_cl_TG_OmL/Grid_spectra_'+str(len(OmL))+'_planck_lmin0/TGsims_'+str(simparams['nside'])+'_OmL'+str(om)+'/' )
    #if not os.path.exists('sims/Cl/Grid_spectra_'+str(len(OmL))+'_planck_lmin0/TGsims_'+str(simparams['nside'])+'_OmL'+str(om)+'/' ):
    #    os.makedirs('sims/Cl/Grid_spectra_'+str(len(OmL))+'_planck_lmin0/TGsims_'+str(simparams['nside'])+'_OmL'+str(om)+'/')

    if not os.path.exists('output_cl_TG_OmL/Grid_spectra_'+str(len(OmL))+'_planck_old/TGsims_'+str(simparams['nside'])+'_OmL'+str(om)+'/' ):
        os.makedirs('output_cl_TG_OmL/Grid_spectra_'+str(len(OmL))+'_planck_old/TGsims_'+str(simparams['nside'])+'_OmL'+str(om)+'/' )
    if not os.path.exists('sims/Cl/Grid_spectra_'+str(len(OmL))+'_planck_old/TGsims_'+str(simparams['nside'])+'_OmL'+str(om)+'/' ):
        os.makedirs('sims/Cl/Grid_spectra_'+str(len(OmL))+'_old/TGsims_'+str(simparams['nside'])+'_OmL'+str(om)+'/')


    fname_xcspectra.append('spectra/Grid_spectra_30/CAMBSpectra_OmL'+str(om)+'.dat')#.replace('.', '') np.chararray(len(OmL))
    sim_dir.append('sims/Needlet/Grid_spectra_'+str(len(OmL))+'/TGsims_'+str(simparams['nside'])+'_OmL'+str(om)+'/') #np.chararray(len(OmL))
    out_dir.append('output_cl_TG_OmL/Grid_spectra_'+str(len(OmL))+'_old/TGsims_'+str(simparams['nside'])+'_OmL'+str(om)+'/' ) #np.chararray(len(OmL))

#dir_spectra_fid=f'spectra/Grid_spectra_{len(OmL)}_planck/CAMBSpectra_OmL_fiducial.dat'
#dir_out_fid = f'output_needlet_TG_OmL/Grid_spectra_{len(OmL)}_planck/'
dir= {'fname_xcspectra':fname_xcspectra, 'sim_dir':sim_dir, 'out_dir':out_dir}


sims_analysis = grid.Analysis_sims_grid_cl(OmL, dir, simparams, lmax, delta_ell )

lbins = sims_analysis[0].ell_binned
print(lbins)

cl_sims_TS_galS_grid = grid.Compute_cl_grid(OmL, delta_ell, sims_analysis, simparams['nside'], lmax)

#grid.Make_plot(betaj_sims_TS_galS_grid, OmL, jmax, sims_analysis, dir, simparams['nside'])

#fig = plt.figure(figsize=(17,10))
#ax = fig.add_subplot(1, 1, 1)
#
#ax.plot(delta_ell)