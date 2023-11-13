import numpy as np
import camb
from camb import model, initialpower
from camb.sources import GaussianSourceWindow, SplinedSourceWindow
import matplotlib.pyplot as plt
import os
import grid_spectra as spectra


OmL = np.linspace(0.0,0.95,30)
lmax = 500
path_pickle = '/ehome/bdecaro/grid_spectra/OmL_30_grid_WUCLID_lmin0_lmax500'

pickle_Oml = spectra.Read_spectra(path_pickle)[0]

out_dir = f'spectra/Grid_spectra_{len(OmL)}_EUCLID/'
if not os.path.exists(out_dir):
        os.makedirs(out_dir)

ell = np.arange(lmax+1)

header = 'L ,TxT, TxW1 ,W1xW1'


filename_fiducial = out_dir+f'EUCLID_cl_OmL_fiducial.dat'
np.savetxt(filename_fiducial,np.array([ell,  pickle_Oml['nbins1']['cls_fid']['TxT'][:lmax+1], pickle_Oml['nbins1']['cls_fid']['TxW1'][:lmax+1], pickle_Oml['nbins1']['cls_fid']['W1xW1'][:lmax+1]]), header= header )

print(pickle_Oml['nbins1']['cls_grid'].shape)

for p, oml in enumerate(OmL):
    filename = f'spectra/Grid_spectra_{len(OmL)}_EUCLID/EUCLID_cl_OmL{oml}.dat'#.replace('.', '')+'.dat'
    #filename = f'spectra/Grid_spectra_{len(OmL)}_no_mnu_old/CAMBSpectra_OmL{oml}_lmin0.dat'#.replace('.', '')+'.dat'
    np.savetxt(filename,np.array([ell, pickle_Oml['nbins1']['cls_grid'][p][0][:lmax+1]]), header=header)
    #print(cls[p]['TxT'][pars_fid.min_l:(pars_fid.max_l+1)].shape)


