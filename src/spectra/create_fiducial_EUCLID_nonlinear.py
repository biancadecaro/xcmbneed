import numpy as np
import grid_spectra as spectra

filename ='../../../grid_spectra/fiducial_IST_non_linear_lmax500_lmin2'
pkl = spectra.Read_spectra(filename)[0]

lmax=1000

ell =  np.arange(lmax+1)

print(ell)

header = 'L, TxT, TxW1, W1xW1'
filename_inifiles='inifiles/fiducial_EUCLID_nonlinear.dat'
np.savetxt(filename_inifiles,np.array([ell, pkl['nbins1']['cls_fid']['TxT'][ell],pkl['nbins1']['cls_fid']['TxW1'][ell],pkl['nbins1']['cls_fid']['W1xW1'][ell]]), header=header)