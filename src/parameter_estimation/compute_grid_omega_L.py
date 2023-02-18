import numpy as np
import matplotlib
import euclid_windows as EW
import grid_spectra as spectra

import matplotlib.pyplot as plt

FidParams_Planck2018 = {"H0": 67.36, "ombh2": 0.02237, "omch2": 0.1200, "As": 2.0989e-09, "ns": 0.9649, "tau" : 0.0544}
#FidParams = {"H0": 67.32, "ombh2": 0.022, "As": 2.1e-9, "ns": 0.966} #Planck 2018
#galaxy count: gaussian, z=1, sigma = 0.25, bias = 1.
lmax = 2048

lmin = 0

settings = ['nbins1']

OmL = np.linspace(0.0,0.95,60)

#grid_spectra_OmL = spectra.Compute_grid(FidParams_Planck2018, lmax=lmax, lmin=lmin, settings=settings, WantArray = False, OmL=OmL )

filename_OmL = 'grid_OmL_len_'+str(len(OmL))+'_lmax_'+str(lmax)

#spectra.Save_spectra(filename_OmL, grid_spectra_OmL)
data_OmL = spectra.Read_spectra(filename_OmL)

ell = np.arange(lmin, lmax+1)
#print(grid_spectra_OmL['nbins1']['cls_grid']['OmL=0.4']['TxT'][lmin:lmax])
ell_ts = np.arange(data_OmL['nbins1']['cls_grid']['OmL='+str(OmL[4])+'']['TxT'].shape[0])
#print(len(ell_ts),len(grid_spectra_OmL['nbins1']['cls_grid']['OmL=0.4']['TxT'] ))
#print(ell_ts[-1], ell_ts[0])

fig2 = plt.figure(figsize=(10,7))
plt.loglog(ell, data_OmL['nbins1']['cls_grid']['OmL='+str(OmL[4])+'']['TxT'][ell], label = 'OmL'+str(OmL[4]))
plt.loglog(ell, data_OmL['nbins1']['cls_grid']['OmL='+str(OmL[8])+'']['TxT'][ell], label = 'OmL'+str(OmL[8]))
plt.loglog(ell, data_OmL['nbins1']['cls_grid']['OmL='+str(OmL[16])+'']['TxT'][ell], label = 'OmL'+str(OmL[16]))
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell(\ell+1)/2\pi C_l^{TG}$')
plt.title(r'nbins='+settings[0]+r', $l_{max}$ ='+str(lmax)+r', OmL ')
plt.legend(loc = 'best' )

plt.savefig('prova_spectra_OmL_grid_60.png')



header = 'L ,TxT, TxW1 ,W1xW1'

for p, oml in enumerate(OmL):
    filename = '../spectra/Grid_spectra_60/CAMBSpectra_OmL'+str(oml)+'.dat'#.replace('.', '')+'.dat'
    np.savetxt(filename,np.array([ell_ts, data_OmL['nbins1']['cls_grid']['OmL='+str(oml)]['TxT'],data_OmL['nbins1']['cls_grid']['OmL='+str(oml)]['TxW1'],data_OmL['nbins1']['cls_grid']['OmL='+str(oml)]['W1xW1']]), header=header)

#filename_1 = '../spectra/CAMBSpectra_OmL04.dat'
#np.savetxt(filename_1,np.array([ell_ts, grid_spectra_OmL['nbins1']['cls_grid']['OmL=0.4']['TxT'],grid_spectra_OmL['nbins1']['cls_grid']['OmL=0.4']['TxW1'],grid_spectra_OmL['nbins1']['cls_grid']['OmL=0.4']['W1xW1']]), header=header)
#
#filename_2 = '../spectra/CAMBSpectra_OmL05.dat'
#np.savetxt(filename_2,np.array([ell_ts, grid_spectra_OmL['nbins1']['cls_grid']['OmL=0.5']['TxT'],grid_spectra_OmL['nbins1']['cls_grid']['OmL=0.5']['TxW1'],grid_spectra_OmL['nbins1']['cls_grid']['OmL=0.5']['W1xW1']]), header=header)
#
#filename_3 = '../spectra/CAMBSpectra_OmL06.dat'
#np.savetxt(filename_3,np.array([ell_ts, grid_spectra_OmL['nbins1']['cls_grid']['OmL=0.6']['TxT'],grid_spectra_OmL['nbins1']['cls_grid']['OmL=0.6']['TxW1'],grid_spectra_OmL['nbins1']['cls_grid']['OmL=0.6']['W1xW1']]), header=header)
#
#filename_4 = '../spectra/CAMBSpectra_OmL07.dat'
#np.savetxt(filename_4,np.array([ell_ts, grid_spectra_OmL['nbins1']['cls_grid']['OmL=0.7']['TxT'],grid_spectra_OmL['nbins1']['cls_grid']['OmL=0.7']['TxW1'],grid_spectra_OmL['nbins1']['cls_grid']['OmL=0.7']['W1xW1']]), header=header)
#
#filename_5 = '../spectra/CAMBSpectra_OmL08.dat'
#np.savetxt(filename_5,np.array([ell_ts, grid_spectra_OmL['nbins1']['cls_grid']['OmL=0.8']['TxT'],grid_spectra_OmL['nbins1']['cls_grid']['OmL=0.8']['TxW1'],grid_spectra_OmL['nbins1']['cls_grid']['OmL=0.8']['W1xW1']]), header=header)

filename_fid = '../spectra/CAMBSpectra_fiducial.dat'
np.savetxt(filename_fid,np.array([ell_ts, data_OmL['nbins1']['cls_fid']['TxT'],data_OmL['nbins1']['cls_fid']['TxW1'],data_OmL['nbins1']['cls_fid']['W1xW1']]), header=header)
