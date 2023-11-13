import numpy as np
import camb
from camb import model, initialpower
from camb.sources import GaussianSourceWindow, SplinedSourceWindow
import matplotlib.pyplot as plt
import os

#FidParams_Planck2018 = {"H0": 67.36, "ombh2": 0.02237, "omch2": 0.1200, "As": 2.0989e-09, "ns": 0.9649, "tau" : 0.0544, "omnuh2":0.000645138398938178, "num_nu_massless":2.046,"num_nu_massive":1}
#
#H0=FidParams_Planck2018['H0']
#ombh2 = FidParams_Planck2018['ombh2']
#omnuh2=FidParams_Planck2018['omnuh2']#[0.0006451383989381787,0]
#As= FidParams_Planck2018['As']
#ns= FidParams_Planck2018['ns']
#tau= FidParams_Planck2018['tau']
#num_nu_massless=FidParams_Planck2018['num_nu_massless']
#num_nu_massive=FidParams_Planck2018['num_nu_massive']
#omk=0
#mnu=0.06
#omch2_= omch2(OmL,ombh2, omnuh2, H0 )
lmax=2050
#print(OmL,omch2_)


pars_fid=camb.read_ini('./CAMB_01_planck.ini')
results_fid = camb.get_results(pars_fid)
cls_fid=results_fid.get_cmb_unlensed_scalar_array_dict(raw_cl=True)

ell = np.arange(pars_fid.min_l,pars_fid.max_l+1)

#print(cls_fid['TxW1'].shape, ell.shape, cls_fid['W1xW1'][pars_fid.min_l:(pars_fid.max_l+1)].shape)


print('... Fiducial cls computed ...')

header = 'L ,TxT, TxW1 ,W1xW1'

print(f"lmin={pars_fid.min_l}, lmax={pars_fid.max_l}")
print(f"omnuh2={pars_fid.omnuh2}, omch2={pars_fid.omch2}, omega_de={results_fid.omega_de}")

filename1 = f'CAMBSpectra_planck_fiducial_lmin{pars_fid.min_l}_{pars_fid.max_l}.dat'#.replace('.', '')+'.dat'
np.savetxt(filename1,np.array([ell, cls_fid['TxT'][pars_fid.min_l:(pars_fid.max_l+1)], cls_fid['TxW1'][pars_fid.min_l:(pars_fid.max_l+1)], cls_fid['W1xW1'][pars_fid.min_l:(pars_fid.max_l+1)]]), header=header)

