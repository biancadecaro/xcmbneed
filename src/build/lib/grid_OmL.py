import numpy as np
import camb
from camb import model, initialpower
from camb.sources import GaussianSourceWindow, SplinedSourceWindow
import matplotlib.pyplot as plt
import os



def omch2(OmL_, ombh2_, omnuh2_, H0_ ):
    omch2=[]
    for om in OmL_:
        omch2.append((1-om)*(H0_/100)**2 - ombh2_-omnuh2_)
    return np.asarray(omch2)


OmL = np.linspace(0.0,0.95,30)

FidParams_Planck2018 = {"H0": 67.36, "ombh2": 0.02237, "omch2": 0.1200, "As": 2.0989e-09, "ns": 0.9649, "tau" : 0.0544, "omnuh2":0.000645138398938178, "num_nu_massless":2.046,"num_nu_massive":1}

H0=FidParams_Planck2018['H0']
ombh2 = FidParams_Planck2018['ombh2']
omnuh2=FidParams_Planck2018['omnuh2']#[0.0006451383989381787,0]
As= FidParams_Planck2018['As']
ns= FidParams_Planck2018['ns']
tau= FidParams_Planck2018['tau']
num_nu_massless=FidParams_Planck2018['num_nu_massless']
num_nu_massive=FidParams_Planck2018['num_nu_massive']
omk=0
#mnu=0.06
omch2_= omch2(OmL,ombh2, omnuh2, H0 )
lmax=2050
#print(OmL,omch2_)

idx = (np.abs(omch2_ - FidParams_Planck2018['omch2'])).argmin()
print(idx)

pars_fid=camb.read_ini('spectra/inifiles/CAMB_01_planck.ini')
results_fid = camb.get_results(pars_fid)
cls_fid=results_fid.get_cmb_unlensed_scalar_array_dict(raw_cl=True)

ell = np.arange(pars_fid.min_l,pars_fid.max_l+1)

#print(cls_fid['TxW1'].shape, ell.shape, cls_fid['W1xW1'][pars_fid.min_l:(pars_fid.max_l+1)].shape)


print('... Fiducial cls computed ...')

header = 'L ,TxT, TxW1 ,W1xW1'

print(f"lmin={pars_fid.min_l}, lmax={pars_fid.max_l}")

filename1 = f'spectra/inifiles/CAMBSpectra_OmL_fiducial_lmin0.dat'#.replace('.', '')+'.dat'
np.savetxt(filename1,np.array([ell, cls_fid['TxT'][pars_fid.min_l:(pars_fid.max_l+1)], cls_fid['TxW1'][pars_fid.min_l:(pars_fid.max_l+1)], cls_fid['W1xW1'][pars_fid.min_l:(pars_fid.max_l+1)]]), header=header)

if not os.path.exists(f'spectra/Grid_spectra_{len(OmL)}_planck_1'):
        os.makedirs(f'spectra/Grid_spectra_{len(OmL)}_planck_1')
filename = f'spectra/Grid_spectra_{len(OmL)}_planck_1/CAMBSpectra_OmL_fiducial_lmin0.dat'#.replace('.', '')+'.dat'
np.savetxt(filename,np.array([ell, cls_fid['TxT'][pars_fid.min_l:(pars_fid.max_l+1)], cls_fid['TxW1'][pars_fid.min_l:(pars_fid.max_l+1)], cls_fid['W1xW1'][pars_fid.min_l:(pars_fid.max_l+1)]]), header=header)
#if not os.path.exists(f'spectra/Grid_spectra_{len(OmL)}_no_mnu_old'):
#        os.makedirs(f'spectra/Grid_spectra_{len(OmL)}_no_mnu_old')


cls=[]
for i,om in enumerate(omch2_):
    pars=camb.read_ini('spectra/inifiles/CAMB_01_planck.ini')
    pars.omch2=om
    print('omnuh2=%0.8e'%pars.omnuh2)
    print(f"lmin={pars.min_l}, lmax={pars.max_l}")
    results = camb.get_results(pars)
    cls1=results.get_cmb_unlensed_scalar_array_dict(raw_cl=True)
    cls.append(cls1)
    print('omega_de=%0.8f'%results.omega_de,'omch2=%0.8e'%pars.omch2)
    print(cls1['TxW1'].shape)
    pars=0
    results=0
    print(i)




for p, oml in enumerate(OmL):
    filename = f'spectra/Grid_spectra_{len(OmL)}_planck_1/CAMBSpectra_OmL{oml}_lmin0.dat'#.replace('.', '')+'.dat'
    #filename = f'spectra/Grid_spectra_{len(OmL)}_no_mnu_old/CAMBSpectra_OmL{oml}_lmin0.dat'#.replace('.', '')+'.dat'
    np.savetxt(filename,np.array([ell, cls[p]['TxT'][pars_fid.min_l:(pars_fid.max_l+1)],cls[p]['TxW1'][pars_fid.min_l:(pars_fid.max_l+1)],cls[p]['W1xW1'][pars_fid.min_l:(pars_fid.max_l+1)]]), header=header)
    print(cls[p]['TxT'][pars_fid.min_l:(pars_fid.max_l+1)].shape)


