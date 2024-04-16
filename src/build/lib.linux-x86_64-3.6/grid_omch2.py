import numpy as np
import camb
from camb import model, initialpower
from camb.sources import GaussianSourceWindow, SplinedSourceWindow
import matplotlib.pyplot as plt
import os


FidParams_Planck2018 = {"H0": 67.36, "ombh2": 0.02237, "omch2": 0.1200, "As": 2.0989e-09, "ns": 0.9649, "tau" : 0.0544, "omnuh2":0.000645138398938178, "num_nu_massless":2.046,"num_nu_massive":1}
omch2 = np.linspace(0.0,(FidParams_Planck2018["H0"]/100)**2,30)

H0=FidParams_Planck2018['H0']
ombh2 = FidParams_Planck2018['ombh2']
omnuh2=FidParams_Planck2018['omnuh2']#[0.0006451383989381787,0]
As= FidParams_Planck2018['As']
ns= FidParams_Planck2018['ns']
tau= FidParams_Planck2018['tau']
num_nu_massless=FidParams_Planck2018['num_nu_massless']
num_nu_massive=FidParams_Planck2018['num_nu_massive']
omk=0


#idx = (np.abs(omch2_ - FidParams_Planck2018['omch2'])).argmin()
#print(idx)

#pars_fid = camb.CAMBparams(min_l=2)
#pars_fid.set_cosmology(
#        H0=H0,
#        ombh2=ombh2,
#        omch2=0.1200,
#        mnu=mnu,
#        omk=omk,
#        tau=tau,
#    )
#pars_fid.set_for_lmax(lmax, lens_potential_accuracy=0)
#pars_fid.set_dark_energy(w=-1, wa=0, dark_energy_model="ppf")
#pars_fid.Want_CMB = True
#pars_fid.Want_CMB_lensing = False
#pars_fid.SourceTerms.limber_windows = False
#pars_fid.SourceTerms.DoPotential = False
#pars_fid.SourceTerms.counts_density = True
#pars_fid.SourceTerms.counts_lensing = False
#pars_fid.SourceTerms.counts_velocity = False
#pars_fid.SourceTerms.counts_radial = False
#pars_fid.SourceTerms.counts_redshift = False
#pars_fid.SourceTerms.counts_potential = False
#pars_fid.SourceTerms.counts_timedelay = False
#pars_fid.SourceTerms.counts_ISW = False
#pars_fid.NonLinear = model.NonLinear_none
#pars_fid.SourceWindows=GaussianSourceWindow(redshift=1., source_type='counts', bias=1., sigma=0.25, dlog10Ndm=-0.2),
#results_fid = camb.get_results(pars_fid)
#cls_fid=results_fid.get_cmb_unlensed_scalar_array_dict()
pars_fid=camb.read_ini('spectra/inifiles/CAMB_01_planck.ini')
results_fid = camb.get_results(pars_fid)
cls_fid=results_fid.get_cmb_unlensed_scalar_array_dict()

lmax = pars_fid.max_l
cls=[]
for i,om in enumerate(omch2):
    pars=camb.read_ini('spectra/inifiles/CAMB_01_planck.ini')
    pars.omch2=om
    print('omnuh2=%0.8e'%pars.omnuh2)
    #pars = camb.CAMBparams(min_l=2)
    #pars.set_cosmology(
    #        H0=H0,
    #        ombh2=ombh2,
    #        omch2=om,
    #        #mnu=mnu,
    #        omk=omk,
    #        tau=tau,
    #        omnuh2=omnuh2,
    #        num_nu_massless=num_nu_massless,
    #        num_nu_massive=num_nu_massive
#
#
    #    )
    #pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    #pars.set_dark_energy(w=-1, wa=0, dark_energy_model="ppf")
#
    #pars.Want_CMB = True
    #pars.Want_CMB_lensing = False
    #pars.SourceTerms.limber_windows = False
    #pars.SourceTerms.DoPotential = False
    #pars.SourceTerms.counts_density = True
    #pars.SourceTerms.counts_lensing = False
    #pars.SourceTerms.counts_velocity = False
    #pars.SourceTerms.counts_radial = False
    #pars.SourceTerms.counts_redshift = False
    #pars.SourceTerms.counts_potential = False
    #pars.SourceTerms.counts_timedelay = False
    #pars.SourceTerms.counts_ISW = False
    #pars.NonLinear = model.NonLinear_none
#
    #pars.SourceWindows=GaussianSourceWindow(redshift=1., source_type='counts', bias=1., sigma=0.25, dlog10Ndm=-0.2),
    results = camb.get_results(pars)
    cls1=results.get_cmb_unlensed_scalar_array_dict()
    cls.append(cls1)
    print('omega_de=%0.8f'%results.omega_de,'omch2=%0.8e'%pars.omch2)
    pars=0
    results=0
    print(i)


ell = np.arange(2,lmax)
#
#plt.plot(ell,cls[idx]['TxW1'][ell])
#plt.savefig('grid_TxD.png')

header = 'L ,TxT, TxW1 ,W1xW1'

if not os.path.exists(f'spectra/Grid_spectra_omch2{len(omch2)}_planck'):
        os.makedirs(f'spectra/Grid_spectra_omch2{len(omch2)}_planck')
filename = f'spectra/Grid_spectra_omch2{len(omch2)}_planck/CAMBSpectra_OmL_fiducial.dat'#.replace('.', '')+'.dat'
np.savetxt(filename,np.array([ell, cls_fid['TxT'][ell],cls_fid['TxW1'][ell],cls_fid['W1xW1'][ell]]), header=header)

for p, omc in enumerate(omch2):
    filename = f'spectra/Grid_spectra_omch2{len(omch2)}_planck/CAMBSpectra_OmL{omc}.dat'#.replace('.', '')+'.dat'
    np.savetxt(filename,np.array([ell, cls[p]['TxT'][ell],cls[p]['TxW1'][ell],cls[p]['W1xW1'][ell]]), header=header)


