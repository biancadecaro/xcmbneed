import numpy as np
import pickle
import euclid_windows as EW
import camb
from camb.sources import GaussianSourceWindow, SplinedSourceWindow
from camb import model, initialpower
import time as time 


#print('CAMB version '+ str(camb.__version__))
#print(camb.__file__)

#print('If WantMG = True:')
#print('1) you need MGCAMB - check https://github.com/sfu-cosmo/MGCAMB')
#print('2) set w0=-1 and wa=0 (this is configuration is automatically set)')

def Compute_ingredients(fiducial, l_max, l_min, Win):

    #Compute the cosmology with CAMB given a set of parameters. 
    #INPUT:
    #fiducial : dictionary with the cosmlogical parameters ;
    #lmax (lmin) : max (min) multiple at with calculate the Cl's;
    #Win: return of Win.get_distributions();
    #RETURN: 'results' class of CAMB.

    if l_min:
        pars = camb.CAMBparams(min_l = l_min)
    else:
        pars = camb.CAMBparams()

    if "H0" in fiducial.keys():
        H0 = fiducial["H0"]
    else:
        H0 = 67.5

    #if "OmL" in fiducial.keys() and "ombh2" in fiducial.keys():
    #    OmL = fiducial['OmL']
    #    ombh2 = fiducial["ombh2"]
    #    omch2 = (1-OmL)*(H0/100)**2 - ombh2
    
    #else:
    if "ombh2" in fiducial.keys():
        ombh2 = fiducial["ombh2"]
    else:
        ombh2 = 0.022
    if "omch2" in fiducial.keys():
        omch2 = fiducial["omch2"]
    else:
        omch2 = 0.122
    

    if "As" in fiducial.keys():
        As = fiducial["As"]
    else:
        As = 2e-9

    if "ns" in fiducial.keys():
        ns = fiducial["ns"]
    else:
        ns = 0.965

    if "mnu" in fiducial.keys():
        mnu = fiducial["mnu"]
    else:
        mnu = 0.06

    if "omk" in fiducial.keys():
        omk = fiducial["omk"]
    else:
        omk = 0

    if "w0" in fiducial.keys():
        w0 = fiducial["w0"]
    else:
        w0 = -1.0

    if "wa" in fiducial.keys():
        wa = fiducial["wa"]
    else:
        wa = 0
    if "tau" in fiducial.keys():
        tau = fiducial["tau"]
    else:
        tau = 0.055
        

    #Om = (ombh2 + omch2) / H0 ** 2 * 1e4

    pars.set_cosmology(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        mnu=mnu,
        omk=omk,
        tau=tau,
    )

    pars.InitPower.set_params(As=As, ns=ns, r=0)
    if l_max:
        pars.set_for_lmax(l_max + 20, lens_potential_accuracy=0)
    else:
        pars.set_for_lmax(1000, lens_potential_accuracy=0)

    pars.set_dark_energy(w=w0, wa=wa, dark_energy_model="ppf")

    pars.Want_CMB = True
    pars.Want_CMB_lensing = False
    pars.SourceTerms.limber_windows = False
    pars.SourceTerms.DoPotential = False
    pars.SourceTerms.counts_density = True
    pars.SourceTerms.counts_lensing = False
    pars.SourceTerms.counts_velocity = False
    pars.SourceTerms.counts_radial = False
    pars.SourceTerms.counts_redshift = False
    pars.SourceTerms.counts_potential = False
    pars.SourceTerms.counts_timedelay = False
    pars.SourceTerms.counts_ISW = False
    pars.NonLinear = model.NonLinear_none
    print('pars.NonLinear=',pars.NonLinear)

    pars.SourceWindows = Win.get_camb_distributions()
    
    results = camb.get_results(pars)

    return results

def Compute_spectra(results, Win, WantTT,WantGG):

    #Compute the Cls. 
    #INPUT:
    #results : 'results' class of CAMB;
    #Win : return of Win.get_distributions();
    #WantTT: if True calculate TT spectrum;
    #WantGG: if True calculate GG spectrum;
    #RETURN: 
    #cls : dictionary with Cl^TG spectra for each redshift bin, Cl^TT if WantTT=True, Cl^GG if WantGG=True.

    cls_camb = results.get_cmb_unlensed_scalar_array_dict(raw_cl=True)
    if WantTT == False:
       cls_camb.pop('TxT')
    remove_keys= ['TxE','TxP','ExT','ExP','ExE','PxP','PxE','PxT']
    for k in remove_keys:
        cls_camb.pop(k)
    for i in range(1,Win.nbin+1):
        remove_keys_1 = ['ExW'+str(i),'PxW'+str(i),'W'+str(i)+'xE','W'+str(i)+'xP','W'+str(i)+'xT' ]
        for key in remove_keys_1:
            cls_camb.pop(key)
        if WantGG == False:
            for j in range(1,Win.nbin+1):
                remove_keys_2 = ['W'+str(i)+'xW'+str(j)]
                for key in remove_keys_2:
                    cls_camb.pop(key)
        else:
            for j in range(1,Win.nbin+1):
                if i > j:
                    remove_keys_3 = ['W'+str(i)+'xW'+str(j)]
                    for key in remove_keys_3:
                        cls_camb.pop(key)

    return cls_camb


def Compute_fiducial_cls(fid, l_min, l_max, Win):

    #Compute the fiducial Cls .
    #INPUT:
    #grid : dictionary of cls
    #fid : dictionary with the cosmlogical parameters;
    #lmax (lmin) : max (min) multiple at with calculate the Cl's; 
    #Win : return of Win.get_distributions();
    #RETURN:
    #fid_cls : dictionary of fiducial cls ['TxT', 'TxW', 'WxW']

    cosmo_fid = Compute_ingredients(fiducial = fid, l_min=l_min, l_max=l_max, Win=Win)
    fid_cls = Compute_spectra(results = cosmo_fid, Win=Win, WantGG=True, WantTT=True)
    #print(fid_cls.keys())
    return fid_cls





def Compute_grid(fiducial,lmin,lmax, settings, WantEst = False):
     
    #Compute the fiducial Cls and theorical Cls over a grid of given parameters.
    #INPUT:
    #fiducial : dictionary with the cosmlogical parameters;
    #settings: list of strings with settings. Options are: [nbins3, nbins5, nbins10, sigmaz0.001,sigmaz0.05, sigmaz0.1, sigmaz0.5];
    #lmax (lmin) : max (min) multiple at with calculate the Cl's; 
    #WantArray : if True returns the theoretical spectra calculated over a grid of parameter as a numpy array, is False returns a dictionary;
    #WantGrid : if True returns the theoretical spectra calculated over a grid of parameter, if False only fiducial Cl's. DEFAULT = True;
    #kwargs = keywords arguments with the array of the parameters over which calculate the grid of spectra;
    #RETURN:
    #grid_spectra: object with fiducial Cl's (TG, TT, GG) and theoretical Cl's for each number of grid and 
    #               with a grid of parameters over which the theoretical spectra are calculated.
    
    if "nbins" in fiducial.keys():
        nbins_fid = fiducial["nbins"]
    else:
        nbins_fid = 10
    
    if "sigmab" in fiducial.keys():
        sigmab_fid = fiducial["sigmab"]
    else:
        sigmab_fid = 0.05
    
    if "sigma0" in fiducial.keys():
        sigma0_fid = fiducial["sigma0"]
    else:
        sigma0_fid = 0.05
    
    if "zb" in fiducial.keys():
        zb_fid = fiducial["zb"]
    else:
        zb_fid = 0.0
        
    if "z0" in fiducial.keys():
        z0_fid = fiducial["z0"]
    else:
        z0_fid = 0.1 ## Original: 0.0 ##
    
    if "cb" in fiducial.keys():
        cb_fid = fiducial["cb"]
    else:
        cb_fid = 1.0
    
    if "c0" in fiducial.keys():
        c0_fid = fiducial["c0"]
    else:
        c0_fid = 1.0
        
    if "fout" in fiducial.keys():
        fout_fid = fiducial["fout"]
    else:
        fout_fid = 0.1 ## Original: 0.0 ##
    
    if "bintype" in fiducial.keys():
        bintype_fid = fiducial["bintype"]
    else:
        bintype_fid = "equipopulated"
    
    if "biastype" in fiducial.keys():
        biastype_fid = fiducial["biastype"]
    else:
        biastype_fid = "stepwise"
    
            
    print("Fiducial values:", fiducial)
    print('#####################')
    
    grid_spectra = {}
    for s in settings:
        
        nbins=nbins_fid
        
        cb=cb_fid
        sigmab=sigmab_fid
        zb=zb_fid
        
        c0=c0_fid
        z0=z0_fid
        sigma0=sigma0_fid
        
        fout=fout_fid
        
        bintype = bintype_fid
        biastype = biastype_fid
        
        subsettings = s.rsplit(',')
        
        for ss in subsettings:
            if 'nbins' in ss: nbins = int(ss.rsplit('nbins')[1])
            
            if 'sigmab' in ss: sigmab = float(ss.rsplit('sigmab')[1])
            if 'zb' in ss: zb = float(ss.rsplit('zb')[1])
            
            if 'sigma0' in ss: sigma0 = float(ss.rsplit('sigma0')[1])
            if 'z0' in ss: z0 = float(ss.rsplit('z0')[1])
            
            if 'fout' in ss: fout = float(ss.rsplit('fout')[1])
            
            if 'bintype' in ss: bintype = str(ss.rsplit('bintype')[1])
            if 'biastype' in ss: biastype = str(ss.rsplit('biastype')[1])
            
        grid_spectra[s]={}
        print("Setting name:", s)
        print("nbins:", nbins, "sigmab:", sigmab, "sigma0:", sigma0, "zb:", zb, "z0:", z0, "fout:", fout)
        print("bintype:", bintype, "biastype:", biastype)
        print('###################')
        print('Computing '+s)

        ## Original EW ##
        # Win = EW.Windows(bintype="equipopulated", normalize=False, nbin=nbins, sigma0=sigma0, sigmab=sigmab, zb=zb, z0=z0, fout=fout, cb=cb, c0=c0)

        # New EW ##
        if WantEst and nbins == 1 :
            Win = EW.Windows(normalize = False, bintype = [0.001,2.50],nbin = 1, biastype="continuous", use_true_galactic_dist=True)
            print('sono dentro nbins=1 con bias continuous')
        else:
            Win = EW.Windows(normalize=False, bintype = bintype, biastype = biastype,
                                nbin=nbins, sigma0=sigma0, sigmab=sigmab, zb=zb, z0=z0, fout=fout, cb=cb, c0=c0)
            # setting 'bintype="equipopulated" ' returns bin edges that are different from those of S.Ilic

        Win.get_distributions()
        grid_spectra[s]['cls_fid'] = Compute_fiducial_cls(fid=fiducial, l_min=lmin, l_max=lmax, Win=Win)
        #cosmo_fid = Compute_ingredients(fiducial, l_min=lmin, l_max=lmax, Win=Win)
        #grid_spectra[s]['cls_fid'] = Compute_spectra(cosmo_fid, Win, WantGG=True, WantTT=True)
        print('Fiducial cls for', s,' setting computed!')
        
    return grid_spectra

def Save_spectra(filename, spectra):

    #Save the Cl's in a pickle file.
    #INPUT:
    #filename: string of the name of the file;
    #spectra: cls. 

    with open(filename+'.pkl', 'wb') as f:
        pickle.dump(spectra, f)
    return

def Read_spectra(filename):

    #Read the Cl's from a pickle file.
    #INPUT:
    #filename: string of the name of the file;
    #RETURN: cls. 
    with open(filename+'.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":

    FidParams_IST = {"H0": 67.0, "ombh2": 0.0224, "omch2": 0.12, "As": 2.1115e-09, "ns": 0.96, "tau" : 0.058, "mnu": 0.06}

    lmin = 0
    lmax = 2050
    ell = np.arange(lmin, lmax+1)

    setting = ['nbins3']

    fid_nbins3 = Compute_grid(FidParams_IST, lmax=lmax, lmin=lmin, settings=setting)

    filename = 'fiducial_IST_nbins3_lmax2050_lmin0'


    Save_spectra(filename, fid_nbins3)

