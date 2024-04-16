import numpy as np
import random as rn
import os
import matplotlib.pyplot as plt
import healpy as hp

def GetCorrAlms(clxy, clxx, clyy, nside, lmax=None):
    """
    The purpose of this function is to generate correlated gaussian realizations of 
    two fields X and Y which statistics is described by the sets of three Auto- and 
    Cross-spectra clxx, clyy, and clxy. 

    ! Realizations are returned as *alms* !

    Params.
    1. clxy:   array containing the cross-power spectrum C_l^{XY}
    2. clxx:   array containing the auto-power  spectrum C_l^{XX} of first field
    3. clyy:   array containing the auto-power  spectrum C_l^{YY} of second field
    4. nside:  Healpix resolution
    5. lmax:   (optional) maximum ell 
    
    The outputs are two numpy arrays containing:
    i.  alm_xx: First field Alm coefficients
    ii. alm_yy: Second field Alm coefficients 

    Based on HealPix routine create_alm.
    """

    if lmax is None:
        lmax = 2*nside 
        lmax = int(lmax)

    tot_alm = hp.sphtfunc.Alm.getsize(lmax)  

    alm_xx = np.zeros(tot_alm, dtype=complex)
    alm_yy = np.zeros(tot_alm, dtype=complex)


    # Loop on l: Let's deal with X field coefficients
    # and first term rms_yy1 of the Y field coefficients
    for l in range(0,lmax+1):
        rms_xx = 0.
        rms_yy1 = 0.
        if clxx[l] != 0.: # To avoid division by zero
            rms_xx = np.sqrt(clxx[l])
            rms_yy1 = clxy[l]/rms_xx

        # Treat the m = 0 case
        rand1_r = np.random.normal()
        rand1_i = 0.
        alm_xx.real[l] = rand1_r*rms_xx
        alm_yy.real[l] = rand1_r*rms_yy1

        #Treat the m > 0 cases
        for m in range(1,l+1):
            rand1_r = np.random.normal()/np.sqrt(2.0)
            rand1_i = np.random.normal()/np.sqrt(2.0)
            alm_xx.real[(m*(2*lmax+1-m)//2)+l] = rand1_r*rms_xx #ho aggiunto // invece di / nella divisione per due (mi casta a integer)
            alm_xx.imag[(m*(2*lmax+1-m)//2)+l] = rand1_i*rms_xx  
            alm_yy.real[(m*(2*lmax+1-m)//2)+l] = rand1_r*rms_yy1
            alm_yy.imag[(m*(2*lmax+1-m)//2)+l] = rand1_i*rms_yy1

    # Loop on l: second term rms_yy2 of Y field coefficients
    for l in range(0,lmax+1):
        rms_yy2 = 0.
        if clxx[l] != 0.: # To avoid division by zero
            rms_yy2 = clyy[l] - (clxy[l]/clxx[l])*clxy[l]
            rms_yy2 = np.sqrt(rms_yy2)

        # Treat the m = 0 case
        rand2_r = np.random.normal()
        rand2_i = 0.
        alm_yy.real[l] = alm_yy.real[l] + rand2_r*rms_yy2

        #Treat the m > 0 cases
        for m in range(1,l+1):
            rand2_r = np.random.normal()/np.sqrt(2.0)
            rand2_i = np.random.normal()/np.sqrt(2.0)
            alm_yy.real[(m*(2*lmax+1-m)//2)+l] = alm_yy.real[(m*(2*lmax+1-m)//2)+l] + rand2_r*rms_yy2 
            alm_yy.imag[(m*(2*lmax+1-m)//2)+l] = alm_yy.imag[(m*(2*lmax+1-m)//2)+l] + rand2_i*rms_yy2
    
    #header = 'alm_xx, alm_yy'
    #np.savetxt('alms.dat', np.array([alm_xx, alm_yy]), header = header)

    return alm_xx, alm_yy

def GetCorrMaps(clxy, clxx, clyy, nside, lmax=None, pixwin=True):
    """
    The purpose of this function is to generate correlated gaussian realizations of 
    two fields X and Y which statistics is described by the sets of three Auto- and 
    Cross-spectra clxx, clyy, and clxy. 
    
    ! Realizations are returned as *maps* !

    Params.
    1. clxy:   array containing the cross-power spectrum C_l^{XY}
    2. clxx:   array containing the auto-power  spectrum C_l^{XX} of first field
    3. clyy:   array containing the auto-power  spectrum C_l^{YY} of second field
    4. nside:  Healpix resolution
    5. lmax:   (optional) maximum ell 
    6. pixwin: (optional) Convolve with pixel window function
    
    The outputs are two numpy arrays containing:
    i.  map_xx: First field map
    ii. map_yy: Second field map 
    """
    #if lmax is None:
    #    lmax = clxy.shape[0]
    #    print(lmax)
    
    alm_xx, alm_yy = GetCorrAlms(clxy, clxx, clyy, nside, lmax=lmax)

    # Creating XX and YY Maps
    map_xx = hp.sphtfunc.alm2map(alm_xx, nside, pixwin=pixwin, lmax=lmax, verbose=False)
    map_yy = hp.sphtfunc.alm2map(alm_yy, nside, pixwin=pixwin, lmax=lmax, verbose=False)

    return map_xx, map_yy

def Counts2Delta(counts, mask=None):
    """
    Converts a number counts Healpix map into a density contrast map.

    Note
    ----
    Use only binary mask.
    """
    counts = np.asarray(counts)
    
    if mask is not None:
        mask   = np.asarray(mask)
        counts = hp.ma(counts)
        counts.mask = np.logical_not(mask)
    
    mean  = np.mean(counts)
    delta = (counts - mean) / mean
    
    return delta

def GetNlgg(counts, dim,mask=None, lmax=None, return_ngal=False):
    """
    Returns galaxy shot-noise spectra given a number counts Healpix map. 
    If return_ngal is True, it returns also the galaxy density in gal/ster.

    Note
    ----
    1. Use only binary mask.
    2. If mask is not None, yielded spectrum is not pseudo
    """
    counts = np.asarray(counts)

    if lmax is None: lmax = hp.npix2nside(counts.size) * 2
    if mask is not None: 
        mask = np.asarray(mask)
        fsky = np.mean(mask)
    else: 
        mask = 1.
        fsky = 1.

    N_tot = np.sum(counts * mask)
    if dim=='ster':
        ngal  = N_tot / fsky
    else:
        ngal  = N_tot / 4. / np.pi / fsky

    if return_ngal:
        return np.ones(lmax+1) / ngal, ngal
    else:
        return np.ones(lmax+1)/ ngal

def GetCountsTot(delta, ngal, dim='pix'):
    """
    Returns total galaxy *counts* map (signal + poisson noise) given an overdensity 
    *delta* = (n-ngal)/ngal Healpix map and mean galaxy density ngal (in gal/pix).
    """
    if dim == 'ster':
        ngal = ngal*hp.nside2pixarea(hp.npix2nside(delta.size))
    # elif dim == 'arcmin':
        # ngal = ngal/hp.nside2pixarea(hp.npix2nside(delta.size), degrees=True)/0.000278 # Sq. deg -> Sq. arcmin

    # Delta has to be > -1
    delta[delta < -1.] = -1.

    return np.random.poisson(lam=ngal*(1.+delta)) * 1.

def GetGalNoiseMap(nside, ngal, dim='pix', delta=False):
    '''
    Returns (poissonian) noise *counts* galaxy map given a mean galaxy density.
    If delta = True, it returns the noise *overdensity* galaxy map.
    '''
    if dim == 'ster':
        ngal = ngal*hp.nside2pixarea(nside)

    npix   = hp.nside2npix(nside)
    sigma2 = 1./ngal

    # Creating the galaxy noise map
    map_ngg = np.random.poisson(lam=1./sigma2, size=npix)
    
    # Converting it to density contrast map
    if delta:
        nbar = np.mean(map_ngg)
        map_ngg = (map_ngg - nbar)/nbar
        map_ngg[map_ngg < -1.] = -1.

    return map_ngg

def GetGalMask(nside, lat=None, fsky=None, nest=False):
    """
    Returns a symmetric Galactic Mask in Healpix format at nside resolution.
    Pixels with latitude < |lat| deg are set to 0.
    Otherwise you input the fsky and evaluates the required latitude.
    """
    if lat is None:
        if fsky is not None:
            lat = np.rad2deg(np.arcsin(1. - fsky))
        else:
            raise ValueError("Missing lat or fsky !")

    mask      = np.zeros(hp.nside2npix(nside))
    theta_cut = np.deg2rad(90. - lat)
    mask[np.where((hp.pix2ang(nside, np.arange(mask.size), nest=nest))[0] >= (np.pi - theta_cut))] = 1.
    mask[np.where((hp.pix2ang(nside, np.arange(mask.size), nest=nest))[0] <= theta_cut)] = 1.

    return mask

