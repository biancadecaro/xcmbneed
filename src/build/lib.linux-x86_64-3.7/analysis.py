import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import argparse, os, sys, warnings, glob
import imp

import cython_mylibc as pippo
#pippo=cython_mylibc.mylibc()

import utils
from master import Master
#from master_needlets import Master_needlets

from IPython import embed

#from src.master import Binner
from master import Binner

class HarmAnalysis(Master):
        """
        Class to perform spherical harmonic analysis (based on MASTER algorithm). 
        @see arXiv:     

        !! WARN: inpainting implemented only for XC so far !!

        """     
        def __init__(self, lmax, OutDir, Sims, lmin=2, delta_ell=1, mask=None, flattening=None, 
                                 pixwin=True, fwhm_smooth=None, fsky_approx=False):
                """
                Parameters
                ----------
                lmax : int
                        Maximum multipole value
                OutDir : str
                    Directory to store the analysis results
                Sims : class
                        Simulations class 
                lmin : int
                        Minimum multipole value
                delta_ell : int
                        Bin-size
                mask : Healpix map
                        Mask to apply
                flattening : str
                        Power spectrum flattening type. Default = None
                        Possible types:
                        1.  None: fact = 1 
                        2. 'Ell': fact = l           
                        3. 'CMBLike': fact = l*(l+1)/2\pi 
                pixwin : boolean
                        Deconvolves for pixel window function if True
                fwhm_smooth : float or tuple
                        Beam FWHM in arcmin 
                fsky_approx : boolean
                        MASTER method if False
                """
                if mask is not None:
                        mask = np.asarray(mask)
                else: 
                        mask = np.ones(hp.nside2npix(Sims.SimPars['nside']))
                
                super(HarmAnalysis, self).__init__(lmin, lmax, delta_ell, mask, flattening, pixwin, fwhm_smooth, fsky_approx)
                #super(HarmAnalysis, self).__init__(mask, lmin=lmin, lmax=lmax, delta_ell=delta_ell, flat=flattening, pixwin=pixwin, fwhm_smooth=fwhm_smooth, MASTER=fsky_approx)
                
                self.lmax        = lmax
                self.Sims        = Sims    
                self.OutDir      = OutDir 
                self.lmin        = lmin 
                self.mask        = mask
                self.delta_ell   = delta_ell
                self.flattening  = flattening
                self.pixwin      = pixwin
                self.fwhm_smooth = fwhm_smooth
                self.fsky_approx = fsky_approx

                if not os.path.exists(self.OutDir):
                    os.makedirs(self.OutDir)

        def GetClSimsFromMaps(self, field1, nsim, field2=None, fix_field=None, fname=None, EuclidSims=False):
                """
                Evaluates (auto- or cross-) angular power spectrum for simulated maps of a given field. 

                Parameters
                ----------
                field1 : str
                        Cosmic field to be analyzed => 'kappaT, kappaN, kappaS, deltaT, deltaS'
                nsim  : int
                        Number of simulations to be analyzed
                field2 : str
                        Second cosmic field to be analyzed => 'kappaT, kappaN, kappaS, deltaT, deltaS'
                fix_field : str or Healpix map
                        Map to cross-correlate with nsim simulated maps of field1. This is for *null-tests*.
                mask  : Healpix map
                        (binary) Mask applied to each map
                fname : str
                        File name to store the array. 

                Returns
                -------
                cl_sims : array
                        array with ((nsim, ell_binned.size)) shape containing angular power spectra for each simulation. 
                """
                try:
                        from mpi4py import MPI
                        comm = MPI.COMM_WORLD
                        myid, nproc = comm.Get_rank(), comm.Get_size()
                except ImportError:
                        myid, nproc = 0, 1

                try:
                        cl_sims_tot  = np.loadtxt(self.OutDir + fname)
                        if myid == 0: print("...Cl's Sims " + self.OutDir + fname + " found...")
                except:
                        # print("...simulations beta_jk's " + fname + " not found...")          
                        if myid == 0: print("...evaluating Cl's Sims...")
                        if myid == 0: print("...field1----->", field1)
                        if myid == 0: print("...field2----->", field2)
                        if myid == 0: print("...fix_field-->", fix_field)

                        if (field2 != None) and (fix_field != None):
                                raise ValueError("Options not compatible! => Either you set field2 or fix_field")

                        if fix_field is not None:
                                if type(fix_field) == str:
                                        fix_field = hp.read_map(fix_field)
                                assert( hp.isnpixok(len(fix_field)) )                           

                        dim = int((nsim) / nproc)
                        print("dim=",dim)

                        if nproc > 1:
                                if myid < (nsim) % nproc:
                                        dim += 1
                        #self.ell_binned = Binner(bin_edges=None, lmin=self.lmin, lmax=self.lmax, delta_ell=self.delta_ell, flat=self.flattening)#bianca
                        cl_sims = np.zeros((dim,self.ell_binned.size))
                       
                        # Loop over simulations
                        k = 0
                        if (field2 is None) and (fix_field is None): # Field1 x Field1
                                for n in range(myid, nsim, nproc): 
                                        if EuclidSims==True:
                                                m = self.Sims.GetSimField_Euclid(field1, n, self.lmax)
                                        else:
                                                m = self.Sims.GetSimField(field1, n)
                                        cl_sims[k,:] = self.get_spectra(m)
                                        k += 1
                        elif (field2 is not None) and (fix_field is None): # Field1 x Field2
                                for n in range(myid, nsim, nproc):
                                        if EuclidSims==True:
                                                m1 = self.Sims.GetSimField_Euclid(field1, n, self.lmax)
                                                m2 = self.Sims.GetSimField_Euclid(field2, n, self.lmax)
                                        else:
                                                m1 = self.Sims.GetSimField(field1, n)
                                                m2 = self.Sims.GetSimField(field2, n)
                                        m1 *= self.mask
                                        m2 *= self.mask
                                        print(np.sum(self.mask))
                                        m1 = hp.remove_dipole(m1, verbose=False)
                                        m2 = hp.remove_dipole(m2, verbose=False)
                                        inpainting = False #bianca 3/3/2022
                                        if inpainting:
                                                m1 = inp.diffusive_inpaint2(m1, self.mask, niter)
                                                m2 = inp.diffusive_inpaint2(m2, self.mask, niter)
                                                #hp.mollview(m1, norm='hist'); hp.mollview(m2, norm='hist'); plt.show()
                                                
                                                cl_ = hp.anafast(m1, map2=m2, lmax=self.lmax)
                                                cl_sims[k,:] = self.bin_spectra(cl_)/self.fsky
                                        else:
                                                cl_sims[k,:] = self.get_spectra(m1, map2=m2)
                                                
                                        k += 1
                        elif (field2 is None) and (fix_field is not None): # Field1 x Fix_field (Null-test)
                                for n in range(myid, nsim, nproc): 
                                        if EuclidSims==True:
                                                m1 = self.Sims.GetSimField_Euclid(field1, n, self.lmax)
                                        else:
                                                m1 = self.Sims.GetSimField(field1, n)
                                        cl_sims[k,:] = self.get_spectra(m1, map2=fix_field)
                                        k += 1

                        assert (k == dim)

                        if nproc > 1:
                                cl_sims_tot = comm.gather(cl_sims, root=0)
                                if myid == 0:
                                    cl_sims_tot = np.vstack((_sims for _sims in cl_sims_tot))
                        else:
                                cl_sims_tot = cl_sims
                        
                        if myid == 0:  
                                if fname is not None:
                                        print("...saving Cl's Sims to output " + self.OutDir + fname + "...")
                                        np.savetxt(self.OutDir + fname, cl_sims_tot, header='Cls sims')

                        if nproc > 1:
                                comm.Barrier() 

                return cl_sims_tot

        def GetCovMatrixFromMaps(self, field1, nsim, field2=None, fix_field=None, fname=None, fname_sims=None):
                """
                Evaluates the Cl covariance (and correlation) matrix for simulated maps of a given field. 

                Parameters
                ----------
                field1 : str
                        Cosmic field to be analyzed => 'kappaT, kappaN, kappaS, deltaT, deltaS'
                nsim  : int
                        Number of simulations to be analyzed
                field2 : str
                        Second cosmic field to be analyzed => 'kappaT, kappaN, kappaS, deltaT, deltaS'
                fix_field : str or Healpix map
                        Map to cross-correlate with nsim simulated maps of field1. This is for *null-tests*.
                fname : str
                        File name to store the covariance matrix. 
                fname_sims : str
                        File name to look for array with stored simulated Cls. 

                Returns
                -------
                cov_cl : array
                        array with ((ell_binned.size, ell_binned.size)) shape containing covariance matrix
                """
                
                #try:
                #       cov_cl  = np.loadtxt(self.OutDir + fname)
                #       print("...Covariance Matrix " + self.OutDir + fname + " found...")
                #except:
                #       # print("...simulations beta_jk's " + fname + " not found...")
                #       print("...evaluating Covariance Matrix...")
#
                #       if (field2 != None) and (fix_field != None):
                #               raise ValueError("Options not compatible! => Either you set field2 or fix_field")
#
                #       if fix_field is not None:
                #               if type(fix_field) == str:
                #                       fix_field = hp.read_map(fix_field)
                #               assert( hp.isnpixok(len(fix_field)) )                           
#
                #       cl_sims = self.GetClSimsFromMaps(field1, nsim, field2=field2, fix_field=fix_field, fname=fname_sims)
                #       cov_cl  = np.cov(cl_sims.T)
                #       
#
                #       if fname is not None:
                #               print("...saving Covariance Matrix to output " + self.OutDir + fname + "...")
                #               np.savetxt(self.OutDir + fname, cov_cl, header='Covariance matrix <C_l C_l>')
#
                #       corr_cl = np.corrcoef(cl_sims.T)
#
                #return cov_cl, corr_cl
##
                # print("...simulations beta_jk's " + fname + " not found...")
                print("...evaluating Covariance Matrix...")
                if (field2 != None) and (fix_field != None):
                        raise ValueError("Options not compatible! => Either you set field2 or fix_field")
                if fix_field is not None:
                        if type(fix_field) == str:
                                fix_field = hp.read_map(fix_field)
                        assert( hp.isnpixok(len(fix_field)) )                           
                cl_sims = self.GetClSimsFromMaps(field1, nsim, field2=field2, fix_field=fix_field, fname=fname_sims)
                cov_cl  = np.cov(cl_sims.T)
                
                if fname is not None:
                        print("...saving Covariance Matrix to output " + self.OutDir + fname + "...")
                        np.savetxt(self.OutDir + fname, cov_cl, header='Covariance matrix <C_l C_l>')
                corr_cl = np.corrcoef(cl_sims.T)
                
                return cov_cl,corr_cl
        

#class NeedAnalysis(Master_needlets):
class NeedAnalysis(object):
        """
        Class to perform needlet analysis (based on Alessandro Renzi's library). 
        @see arXiv:0707.0844    
        """     
        def __init__(self, jmax, lmax, OutDir, Sims, mask=None, flattening=None, 
                                 pixwin=False, fwhm_smooth=None, fsky_approx=False):
        #def __init__(self, jmax, lmin, lmax, OutDir, Sims, mask=None, flattening=None, 
        
                """
                Parameters
                ----------
                jmax : int
                    Maximum needlet frequency
                lmax : int
                        Maximum multipole value
                OutDir : str
                    Directory to store the analysis results
                Sims : class
                        Simulations class 
                """
                if mask is not None:
                        mask = np.asarray(mask)
                else: 
                        mask = np.ones(hp.nside2npix(Sims.SimPars['nside']))
                
                #super(NeedAnalysis, self).__init__(lmin, lmax, jmax, mask, flattening, pixwin, fwhm_smooth, fsky_approx)
                
                self.jmax   = jmax
                self.lmax   = lmax
                self.Sims   = Sims    
                self.OutDir = OutDir  
                #self.fsky_approx = fsky_approx
                #print(f'fsky_approx = {fsky_approx}')

                if not os.path.exists(self.OutDir):
                    os.makedirs(self.OutDir)

                # Initialize Needlet library
                print("...Initializing Needlet library...")
                self.B = pippo.mylibpy_jmax_lmax2B(self.jmax, self.lmax)
		#self.B = mylibc.jmax_lmax2B(self.jmax, self.lmax)
                print("==>lmax={:d}, jmax={:d}, nside={:d}, B={:e}".format(self.lmax,self.jmax,self.Sims.SimPars['nside'],self.B)) 
                self.b_values = pippo.mylibpy_needlets_std_init_b_values(self.B, self.jmax, self.lmax)
                #print(self.b_values, self.b_values.shape)
                #np.savetxt('./b_values_B=%1.2f' %self.B+'.txt',self.b_values)
                pippo.mylibpy_needlets_check_windows(self.jmax, self.lmax, self.b_values)
                self.jvec = np.arange(self.jmax+1)
                print("...done...")

        def Betajk2Betaj(self, betajk1, betajk2=None, mask=None):
                """
                Returns the needlet (auto- or cross-) power spectrum \beta_j given the needlet coefficients.

                Parameters
                ----------
                betajk1 : array
                        Map 1 needlet coefficients 
                betajk2 : array
                        Map 2 needlet coefficients 
                """
                # if mask is None:
                #       maskarr = 1.
                # else:
                #       maskarr = np.vstack([mask for i in range(self.jmax+1)])
 
                maskarr = 1.

                if betajk2 is None: # auto-spectrum
                        betaj = np.mean(betajk1[:self.jmax+1,:]*betajk1[:self.jmax+1,:]*maskarr, axis=1)
                else:
                        betaj = np.mean(betajk1[:self.jmax+1,:]*betajk2[:self.jmax+1,:]*maskarr, axis=1)

                return betaj

        def Map2Betaj(self, map1,nsim, map2=None, mask=None, noise=0., inpainting=False, niter=2000, path_inpainting="python/inpainting.py"):
                """
                Returns the needlet (auto- or cross-) power spectrum \beta_j given Healpix maps.

                Parameters
                ----------
                map1 : Healpix map
                        Map 1
                map2 : Healpix map
                        Map 2 
                mask : Healpix map
                        (binary) Mask applied to each map 
                noise : 

                inpainting :

                niter :

                path_inpainting :

                Returns
                -------
                betaj : array
                        Array w/ shape (jmax+1) containing auto(cross) needlet power spectrum

                Notes
                -----
                i.   In case of cross-spectrum the mask is applied to both maps
                ii.  In case of masked sky, extracted spectrum is divided by f_sky factor 
                iii. In case of masked sky, noise bias is *not* divided by f_sky factor
                """
                #print('sono dentro Map2Betaj')
                if inpainting:
                        inp = imp.load_source("module.name", path_inpainting)

                #if not isinstance(noise, list):
                #       assert (len(noise)==self.jmax+1)

                fsky = 1.
                
                if mask is not None:
                        fsky  = np.mean(mask) 
                        map1 *= mask 
                        print(f'fsky mappa 1={fsky:0.2f}')

                map1 = hp.remove_dipole(map1, verbose=False)#.compressed()
                
                if inpainting:
                        map1 = inp.diffusive_inpaint2(map1, mask, niter)

                betajk1 = pippo.mylibpy_needlets_f2betajk_healpix_harmonic(map1, self.B, self.jmax, self.lmax)
                #print('shape betajk1='+str(betajk1.shape))
                if map2 is None: # Auto-
                       return self.Betajk2Betaj(betajk1, mask=mask)/fsky - noise
                else: # Cross-
                        if mask is not None:
                               map2 *= mask 
                               print(f'fsky mappa 2={np.mean(mask) :0.2f}')
                        map2 = hp.remove_dipole(map2, verbose=False)#.compressed()
                        if inpainting:
                               map2 = inp.diffusive_inpaint2(map2, mask, niter)
                               print('sono dentro inpainting')
                        betajk2 = pippo.mylibpy_needlets_f2betajk_healpix_harmonic(map2, self.B, self.jmax, self.lmax)
                        
                        #for j in range(self.jmax):
                        #        hp.write_map(f'maps_beta/map_beta_{j}_B = %1.2f ' %self.B+f'_nsim_{nsim}_fsky={fsky}'+'T', betajk1[j,:], overwrite= True)     
                        #        hp.write_map(f'maps_beta/map_beta_{j}_B = %1.2f ' %self.B+f'_nsim_{nsim}_fsky={fsky}'+'G', betajk2[j,:], overwrite= True)     
                        return self.Betajk2Betaj(betajk1, betajk2=betajk2, mask=mask)#/fsky #!! WARN /fsky, result unbiased

<<<<<<< HEAD
                        #for j in range(self.jmax):
                        #        hp.write_map(f'maps_beta/map_beta_{j}_B = %1.2f ' %self.B+f'_nsim_{nsim}_fsky={fsky}'+'T', betajk1[j,:], overwrite= True)     
                        #        hp.write_map(f'maps_beta/map_beta_{j}_B = %1.2f ' %self.B+f'_nsim_{nsim}_fsky={fsky}'+'G', betajk2[j,:], overwrite= True)     
                        return self.Betajk2Betaj(betajk1, betajk2=betajk2, mask=mask)/fsky #!! WARN /fsky, result unbiased

        def GetBetajkSims(self, field, nsim, mask=None, fname=None):
=======
        def GetBetajkSims(self, field, nsim, mask=None, fname=None, EuclidSims=False):
>>>>>>> euclid_implementation
                """
                Evaluates needlet coefficients for simulated maps of a given field. 

                Parameters
                ----------
                field : str
                        Cosmic field to be analyzed => 'kappaT, kappaN, kappaS, deltaT, deltaS'
                nsim  : int
                        Number of simulations to be analyzed
                mask  : Healpix map
                        (binary) Mask applied 
                fname : str
                        File name to store the array. 

                Returns
                -------
                betajk_sims : array
                        array with ((nsim, jmax+1, npix)) shape containing needlet coefficients 
                """
                try:
                        betajk_sims_tot = np.load(self.OutDir + fname)
                        print("...simulations beta_jk's " + self.OutDir + fname + "found...")
                except:
                        print("...simulations beta_jk's " + self.OutDir + fname + " not found...")
                        print("...evaluating...")

                        try:
                                from mpi4py import MPI
                                comm = MPI.COMM_WORLD
                                myid, nproc = comm.Get_rank(), comm.Get_size()
                        except ImportError:
                                myid, nproc = 0, 1

                        dim = (nsim) / nproc
                        if nproc > 1:
                                if myid < (nsim) % nproc:
                                        dim += 1

                        betajk_sims = np.zeros((dim, self.jmax+1, hp.nside2npix(self.Sims.SimPars['nside'])))

                        ## Cycle over simulations 
                        k = 0
                        for n in range(myid, nsim, nproc):
                               if EuclidSims==True:
                                        m = self.Sims.GetSimField_Euclid(field, n, self.lmax)
                               else:
                                m = self.Sims.GetSimField(field, n)
                               if mask is not None:
                                       #if hp.isnpixok(mask.size)
                                       m *= mask # TODO: does the library work with masked arrays??
                               betajk = pippo.mylibpy_needlets_f2betajk_healpix_harmonic(m, self.B, self.jmax, self.lmax)
                               betajk_sims[k, :] = betajk
                               del betajk
                               k += 1
                        assert (k == int(dim)) 
#
                        if nproc > 1:
                               betajk_sims_tot = comm.gather(betajk_sims, root=0)
                               if myid == 0:
                                   betajk_sims_tot = np.vstack((_sims for _sims in betajk_sims_tot)) 
                        else:
                               betajk_sims_tot = betajk_sims

                        if myid == 0: 
                                if fname is None:
                                        fname = ''
                                print("...evaluation terminated...")
                                np.save(self.OutDir + fname, betajk_sims_tot)
                                print("...saved to output " + self.OutDir + fname + "...")

                        if nproc > 1:
                                comm.Barrier() 

                        #print('shape betajk sim = '+str(betajk_sims_tot.shape))

                return betajk_sims_tot
        
        def GetBetajSimsFromMaps(self, field1, nsim, field2=None, fix_field=None, mask=None, fname=None, noise=0., EuclidSims=False, inpainting=False, niter=1000, path_inpainting="/u/ap/fbianchini/codes/inpainting/python/inpainting.py"):
                """
                Evaluates needlet (auto- or cross-) power spectrum for simulated maps of a given field. 

                Parameters
                ----------
                field1 : str
                        Cosmic field to be analyzed => 'kappaT, kappaN, kappaS, deltaT, deltaS'
                nsim  : int
                        Number of simulations to be analyzed
                field2 : str
                        Second cosmic field to be analyzed => 'kappaT, kappaN, kappaS, deltaT, deltaS'
                fix_field : str or Healpix map
                        Map to cross-correlate with nsim simulated maps of field1. This is for *null-tests*.
                mask  : Healpix map
                        (binary) Mask applied to each map
                fname : str
                        File name to store the array. 

                Returns
                -------
                betaj_sims : array
                        array with ((nsim, jmax+1)) shape containing needlet coefficients 
                """
                #print('sono dentro GetBetajSimsFromMaps')
                try:
                        from mpi4py import MPI
                        comm = MPI.COMM_WORLD
                        myid, nproc = comm.Get_rank(), comm.Get_size()
                except ImportError:
                        myid, nproc = 0, 1
                
                try:
                        betaj_sims_tot  = np.loadtxt(self.OutDir + fname)
                        if myid == 0: print("...Beta_J Sims " + self.OutDir + fname + " found...")
                except:
                        if myid == 0: print("...evaluating Beta_j Sim...")
                        if myid == 0: print("...field1----->", field1)
                        if myid == 0: print("...field2----->", field2)
                        if myid == 0: print("...fix_field-->", fix_field)

                        if (field2 != None) and (fix_field != None):
                                raise ValueError("Options not compatible! => Either you set field2 or fix_field")

                        if fix_field is not None:
                                if type(fix_field) == str:
                                        fix_field = hp.read_map(fix_field)
                                assert( hp.isnpixok(len(fix_field)) )                           

                        dim = (nsim) / nproc

                        if nproc > 1:
                                if myid < (nsim) % nproc:
                                        dim += 1
                        
                        print(dim)

                        betaj_sims = np.zeros((int(dim), self.jmax+1)) 

                        # Cycle over simulations 
                        k = 0
                        if (field2 is None) and (fix_field is None): # Field1 x Field1
                                for n in range(myid, nsim, nproc): 
                                        
                                        if EuclidSims==True:
                                                m1    = self.Sims.GetSimField_Euclid(field1, n, self.lmax)
                                        else:
                                                m1    = self.Sims.GetSimField(field1, n)
                                        betaj = self.Map2Betaj(m1, k, mask=mask, noise=noise, inpainting=inpainting, niter=niter, path_inpainting=path_inpainting)
                                        betaj_sims[k, :] = betaj
                                        k += 1
                        elif (field2 is not None) and (fix_field is None): # Field1 x Field2
                                for n in range(myid, nsim, nproc): 
<<<<<<< HEAD
                                        m1    = self.Sims.GetSimField(field1, n)
                                        m2    = self.Sims.GetSimField(field2, n)
=======
                                        print(f'Simulation number:{n}')
                                        if EuclidSims==True:
                                                m1    = self.Sims.GetSimField_Euclid(field1, n, self.lmax)
                                                m2    = self.Sims.GetSimField_Euclid(field2, n, self.lmax)
                                                print(f'field1={field1}, field2={field2}')
                                        else:
                                                m1    = self.Sims.GetSimField(field1, n)
                                                m2    = self.Sims.GetSimField(field2, n)
>>>>>>> euclid_implementation
                                        #print('sto calcolando betaj')
                                        betaj = self.Map2Betaj(m1,k, map2=m2, mask=mask, noise=noise, inpainting=inpainting, niter=niter, path_inpainting=path_inpainting)
                                        betaj_sims[k, :] = betaj
                                        k += 1
                        elif (field2 is None) and (fix_field is not None): # Field1 x Fix_field (Null-test)
                                for n in range(myid, nsim, nproc): 
                                        if EuclidSims==True:
                                                m1    = self.Sims.GetSimField_Euclid(field1, n, self.lmax)
                                        else:
                                                m1    = self.Sims.GetSimField(field1, n)
                                        betaj = self.Map2Betaj(m1, k, map2=fix_field, mask=mask, noise=noise, inpainting=inpainting, niter=niter, path_inpainting=path_inpainting, nsim=nsim)
                                        betaj_sims[k, :] = betaj
                                        k += 1
                        print(k, dim)

                        #assert (k == int(dim)) #Bianca

                        if nproc > 1:
                                betaj_sims_tot = comm.gather(betaj_sims, root=0)
                                if myid == 0:
                                    betaj_sims_tot = np.vstack((_sims for _sims in betaj_sims_tot)) 
                        else:
                                betaj_sims_tot = betaj_sims

                        if myid == 0: 
                                if fname is not None:
                                        print("...evaluation terminated...")
                                        print("...saving to output " + self.OutDir + fname + "...")
                                        np.savetxt(self.OutDir + fname, betaj_sims_tot)
                        if nproc > 1:
                                comm.Barrier() 
                #print('esco da questa funzione?')
                return betaj_sims_tot

        def GetCovMatrixFromMaps(self, field1, nsim, field2=None, fix_field=None, mask=None, fname=None, fname_sims=None):
                """
                Evaluates covariance (and correlation) matrix for simulated maps of a given field. 

                Parameters
                ----------
                field1 : str
                        Cosmic field to be analyzed => 'kappaT, kappaN, kappaS, deltaT, deltaS'
                nsim  : int
                        Number of simulations to be analyzed
                field2 : str
                        Second cosmic field to be analyzed => 'kappaT, kappaN, kappaS, deltaT, deltaS'
                fix_field : str or Healpix map
                        Map to cross-correlate with nsim simulated maps of field1. This is for *null-tests*.
                mask  : Healpix map
                        (binary) Mask applied to each map
                fname : str
                        File name to store the array. 

                Returns
                -------
                cov_betaj : array
                        array with ((jmax+1, jmax+1)) shape containing covariance matrix
                """
                
                #try:
                #        cov_betaj  = np.loadtxt(self.OutDir + fname, unpack=True)
                #        print("...Covariance Matrix " + self.OutDir + fname + " found...")
                #except:
                #        # print("...simulations beta_jk's " + fname + " not found...")
                #        print("...evaluating Covariance Matrix...")
#
                #        betaj_sims = self.GetBetajSimsFromMaps(field1, nsim, field2=field2, fix_field=fix_field, mask=None, fname=fname_sims)
                #        cov_betaj  = np.cov(betaj_sims.T)       
#
                #        if fname is not None:
                #                print("...saving Covariance Matrix to output " + self.OutDir + fname + "...")
                #                np.savetxt(self.OutDir + fname, cov_betaj, header='Covariance matrix <beta_j1 beta_j2>')
                #        corr_betaj = np.corrcoef(cov_betaj)
#
                #return cov_betaj, corr_betaj

                print("...evaluating Covariance Matrix...")

                betaj_sims = self.GetBetajSimsFromMaps(field1, nsim, field2=field2, fix_field=fix_field, mask=None, fname=fname_sims)
                cov_betaj  = np.cov(betaj_sims.T) 
                #print('fin qui ci sono')
                   
                if fname is not None:
                        print("...saving Covariance Matrix to output " + self.OutDir + fname + "...")
                        np.savetxt(self.OutDir + fname, cov_betaj, header='Covariance matrix <beta_j1 beta_j2>')
               
                corr_betaj = np.corrcoef(betaj_sims.T)

                return cov_betaj, corr_betaj

        def GetDjkFromMaps(self, field, mask, nsim, EuclidSims=False, fname=None):
                """
                Returns MC averaged difference between needlet coefficients w/ and w/o masking applied.
                @see arXiv:0707.0844 (Eq. 7)

                Parameters
                ----------
                field : str
                        Cosmic field to be analyzed => 'kappaT, kappaN, kappaS, deltaT, deltaS'
                mask  : Healpix map
                        (binary) Mask applied to each map
                nsim  : int
                        Number of simulations to be analyzed
                fname : str
                        File name to store the array. 

                Returns
                -------
                Djk : array
                        array with ((jmax+1, npix)) shape containing Djk quantity.
                """
                
                try:
                        Djk = np.load(self.OutDir+fname)
                        print("...Djk array " + self.OutDir + fname + " found...")
                except:
                        # print("...simulations beta_jk's " + fname + " not found...")
                        print("...evaluating Djk array...")

                        delta2_betajk_mean = np.zeros((self.jmax+1, hp.nside2npix(self.Sims.SimPars['nside']))) 
                        betajk2_mean       = np.zeros((self.jmax+1, hp.nside2npix(self.Sims.SimPars['nside'])))

                        for n in range(0,nsim):
                               if EuclidSims==True:
                                       m = self.Sims.GetSimField_Euclid(field, n, self.lmax)
                               else:
                                m = self.Sims.GetSimField(field, n)
                               betajk      = pippo.mylibpy_needlets_f2betajk_healpix_harmonic(m, self.B, self.jmax, self.lmax)
                               betajk_mask = pippo.mylibpy_needlets_f2betajk_healpix_harmonic(m*mask, self.B, self.jmax, self.lmax)
                               delta2_betajk_mean += (betajk_mask - betajk)**2
                               betajk2_mean       += betajk**2
                               del betajk, betajk_mask

                        delta2_betajk_mean /= nsim
                        betajk2_mean       /= nsim

                        Djk = delta2_betajk_mean/betajk2_mean

                        if fname is not None:
                                print("...evaluation terminated...")
                                np.save(self.OutDir + fname, Djk)
                                print("...saved to output " + self.OutDir + fname + "...")

                #return Djk

        def GetBetajMeanFromMaps(self, field1, nsim, field2=None, fix_field=None, mask=None, fname=None, fname_sims=None):
                """
                Evaluates *mean* needlet (auto- or cross-) power spectrum for simulated maps of a given field. 

                Parameters
                ----------
                field1 : str
                        Cosmic field to be analyzed => 'kappaT, kappaN, kappaS, deltaT, deltaS'
                nsim  : int
                        Number of simulations to be analyzed
                field2 : str
                        Second cosmic field to be analyzed => 'kappaT, kappaN, kappaS, deltaT, deltaS'
                fix_field : str or Healpix map
                        Map to cross-correlate with nsim simulated maps of field1. This is for *null-tests*.
                mask  : Healpix map
                        (binary) Mask applied to each map
                fname : str
                        File name to store the array. 

                Returns
                -------
                betaj_mean : array
                        array with (jmax+1) shape containing *mean* needlet auto (or cross) power spectrum.
                
                Notes
                -----
                Evaluates the mean needlet spectra beta_j from nsim.
                It can be used for:
                1. pipeline validation: cycle over field1 (and/or field2)
                2. null test: cycle over field1 and fix_field is the field kept fixed  
                """
                try:
                        betaj_mean = np.loadtxt(self.OutDir + fname, unpack=True, usecols=[1])
                        print("...<beta_j>_MC " + fname + " found...")
                except:
                        # print("...simulations beta_jk's " + fname + " not found...")
                        print("...evaluating <beta_j>_MC...")

                        betaj_sims = self.GetBetajSimsFromMaps(field1, nsim, field2=field2, fix_field=fix_field, mask=mask, fname=fname_sims)
                        betaj_mean = np.mean(betaj_sims, axis=0)

                        if fname is not None:
                                print("...evaluation terminated...")
                                np.savetxt(self.OutDir + fname, np.c_[self.jvec, betaj_mean], header = 'j, <beta_j>_MC averaged over nsim' + str(nsim) + ' MC simulations')
                                print("...saved to output " + self.OutDir + fname + "...")
                
                return betaj_mean

        def GetDjkThresholded(Djk, j, mask=None, threshold=0.1):
                """
                Evaluates *mean* needlet (auto- or cross-) power spectrum for simulated maps of a given field. 

                Parameters
                ----------
                Djk : array
                        Array with ((jmax+1, npix)) shape containing Djk quantity.
                j : int
                        Needlet frequency
                mask  : Healpix map
                        (binary) Mask applied to each map
                threshold : float
                        Cut-off to select pixels with Djk > threshold

                Returns
                -------
                m_cut : Healpix map
                        Binary map (i.e. pixels = 0 or 1) with 1 where pixel > threshold
                """
                m = Djk[j,:]
                if mask is not None:
                        m *= mask
                m_cut = np.zeros(m.size)
                m_cut[m > threshold] = 1.
                return m_cut


        # def GetBetajMeanFromBetajkSims(self, betajk1_sims, nsim, betajk2_sims=None, fix_field=None, mask=None, fname=None):
        #       try:
        #               betaj_mean = np.loadtxt(self.OutDir + fname, unpack=True, usecols=[1])
        #               print("...<beta_j>_MC " + self.OutDir + fname + "found...")
        #       except:
        #               # print("...simulations beta_jk's " + fname + " not found...")
        #               print("...evaluating <beta_j>_MC...")

        #               if (betajk2_sims != None) and (fix_field != None):
        #                       raise ValueError("Options not compatible! => Either you set betajk2_sims or fix_field")

        #               if fix_field is not None:
        #                       if type(fix_field) == str:
        #                               fix_field = hp.read_map(fix_field)
        #                       assert( hp.isnpixok(len(fix_field)) )                           

        #               if (betajk2_sims is None) and (fix_field is None): # Field1 x Field1
        #                       betaj_mean = np.mean(betajk1_sims[:nsim,:self.jmax+1,:]**2, axis=(2,0))
        #               elif (betajk2_sims is not None) and (fix_field is None): # Field1 x Field2
        #                       betaj_mean = np.mean(betajk1_sims[:nsim,:self.jmax+1,:]*betajk2_sims[:nsim,:self.jmax+1,:], axis=(2,0))
        #               elif (field2 is None) and (fix_field is not None): # Field1 x Fix_field (Null-test)
        #                       if mask is not None:
        #                               fix_field = fix_field*mask
        #                       fix_field_betajk = mylibc.needlets_f2betajk_healpix_harmonic(fix_field, self.B, self.jmax, self.lmax)
        #                       betaj_mean = np.mean(betajk1_sims[:nsim,:self.jmax+1,:]*fix_field_betajk[:self.jmax+1,:], axis=(2,0))

        #               print("...mean beta_j evaluated...")

        #               if fname is not None:
        #                       np.savetxt(self.OutDir + fname, np.c_[self.jvec, betaj_mean], header = 'j, <beta_j>_MC averaged over nsim' + str(nsim) + ' MC simulations')
        #                       print("...saved to output " + self.OutDir + fname + "...")

        #       return betaj_mean


        # def GetDjkFromBetajkSims(self, betajk_sims, betajk_sims_mask, nsim, fname=None):
        #       try:
        #               Djk = np.load(self.OutDir + fname)
        #               print("...Djk array " + self.OutDir + fname + "found...")
        #       except:
        #               # print("...simulations beta_jk's " + fname + " not found...")
        #               print("...evaluating Djk...")

        #               delta2_betajk_mean = np.mean((betajk_sims_mask[:nsim,:,:] - betajk_sims[:nsim,:,:])**2, axis=0)
        #               betajk2_mean       = np.mean(betajk_sims[:nsim,:,:]**2, axis=0)

        #               Djk = delta2_betajk_mean/betajk2_mean
                        
        #               if fname is not None:
        #                       print("...evaluation terminated...")
        #                       np.save(self.OutDir + fname, Djk)
        #                       print("...saved to output " + self.OutDir + fname + "...")

        #       return Djk

        # def GetBetajSimsFromBetajkSims(self, betajk1_sims, nsim, betajk2_sims=None, fname=None):#, betajk1_sims=None, betajk2_sims=None, fname=None):
        #       """
        #       field = 'kappaT, kappaN, kappaS, deltaT, deltaS'

        #       2d Array with [Nsim, betaj]

        #       """
        #       if type(betajk1_sims) == str:
        #               betajk1_sims = np.load(self.OutDir + betajk1_sims)      

        #       if betajk2_sims is not None:
        #               if type(betajk2_sims) == str:
        #                       betajk2_sims = np.load(self.OutDir + betajk2_sims)

        #       try:
        #               from mpi4py import MPI
        #               comm = MPI.COMM_WORLD
        #               myid, nproc = comm.Get_rank(), comm.Get_size()
        #       except ImportError:
        #               myid, nproc = 0, 1

        #       dim = (nsim) / nproc
        #       if nproc > 1:
        #               if myid < (nsim) % nproc:
        #                       dim += 1

        #       betaj_sims = np.zeros((dim, self.jmax+1))

        #       # Cycle over simulations 
        #       k = 0
        #       if betajk2_sims is None: # Auto-
        #               for n in range(myid, nsim, nproc): 
        #                       betaj = self.Betajk2Betaj(betajk1_sims[n,:,:])
        #                       betaj_sims[k, :] = betaj
        #                       k += 1
        #       else: # Cross-
        #               for n in range(myid, nsim, nproc): 
        #                       betaj = self.Betajk2Betaj(betajk1_sims[n,:,:], betajk2=betajk2_sims[n,:,:])
        #                       betaj_sims[k, :] = betaj
        #                       k += 1
        #       assert (k == dim)

        #       if nproc > 1:
        #               betaj_sims_tot = comm.gather(betaj_sims, root=0)
        #               if myid == 0:
        #                   betaj_sims_tot = np.vstack((_sims for _sims in betaj_sims_tot)) 
        #       else:
        #               betaj_sims_tot = betaj_sims

        #       if myid == 0: 
        #               if fname is not None:
        #                       print("...evaluation terminated...")
        #                       print("...saving to output " + self.OutDir + fname + "...")
        #                       np.save(self.OutDir + fname, betaj_sims_tot)

        #       comm.Barrier()

        #       return betaj_sims_tot


        # def GetCovMatrixFromBetajkSims(self, betajk1_sims, nsim, betajk2_sims=None, fix_field=None, mask=None, fname=None, cov=False):
        #       betaj_sims = self.GetBetajSimsFromBetajkSims(betajk1_sims, nsim, betajk2_sims=betajk2_sims)

        #       cov_betaj  = np.cov(betaj_sims.T)       
        #       corr_betaj = np.corrcoef(cov_betaj)

        #       if fname is not None:
        #               print("...saving Correlation Matrix to output " + self.OutDir + fname + "...")
        #               np.savetxt(self.OutDir + fname, corr_betaj, header='Covariance matrix <beta_j1 beta_j2>')

        #       if cov:
        #               return corr_betaj, cov_betaj
        #       else:
        #               return corr_betaj
