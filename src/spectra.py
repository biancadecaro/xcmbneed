import numpy as np
import matplotlib.pyplot as plt
from mll import mll
import healpy as hp
import cython_mylibc as pippo

class XCSpectraFile(object):
    """ class to load and store Cls from the output files related to CMB lensing and galaxy field.
        If ncol = 4 -> l, clkg, clgg, clkk
           ncol = 5 -> l, clkg, clgg, clkk, nlkk
           ncol = 8 -> l, clkg, clkmu, clgg, clgmu, clmumu, clkk, nlkk
    """
    def __init__(self, clfname, WantTG,  lmin=None, lmax=None, b=1, alpha=1 ): 
        """ load Cls.
            ! All spectra must be passed from lmin = 0 !
             * clfname          = file name to load from.
             * (optional) lmax  = maximum multipole to load (all multipoles in file will be loaded by default).
             * (optional) bias  = linear galaxy bias.
             * (optional) alpha = number counts slope N(>F) \propto F^{-\alpha}.
        """
        self.clfname = clfname

        if WantTG == True:
            data = np.genfromtxt(clfname)
            self.ell = data[0].astype(int)
                        
            if lmin == None:
                lmin = data[0][0].astype(int)
                
            
            if lmax == None:
                lmax = data[0][-1].astype(int)
        
            self.lmax = lmax 
            self.lmin = lmin
            #self.ell   = np.arange(lmin, lmax, dtype=np.float) #lmax+1
        
            print( 'lmin =', lmin)
            print( 'lmax =', lmax)

            self.ell = np.arange(self.lmin, self.lmax+1)
            self.cltt = data[1]#np.zeros((lmin,(lmax+1))) #np.divide(data[1][lmin:(lmax+1)],(self.ell*(self.ell+1)/(2*np.pi)))
            self.cltg =  data[2]#np.zeros((lmin,(lmax+1))) #np.divide(data[2][lmin:(lmax+1)],(self.ell*(self.ell+1)/(2*np.pi)))
            self.clg1g1 = data[3]#np.zeros((lmin,(lmax+1))) #np.divide(data[3][lmin:(lmax+1)],(self.ell*(self.ell+1)/(2*np.pi)))
            
            #self.cltt = data[1][lmin:]#np.divide(data[1][lmin:(lmax+1)],(self.ell*(self.ell+1)/(2*np.pi)))#np.divide(data[1][lmin:(lmax+1)],(ell*(ell+1)))#np.zeros((lmin,(lmax+1))) #
            #self.cltg =  data[2][lmin:]#np.divide(data[2][lmin:(lmax+1)],(self.ell*(self.ell+1)/(2*np.pi)))#np.divide(data[2][lmin:(lmax+1)],(ell*(ell+1)))#np.zeros((lmin,(lmax+1))) #
            #self.clg1g1 = data[3][lmin:]#np.divide(data[3][lmin:(lmax+1)],(self.ell*(self.ell+1)/(2*np.pi)))#np.divide(data[3][lmin:(lmax+1)],(ell*(ell+1)))#np.zeros((lmin,(lmax+1))) #
            print(self.ell.shape, self.cltg.shape, self.cltt.shape, self.clg1g1.shape)


            #print(self.cltg.shape, self.ell, self.ell.shape, lmin, lmax)
            #self.cltg_tot = self.get_kg_tot(b=b, alpha=1)
            #self.clgg_tot = self.get_gg_tot(b=b, alpha=1)
            #for l in range(lmin, lmax-1):
            #	print(l)
            #	self.cltt[l] = self.cltt[l]/(l*(l+1))#[lmin:(lmax+1)]
            #	self.cltg[l] = self.cltg[l]/(l*(l+1))#[lmin:(lmax+1)]
            #	self.clg1g1[l] = self.clg1g1[l]/(l*(l+1))#[lmin:(lmax+1)]
            #print(self.ell.shape, self.cltg.shape, self.cltt.shape, self.clg1g1.shape)
            #header = 'ell, TT, TG, GG'
            #np.savetxt('spectra/cl_spectra.dat', np.array([self.ell,self.cltt, self.cltg, self.clg1g1]), header = header )
        
        else:
            clarray = np.loadtxt(clfname)
            assert(int(clarray[0,0]) == 0)
    
            if lmin == None:
                lmin = 0
    
            if lmax == None:
                lmax = int(clarray[-1,0]) - 1
    
            print( 'lmin =', lmin)
            print( 'lmax =', lmax)
    
            ncol = np.shape(clarray)[1]
    
            self.lmax = lmax
            self.ls   = np.arange(lmin, lmax+1, dtype=np.float)
            
    
            if ncol == 4:                                                                            
                self.clkg = clarray[lmin:(lmax+1),1]
                self.clgg = clarray[lmin:(lmax+1),2]
                self.clkk = clarray[lmin:(lmax+1),3]
    
            elif ncol == 5:                                                                          
                self.clkg = clarray[lmin:(lmax+1),1]
                self.clgg = clarray[lmin:(lmax+1),2]
                self.clkk = clarray[lmin:(lmax+1),3]
                self.nlkk = clarray[lmin:(lmax+1),4]
    
            elif ncol == 8:                                                                          
                self.clkg   = clarray[lmin:(lmax+1),1]
                self.clkmu  = clarray[lmin:(lmax+1),2]
                self.clgg   = clarray[lmin:(lmax+1),3]
                self.clgmu  = clarray[lmin:(lmax+1),4]
                self.clmumu = clarray[lmin:(lmax+1),5]
                self.clkk   = clarray[lmin:(lmax+1),6]
                self.nlkk   = clarray[lmin:(lmax+1),7]
            print(self.ls, lmax, self.ls.shape, self.clkg.shape)

            self.clkg_tot = self.get_kg_tot(b=b, alpha=alpha)
            self.clgg_tot = self.get_gg_tot(b=b, alpha=alpha)

    def get_kg_tot(self, b=1., alpha=1.):
        return b*self.clkg + b*(alpha-1)*self.clkmu

    def get_gg_tot(self, b=1., alpha=1.):
        return b**2*self.clgg + 2*b*(alpha-1)*self.clgmu + (alpha-1)**2*self.clmumu

    def get_s2n_ell(self, spec, b=1., alpha=1., fsky=1., ngg=0., lmax=None):
        if lmax is None: lmax = self.lmax
        if spec == 'kg':
            kg = self.get_kg_tot(b=b, alpha=alpha)
            gg = self.get_gg_tot(b=b, alpha=alpha)  
            if ngg == 0.:
                gg_tot = gg
            else:
                gg_tot = gg + 1./ngg
            kk_tot   = self.clkk + self.nlkk
            delta_cl = np.sqrt(1./(2.*(self.ls+1.)*fsky)*(kg**2 + kk_tot*gg_tot))        
            # sig = getattr(self, spec)  
            s2n_ell = kg/delta_cl
            s2n_ell[np.isnan(s2n_ell)] = 0.
            return s2n_ell

    def get_s2n_cum(self, spec, b=1., alpha=1., fsky=1., ngg=0., lmin=0, lmax=None):
        if lmax is None: lmax = self.lmax
        s2n_ell = self.get_s2n_ell(spec, b=b, alpha=alpha, fsky=fsky, ngg=ngg, lmax=lmax)
        return np.array([np.sqrt(np.sum(s2n_ell[int(lmin):i+1]**2)) for i in range(int(lmin),int(lmax)+1)])

    # def copy(self, lmax=None, lmin=None):
    #     """ clone this object.
    #          * (optional) lmax = restrict copy to L<=lmax.
    #          * (optional) lmin = set spectra in copy to zero for L<lmin.
    #     """
    #     if (lmax == None):
    #         return copy.deepcopy(self)
    #     else:
    #         assert( lmax <= self.lmax )
    #         ret      = copy.deepcopy(self)
    #         ret.lmax = lmax
    #         ret.ls   = np.arange(0, lmax+1)
    #         for k, v in self.__dict__.items():
    #             if k[0:2] == 'cl':
    #                 setattr( ret, k, copy.deepcopy(v[0:lmax+1]) )

    #         if lmin != None:
    #             assert( lmin <= lmax )
    #             for k in self.__dict__.keys():
    #                 if k[0:2] == 'cl':
    #                     getattr( ret, k )[0:lmin] = 0.0
    #         return ret

    # def hashdict(self):
    #     """ return a dictionary uniquely associated with the contents of this clfile. """
    #     ret = {}
    #     for attr in ['lmax', 'clkg', 'clkmu', 'clgg', 'clgmu', 'clmumu', 'clkk', 'nlkk']:
    #         if hasattr(self, attr):
    #             ret[attr] = getattr(self, attr)
    #     return ret


    def plot(self, spec='clkg', p=plt.plot, t=lambda l:1., **kwargs):
        """ plot the spectrum
             * spec = spectrum to display (e.g. cltt, clee, clte, etc.)
             * p    = plotting function to use p(x,y,**kwargs)
             * t    = scaling to apply to the plotted Cl -> t(l)*Cl
        """
        p( self.ls, t(self.ls) * getattr(self, spec), **kwargs )

class NeedletTheory(object):
    """ class to compute theoretical quantities related to needlet, i.e. needlet power spectrum
        given an angular power spectrum Cl
    """
    def __init__(self, B, mergej=None, npoints=1000):
        """ 
        * B       = Needlet width parameter
        * npoints = # of points to sample the integrals
        """
        self.B = B
        self.npoints = npoints
        self.norm = self.get_normalization()
    
    def get_bneed(self,jmax, lmax, mergej=None):
        self.b_values = pippo.mylibpy_needlets_std_init_b_values(self.B, jmax, lmax)
        if mergej:
            self.b_values = pippo.mylibpy_needlets_std_init_b_values_mergej(self.B, jmax, lmax, mergej)
        return self.b_values
    
    def get_jvec(self):
        self.jvec = np.arange(self.b_values.shape[0])
        return self.jvec

    def ell_binning(self, jmax, lmax):#, ell):
        """
        Returns the binning scheme in  multipole space
        """
        #assert(np.floor(self.B**(jmax+1)) <= ell.size-1) 
        
        bjl  = self.b_values**2#np.zeros((jmax+1,lmax+1))
        ell  = np.arange(lmax+1)*np.ones((bjl.shape[0], lmax+1))
        ellj =np.zeros((bjl.shape[0], lmax+1))

        for j in range(bjl.shape[0]):
            #b2 = self.b_need(ell[j]/self.B**j)**2
            #b2[np.isnan(b2)] = 0.
            bjl[j,:][bjl[j,:]!=0] = 1.
            #bjl[j,:] = b2
            ellj[j,:] = ell[j,:]*bjl[j,:]
        return ellj 

    def get_B_parameter(self):
        """
        Returns the D parameter for needlets
        """
        return np.power(self.lmax, 1./self.jmax)

    def f_need(self, t):
        """
        Standard needlets f function
        @see arXiv:0707.0844
        """
        # WARN: This *vectorized* version works only for *arrays*
        good_idx = np.logical_and(-1. < t, t < 1.)
        f1 = np.zeros(len(t))
        f1[good_idx] = np.exp(-1./(1.-(t[good_idx]*t[good_idx])))
        return f1

    def get_normalization(self):
        """
        Evaluates the normalization of the standard needlets function
        @see arXiv:0707.0844
        """
        from scipy.integrate import simps
        
        t = np.linspace(-1,1,self.npoints)
        return simps(self.f_need(t), t)

    def psi_need(self, u):
        """
        Standard needlets Psi function
        @see arXiv:0707.0844
        """
        from scipy.integrate import simps
           
        # u_ = np.linspace(-1.,u,self.npoints)
        return [simps(self.f_need(np.linspace(-1.,u_,self.npoints)), np.linspace(-1.,u_,self.npoints))/self.norm for u_ in u]

    def phi_need(self, t):
        """
        Standard needlets Phi function
        @see arXiv:0707.0844
        """
        from scipy.integrate import simps

        left_idx = np.logical_and(0 <= t, t <= 1./self.B)
        cent_idx = np.logical_and(1./self.B < t, t < 1.)
        rite_idx = t > 1.

        phi = np.zeros(len(t))
        phi[left_idx] = 1.
        phi[cent_idx] = self.psi_need(1.-2.*self.B/(self.B-1.)*(t[cent_idx]-1./self.B))
        phi[rite_idx] = 0.

        return phi

    def b_need(self, xi):
        """
        Standard needlets windows function
        @see arXiv:0707.0844
        """
        return np.sqrt(np.abs(self.phi_need(xi/self.B)-self.phi_need(xi)))

    def cl2betaj(self, jmax, cl, lmax):
        """
        Returns needlet power spectrum \beta_j given an angular power spectrum Cl.
        """
        #assert(np.floor(self.B**(jmax+1)) <= cl.size-1) 
        #print( np.floor(self.B**(jmax+1)), cl.size-1)
        
        ell = np.arange(0, lmax+1)
        betaj = np.zeros(jmax+1)
        bjl = np.zeros((jmax+1, lmax+1))
        for j in range(jmax+1):
            #lmin = np.floor(self.B**(j-1))
            #lmax = np.floor(self.B**(j+1))
            #if lmax > cl.size-1:
            #    lmax=cl.size-1
            #ell  = np.arange(lmin, lmax+1, dtype=np.int)
            #b2   = self.b_need(ell/self.B**j)*self.b_need(ell/self.B**j)
            b2 = self.b_need(ell/self.B**j)**2
            b2[np.isnan(b2)] = 0.
            bjl[j, :] = b2
            betaj[j] = np.sum(b2*(2.*ell+1.)/4./np.pi*cl[ell])
        
        return betaj

    def get_Mll(self, wl, lmax=None):
        """
        Returns the Coupling Matrix M_ll from l = 0 (Hivon et al. 2002)

        Notes
        -----
        M_ll.shape = (lmax+1, lmax+1)
        """
        if lmax == None:
            lmax = wl.size-1
        assert(lmax <= wl.size-1)
        return np.float64(mll.get_mll(wl[:lmax+1], lmax))

    def gamma_bf(self, wl, jmax, lmax):
        """
        Returns the Gamma Matrix \gamma_lj from Domenico's notes [Brute force calculation]

        Notes
        -----
        gamma_lj.shape = (lmax+1, jmax+1)
        """
        Mll = self.get_Mll(wl, lmax=lmax)
        blj = np.zeros((lmax+1,jmax+1))
        ell = np.arange(0, lmax+1, dtype=np.int)

        # Brute force
        for l in range(lmax+1):
            for j in range(jmax+1):
                b2 = self.b_need(ell/self.B**j)*self.b_need(ell/self.B**j)
                fact = b2*(2*ell+1.)
                blj[l,j] = np.dot(Mll[:,l], fact)

        # for j in range(jmax+1):
        #     lmin = np.floor(self.B**(j-1))
        #     lmax = np.floor(self.B**(j+1))
        #     ell  = np.arange(lmin, lmax+1, dtype=np.int)
        #     b2   = self.b_need(ell/self.B**j)*self.b_need(ell/self.B**j)
        #     betaj[:,j] = b2*(2.*ell+1.)

        return blj

    def gamma_np(self, wl, jmax, lmax):
        """
        Returns the Gamma Matrix \gamma_lj from Domenico's notes

        Notes
        -----
        gamma_lj.shape = (lmax+1, jmax+1)
        """
        Mll  = self.get_Mll(wl, lmax=lmax)
        ell  = np.arange(0, lmax+1, dtype=np.int)
        blj  = np.zeros((lmax+1,jmax+1))
        for j in range(jmax+1):
            b2 = self.b_need(ell/self.B**j)*self.b_need(ell/self.B**j)
            blj[:,j] = b2*(2*ell+1.) 
        return np.dot(Mll.T, blj)

    def gammaJ(self, cl, wl, jmax, lmax):
        """
        Returns the \Gamma_j vector from Domenico's notes

        Notes
        -----
        gamma_lj.shape = (lmax+1, jmax+1)
        """
        Mll  = self.get_Mll(wl, lmax=lmax)
        ell  = np.arange(0, lmax+1, dtype=np.int)
        bjl  = self.b_values**2#np.zeros((jmax+1,lmax+1))
        #filename = f'b_need/bneed_lmax{lmax}_jmax{jmax}_B{self.B:0.2f}.dat'
        #for j in range(jmax+1):
        #    b2 = self.b_need(ell/self.B**j)**2
        #    b2[np.isnan(b2)] = 0.
        #    bjl[j,:] = b2
        #np.savetxt(filename, bjl) 
        return (bjl*(2*ell+1.)*np.dot(Mll, cl[:lmax+1])).sum(axis=1)/(4*np.pi)#np.dot(bjl, np.dot(Mll, cl[:lmax+1]))/(4*np.pi)
    
    def sigmaJ(self, cl, wl, jmax, lmax):
        """
        Returns the \sigma_j from Domenico's notes

        Notes
        -----
        gamma_lj.shape = (lmax+1, jmax+1)
        """
        gammalj = self.gamma_np(wl, jmax, lmax)
        ell     = np.arange(0, lmax+1, dtype=np.int)
        return np.sqrt(np.dot(gammalj.T, 2.*cl[:lmax+1]**2/(2*ell+1)))


    #def delta_beta_j_cov(self, jmax, cltg, cltt, clgg, cov):
    #	"""
    #		Returns the \delta beta_j from https://arxiv.org/abs/astro-ph/0606475
    #		eq 18
    #	"""
    #	delta_beta_j_squared = np.zeros(jmax+1)
    #	cov_diag = np.zeros(jmax+1)
    #	for j in range(jmax+1):
    #		l_min = np.floor(self.B**(j-1))
    #		l_max = np.floor(self.B**(j+1))
    #		ell = np.arange(l_min, l_max+1, dtype=np.int)
    #		#delta_beta_j_squared[j] = np.sum(((2*ell+1)/(16*np.pi**2))*(self.b_need(ell/self.B**j)**4)*(cltg[ell]**2 + cltt[ell]*clgg[ell]))
    #		cov_diag[j] = cov[j][j]
    #	#print(cov, cov_diag)
    #	return np.sqrt(cov_diag)

    def delta_beta_j(self, jmax, lmax,cltg, cltt, clgg,  noise_gal_l=None):
            """
                Returns the \delta beta_j from https://arxiv.org/abs/astro-ph/0606475
                eq 18
            """
            delta_beta_j_squared = np.zeros(jmax+1)
            if noise_gal_l is not None:
                clgg_tot = clgg+noise_gal_l
            else:
                clgg_tot = clgg
    
            #for j in range(jmax+1):
            #    l_min = np.floor(self.D**(j-1))
            #    l_max = np.floor(self.D**(j+1))
            #    if l_max > cltg.size-1:
            #        l_max=cltg.size-1
            #    ell = np.arange(l_min,l_max+1, dtype=np.int)
            #    delta_beta_j_squared[j] = np.sum(((2*ell+1)/(16*np.pi**2))*(self.b_need(ell/self.D**j)**4)*(cltg[ell]**2 + cltt[ell]*clgg_tot[ell]))
#
            ell = np.arange(0,lmax+1)
            bjl = np.zeros((jmax+1, lmax+1))
            for j in range(jmax+1):
                b2 = self.b_need(ell/self.B**j)**2
                b2[np.isnan(b2)] = 0.
                bjl[j, :] = b2
                delta_beta_j_squared[j]= np.sum(((2*ell+1)/(16*np.pi**2))*(b2**2)*(cltg[ell]**2 + cltt[ell]*clgg_tot[ell]))
            return np.sqrt(delta_beta_j_squared)
    


    def variance_gammaj(self, cltg,cltt, clgg, wl, jmax, lmax, noise_gal_l=None):
        """
        Returns the Cov(\Gamma_j, \Gamma_j') 
        Notes
        -----
        Cov(gamma)_jj'.shape = (jmax+1, jmax+1)
        """
        if noise_gal_l is not None:
            clgg_tot = clgg+noise_gal_l
        else:
            clgg_tot = clgg

        Mll  = self.get_Mll(wl, lmax=lmax)
        ell  = np.arange(lmax+1, dtype=int)
        bjl  = self.b_values**2*(2*ell+1.) #np.zeros((jmax+1,lmax+1))
        #delta_gammaj = np.zeros((jmax+1, jmax+1))
        #for j in range(jmax+1):
        #    b2 = self.b_need(ell/self.B**j)**2
        #    b2[np.isnan(b2)] = 0.
        #    bjl[j,:] = b2*(2*ell+1.) 
        covll = np.zeros((lmax+1, lmax+1))
        for ell1 in range(lmax+1):
            for ell2 in range(lmax+1):
                covll[ell1,ell2] = Mll[ell1,ell2]*(cltg[ell1]*cltg[ell2]+np.sqrt(cltt[ell1]*cltt[ell2]*clgg_tot[ell1]*clgg_tot[ell2]))/(2.*ell1+1)
        delta_gammaj = np.dot(bjl, np.dot(covll, bjl.T))
        return delta_gammaj/(4*np.pi)**2
