import numpy as np

class NeedletTheory(object):
    """ class to compute theoretical quantities related to needlet, i.e. needlet power spectrum
        given an angular power spectrum Cl
    """
    def __init__(self,jmax,lmax, npoints=1000):
        """ 
        * B       = Needlet width parameter
        * npoints = # of points to sample the integrals
        """
        self.jmax = jmax
        self.lmax =lmax
        self.B = self.get_B_parameter(self.jmax,self.lmax)
        print(self.B)
        self.npoints = npoints
        self.norm = self.get_normalization()

    def get_B_parameter(self, jmax, lmax):
        return np.power(lmax, 1./jmax)

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

    def cl2betaj(self, jmax, cl):
        """
        Returns needlet power spectrum \beta_j given an angular power spectrum Cl.
        """
        
        #print(cl.size)
        #print(np.floor(self.B**(jmax+1)))

        assert(np.floor(self.B**(jmax+1)) <= cl.size-1) 
        #print( np.floor(self.B**(jmax+1)), cl.size-1)
 
        betaj = np.zeros(jmax+1)
        for j in range(jmax+1):
            lmin = np.floor(self.B**(j-1))
            lmax = np.floor(self.B**(j+1))
            ell  = np.arange(lmin, lmax+1, dtype=np.int)
            #print(lmin, lmax, j)
            b2   = self.b_need(ell/self.B**j)*self.b_need(ell/self.B**j)
            betaj[j] = np.sum(b2*(2.*ell+1.)/4./np.pi*cl[ell])
        return betaj

    def delta_beta_j(self, jmax, cltg, cltt, clgg):
            """
                Returns the \delta beta_j from https://arxiv.org/abs/astro-ph/0606475
                eq 18
            """
            delta_beta_j_squared = np.zeros(jmax+1)
            
            for j in range(jmax+1):
                l_min = np.floor(self.B**(j-1))
                l_max = np.floor(self.B**(j+1))
                ell = np.arange(l_min,l_max+1, dtype=np.int)
                
                delta_beta_j_squared[j] = np.sum(((2*ell+1)/(16*np.pi**2))*(self.b_need(ell/self.B**j)**4)*(cltg[ell]**2 + cltt[ell]*clgg[ell]))
            
            return np.sqrt(delta_beta_j_squared)