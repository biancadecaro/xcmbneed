import numpy as np

class NeedletTheory(object):
    """ class to compute theoretical quantities related to needlet, i.e. needlet power spectrum
        given an angular power spectrum Cl
    """
    def __init__(self,jmax,lmax, npoints=1000):
        """ 
<<<<<<< HEAD
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

=======
        * jmax    = Number of bin 
        * lmax    = Maximum multipole for the analysis
        * B       = Needlet width parameter
        * npoints = Number of points to sample the integrals
        """
        self.jmax = jmax
        self.lmax =lmax
        self.D = self.get_D_parameter(self.jmax,self.lmax)
        self.npoints = npoints
        self.norm = self.get_normalization()
        self.jvec = np.arange(jmax+1)

    def get_D_parameter(self, jmax, lmax):
        """
        Returns the D parameter for needlets
        """
        return np.power(lmax, 1./jmax)

    def ell_binning(self, jmax, lmax, ell):
        """
        Returns the binning scheme in  multipole space
        """
        assert(np.floor(self.D**(jmax+1)) <= ell.size-1) 
        ellj = []
        for j in range(jmax+1):
            lmin = np.floor(self.D**(j-1))
            lmax = np.floor(self.D**(j+1))
            ell1  = np.arange(lmin, lmax+1, dtype=int)
            ellj.append(ell1)
        return np.asarray(ellj, dtype=object)

>>>>>>> euclid_implementation
    def f_need(self, t):
        """
        Standard needlets f function
        @see arXiv:0707.0844
        """
<<<<<<< HEAD
        # WARN: This *vectorized* version works only for *arrays*
=======
>>>>>>> euclid_implementation
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

<<<<<<< HEAD
        left_idx = np.logical_and(0 <= t, t <= 1./self.B)
        cent_idx = np.logical_and(1./self.B < t, t < 1.)
=======
        left_idx = np.logical_and(0 <= t, t <= 1./self.D)
        cent_idx = np.logical_and(1./self.D < t, t < 1.)
>>>>>>> euclid_implementation
        rite_idx = t > 1.

        phi = np.zeros(len(t))
        phi[left_idx] = 1.
<<<<<<< HEAD
        phi[cent_idx] = self.psi_need(1.-2.*self.B/(self.B-1.)*(t[cent_idx]-1./self.B))
=======
        phi[cent_idx] = self.psi_need(1.-2.*self.D/(self.D-1.)*(t[cent_idx]-1./self.D))
>>>>>>> euclid_implementation
        phi[rite_idx] = 0.

        return phi

    def b_need(self, xi):
        """
        Standard needlets windows function
        @see arXiv:0707.0844
        """
<<<<<<< HEAD
        return np.sqrt(np.abs(self.phi_need(xi/self.B)-self.phi_need(xi)))
=======
        return np.sqrt(np.abs(self.phi_need(xi/self.D)-self.phi_need(xi)))
>>>>>>> euclid_implementation

    def cl2betaj(self, jmax, cl):
        """
        Returns needlet power spectrum \beta_j given an angular power spectrum Cl.
<<<<<<< HEAD
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
=======
        @see eq 2.17 https://arxiv.org/abs/1607.05223
        """

        assert(np.floor(self.D**(jmax+1)) <= cl.size-1) 
 
        betaj = np.zeros(jmax+1)
        for j in range(jmax+1):
            lmin = np.floor(self.D**(j-1))
            lmax = np.floor(self.D**(j+1))
            ell  = np.arange(lmin, lmax+1, dtype=int)
            b2   = self.b_need(ell/self.D**j)*self.b_need(ell/self.D**j)
>>>>>>> euclid_implementation
            betaj[j] = np.sum(b2*(2.*ell+1.)/4./np.pi*cl[ell])
        return betaj

    def delta_beta_j(self, jmax, cltg, cltt, clgg):
            """
<<<<<<< HEAD
                Returns the \delta beta_j from https://arxiv.org/abs/astro-ph/0606475
                eq 18
=======
                Returns the \delta beta_j (variance) 
                @see eq 2.19 https://arxiv.org/abs/1607.05223
                
>>>>>>> euclid_implementation
            """
            delta_beta_j_squared = np.zeros(jmax+1)
            
            for j in range(jmax+1):
<<<<<<< HEAD
                l_min = np.floor(self.B**(j-1))
                l_max = np.floor(self.B**(j+1))
                ell = np.arange(l_min,l_max+1, dtype=np.int)
                
                delta_beta_j_squared[j] = np.sum(((2*ell+1)/(16*np.pi**2))*(self.b_need(ell/self.B**j)**4)*(cltg[ell]**2 + cltt[ell]*clgg[ell]))
            
            return np.sqrt(delta_beta_j_squared)
=======
                l_min = np.floor(self.D**(j-1))
                l_max = np.floor(self.D**(j+1))
                ell = np.arange(l_min,l_max+1, dtype=int)
                
                delta_beta_j_squared[j] = np.sum(((2*ell+1)/(16*np.pi**2))*(self.b_need(ell/self.D**j)**4)*(cltg[ell]**2 + cltt[ell]*clgg[ell]))
            
            return np.sqrt(delta_beta_j_squared)
    
######################################

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import os
    # Matplotlib defaults ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #rc('text',usetex=True)
    #rc('font',**{'family':'serif','serif':['Computer Modern']})
    plt.rcParams['axes.linewidth']  = 5.
    plt.rcParams['axes.labelsize']  =30
    plt.rcParams['xtick.labelsize'] =30
    plt.rcParams['ytick.labelsize'] =30
    plt.rcParams['xtick.major.size'] = 30
    plt.rcParams['ytick.major.size'] = 30
    plt.rcParams['xtick.minor.size'] = 30
    plt.rcParams['ytick.minor.size'] = 30
    plt.rcParams['legend.fontsize']  = 'large'
    plt.rcParams['legend.frameon']  = False
    plt.rcParams['axes.labelsize'] = 'large'
    plt.rcParams['axes.titlesize'] = 'large'
    plt.rcParams["errorbar.capsize"] = 15
    #
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['font.size'] = 40
    plt.rcParams['lines.linewidth']  = 5.
    #plt.rcParams['backend'] = 'WX'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # embed()

    jmax=10
    lmax=256

    need_theory = NeedletTheory(lmax=lmax,jmax=jmax)
    print(f'D={need_theory.D}')

    cl_theory = np.loadtxt('/ehome/bdecaro/xcmbneed/src/spectra/inifiles/EUCLID_fiducial.dat')
    ell_theory = cl_theory[0]
    cl_theory_tt = cl_theory[1]
    cl_theory_tg = cl_theory[2]
    cl_theory_gg = cl_theory[3]

    betatg = need_theory.cl2betaj(jmax, cl=cl_theory_tg)
    delta = need_theory.delta_beta_j(jmax=jmax, cltg=cl_theory_tg, cltt=cl_theory_tt,clgg=cl_theory_gg)

    ell_bin = need_theory.ell_binning(jmax=jmax, lmax=lmax, ell =ell_theory)
    for j in range(1,jmax):
        print(f'Bin {j}={ell_bin[j]}')
    

    fig = plt.figure(figsize=(27,20))

    plt.suptitle(r'$D = %1.2f $' %need_theory.D + r' $\ell_{max} =$'+str(lmax) + r' $j_{max} = $'+str(jmax))
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(need_theory.jvec, betatg,  yerr=delta, color='firebrick', fmt='o', ms=10,capthick=5, label=r'theory')
    ax.legend(loc='best')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_xlabel(r'$j$')
    ax.set_ylabel(r'$\beta_j^{Tgal}$')
    fig.tight_layout()
    plt.savefig(f'betaj_theory_T_gal_noise_jmax{jmax}_D{need_theory.D:0.2f}_lmax{lmax}.png', bbox_inches='tight')
>>>>>>> euclid_implementation
