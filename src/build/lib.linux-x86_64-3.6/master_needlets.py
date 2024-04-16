import numpy as np
import healpy as hp
from numpy import linalg as la
from mll import mll
from spectra import NeedletTheory


__all__ = ['Binner_needlets', 'Master_nesdlets']

arcmin2rad = 0.000290888208666

class Binner_needlets(object):
    """
    Class for computing binning scheme.
    """
    def __init__(self, B, lmin, lmax, jmax, flattening=None):
        """
        Parameters
        ----------
        lmin : int
            Lower bound of the first l bin.
        lmax : int
            Highest l value to be considered. The inclusive upper bound of
            the last l bin is lesser or equal to this value.
        delta_ell :
            The l bin width.
        flattening : str
            Power spectrum flattening type. Default = None
            Possible types:
            1.  None: fact = 1 
            2. 'Ell': fact = l           
            3. 'CMBLike': fact = l*(l+1)/2\pi 
            TODO: IMPLEMENT OTHER TYPES
        """
        
        lmin = int(lmin)
        lmax = int(lmax)
        #if lmin < 1:
        #    raise ValueError('Input lmin is less than 1.')
        #if lmax < lmin:
        #    raise ValueError('Input lmax is less than lmin.')
        self.lmin = lmin
        self.lmax = lmax
        self.flattening = flattening

        #delta_ell      = int(delta_ell)
        #self.delta_ell = delta_ell
        self.jmax = jmax
        self.B = B
        self.P_jl, self.Q_lj = self._bin_ell() #self.ell_binned, self.P_jl, self.Q_lj

        

    def _bin_ell(self):
        #nbins = (self.lmax - self.lmin + 1) // self.delta_ell
        #start = self.lmin + np.arange(nbins) * self.delta_ell
        #stop  = start + self.delta_ell
        need_theory = NeedletTheory(self.B)
        #ell = np.arange(self.lmin, self.lmax+1)
        #ell_binned = need_theory.ell_binning(self.lmin, self.lmax, ell)
        j_binned = np.arange(start=1,stop=self.jmax+1) # ! Starting from 1


        if self.flattening is None:
            flat = np.ones(self.lmax + 1)
        elif self.flattening in ['Ell', 'ell']:
            flat = np.arange(self.lmax + 1)
        elif self.flattening in ['CMBLike', 'cmblike', 'CMBlike']:
            flat = np.arange(self.lmax + 1)
            flat = flat * (flat + 1) / (2 * np.pi)
        else:
            msg = self.flattening + ' is not a flattening style'
            raise RuntimeError(msg)

        _P_jl = np.zeros((self.jmax, self.lmax+1 ))
        _Q_lj = np.zeros((self.lmax+1 , self.jmax))
        #b_sum = np.zeros(self.lmax+1)
        for _j, j in enumerate(j_binned):
            #print(_j, j)
            lminj = np.ceil(self.B**(j-1))
            lmaxj = np.floor(self.B**(j+1))
            if j == self.jmax:
                ellj  = np.arange(lminj, self.lmax+1, dtype=np.int)
            else:
                ellj  = np.arange(lminj, lmaxj+1, dtype=np.int)
            #troncare il j=12 fino a ell = lmax 
            print(f'j={j}. lminj={ellj[0]}, lmaxj={ellj[-1]}, ellj.shape={ellj.shape[0 ]}')
            b = need_theory.b_need(ellj/self.B**j)
            b2   = b**2
            _P_jl[_j,  ellj] =b2*(2. * ellj+1)/(ellj.shape[0])/(4.*np.pi)#b2*(2. * ellj+1) / (4.*np.pi *(ellj[0]-ellj[-1])) #b2*(1. * flat[ellj] / (lminj - lmaxj))
            _Q_lj[ellj, _j] = 4.*np.pi / (b2*(2. * ellj+1))    #b2*(1. / flat[ellj])
        #print(b_sum) controllato fa 1 per ogni ell > j

        #b_need2= mylibc.mylibpy_needlets_std_init_b_values(B, jmax, lmax)**2

        #ell_binned_j = np.zeros((jmax+1,lmax+1))
        #ell_j = np.zeros((jmax+1,lmax+1))
        #for j in range(jmax+1):
        #    #print(_j, j)
        #    lminj = np.ceil(B**(j-1))
        #    lmaxj = np.floor(B**(j+1))
        #    if j == jmax:
        #        ellj  = np.arange(lminj, lmax+1, dtype=np.int)
        #    else:
        #        ellj  = np.arange(lminj, lmaxj+1, dtype=np.int)
        #    ell_binned_j[j,:] = ellj.shape[0]*np.ones(lmax+1)
        #    ell_j[j,ellj] = ellj
        #
        #P_jl = (1./ell_binned_j)*b_need2*(2.*ell_j+1)/(4*np.pi)
        #Q_lj = (4*np.pi)/(b_need2*(2.*ell_j+1))
        #print(Q_lj

        return _P_jl, _Q_lj #ell_binned

    def bin_spectra(self, spectra): #questa non dovrebbe servire
        """
        Average spectra in bins specified by lmin, lmax and delta_ell,
        weighted by flattening term specified in initialization.

        """
        spectra = np.asarray(spectra)
        jmax    = spectra.shape[-1] - 1
        if jmax < self.jmax:
            raise ValueError('The input spectra do not have enough j.')

        return np.dot(self.P_jl, spectra[..., :self.jmax+1])


class Master_needlets(Binner_needlets):
    """
    Class to estimate Cross- and Auto-power spectra using the Xspect/MASTER method. 
    Hivon et al. 2002, Tristam et al. 2004.
    It also implements the fsky_approximation.
    """

    def __init__(self,B ,lmin, lmax, jmax, mask, flattening=None, 
                 pixwin=False, fwhm_smooth=None, fsky_approx=False):
        """
        Parameters
        ----------
        fwhm_smooth arcmin

        # TODO: implement second mask
        """
        super(Master_needlets, self).__init__(B,lmin, lmax, jmax, flattening)

        mask        = np.asarray(mask)
        self.mask   = mask
        self.pixwin = pixwin
        self.nside  = hp.npix2nside(mask.size)
        self.fwhm_smooth = fwhm_smooth
        self.fsky_approx = fsky_approx

        self.fsky = self._get_ith_mask_moment(2)
        print(f'fsky={self.fsky}')

        # TODO: implement pixwin and beam for needlets
        #if pixwin:
        #    self.pw2_l  = ((hp.pixwin(self.nside))**2)[:lmax+1]
        #    self.pw2_ll = np.diag(self.pw2_l)
#
        #if fwhm_smooth is not None:
        #    if np.size(fwhm_smooth) != 0: 
        #        self.B_1_l = hp.gauss_beam(fwhm_smooth[0] * arcmin2rad, lmax=self.lmax)
        #        self.B_2_l = hp.gauss_beam(fwhm_smooth[1] * arcmin2rad, lmax=self.lmax)
        #        self.B2_l  = self.B_1_l * self.B_2_l
        #        self.B2_ll = np.diag(self.B2_l)
        #    else:
        #        self.B_1_l = hp.gauss_beam(fwhm_smooth[0] * arcmin2rad, lmax=self.lmax)
        #        self.B2_l  = self.B_1_l**2
        #        self.B2_ll = np.diag(self.B2_l)

        if fsky_approx: # f_sky approximation
            self.weight = np.ones(lmax+1)
            if pixwin:
                self.weight *= self.pw2_l
            if fwhm_smooth is not None:
                self.weight *= self.B2_l
        else: # MASTER/Xspect
            W_l       = hp.anafast(mask, lmax = self.lmax)
            self.W_l  = W_l
            M_ll      = self._get_Mll()
            self.M_ll = M_ll 
            M_jj      = np.dot(np.dot(self.P_jl, self.M_ll), self.Q_lj)
            self.M_jj = M_jj
            #K_ll      = self.M_ll

            #if pixwin:
            #    K_ll = np.dot(K_ll, self.pw2_ll)
            #if fwhm_smooth is not None:
            #    K_ll = np.dot(K_ll, self.B2_ll)

            #self.K_ll = K_ll
            K_jj      = np.dot(np.dot(self.P_jl, self.M_ll), self.Q_lj)
            self.K_jj = K_jj
            K_jj_inv  = la.inv(K_jj)
            self.K_jj_inv = K_jj_inv
            
    def get_spectra(self, beta_j):
        """
        Method to return extracted auto- or cross-spectrum of a Healpix map if *map2* 
        is provided. User can choose to use MASTER method (default) of f_sky approximation 
        if *fsky_approx* is True.
        It is also possible to input a *pseudo* noise power spectrum for debiasing.

        Parameters
        ----------
        map1 : array
            Healpix map #1.
        map2 : array, optional. Default : None
            Healpix map #2.
        nl   : array, optional
            Noise power spectrum. Default : None
        pseudo : boolean
            If true, the passed nl array is the pseudo-noisep woer spectrum. Default : False
           analytic_errors : boolean
               Flag for analytical error bars estimation. Default : False

        Returns
        -------
        cl(, cl_err) : array(s)
            Array containing extracted (auto- or cross-) spectrum.
            If *analytic_errors* is True, it also returns error bars.e
            Corresponding bins are stored in *Master.ell_binned*.

        Example of usage
        ----------------
        kg    = Master(lmin, lmax, delta_ell, mask, *args)
        kappa = hp.read_map('convergence_map.fits')
        delta = hp.read_map('galaxy_map.fits')

        cl_kg = kg.get_spectra(kappa, mappa2 = delta) 

        or

        cl_kg, err_cl_kg = kg.get_spectra(kappa, mappa2 = delta, analytic_errors = True) 

        Notes
        -----
        1. Noise bias can be subtracted from cross-correlation too.
        2. Especially for auto-correlation, it assumes that maps
           signal are treated in the same way i.e., 
           have been smoothed with the same beam.
        3. Noise is not weighted by pixel or beam window function.
        """
        
        # Use MASTER or f_sky
        if self.fsky_approx:
            beta_j = beta_j / self.fsky 

        else: # a' la MASTER/Xspect
            beta_j = np.dot(self.K_jj_inv, beta_j)
           
        return beta_j


    def _get_Mll(self):
        """
        Returns the Coupling Matrix M_ll from l = 0 
        (Hivon et al. 2002)

        Notes
        -----
        M_ll.shape = (lmax+1, lmax+1)
        """
        _M_ll = mll.get_mll(self.W_l, self.lmax)
        return np.float64(_M_ll)

    def _get_ith_mask_moment(self, i):
        """
        Returns the i-th momenent of the mask as:
        w^i = 1/4\pi \int d^2n W^i(n) = \Omega_p /4\pi \sum_p W^i(p)
        where W(n) is the mask and \Omega_p is the surface area of the pixel

        Parameters
        ----------
        i : int
            i-th moment of the mask	

        """
        return hp.nside2pixarea(self.nside) * np.sum(self.mask**i) / 4. / np.pi

'''
class Master_python(object):
    """
    Cross- and Auto-power spectra estimation using the Xspect/MASTER method. 
    Hivon et al. 2002, Tristam et al. 2004
    """

    def __init__(self, lmin, lmax, delta_ell, mask = None):
        """
        Parameters
        ----------
        mask : boolean Healpix map
            Mask defining the region of interest (of value True)
        lmin : int
            Lower bound of the first l bin.
        lmax : int
            Highest l value to be considered. The inclusive upper bound of
            the last l bin is lesser or equal to this value.
        delta_ell :
            The l bin width.
        """
        
        lmin = int(lmin)
        lmax = int(lmax)
        if lmin < 1:
            raise ValueError('Input lmin is less than 1.')
        if lmax < lmin:
            raise ValueError('Input lmax is less than lmin.')
        self.lmin = lmin
        self.lmax = lmax

        delta_ell      = int(delta_ell)
        self.delta_ell = delta_ell
        self.ell_binned, self._P_bl, self._Q_lb = self._bin_ell()

        if mask is not None:
            mask      = np.asarray(mask)
            self.mask = mask
            W_l       = hp.anafast(mask)[:lmax+1]
            self.W_l  = W_l
            
            M_ll      = self._get_Mll()
            self.M_ll = M_ll 
            M_bb      = np.dot(np.dot(self._P_bl, self.M_ll), self._Q_lb)
            self.M_bb = M_bb

    def _bin_ell(self):
        nbins = (self.lmax - self.lmin + 1) // self.delta_ell
        start = self.lmin + np.arange(nbins) * self.delta_ell
        stop  = start + self.delta_ell
        ell_binned = (start + stop - 1) / 2

        P_bl = np.zeros((nbins, self.lmax + 1))
        Q_lb = np.zeros((self.lmax + 1, nbins))

        for b, (a, z) in enumerate(zip(start, stop)):
            P_bl[b, a:z] = 1. / (z - a)
            Q_lb[a:z, b] = 1.

        return ell_binned, P_bl, Q_lb

    def _get_Mll(self):
        """
        Returns the Coupling Matrix M_ll from l = 0 
        (Hivon et al. 2002)
        """
        M_ll = np.zeros((self.lmax + 1, self.lmax + 1))
        for l1 in xrange(0, self.lmax + 1):
            for l2 in xrange(0, self.lmax + 1):
                numb2     = 2. * l2 + 1.
                coupl_sum = 0.
                dim = l1 + l2 - abs(l1-l2) + 1
                l3min, l3max, wig, ier = wigner_3j.rc3jj(l1,l2,0.,0.,dim)
                numb3 = 2. * np.arange(int(l3min), min(int(l3max), self.lmax) + 1) + 1.
                wig2  = wig[:numb3.size]
                wig2  = wig2**2
                coupl_sum    = np.dot(numb3, wig2 * self.W_l[np.arange(int(l3min), min(int(l3max), self.lmax) + 1)])
                M_ll[l1][l2] = numb2 * coupl_sum / (4. * np.pi)

        return M_ll;
'''
