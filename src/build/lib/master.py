import numpy as np
import healpy as hp
from numpy import linalg as la
from mll import mll

__all__ = ['Binner', 'Master']

arcmin2rad = 0.000290888208666

class Binner(object):
	"""
	Class for computing binning scheme.
	"""
	def __init__(self, lmin, lmax, delta_ell, flattening=None):
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
		if lmin < 1:
		    raise ValueError('Input lmin is less than 1.')
		if lmax < lmin:
		    raise ValueError('Input lmax is less than lmin.')
		self.lmin = lmin
		self.lmax = lmax
		self.flattening = flattening

		delta_ell      = int(delta_ell)
		self.delta_ell = delta_ell
		self.ell_binned, self.P_bl, self.Q_lb = self._bin_ell()

	def _bin_ell(self):
		nbins = (self.lmax - self.lmin + 1) // self.delta_ell
		start = self.lmin + np.arange(nbins) * self.delta_ell
		stop  = start + self.delta_ell
		ell_binned = (start + stop - 1) / 2

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

		_P_bl = np.zeros((nbins, self.lmax + 1))
		_Q_lb = np.zeros((self.lmax + 1, nbins))

		for b, (a, z) in enumerate(zip(start, stop)):
			_P_bl[b, a:z] = 1. * flat[a:z] / (z - a)
			_Q_lb[a:z, b] = 1. / flat[a:z]

		return ell_binned, _P_bl, _Q_lb

	def bin_spectra(self, spectra):
		"""
		Average spectra in bins specified by lmin, lmax and delta_ell,
		weighted by flattening term specified in initialization.

		"""
		spectra = np.asarray(spectra)
		lmax    = spectra.shape[-1] - 1
		if lmax < self.lmax:
			raise ValueError('The input spectra do not have enough l.')

		return np.dot(self.P_bl, spectra[..., :self.lmax+1])


class Master(Binner):
	"""
	Class to estimate Cross- and Auto-power spectra using the Xspect/MASTER method. 
	Hivon et al. 2002, Tristam et al. 2004.
	It also implements the fsky_approximation.
	"""

	def __init__(self, lmin, lmax, delta_ell, mask, flattening=None, 
				 pixwin=True, fwhm_smooth=None, fsky_approx=False):
		"""
		Parameters
		----------
		fwhm_smooth arcmin

		# TODO: implement second mask
		"""
		super(Master, self).__init__(lmin, lmax, delta_ell, flattening)

		mask        = np.asarray(mask)
		self.mask   = mask
		self.pixwin = pixwin
		self.nside  = hp.npix2nside(mask.size)
		self.fwhm_smooth = fwhm_smooth
		self.fsky_approx = fsky_approx

		self.fsky = self._get_ith_mask_moment(2)

		if pixwin:
			self.pw2_l  = ((hp.pixwin(self.nside))**2)[:lmax+1]
			self.pw2_ll = np.diag(self.pw2_l)

		if fwhm_smooth is not None:
			if np.size(fwhm_smooth) != 0: 
				self.B_1_l = hp.gauss_beam(fwhm_smooth[0] * arcmin2rad, lmax=self.lmax)
				self.B_2_l = hp.gauss_beam(fwhm_smooth[1] * arcmin2rad, lmax=self.lmax)
				self.B2_l  = self.B_1_l * self.B_2_l
				self.B2_ll = np.diag(self.B2_l)
			else:
				self.B_1_l = hp.gauss_beam(fwhm_smooth[0] * arcmin2rad, lmax=self.lmax)
				self.B2_l  = self.B_1_l**2
				self.B2_ll = np.diag(self.B2_l)

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
			M_bb      = np.dot(np.dot(self.P_bl, self.M_ll), self.Q_lb)
			self.M_bb = M_bb
			K_ll      = self.M_ll

			if pixwin:
				K_ll = np.dot(K_ll, self.pw2_ll)
			if fwhm_smooth is not None:
				K_ll = np.dot(K_ll, self.B2_ll)

			self.K_ll = K_ll
			K_bb      = np.dot(np.dot(self.P_bl, self.K_ll), self.Q_lb)
			self.K_bb = K_bb
			K_bb_inv  = la.inv(K_bb)
			self.K_bb_inv = K_bb_inv

	def get_spectra(self, map1, map2=None, nl=None, pseudo=False, analytic_errors=False):
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
		map1 = np.asarray(map1)
		if map2 is None: # Auto-power spectrum
			pcl = hp.anafast(map1 * self.mask, lmax=self.lmax)
		else:            # Cross-power spectrum
			map2 = np.asarray(map2)
			pcl  = hp.anafast(map1 * self.mask, map2=map2 * self.mask, lmax=self.lmax)
		
		if analytic_errors: 
			pcl_tot = pcl

		if nl is not None:
			if nl.size - 1 < self.lmax:
				raise ValueError('The noise power spectrum does not have enough l.')

		# Use MASTER or f_sky
		if self.fsky_approx:
			if nl is None:
				cl = np.dot(self.P_bl, pcl/self.weight)/self.fsky
			else:
				if pseudo:
					cl = self.bin_spectra(pcl/self.weight - nl[:self.lmax+1]) / self.fsky
				else:
					cl = self.bin_spectra(pcl/self.weight) / self.fsky - self.bin_spectra(nl[:self.lmax+1])
			if analytic_errors and map2 is None:
				cl_tot = self.bin_spectra(pcl_tot/self.weight) / self.fsky
		else: # a' la MASTER/Xspect
			if nl is  None: 
				cl = np.dot(self.K_bb_inv, np.dot(self.P_bl, pcl))
			else:
				if pseudo:
					cl = np.dot(self.K_bb_inv, np.dot(self.P_bl, pcl - nl[:self.lmax+1]))
				else:
					cl = np.dot(self.K_bb_inv, np.dot(self.P_bl, pcl)) - self.bin_spectra(nl[:self.lmax+1])			
			if analytic_errors and map2 is None:
				cl_tot = np.dot(self.K_bb_inv, np.dot(self.P_bl, pcl_tot))

		# Error bars analytic estimation 
		# TODO: could think of move this into another method
		if analytic_errors:
			if map2 is None: # Auto-power spectrum
				cl_err = np.sqrt(2./((2. * self.ell_binned + 1) * self.delta_ell * self.fsky)) * cl_tot
			else: # Cross-spectrum
				# Extracting TOTAL pseudo-power spectra
				pcl_1 = hp.anafast(map1 * self.mask, lmax=self.lmax)
				pcl_2 = hp.anafast(map2 * self.mask, lmax=self.lmax)
				
				if self.fwhm_smooth is not None:
					B2_1_ll = np.diag(self.B_1_l**2)
					B2_2_ll = np.diag(self.B_2_l**2)

				if self.fsky_approx:
					weight_1 = np.ones(self.lmax+1)
					weight_2 = np.ones(self.lmax+1)

					if self.pixwin:
						weight_1 *= self.pw2_l
						weight_2 *= self.pw2_l
					if self.fwhm_smooth is not None:
						weight_1 *= B2_1_ll
						weight_2 *= B2_2_ll

					cl1 = np.dot(self.P_bl, pcl_1/weight_1) / self.fsky
					cl2 = np.dot(self.P_bl, pcl_2/weight_2) / self.fsky

				else:
					K_ll_1 = self.M_ll
					K_ll_2 = self.M_ll
					
					if self.pixwin:
						K_ll_1 = np.dot(K_ll_1, self.pw2_ll)
						K_ll_2 = np.dot(K_ll_2, self.pw2_ll)
					if self.fwhm_smooth is not None:
						K_ll_1 = np.dot(K_ll_1, B2_1_ll)
						K_ll_2 = np.dot(K_ll_2, B2_2_ll)

					K_bb_1 = np.dot(np.dot(self.P_bl, K_ll_1), self.Q_lb)
					K_bb_2 = np.dot(np.dot(self.P_bl, K_ll_2), self.Q_lb)

					K_bb_inv_1  = la.inv(K_bb_1)
					K_bb_inv_2  = la.inv(K_bb_2)

					cl1 = np.dot(K_bb_inv_1, np.dot(self.P_bl, pcl_1))
					cl2 = np.dot(K_bb_inv_2, np.dot(self.P_bl, pcl_2))

				cl_err = np.sqrt(2./((2. * self.ell_binned + 1) * self.delta_ell * self.fsky) * (cl**2 + cl1 * cl2))

			return cl, cl_err
		else:
			return cl


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
