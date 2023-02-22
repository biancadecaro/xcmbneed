import numpy as np
import healpy as hp
import scipy.linalg as la
from mll import mll

__all__ = ['Binner', 'Master']

arcmin2rad = 0.000290888208666

class Binner(object):
	"""
	Class for computing binning scheme.
	"""
	def __init__(self, bin_edges=None, lmin=2, lmax=500, delta_ell=50, flat=None):
		"""
		Parameters
		----------
		bin_edges: array
			Edges of bandpowers
		lmin : int
		    Lower bound of the first l bin.
		lmax : int
		    Highest l value to be considered. The inclusive upper bound of
		    the last l bin is lesser or equal to this value.
		delta_ell :
		    The l bin width.
		flat : function flat(l)
			Power spectrum flattening type. Default = None
		"""
		
		if bin_edges is None:
			if lmin < 1:
			    raise ValueError('Input lmin is less than 1.')
			if lmax < lmin:
			    raise ValueError('Input lmax is less than lmin.')

			self.lmin      = int(lmin)
			self.lmax      = int(lmax)
			self.delta_ell = int(delta_ell)

			nbins = (self.lmax - self.lmin + 1) // self.delta_ell
			start = self.lmin + np.arange(nbins) * self.delta_ell
			stop  = start + self.delta_ell
			self.lmax = stop[-1]
			self.bin_edges = np.append(start, stop[-1])

		else: 
			self.bin_edges = np.asarray(bin_edges)
			self.lmin      = int(self.bin_edges[0])
			self.lmax      = int(self.bin_edges[-1])
			self.delta_ell = self.bin_edges[1:] - self.bin_edges[:-1]
			
			nbins = len(self.delta_ell)
			start = np.floor(self.bin_edges[:-1])
			stop  = np.ceil(self.bin_edges[1:])

		# Centers of bandpowers
		self.lb = (start + stop - 1) / 2

		# Apply prewhitening to bandpowers
		self.flat = flat
		if self.flat == None:
		    flat_ = np.ones(self.lmax + 1)
		else:
		    flat_ = self.flat(np.arange(self.lmax + 1))

		# Creating binning operators 		 
		self.P_bl = np.zeros((nbins, self.lmax + 1))
		self.Q_lb = np.zeros((self.lmax + 1, nbins))

		for b, (a, z) in enumerate(zip(start, stop)):
			a = int(a); z = int(z)
			self.P_bl[b, a:z] = np.nan_to_num( 1. * flat_[a:z] / (z - a) )
			self.Q_lb[a:z, b] = np.nan_to_num( 1. / flat_[a:z] )

	def bin_spectra(self, spectra):
		"""
		Average spectra in bins.
		"""
		spectra = np.asarray(spectra)
		lmax    = spectra.shape[-1] - 1
		if lmax < self.lmax:
			raise ValueError('The input spectra do not have enough l.')

		return np.dot(self.P_bl, spectra[..., :self.lmax+1])

class Master(Binner):
	"""
	Class to estimate Cross- and Auto-power spectra using the Xspect/MASTER method.
	User can choose between MASTER method or f_sky approximation (default). 
	See Hivon et al. 2002, Tristam et al. 2004 for equations.

	The idea is that you set up a Master object specified by:
	i)   Binning scheme
	ii)  Mask 
	iii) Transfer functions (beam, pixel, filtering)
	and then extract auto/cross-power spectra by calling Master.get_spectra(...)

	Parameters
	----------

	~ Note: If you need the power spectrum up to lmax, use lmax + 200.
	"""

	def __init__(self, mask, bin_edges=None, lmin=2, lmax=500, delta_ell=50, flat=None, 
				 pixwin=True, fwhm_smooth=None, transfer=None, MASTER=False, inv_routine=la.inv):
		"""
		Parameters
		----------
		mask: array
			Healpix map containing mask to apply to maps 
		pixwin: boolean
			If True, deconvolve spectra for the pixel window function
		fwhm_smooth: 
			If not None, deconvolve for Gaussian Beam smoothing effect (expressed in *arcmin*)
			It can be a number or a tuple/list/array containing two values
		transfer: array
			If not None, deconvolve for *effective* transfer function F_l => transf = F^{AB}_l = \sqrt{F_l^A * F_l^B}
		MASTER: boolean
			If True, compute Coupling matrix and deconvolve for mask-induced mode-coupling effects

		# TODO: implement second mask
		"""
		super(Master, self).__init__(bin_edges, lmin, lmax, delta_ell, flat)

		mask             = np.asarray(mask)
		self.mask        = mask
		self.pixwin      = pixwin
		self.nside       = hp.npix2nside(mask.size)
		self.fwhm_smooth = fwhm_smooth
		self.MASTER      = MASTER
		self.inv_routine = inv_routine

		self.eps = 1e-6 # Conditioning param: HARDCODED !

		self.fsky = self.get_ith_mask_moment(2)

		# Deconvolve for pixel window function
		if pixwin:
			self.pw2_l  = ((hp.pixwin(self.nside))**2)[:self.lmax+1]
			self.pw2_ll = np.diag(self.pw2_l)

		# Deconvolve for gaussian beam
		if fwhm_smooth is not None:
			if hasattr(fwhm_smooth, "__len__"): 
				self.B_1_l = hp.gauss_beam(fwhm_smooth[0] * arcmin2rad, lmax=self.lmax)
				self.B_2_l = hp.gauss_beam(fwhm_smooth[1] * arcmin2rad, lmax=self.lmax)
				self.B2_l  = self.B_1_l * self.B_2_l
				self.B2_ll = np.diag(self.B2_l)
			else:
				self.B_1_l = hp.gauss_beam(fwhm_smooth * arcmin2rad, lmax=self.lmax)
				self.B_2_l = self.B_1_l.copy()
				self.B2_l  = self.B_1_l**2
				self.B2_ll = np.diag(self.B2_l)

		# Deconvolve for transfer function
		if transfer is not None:
			assert( len(transfer) > self.lmax+1)
			self.F_l  = transfer[:self.lmax+1]
			self.F_ll = np.diag(transfer[:self.lmax+1])

		# Mode-coupling matrix
		if MASTER:
			self.W_l  = hp.anafast(mask, lmax=self.lmax)
			self.M_ll = self._get_Mll()
			self.M_bb = np.dot(np.dot(self.P_bl, self.M_ll), self.Q_lb)
			K_ll      = self.M_ll

			if pixwin:
				K_ll = np.dot(K_ll, self.pw2_ll)
			if fwhm_smooth is not None:
				K_ll = np.dot(K_ll, self.B2_ll)
			if transfer is not None:
				K_ll = np.dot(K_ll, self.F_ll)

			self.K_ll = K_ll
			self.K_bb = np.dot(np.dot(self.P_bl, self.K_ll), self.Q_lb)
			try:
				K_bb_inv = self.inv_routine(self.K_bb)
			except:
				print("\t! Problem with Coupling Matrix inversion: let me try a little trick ! ")
				K_bb_inv = self.inv_routine(self.K_bb + np.eye(self.K_bb.shape[0])*self.eps)

			self.K_bb_inv = K_bb_inv

		else:  # f_sky approximation
			self.weight = np.ones(self.lmax+1)
			if pixwin:
				self.weight *= self.pw2_l
			if fwhm_smooth is not None:
				self.weight *= self.B2_l
			if transfer is not None:
				self.weight *= self.F_l

	def get_spectra(self, map1, map2=None, nl=None, pseudo=False, analytic_errors=False):
		"""
		Returns extracted auto- or cross-spectrum if *map2* is provided. 
		It is also possible to input a noise power spectrum for debiasing and specify 
		whether it a pseudo one or not.

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
        	Corresponding bins are stored in *Master.lb*.

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
		if map2 is None: # Auto
			pcl = hp.anafast(map1 * self.mask, lmax=self.lmax)
		else:            # Cross
			map2 = np.asarray(map2)
			pcl  = hp.anafast(map1 * self.mask, map2=map2 * self.mask, lmax=self.lmax)
		
		if analytic_errors: 
			pcl_tot = pcl

		if nl is not None:
			if nl.size - 1 < self.lmax:
				raise ValueError('The noise power spectrum does not have enough l.')

		if self.MASTER:
			if nl is  None: 
				cl = np.dot(self.K_bb_inv, np.dot(self.P_bl, pcl))
			else:
				if pseudo:
					cl = np.dot(self.K_bb_inv, np.dot(self.P_bl, pcl - nl[:self.lmax+1]))
				else:
					cl = np.dot(self.K_bb_inv, np.dot(self.P_bl, pcl)) - self.bin_spectra(nl[:self.lmax+1])			
			if analytic_errors and map2 is None:
				cl_tot = np.dot(self.K_bb_inv, np.dot(self.P_bl, pcl_tot))
		else: # f_sky approx
			if nl is None:
				cl = np.dot(self.P_bl, pcl/self.weight)/self.fsky
			else:
				if pseudo:
					cl = self.bin_spectra(pcl/self.weight - nl[:self.lmax+1]) / self.fsky
				else:
					cl = self.bin_spectra(pcl/self.weight) / self.fsky - self.bin_spectra(nl[:self.lmax+1])
			if analytic_errors and map2 is None:
				cl_tot = self.bin_spectra(pcl_tot/self.weight) / self.fsky

		# Analytic error bars  estimation 
		# TODO: moving this into another method?
		if analytic_errors:
			if map2 is None: # Auto
				cl_err = np.sqrt(2./((2. * self.lb + 1) * self.delta_ell * self.fsky)) * cl_tot
			else: # Cross
				# Extracting TOTAL pseudo-power spectra
				pcl_1 = hp.anafast(map1 * self.mask, lmax=self.lmax)
				pcl_2 = hp.anafast(map2 * self.mask, lmax=self.lmax)
				
				if self.fwhm_smooth is not None:
					B2_1_ll = np.diag(self.B_1_l**2)
					B2_2_ll = np.diag(self.B_2_l**2)

				if self.MASTER:
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

					try:
						K_bb_inv_1 = self.inv_routine(K_bb_1)
					except:
						print("\t! Problem with Coupling Matrix inversion: let me try a little trick ! ")
						K_bb_inv_1 = self.inv_routine(K_bb_1 + np.eye(K_bb_1.shape[0])*self.eps)

					try:
						K_bb_inv_2 = self.inv_routine(K_bb_2)
					except:
						print("\t! Problem with Coupling Matrix inversion: let me try a little trick ! ")
						K_bb_inv_2 = self.inv_routine(K_bb_2 + np.eye(K_bb_2.shape[0])*self.eps)

					# K_bb_inv_1  = self.inv_routine(K_bb_1)
					# K_bb_inv_2  = self.inv_routine(K_bb_2)

					cl1 = np.dot(K_bb_inv_1, np.dot(self.P_bl, pcl_1))
					cl2 = np.dot(K_bb_inv_2, np.dot(self.P_bl, pcl_2))


				else:
					weight_1 = np.ones(self.lmax+1)
					weight_2 = np.ones(self.lmax+1)

					if self.pixwin:
						weight_1 *= self.pw2_l
						weight_2 *= self.pw2_l
					if self.fwhm_smooth is not None:
						weight_1 *= np.diag(B2_1_ll)
						weight_2 *= np.diag(B2_2_ll)

					cl1 = np.dot(self.P_bl, pcl_1/weight_1) / self.fsky
					cl2 = np.dot(self.P_bl, pcl_2/weight_2) / self.fsky

				cl_err = np.sqrt(2./((2. * self.lb + 1) * self.delta_ell * self.fsky) * (cl**2 + cl1 * cl2))

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

	def get_ith_mask_moment(self, i, mask=None):
		"""
		Returns the i-th momenent of the mask as:
		w^i = 1/4\pi \int d^2n W^i(n) = \Omega_p /4\pi \sum_p W^i(p)
		where W(n) is the mask and \Omega_p is the surface area of the pixel

		Parameters
		----------
		i : int
		    i-th moment of the mask	

		"""
		if mask is None:
			mask = self.mask

		nside = hp.npix2nside(len(mask))

		return hp.nside2pixarea(nside) * np.sum(mask**i) / 4. / np.pi
