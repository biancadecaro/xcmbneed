import numpy as np
# import matplotlib.pyplot as plt
import healpy as hp
import argparse, os, sys, warnings, glob
import utils
from seed import rnd_stream as rs
import matplotlib.pyplot as plt

class KGsimulations(object):
	""" 
	Class to manage a set of simulations generated with theory spectra contained in class
	XCSpectraFile, stored in directory LibDir and with specifics given by SimPars dictionary 
	"""
	def __init__(self, XCSpectraFile, LibDir, SimPars, PlanckPath=None , WantTG= False):
		"""
		PlanckPath : path to Planck CMB lensing simulations
		"""
		self.XCSpectraFile = XCSpectraFile
		self.LibDir        = LibDir
		self.SimPars       = SimPars
		self.PlanckPath    = PlanckPath

		if not os.path.exists(self.LibDir):
			os.makedirs(self.LibDir)
			
	def Run(self, nsim, WantTG, EuclidSims=False):
		"""
		It generates nsim realizations of the CMB lensing and galaxy fields.
		The outputs are *correlated* maps of Kappa and Delta, signal-only and
		with noise too. 
		"""
		try:
			from mpi4py import MPI
			comm = MPI.COMM_WORLD
			myid, nproc = comm.Get_rank(), comm.Get_size()
		except ImportError:
			myid, nproc = 0, 1

		if WantTG == True:
			nsim_found = len(glob.glob(self.LibDir + "sim_*_*_" + ('%04d' % self.SimPars['nside']) + ".fits"))/4
		else:
			nsim_found = len(glob.glob(self.LibDir + "sim_*_*_" + ('%04d' % self.SimPars['nside']) + ".fits"))/6
		
		if EuclidSims == True:
			print('... Euclid sims requested ...')
			nsim_found = len(glob.glob(self.LibDir + "map_nbin1_NSIDE" + str(self.SimPars['nside']) + "_lmax*_*_*.fits"))/2
		

		print('nsim_found = ', nsim_found)

		if nsim_found >= nsim:
			if myid == 0: print("... " + ('%d' % nsim_found) + " simulations found...")
		else:
			nleft = nsim-nsim_found
			if myid == 0: print("... " + ('%d' % nsim_found) + " simulations found...")
			if myid == 0: print("...running remaining " + ('%d' % nleft) + " simulations ...")
			if myid == 0: print("-->Starting...")

			# Loop over simulations
			for n in range(int(myid+nsim_found), nsim, nproc): 
				print("...myid = " + str(myid) + "  nsim = " + str(n))
				
				np.random.seed(rs[n])
				print("--> myid ", myid, "-------> seed ", rs[n])	
				if WantTG == True:
					
					#Correlated signal temperature and density maps
					TS_map, galS_map = utils.GetCorrMaps(self.XCSpectraFile.cltg, self.XCSpectraFile.cltt, self.XCSpectraFile.clg1g1, self.SimPars['nside'], pixwin=self.SimPars['pixwin'])
					 #continuare qua e capire se mettere anche noise e total maps (ma forse per ora no perche non ho un rumore)
					
					galN_map = utils.GetGalNoiseMap(self.SimPars['nside'], self.SimPars['ngal'], dim=self.SimPars['ngal_dim'], delta=True) #! questa mappa non serve a niente
					
					# Total maps
					galT_map = utils.Counts2Delta(utils.GetCountsTot(galS_map, self.SimPars['ngal'], dim=self.SimPars['ngal_dim']))
						#Noise maps
					#if self.XCSpectraFile.nltt is not None:
					#	TN_map = hp.synfast(self.XCSpectraFile.nltt, self.SimPars['nside'], pixwin=self.SimPars['pixwin'], verbose=False)
					#	TT_map = TS_map + TN_map
					#else:
					#	TN_map = np.zeros_like(TS_map)
						#TT_map = TS_map
					

					#Saving maps
					fname_TS = self.LibDir + "sim_" + ('%04d' % n) + "_TS_" + ('%04d' % self.SimPars['nside']) + ".fits"
					fname_galS = self.LibDir + "sim_" + ('%04d' % n) + "_galS_" + ('%04d' % self.SimPars['nside']) + ".fits"
					#fname_TN = self.LibDir + "sim_" + ('%04d' % n) + "_TN_" + ('%04d' % self.SimPars['nside']) + ".fits"
					fname_galN = self.LibDir + "sim_" + ('%04d' % n) + "_galN_" + ('%04d' % self.SimPars['nside']) + ".fits"
					#fname_TT = self.LibDir + "sim_" + ('%04d' % n) + "_TT_" + ('%04d' % self.SimPars['nside']) + ".fits"
					fname_galT = self.LibDir + "sim_" + ('%04d' % n) + "_galT_" + ('%04d' % self.SimPars['nside']) + ".fits"
					

					hp.write_map(fname_TS, TS_map, nest=False)#,overwrite = True)
					hp.write_map(fname_galS, galS_map, nest=False)#,overwrite = True)
					#hp.write_map(fname_TN, TN_map, nest=False)#,overwrite = True)
					hp.write_map(fname_galN, galN_map, nest=False)#,overwrite = True)
					#hp.write_map(fname_TT, TT_map, nest=False)#,overwrite = True)
					hp.write_map(fname_galT, galT_map, nest=False)#,overwrite = True)
					#cls_from_sims = hp.sphtfunc.anafast(map1=TS_map, map2=galS_map)
					#np.savetxt('spectra/clsTG_from_sims_null.dat', cls_from_sims)

				else:	
					# Correlated *signal* lensing and density maps
					kappaS_map, deltaS_map = utils.GetCorrMaps(self.XCSpectraFile.clkg_tot, self.XCSpectraFile.clkk, self.XCSpectraFile.clgg_tot, self.SimPars['nside'], pixwin=self.SimPars['pixwin'])

					# Noise maps
					kappaN_map = hp.synfast(self.XCSpectraFile.nlkk, self.SimPars['nside'], pixwin=self.SimPars['pixwin'], verbose=False)
					deltaN_map = utils.GetGalNoiseMap(self.SimPars['nside'], self.SimPars['ngal'], dim=self.SimPars['ngal_dim'], delta=True)

					# Total maps
					deltaT_map = utils.Counts2Delta(utils.GetCountsTot(deltaS_map, self.SimPars['ngal'], dim=self.SimPars['ngal_dim']))
					kappaT_map = kappaS_map + kappaN_map

					# Saving maps
					fname_kappaS = self.LibDir + "sim_" + ('%04d' % n) + "_kappaS_" + ('%04d' % self.SimPars['nside']) + ".fits"
					fname_kappaN = self.LibDir + "sim_" + ('%04d' % n) + "_kappaN_" + ('%04d' % self.SimPars['nside']) + ".fits"
					fname_kappaT = self.LibDir + "sim_" + ('%04d' % n) + "_kappaT_" + ('%04d' % self.SimPars['nside']) + ".fits"
					fname_deltaS = self.LibDir + "sim_" + ('%04d' % n) + "_deltaS_" + ('%04d' % self.SimPars['nside']) + ".fits"
					fname_deltaN = self.LibDir + "sim_" + ('%04d' % n) + "_deltaN_" + ('%04d' % self.SimPars['nside']) + ".fits"
					fname_deltaT = self.LibDir + "sim_" + ('%04d' % n) + "_deltaT_" + ('%04d' % self.SimPars['nside']) + ".fits"

					hp.write_map(fname_kappaS, kappaS_map, nest=False)
					hp.write_map(fname_kappaN, kappaN_map, nest=False)
					hp.write_map(fname_kappaT, kappaT_map, nest=False)
					hp.write_map(fname_deltaS, deltaS_map, nest=False)
					hp.write_map(fname_deltaN, deltaN_map, nest=False)
					hp.write_map(fname_deltaT, deltaT_map, nest=False)
				
			if nproc > 1:
				comm.Barrier() 
			
			if myid == 0: print("-->simulations done...")

	def GetSimField(self, field, idx):
		fname = self.LibDir + "sim_" + ('%04d' % idx) + "_" + field + "_" + ('%04d' % self.SimPars['nside']) + ".fits"
		return hp.read_map(fname, verbose=False)
	
	def GetSimField_Euclid(self, field, idx,lmax):
		fname = self.LibDir + "map_nbin1_NSIDE" + str(self.SimPars['nside']) + "_lmax" + str(lmax)  + "_" + ('%05d' % (idx+1))+ "_" + field + ".fits"#map_nbin1_NSIDE128_lmax256_00327_T.fits
		return hp.read_map(fname, verbose=False)
	
	def GetMeanField(self, field, idx):
		fname = self.LibDir + "sim_" + ('%04d' % idx) + "_" + field + "_" + ('%04d' % self.SimPars['nside']) + ".fits"
		m = hp.read_map(fname, verbose=False)
		m = hp.remove_monopole(m)
		return m.mean()

	# TODO: implement Planck CMB lensing sims

	def GetSimPlanck(self, idx, obs=True):
		"""
		If obs=True, it reads estimated CMB lensing convergence maps, otherwise
		it loads input CMB kappa maps.
		"""
		if obs:
			fname = self.PlanckPath + "obs_klms/sim_" + ('%04d' % n) + "_klm.fits"
		else:
			fname = self.PlanckPath + "sky_klms/sim_" + ('%04d' % n) + "_klm.fits"
		kappa_sim_lm = hp.read_alm(fname)
		kappa_sim    = hp.alm2map(kappa_sim_lm, nside=self.SimPars['nside'], pixwin=self.SimPars['pixwin'])
		return kappa_sim
