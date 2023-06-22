import numpy as np
import matplotlib.pyplot as plt
import cython_mylibc as pippo
import analysis, spectra

#def bin_ell(lmax, lmin, delta_ell):
#		nbins = ( lmax -  lmin + 1) //  delta_ell
#		start =  lmin + np.arange(nbins) *  delta_ell
#		stop  = start +  delta_ell
#		ell_binned = (start + stop - 1) / 2
#
#		flat = np.ones( lmax + 1)
#		_P_bl = np.zeros((nbins,  lmax + 1))
#
#		for b, (a, z) in enumerate(zip(start, stop)):
#			#print(b, a, z)
#			_P_bl[b, a:z] = 1. * flat[a:z] / (z - a)
#
#		return ell_binned, _P_bl
#
#
#def bin_spectra(spectra, P_bl):
#		"""
#		Average spectra in bins specified by lmin, lmax and delta_ell,
#		weighted by flattening term specified in initialization.
#
#		"""
#		spectra = np.asarray(spectra)
#		lmax    = spectra.shape[-1] - 1
#
#		return np.dot(P_bl, spectra[..., :lmax+1])


OmL = np.linspace(0.0,0.95,30)

nside = 128#512
jmax = 12
lmax_n = 512#782
B = pippo.mylibpy_jmax_lmax2B(jmax, lmax_n)
print(B)

#NEEDLETS



beta_theory = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/TG_128/beta_TS_galS_theoretical_fiducial_B1.681792830507429.dat')#np.loadtxt(f'output_needlet_TG_OmL/Grid_spectra_{len(OmL)}_planck/beta_TS_galS_theoretical_OmL_fiducial_B{B}.dat') 
beta_sim_array = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/TG_128/betaj_sims_TS_galS_jmax12_B1.681792830507429_nside128.dat')#np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/TG_256_planck_2/betaj_sims_TS_galS_jmax12_B1.7422102429630426_nside256.dat')
beta_sim = np.mean(beta_sim_array, axis=0)


fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.plot(np.arange(1,jmax), beta_sim[1:-1]/beta_theory[1:-1] -1 )

ax.set_xlabel(r'j')
ax.set_ylabel(r'$\Delta \beta_{j}^{TG}$-1')

plt.savefig(f'comparison_betaj_theory_sims_relative_diff_nside{nside}_jmax{jmax}_B{B:.2f}.png')

fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.plot(np.arange(1,jmax), beta_theory[1:-1], label = r'$\beta_{j}^{TG}$ theory')
ax.plot(np.arange(1,jmax), beta_sim[1:-1], label = r'$\beta_{j}^{TG}$ sims')

ax.set_xlabel(r'j')
ax.set_ylabel(r'$\beta_{j}^{TG}$')
plt.legend()
plt.savefig(f'comparison_betaj_theory_sims_nside{nside}_jmax{jmax}_B{B:.2f}.png')


