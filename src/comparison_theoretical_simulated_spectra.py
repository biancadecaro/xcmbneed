import numpy as np
import matplotlib.pyplot as plt
import cython_mylibc as pippo
import analysis

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

nside = 512
jmax = 12
lmax_n = 782
B = pippo.mylibpy_jmax_lmax2B(jmax, lmax_n)
print(B)

#NEEDLETS

beta_theory = np.loadtxt(f'output_needlet_TG_OmL/Grid_spectra_{len(OmL)}_planck/beta_TS_galS_theoretical_OmL_fiducial_B{B}.dat') 
beta_sim_array = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/TG_256_planck_2/betaj_sims_TS_galS_jmax12_B1.7422102429630426_nside256.dat')
beta_sim = np.mean(beta_sim_array, axis=0)


fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.plot(np.arange(1,jmax), beta_sim[1:-1]/beta_theory[1:-1] -1 )

ax.set_xlabel(r'j')
ax.set_ylabel(r'$\Delta \beta_{j}^{TG}$-1')

plt.savefig('comparison_betaj_theory_sims_relative_diff.png')

fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.plot(np.arange(1,jmax), beta_theory[1:-1], label = r'$\beta_{j}^{TG}$ theory')
ax.plot(np.arange(1,jmax), beta_sim[1:-1], label = r'$\beta_{j}^{TG}$ sims')

ax.set_xlabel(r'j')
ax.set_ylabel(r'$\beta_{j}^{TG}$')
plt.legend()

plt.savefig('comparison_betaj_theory_sims.png')

## Cl's

jmax = 12
lmax = 800
nsim = 500
lmin=0
delta_ell = 50

cl_theory = np.loadtxt(f'/ehome/bdecaro/xcmbneed/src/spectra/inifiles/CAMBSpectra_OmL_fiducial_lmin{lmin}.dat')
cl_theory_tg = cl_theory[2]
cl_theory_tt = cl_theory[1]
cl_theory_gg = cl_theory[3]

print(cl_theory_tg.shape)
cl_sim_tg_array = np.loadtxt(f'cls_Tgal_anafast_nside{nside}_lmax{lmax}_lmin{lmin}.dat')
cl_sim_tg = np.mean(cl_sim_tg_array,axis=0)

cl_sim_tt_array = np.loadtxt(f'cls_TT_anafast_nside{nside}_lmax{lmax}_lmin{lmin}.dat')
cl_sim_tt = np.mean(cl_sim_tt_array,axis=0)

cl_sim_gg_array = np.loadtxt(f'cls_galgal_anafast_nside{nside}_lmax{lmax}_lmin{lmin}.dat')
cl_sim_gg = np.mean(cl_sim_gg_array,axis=0)



fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.plot( cl_sim_tg[2:]/cl_theory_tg[2:lmax+1] -1 )

ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\Delta C_{\ell}^{TG}$-1')

plt.savefig('comparison_cl_tg_theory_sims_relative_diff_lmin0.png')

fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.plot( cl_sim_tt[2:]/cl_theory_tt[2:lmax+1] -1 )

ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\Delta C_{\ell}^{TT}$-1')

plt.savefig('comparison_cl_tt_theory_sims_relative_diff_lmin0.png')


fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.plot( cl_sim_gg[2:]/cl_theory_gg[2:lmax+1] -1 )

ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\Delta C_{\ell}^{GG}$-1')

plt.savefig('comparison_cl_gg_theory_sims_relative_diff_lmin0s.png')

 
