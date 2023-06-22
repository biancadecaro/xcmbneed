import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import analysis, spectra
import cython_mylibc as pippo

sns.color_palette("hls", 10)
sns.set()
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

nsim=1000
nside=128
jmax=12
lmax_cl=256
jvec = np.arange(0,jmax+1)
lbins=255
lbinsvec = np.arange(0,lbins)

cl_tg = np.loadtxt('/ehome/bdecaro/xcmbneed/src/spectra/inifiles/EUCLID_fiducial.dat')[2]


cl_sims=np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_Cl_TG/EUCLID/Mask_noise/TG_128_fsky0.35/cl_sims_TS_galT_lmax256_nside128.dat')

print(cl_sims.T[6].shape)
cl_sims_mean=np.mean(cl_sims, axis=0)

fig, ax = plt.subplots(1,1, figsize=(17,10))

for n in range(nsim):
    plt.plot(lbinsvec, cl_sims[n], 'bo')
plt.plot(lbinsvec, cl_sims_mean, 'ro')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$C_{\ell}^{TG}$')


plt.savefig('Euclid_scatter_sims_cl.png')

fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel(r'$\ell$=0')
sns.histplot(cl_sims.T[0], stat='density',bins=100,element='step',fill=True, color='b',ax=ax)
#ax.set_xlim(0.3, 1.3)
ax.axvline(np.mean(cl_sims.T[0]))

plt.tight_layout()
plt.savefig('Euclid_his_scatter_sims_cl.png')



beta_sims=np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000/betaj_sims_TS_galT_jmax12_B_1.59_nside128.dat')

beta_sims_mean=np.mean(beta_sims, axis=0)

B = pippo.mylibpy_jmax_lmax2B(jmax, lmax_cl)

need_theory = spectra.NeedletTheory(B)
betatg    = need_theory.cl2betaj(jmax=jmax, cl=cl_tg)

fig, ax = plt.subplots(1,1, figsize=(17,10))

for n in range(nsim):
    plt.plot(jvec, beta_sims[n], 'bo')
plt.plot(jvec, betatg, 'ro', label='Theory')
plt.plot(jvec, beta_sims_mean, 'go', label= 'Mean Sims')
ax.set_xlabel('j')
ax.set_ylabel(r'$\beta_{j}^{TG}$')
plt.legend()

plt.savefig('Euclid_scatter_sims_beta.png')

fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('j=5')
sns.histplot(beta_sims.T[6], stat='density',bins=100,element='step',fill=True, color='b',ax=ax)
#ax.set_xlim(0.3, 1.3)
ax.axvline(np.mean(beta_sims.T[6]))

plt.tight_layout()
plt.savefig('Euclid_his_scatter_sims_beta.png')