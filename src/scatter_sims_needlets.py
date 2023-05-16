import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.color_palette("hls", 10)
sns.set()
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

nsim=500
nside=512
jmax=12
lmax_cl=800
jvec = np.arange(0,jmax+1)
lbins=15
lbinsvec = np.arange(0,lbins)

cl_sims=np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_Cl_TG/Planck/TG_512_planck_2_lmin0/cl_sims_TS_galS_lmax800_nside512.dat')

print(cl_sims.T[6].shape)
cl_sims_mean=np.mean(cl_sims, axis=0)

fig, ax = plt.subplots(1,1, figsize=(17,10))

for n in range(nsim):
    plt.plot(lbinsvec, cl_sims[n], 'bo')
plt.plot(lbinsvec, cl_sims_mean, 'ro')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$C_{\ell}^{TG}$')


plt.savefig('scatter_sims_cl.png')

fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel(r'$\ell$=0')
sns.histplot(cl_sims.T[0], stat='density',bins=100,element='step',fill=True, color='b',ax=ax)
#ax.set_xlim(0.3, 1.3)
ax.axvline(np.mean(cl_sims.T[0]))

plt.tight_layout()
plt.savefig('his_scatter_sims_cl.png')



beta_sims=np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/TG_512_planck_2_lmin0/betaj_sims_TS_galS_jmax12_B1.7422102429630426_nside512.dat')

beta_sims_mean=np.mean(beta_sims, axis=0)

fig, ax = plt.subplots(1,1, figsize=(17,10))

for n in range(nsim):
    plt.plot(jvec, beta_sims[n], 'bo')
plt.plot(jvec, beta_sims_mean, 'ro')
ax.set_xlabel('j')
ax.set_ylabel(r'$\beta_{j}^{TG}$')


plt.savefig('scatter_sims_beta.png')

fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('j=5')
sns.histplot(beta_sims.T[6], stat='density',bins=100,element='step',fill=True, color='b',ax=ax)
#ax.set_xlim(0.3, 1.3)
ax.axvline(np.mean(beta_sims.T[6]))

plt.tight_layout()
plt.savefig('his_scatter_sims_beta.png')