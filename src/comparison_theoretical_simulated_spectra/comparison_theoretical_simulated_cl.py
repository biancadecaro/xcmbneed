import numpy as np
import matplotlib.pyplot as plt
import cython_mylibc as pippo
import analysis, spectra

## Cl's

nside = 128
lmax = 256
nsim = 100
lmin=0
delta_ell = 17

cl_theory = np.loadtxt('/ehome/bdecaro/xcmbneed/src/spectra/inifiles/EUCLID_fiducial_lmin0.dat')#np.loadtxt('/ehome/bdecaro/xcmbneed/src/spectra/inifiles/CAMBSpectra_planck_fiducial_lmin0_2050.dat')#np.loadtxt(f'/ehome/bdecaro/xcmbneed/src/spectra/inifiles/CAMBSpectra_OmL_fiducial_lmin{lmin}.dat')
cl_theory_tg = cl_theory[2]
cl_theory_tt = cl_theory[1]
cl_theory_gg = cl_theory[3]
ell_theory = cl_theory[0]

#print(ell_theory[:lmax])f'EUCLID/cls_galgalnoise_anafast_nside{nside}_lmax{lmax}_Euclidnoise_bianca.dat'

cl_sim_tg_array = np.loadtxt(f'../cls_from_maps/EUCLID/cls_Tgalnoise_anafast_nside{nside}_lmax{lmax}_Euclidnoise_bianca.dat')
cl_sim_tg = np.mean(cl_sim_tg_array,axis=0)

#cl_sim_tg_noise_array = np.loadtxt(f'../cls_from_maps/EUCLID/cls_Tgalnoise_anafast_nside{nside}_lmax{lmax}_bianca.dat')
#cl_sim_tg_noise = np.mean(cl_sim_tg_noise_array,axis=0)[2:]

cl_sim_tt_array = np.loadtxt(f'../cls_from_maps/EUCLID/cls_TT_anafast_nside{nside}_lmax{lmax}_Euclidnoise_bianca.dat')
cl_sim_tt = np.mean(cl_sim_tt_array,axis=0)

cl_sim_gg_array = np.loadtxt(f'../cls_from_maps/EUCLID/cls_galgalnoise_anafast_nside{nside}_lmax{lmax}_Euclidnoise_bianca.dat')
cl_sim_gg = np.mean(cl_sim_gg_array,axis=0)

#cl_sim_gg_noise_array = np.loadtxt(f'../cls_from_maps/EUCLID/cls_galgalnoise_anafast_nside{nside}_lmax{lmax}_bianca.dat')
#cl_sim_gg_noise = np.mean(cl_sim_gg_noise_array,axis=0)[2:]
ell_sim = np.arange(0, lmax+1)

print(ell_sim, ell_sim.shape)
print(ell_theory[:lmax+1], ell_theory[:lmax+1].shape)

#cl_ana_tg = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_Cl_TG/Planck/TG_512_1/cl_sims_TS_galS_lmax800_nside512.dat')s
print(cl_sim_tg.shape)

fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.plot( ell_sim*(ell_sim+1)/(2*np.pi)*cl_sim_tg , 'o', label=' sims')
ax.plot( ell_theory[:lmax+1]*(ell_theory[:lmax+1]+1)/(2*np.pi)*cl_theory_tg[:lmax+1] , 'o', label=' theory')
#ax.axhline(y=1)
#ax.set_xlim(0,50)
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$D_{\ell}^{TG}$-1')
plt.legend()
plt.savefig(f'EUCLID/dl_tg_theory_sims_nside{nside}_lmax{lmax}_lmin{lmin}_bianca.png')

fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.plot( cl_sim_tg/cl_theory_tg[:lmax+1] -1, 'o' )
ax.axhline(y=0)
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\Delta C_{\ell}^{TG}$-1')

plt.savefig(f'EUCLID/comparison_cl_tg_theory_sims_relative_diff_nside{nside}_lmax{lmax}_lmin{lmin}_bianca.png')

fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.plot( cl_sim_tt/cl_theory_tt[:lmax+1] -1, 'o' )
ax.axhline(y=0)
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\Delta C_{\ell}^{TT}$-1')

plt.savefig(f'EUCLID/comparison_cl_tt_theory_sims_relative_diff_nside{nside}_lmax{lmax}_lmin{lmin}_bianca.png')


fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.plot( cl_sim_gg/cl_theory_gg[:lmax+1] -1 , 'o')
ax.axhline(y=0)
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\Delta C_{\ell}^{GG}$-1')

plt.savefig(f'EUCLID/comparison_cl_gg_theory_sims_relative_diff_nside{nside}_lmax{lmax}_lmin{lmin}_bianca.png')


#Proviamo a binnare
 

Cls_TG_bin_ell = np.array([cl_sim_tg[l:(l+delta_ell)] for l in range(0,lmax-delta_ell,delta_ell)])
#print(Cls_TG_bin_ell)
Cls_TG_bin_mean = np.array([np.mean(Cls_TG_bin_ell[l]) for l in range(Cls_TG_bin_ell.shape[0])])
#print(Cls_TG_bin_mean)
Dls_TG_bin_ell = np.array([ell_sim[l:l+delta_ell]/(2*np.pi)*(ell_sim[l:l+delta_ell]+1)*cl_sim_tg[l:l+delta_ell] for l in range(0,lmax-delta_ell,delta_ell)])
#print(Dls_TG_bin_ell)
Dls_TG_bin_mean = np.array([np.mean(Dls_TG_bin_ell[l], axis = 0) for l in range(Dls_TG_bin_ell.shape[0])])

#print(Dls_TG_bin_mean.shape, Cls_TG_bin_ell.shape, Cls_TG_bin_mean.shape)
ell_d = np.array([ell_sim[l:l+delta_ell] for l in range(0,lmax-delta_ell,delta_ell)])

ell_d_mean=np.array([np.mean(ell_d[l], axis = 0) for l in range(ell_d.shape[0])])
print(ell_d_mean)

fig, ax = plt.subplots(1,1,figsize = (17,10))
ax.plot(ell_theory[:lmax+1],ell_theory[:lmax+1]*(ell_theory[:lmax+1]+1)/(2*np.pi)*cl_theory_tg[:lmax+1], '-', label ='Camb')
ax.plot(ell_sim,ell_sim*(ell_sim+1)/(2*np.pi)*cl_sim_tg, 'o', label ='Sims noise')
#ax.plot(ell_sim,ell_sim*(ell_sim+1)/(2*np.pi)*cl_sim_tg_noise, 'o', label ='Sims noise')
ax.errorbar(ell_d_mean,Dls_TG_bin_mean, yerr = np.std(Dls_TG_bin_ell, axis=1),fmt='o', label ='Healpix mean binned') #/np.sqrt(delta_ell)
#ax.plot(ell_d_mean,Dls_TG_bin_mean, 'o', label ='Healpix mean')
ax.set_xlabel(r'$\ell$', fontsize = 20)
ax.set_ylabel(r'$D_{\ell}^{TG}$',fontsize = 20)

ax.legend(loc = 'best',frameon = True,fontsize=15)

fig.tight_layout()
plt.savefig('EUCLID/dls_tg_ell_bin_euclid_bianca.png')

print(np.std(Dls_TG_bin_ell, axis=1))

#
fig, ax = plt.subplots(1,1,figsize = (17,10))
ax.plot(ell_sim,cl_theory_tg[ell_sim], '-', label ='Camb')
ax.plot(ell_sim,cl_sim_tg, 'o', label ='Sims')
ax.errorbar(ell_d_mean,Cls_TG_bin_mean, yerr = np.std(Cls_TG_bin_ell, axis=1),fmt='o', label ='Healpix mean binned') #/np.sqrt(delta_ell)
#ax.plot(ell_d, Cls_TG_bin_mean, 'o', label ='Healpix mean')
ax.set_xlabel(r'$\ell$', fontsize = 20)
ax.set_ylabel(r'$C_{\ell}^{TG}$',fontsize = 20)

ax.legend(loc = 'best',frameon = True,fontsize=15)

fig.tight_layout()
plt.savefig('EUCLID/cls_tg_ell_bin_euclid_bianca.png')



Cls_TT_bin_ell = np.array([cl_sim_tt[l:l+delta_ell] for l in range(0,lmax-delta_ell,delta_ell)])
Cls_TT_bin_mean = np.array([np.mean(Cls_TT_bin_ell[l], axis = 0) for l in range(Cls_TT_bin_ell.shape[0])])

Dls_TT_bin_ell = np.array([ell_theory[l:l+delta_ell]/(2*np.pi)*(ell_theory[l:l+delta_ell]+1)*cl_sim_tt[l:l+delta_ell] for l in range(0,lmax-delta_ell,delta_ell)])
Dls_TT_bin_mean = np.array([np.mean(Dls_TT_bin_ell[l], axis = 0) for l in range(Dls_TT_bin_ell.shape[0])])#np.array([np.mean(cl_sim_tg[l:l+delta_ell], axis = 0) for l in range(2,lmax+1,delta_ell)])
print(Dls_TT_bin_mean.shape, Cls_TT_bin_ell.shape, Cls_TT_bin_mean.shape)

fig, ax = plt.subplots(1,1,figsize = (17,10))
ax.plot(ell_theory[:lmax+1],ell_theory[:lmax+1]*(ell_theory[:lmax+1]+1)/(2*np.pi)*cl_theory_tt[:lmax+1], '-', label ='Camb')
ax.plot(ell_sim,ell_sim*(ell_sim+1)/(2*np.pi)*cl_sim_tt, 'o', label ='Sims')
ax.errorbar(ell_d_mean,Dls_TT_bin_mean, yerr = np.std(Dls_TT_bin_ell, axis=1),fmt='o', label ='Healpix mean binned') #/np.sqrt(delta_ell)
#ax.plot(ell_d, Cls_TG_bin_mean, 'o', label ='Healpix mean')
ax.set_xlabel(r'$\ell$', fontsize = 20)
ax.set_ylabel(r'$D_{\ell}^{TT}$',fontsize = 20)

ax.legend(loc = 'best',frameon = True,fontsize=15)

fig.tight_layout()
plt.savefig('EUCLID/dls_tt_ell_bin_euclid_bianca.png')

fig, ax = plt.subplots(1,1,figsize = (17,10))
ax.plot(ell_sim,cl_theory_tt[ell_sim], '-', label ='Camb')
ax.plot(ell_sim,cl_sim_tt, 'o', label ='Sims')
ax.errorbar(ell_d_mean,Cls_TT_bin_mean, yerr = np.std(Cls_TT_bin_ell, axis=1),fmt='o', label ='Healpix mean binned') #/np.sqrt(delta_ell)
#ax.plot(ell_d, Cls_TG_bin_mean, 'o', label ='Healpix mean')
ax.set_xlabel(r'$\ell$', fontsize = 20)
ax.set_ylabel(r'$C_{\ell}^{TT}$',fontsize = 20)

ax.legend(loc = 'best',frameon = True,fontsize=15)

fig.tight_layout()
plt.savefig('EUCLID/cls_tt_ell_bin_euclid_bianca.png')

### GG

Cls_GG_bin_ell = np.array([cl_sim_gg[l:l+delta_ell] for l in range(0,lmax-delta_ell,delta_ell)])
Cls_GG_bin_mean = np.array([np.mean(Cls_GG_bin_ell[l], axis = 0) for l in range(Cls_GG_bin_ell.shape[0])])#np.array([np.mean(cl_sim_tg[l:l+delta_ell], axis = 0) for l in range(2,lmax+1,delta_ell)])

Dls_GG_bin_ell = np.array([ell_theory[l:l+delta_ell]/(2*np.pi)*(ell_theory[l:l+delta_ell]+1)*cl_sim_gg[l:l+delta_ell] for l in range(0,lmax-delta_ell,delta_ell)])
Dls_GG_bin_mean = np.array([np.mean(Dls_GG_bin_ell[l], axis = 0) for l in range(Dls_GG_bin_ell.shape[0])])

#Dls_GG_noise_bin_ell = np.array([ell_theory[l:l+delta_ell]/(2*np.pi)*(ell_theory[l:l+delta_ell]+1)*cl_sim_gg_noise[l:l+delta_ell] for l in range(0,lmax-delta_ell,delta_ell)])
#Dls_GG_noise_bin_mean = np.array([np.mean(Dls_GG_noise_bin_ell[l], axis = 0) for l in range(Dls_GG_bin_ell.shape[0])])
#np.array([np.mean(cl_sim_tg[l:l+delta_ell], axis = 0) for l in range(2,lmax+1,delta_ell)])
print(Dls_GG_bin_mean.shape, Cls_GG_bin_ell.shape, Cls_GG_bin_mean.shape)

fig, ax = plt.subplots(1,1,figsize = (17,10))
ax.plot(ell_theory[:lmax+1],ell_theory[:lmax+1]*(ell_theory[:lmax+1]+1)/(2*np.pi)*cl_theory_gg[:lmax+1], '-', label ='Camb')
ax.plot(ell_sim,ell_sim*(ell_sim+1)/(2*np.pi)*cl_sim_gg, 'o', label ='Sims no noise')
#ax.plot(ell_sim,ell_sim*(ell_sim+1)/(2*np.pi)*cl_sim_gg_noise, 'o', label ='Sims noise')

ax.errorbar(ell_d_mean,Dls_GG_bin_mean, yerr = np.std(Dls_GG_bin_ell, axis=1),fmt='o', label ='Healpix mean binned') #/np.sqrt(delta_ell)
#ax.plot(ell_d, Cls_TG_bin_mean, 'o', label ='Healpix mean')
ax.set_xlabel(r'$\ell$', fontsize = 20)
ax.set_ylabel(r'$D_{\ell}^{GG}$',fontsize = 20)



ax.legend(loc = 'best',frameon = True,fontsize=15)

fig.tight_layout()
plt.savefig('EUCLID/dls_gg_ell_bin_euclid_bianca.png')

fig, ax = plt.subplots(1,1,figsize = (17,10))
ax.plot(ell_sim,cl_theory_gg[ell_sim], '-', label ='Camb')
ax.plot(ell_sim,cl_sim_gg, 'o', label ='Sims')
ax.errorbar(ell_d_mean,Cls_GG_bin_mean, yerr = np.std(Cls_GG_bin_ell, axis=1),fmt='o', label ='Healpix mean binned') #/np.sqrt(delta_ell)
#ax.plot(ell_d, Cls_TG_bin_mean, 'o', label ='Healpix mean')
ax.set_xlabel(r'$\ell$', fontsize = 20)
ax.set_ylabel(r'$C_{\ell}^{GG}$',fontsize = 20)

ax.legend(loc = 'best',frameon = True,fontsize=15)

fig.tight_layout()
plt.savefig('EUCLID/cls_gg_ell_bin_euclid_bianca.png')

