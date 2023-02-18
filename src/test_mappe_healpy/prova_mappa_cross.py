import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = '18'

map_galS = hp.read_map('../sims/Needlet/TGsims_512/sim_0125_galS_0512.fits')
map_TS = hp.read_map('../sims/Needlet/TGsims_512/sim_0125_TS_0512.fits')

Cls_camb = np.loadtxt('../spectra/cl_camb.dat')

Cls_TT = hp.sphtfunc.anafast(map1=map_TS, map2=map_TS)#, use_pixel_weights= True)
Cls_TG = hp.sphtfunc.anafast(map1=map_TS, map2=map_galS)#, use_pixel_weights= True)
Cls_GG = hp.sphtfunc.anafast(map1=map_galS, map2=map_galS)#, use_pixel_weights= True)

Cls_camb_TT = Cls_camb[1]
Cls_camb_TG = Cls_camb[2]
Cls_camb_GG = Cls_camb[3]

ell = np.arange(500)
delta_ell = 20

Cls_TG_bin_ell = np.array([ell[l:l+delta_ell]*(ell[l:l+delta_ell]+1)*Cls_TG[l:l+delta_ell] for l in range(0,500,delta_ell)])
Cls_TG_bin = np.array([Cls_TG[l:l+delta_ell] for l in range(0,500,delta_ell)])
Cls_TG_bin_mean = np.array([np.mean(Cls_TG[l:l+delta_ell], axis = 0) for l in range(0,500,delta_ell)])
ell_d = np.arange(stop=500, step= delta_ell)

print(Cls_TG_bin_ell)

fig, ax = plt.subplots(1,1,figsize = (17,10))
ax.plot(ell*(ell+1)*Cls_camb_TG[ell], '-', label ='Camb')
ax.errorbar(ell_d, ell_d*(ell_d+1)*Cls_TG_bin_mean, yerr = np.std(Cls_TG_bin_ell)/np.sqrt(delta_ell),fmt='o', label ='Healpix mean')
#ax.plot(ell_d, Cls_TG_bin_mean, 'o', label ='Healpix mean')
ax.set_xlabel(r'$\ell$', fontsize = 20)
ax.set_ylabel(r'$C_{\ell}^{TG}$',fontsize = 20)

ax.legend(loc = 'best',frameon = True,fontsize=15)

fig.tight_layout()
plt.savefig('prova_cls_ell_bin.png')

#prova con ell*(ell+1)

fig, ax = plt.subplots(1,1,figsize = (17,10))
ax.plot(Cls_camb_TG[ell], '-', label ='Camb')
ax.errorbar(ell_d, Cls_TG_bin_mean, yerr = np.std(Cls_TG_bin)/np.sqrt(delta_ell),fmt='o', label ='Healpix mean')
#ax.plot(ell_d, Cls_TG_bin_mean, 'o', label ='Healpix mean')
ax.set_xlabel(r'$\ell$', fontsize = 20)
ax.set_ylabel(r'$C_{\ell}^{TG}$',fontsize = 20)

ax.legend(loc = 'best',frameon = True,fontsize=15)

fig.tight_layout()
plt.savefig('prova_cls_bin.png')