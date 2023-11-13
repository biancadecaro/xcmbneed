import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = '18'

map_galS = hp.read_map('/ehome/bdecaro/xcmbneed/src/sims/Needlet/Planck/TGsims_512_planck_2_lmin0/sim_0125_galS_0512.fits')
map_TS = hp.read_map('/ehome/bdecaro/xcmbneed/src/sims/Needlet/Planck/TGsims_512_planck_2_lmin0/sim_0125_TS_0512.fits')

Cls_camb = np.loadtxt('/ehome/bdecaro/xcmbneed/src/spectra/inifiles/CAMBSpectra_OmL_fiducial_lmin0.dat')

Cls_TT = hp.sphtfunc.anafast(map1=map_TS, map2=map_TS)#, use_pixel_weights= True)
Cls_TG = hp.sphtfunc.anafast(map1=map_TS, map2=map_galS)#, use_pixel_weights= True)
Cls_GG = hp.sphtfunc.anafast(map1=map_galS, map2=map_galS)#, use_pixel_weights= True)
Cls_TG_analysis_array = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_Cl_TG/Planck/TG_512_planck_2_lmin0/cl_sims_TS_galS_lmax800_nside512.dat')
Cls_TG_analysis = np.mean(Cls_TG_analysis_array, axis=0)
print(Cls_TG_analysis.shape)


Cls_camb_TT = Cls_camb[1]
Cls_camb_TG = Cls_camb[2]
Cls_camb_GG = Cls_camb[3]

ell = np.arange(0,500)
delta_ell = 20

Cls_TG_bin_ell = np.array([ell[l:l+delta_ell]*(ell[l:l+delta_ell]+1)*Cls_TG[l:l+delta_ell] for l in range(0,500,delta_ell)])
Cls_TG_bin = np.array([Cls_TG[l:l+delta_ell] for l in range(0,500,delta_ell)])
Cls_TG_bin_mean = np.array([np.mean(Cls_TG[l:l+delta_ell], axis = 0) for l in range(0,500,delta_ell)])
ell_d = np.arange(stop=500, step= delta_ell)

#print(Cls_TG_bin_ell)
#prova con ell*(ell+1)
fig, ax = plt.subplots(1,1,figsize = (17,10))
ax.plot(ell*(ell+1)*Cls_camb_TG[ell], '-', label ='Camb')
ax.errorbar(ell_d, ell_d*(ell_d+1)*Cls_TG_bin_mean, yerr = np.std(Cls_TG_bin_ell)/np.sqrt(delta_ell),fmt='o', label ='Healpix mean')
#ax.plot(ell_d, Cls_TG_bin_mean, 'o', label ='Healpix mean')
ax.set_xlabel(r'$\ell$', fontsize = 20)
ax.set_ylabel(r'$C_{\ell}^{TG}$',fontsize = 20)

ax.legend(loc = 'best',frameon = True,fontsize=15)

fig.tight_layout()
plt.savefig('prova_dls_ell_bin_planck_lmin0.png')



fig, ax = plt.subplots(1,1,figsize = (17,10))
ax.plot(Cls_camb_TG[ell], '-', label ='Camb')
ax.errorbar(ell_d, Cls_TG_bin_mean, yerr = np.std(Cls_TG_bin)/np.sqrt(delta_ell),fmt='o', label ='Healpix mean')
#ax.plot(ell_d, Cls_TG_bin_mean, 'o', label ='Healpix mean')
ax.set_xlabel(r'$\ell$', fontsize = 20)
ax.set_ylabel(r'$C_{\ell}^{TG}$',fontsize = 20)

ax.legend(loc = 'best',frameon = True,fontsize=15)

fig.tight_layout()
plt.savefig('prova_cls_ell_bin_planck_lmin0.png')

ell1=np.arange(0,800)
ell_d1=np.linspace(0,800, 15)

fig, ax = plt.subplots(1,1,figsize = (17,10))
ax.plot(Cls_camb_TG[ell1], '-', label ='Camb')
ax.errorbar(ell_d1, Cls_TG_analysis, yerr = np.std(Cls_TG_analysis_array)/np.sqrt(delta_ell),fmt='o', label ='Cls from analysis')
#ax.plot(ell_d, Cls_TG_bin_mean, 'o', label ='Healpix mean')
ax.set_xlabel(r'$\ell$', fontsize = 20)
ax.set_ylabel(r'$C_{\ell}^{TG}$',fontsize = 20)

ax.legend(loc = 'best',frameon = True,fontsize=15)

fig.tight_layout()
<<<<<<< HEAD
plt.savefig('prova_cls_analysis_ell_bin_planck_lmin0.png')

=======
plt.savefig('prova_cls_analysis_ell_bin_planck_lmin0.png')
>>>>>>> euclid_implementation
