import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

map_galS = hp.read_map('../sims/Needlet/TGsims_512/sim_0002_galS_0512.fits')
map_TS = hp.read_map('../sims/Needlet/TGsims_512/sim_0002_TS_0512.fits')

#map_galS = hp.read_map('../sims/Needlet/sims_256/sim_0002_deltaS_0256.fits')
#map_TS = hp.read_map('../sims/Needlet/sims_256/sim_0002_kappaS_0256.fits')

map_galS_null = hp.read_map('../sims/Needlet/NullTGsims_512/sim_0002_galS_0512.fits')
map_TS_null = hp.read_map('../sims/Needlet/NullTGsims_512/sim_0002_TS_0512.fits')

Cls_camb = np.loadtxt('../spectra/CAMBSpectra.dat')

#alm_gal = hp.sphtfunc.map2alm(maps= map_galS,lmax=1000)
#alm_T = hp.sphtfunc.map2alm(maps= map_TS,lmax=1000)
Cls_TT = hp.sphtfunc.anafast(map1=map_TS, map2=map_TS, iter = 9)#, use_pixel_weights= True)
Cls_TG = hp.sphtfunc.anafast(map1=map_TS, map2=map_galS, iter = 9)#, use_pixel_weights= True)
Cls_GG = hp.sphtfunc.anafast(map1=map_galS, map2=map_galS, iter = 9)#, use_pixel_weights= True)

Cls_null = hp.sphtfunc.anafast(map1=map_TS_null, map2=map_galS_null, iter = 9)#, use_pixel_weights= True)

Cls_camb_TT = Cls_camb[1]
Cls_camb_TG = Cls_camb[2]
Cls_camb_GG = Cls_camb[3]
#Cls = hp.sphtfunc.alm2cl(alms1 =alm_T, alms2=alm_gal )

ell = np.arange(500)

#fig = plt.figure(figsize=(17,10))
#ax1 = fig.add_subplot(1, 1, 1)
#
#ax1.plot(ell, ell*(ell+1)*Cls, '.')
#ax1.set_xlabel(r'$\ell$')
#ax1.set_ylabel(r'$\ell(\ell+1)C_{\ell}^{TG}$')
#
#ax2 = fig.add_subplot(1, 2, 2)
#
#ax2.plot(ell, (Cls-Cls_null)/Cls_null, '.')
#ax2.set_xlabel(r'$\ell$')
#ax2.set_ylabel(r'$(C_{\ell}^{TG}-C_{\ell}^{TG~Null})/C_{\ell}^{TG~Null}$')
plt.rcParams['font.size'] = '18'

#fig, (ax1, ax2,ax3) = plt.subplots(2,1, figsize=(35,15))
fig, (ax2,ax3) = plt.subplots(2,1, figsize=(30,18))
#ax1.plot(ell, ell*(ell+1)*Cls[ell], '.')
#ax1.set_xlabel(r'$\ell$')
#ax1.set_ylabel(r'$\ell(\ell+1)C_{\ell}^{TG}$', fontsize = 18)

ax2.plot(ell, (ell*(ell+1)*Cls_TG[ell]-Cls_camb_TG[ell])/Cls_camb_TG[ell])
ax2.set_xlabel(r'$\ell$')
ax2.set_ylabel(r'$(C_{\ell}^{TG}/C_{\ell}^{TG~Camb})-1$',fontsize = 20)

ax3.plot(ell, (Cls_TG[ell]-Cls_null[ell])/Cls_null[ell])
ax3.set_xlabel(r'$\ell$')
ax3.set_ylabel(r'$(C_{\ell}^{TG}/C_{\ell}^{TG~Null})-1$',fontsize = 20)

fig.tight_layout()
plt.savefig('cls_tg_healpix_ratio_camb.png')

fig, ax = plt.subplots(1,1,figsize = (17,10))
ax.plot(Cls_camb_TG[ell], 'o', label ='CAMB')
ax.plot(ell*(ell+1)*Cls_TG[ell], 'o', label ='Healpix')
ax.set_xlabel(r'$\ell$', fontsize = 20)
ax.set_ylabel(r'$\ell*(\ell+1)*C_{\ell}^{TG}$',fontsize = 20)

ax.legend(loc = 'best',frameon = True,fontsize=15)

fig.tight_layout()
plt.savefig('cls_tg_healpix_cls_tg_camb.png')

#Cls Federico

map_deltaS = hp.read_map('../sims/Needlet/sims_256/sim_0009_deltaS_0256.fits')
map_kappaS = hp.read_map('../sims/Needlet/sims_256/sim_0009_kappaS_0256.fits')

Cls_deltak = hp.sphtfunc.anafast(map1=map_deltaS, map2=map_kappaS)
Cls_deltadelta = hp.sphtfunc.anafast(map1=map_deltaS, map2=map_deltaS)

Cls_lensing_camb = np.loadtxt('../spectra/XCSpectra.dat')

Cls_deltak_camb = Cls_lensing_camb[ell,1]
Cls_deltadelta_camb = Cls_lensing_camb[ell,3]

fig, ax = plt.subplots(1,1,figsize = (17,10))
ax.plot(Cls_deltak_camb[ell], 'o', label ='CAMB')
ax.plot(Cls_deltak[ell], 'o', label ='Healpix')
ax.set_xlabel(r'$\ell$', fontsize = 20)
ax.set_ylabel(r'$C_{\ell}^{kG}$',fontsize = 20)

ax.legend(loc = 'best',frameon = True,fontsize=15)

fig.tight_layout()
plt.savefig('cls_kG_healpix_cls_camb_kG_lensing.png')


# Prova Null-Test

fig, ax = plt.subplots(1,1,figsize = (17,10))
ax.plot(ell*(ell+1)*Cls_TG[ell], 'o', label ='TG')
ax.plot(ell*(ell+1)*Cls_null[ell], 'o', label ='TG Null')
ax.set_xlabel(r'$\ell$', fontsize = 20)
ax.set_ylabel(r'$\ell(\ell+1)C_{\ell}^{TT}$',fontsize = 20)

ax.legend(loc = 'best',frameon = True,fontsize=15)

fig.tight_layout()
plt.savefig('cls_tg_null_test_cls_tg.png')

alms_null = np.loadtxt('../alms_nulltest.dat',dtype = complex)

alms_null_xx = alms_null[0]
alms_null_yy = alms_null[1]

cls_prova_null = hp.sphtfunc.alm2cl(alms1 = alms_null_xx, alms2 = alms_null_yy )

fig, ax = plt.subplots(1,1,figsize = (17,10))
ax.plot(ell*(ell+1)*cls_prova_null[ell], 'o', label ='Null')
ax.set_xlabel(r'$\ell$', fontsize = 20)
ax.set_ylabel(r'$\ell(\ell+1)C_{\ell}^{TG~Null}$',fontsize = 20)

ax.legend(loc = 'best',frameon = True,fontsize=15)

fig.tight_layout()
plt.savefig('cls_tg_null_test_from_alms.png')

# Prova TG 

alms_tg = np.loadtxt('../alms.dat',dtype = complex)

alms_tg_xx = alms_tg[0]
alms_tg_yy = alms_tg[1]

cls_prova_tg = hp.sphtfunc.alm2cl(alms1 = alms_tg_xx, alms2 = alms_tg_yy )

fig, ax = plt.subplots(1,1,figsize = (17,10))
ax.plot(ell*(ell+1)*cls_prova_tg[ell], 'o', label ='TG')
ax.plot(ell*(ell+1)*cls_prova_null[ell], 'o', label ='TG Null')
ax.set_xlabel(r'$\ell$', fontsize = 20)
ax.set_ylabel(r'$\ell(\ell+1)C_{\ell}^{TT}$',fontsize = 20)

ax.legend(loc = 'best',frameon = True,fontsize=15)

fig.tight_layout()
plt.savefig('cls_tg_null_test_cls_tg_from_alms.png')

#print(type(alms_tg_xx), type(alms_tg_yy))
#alms_tg_xy = alms_tg_xx-alms_tg_yy
#fig, ax = plt.subplots(1,1,figsize = (17,10))
#prova_mappa_tg = hp.sphtfunc.alm2map(alms = alms_tg_xx, nside = 512)
#hp.write
#plt.savefig('prova_mappa_tg_alms.png')

cls_prova_gg = hp.sphtfunc.alm2cl(alms1 = alms_tg_yy, alms2 = alms_tg_yy )

fig, ax = plt.subplots(1,1,figsize = (17,10))
ax.plot(ell*(ell+1)*cls_prova_gg[ell], 'o', label ='GG Healpix')
ax.plot(Cls_camb_GG[ell], 'o', label ='CAMB')
ax.set_xlabel(r'$\ell$', fontsize = 20)
ax.set_ylabel(r'$\ell*(\ell+1)*C_{\ell}^{GG}$',fontsize = 20)

ax.legend(loc = 'best',frameon = True,fontsize=15)

fig.tight_layout()
plt.savefig('cls_GG_healpix_alms_cls_camb.png')

# Prova GG

fig, ax = plt.subplots(1,1,figsize = (17,10))
ax.plot(ell*(ell+1)*Cls_GG[ell], 'o', label ='Healpix')
ax.plot(Cls_camb_GG[ell], 'o', label ='CAMB')
ax.set_xlabel(r'$\ell$', fontsize = 20)
ax.set_ylabel(r'$\ell*(\ell+1)*C_{\ell}^{GG}$',fontsize = 20)

ax.legend(loc = 'best',frameon = True,fontsize=15)

fig.tight_layout()
plt.savefig('cls_GG_healpix_cls_GG_camb.png')

#Prova TT

fig, ax = plt.subplots(1,1,figsize = (17,10))
ax.plot(Cls_camb_TT[ell], 'o', label ='CAMB')
ax.plot(ell*(ell+1)*Cls_TT[ell], 'o', label ='Healpix')
ax.set_xlabel(r'$\ell$', fontsize = 20)
ax.set_ylabel(r'$\ell*(\ell+1)*C_{\ell}^{TT}$',fontsize = 20)

ax.legend(loc = 'best',frameon = True,fontsize=15)

fig.tight_layout()
plt.savefig('cls_TT_healpix_cls_TT_camb.png')

# From Sims

cl_from_sims = np.loadtxt('../spectra/clsTG_from_sims.dat')


fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.plot(ell, (ell*(ell+1)*cl_from_sims[ell]-Cls_camb_TG[ell])/Cls_camb_TG[ell], 'o')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\ell(\ell+1)C_{\ell}^{TG}$')

plt.savefig('cl_tg_from_sims_ratio_cl_tg_camb.png')

cl_from_sims_null = np.loadtxt('../spectra/clsTG_from_sims_null.dat')


fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.plot(ell, (ell*(ell+1)*cl_from_sims_null[ell]-Cls_camb_TG[ell])/Cls_camb_TG[ell], 'o')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\ell(\ell+1)C_{\ell}^{Null~TG}$')

plt.savefig('cl_tg_null_from_sims_ratio_cl_tg_camb.png')

fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.plot(ell, ell*(ell+1)*cl_from_sims[ell], 'o', label = 'TG Sims')
ax.plot(ell, ell*(ell+1)*cl_from_sims_null[ell], 'o', label = 'TG Null Sims')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\ell(\ell+1)C_{\ell}^{TG}$')

ax.legend(loc = 'best',frameon = True,fontsize=15)
fig.tight_layout()

plt.savefig('cl_tg_null_sims_cl_tg_sims.png')