import numpy as np
import matplotlib.pyplot as plt

cl = np.loadtxt('/ehome/bdecaro/xcmbneed/src/spectra/inifiles/EUCLID_fiducial.dat')#np.loadtxt('/ehome/bdecaro/xcmbneed/src/spectra/inifiles/CAMBSpectra_OmL_fiducial.dat')

ell = cl[0]
tt = cl[1]
tg = cl[2]
gg = cl[3]

cl_spectra = np.loadtxt('/ehome/bdecaro/xcmbneed/src/spectra/inifiles/CAMBSpectra_OmL_fiducial.dat')#np.loadtxt('/ehome/bdecaro/xcmbneed/src/spectra/cl_spectra.dat')

ell_spectra = cl_spectra[0]
tt_spectra = cl_spectra[1]
tg_spectra = cl_spectra[2]
gg_spectra = cl_spectra[3]


fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.plot(ell_spectra[1:500], (ell_spectra[1:500]*ell_spectra[1:500]+1)/(2*np.pi)*gg_spectra[1:500], label='io')
ax.plot(ell[1:500], (ell[1:500]*ell[1:500]+1)/(2*np.pi)*gg[1:500], label=' marina')

ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$C_{\ell}^{TG}$')

ax.legend()

plt.savefig('cl_gg_EUCLID_marina.png')

#fig = plt.figure(figsize=(17,10))
#ax = fig.add_subplot(1, 1, 1)
#
#ax.plot(ell[1:500], (tg[1:500]/tg_spectra[1:500])-1 )
#ax.set_xlabel(r'$\ell$')
#ax.set_ylabel(r'$\Delta C_{\ell}^{TG}$-1')
#plt.savefig('cl_tg_diff_camb_spectra_module_1.png')
#
#fig = plt.figure(figsize=(17,10))
#ax = fig.add_subplot(1, 1, 1)
#
#ax.plot(ell[1:500], (tt[1:500]/tt_spectra[1:500])-1 )
#ax.set_xlabel(r'$\ell$')
#ax.set_ylabel(r'$\Delta C_{\ell}^{TT}$-1')
#plt.savefig('cl_tt_diff_camb_spectra_module_1.png')
#
fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.plot( (gg[:500]/gg_spectra[1:500]) )
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\Delta C_{\ell}^{gg}$')
plt.savefig('cl_gg_diff_camb_spectra_module_marina.png')

#cl_camb = np.loadtxt('CAMBPietrobon.dat')
#
#ell_camb = cl_camb[0]
#tt_camb = cl_camb[1]
#tg_camb = cl_camb[2]
#gg_camb = cl_camb[3]
#
#fig = plt.figure(figsize=(17,10))
#ax = fig.add_subplot(1, 1, 1)
#
#ax.plot(ell_camb[1:500], tg_camb[1:500])
#ax.set_xlabel(r'$\ell$')
#ax.set_ylabel(r'$\ell(\ell+1)C_{\ell}^{TG}$')
#
#plt.savefig('cl_tg_pietrobon.png')

