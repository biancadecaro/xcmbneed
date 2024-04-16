import numpy as np
import matplotlib.pyplot as plt

OmL = np.linspace(0.0,0.95,30)
jmax = 12
lmax = 782

beta_oml = np.array([np.loadtxt(f'/ehome/bdecaro/xcmbneed/src/output_needlet_TG_OmL/Grid_spectra_30_planck_2_lmin0/TGsims_theoretical_OmL{oml}/beta_TS_galS_theoretical_OmL{oml}_B1.7422102429630426.dat') for oml in OmL])#np.array([np.loadtxt(f'/ehome/bdecaro/xcmbneed/src/output_needlet_TG_OmL/Grid_spectra_30_planck_2_lmin0/TGsims_512_OmL{oml}/betaj_sims_TS_galS_Om{oml}_jmax12_B = 1.74 _nside512.dat') for oml in OmL])[:,1:(jmax)]
beta_oml_no_mnu = np.array([np.loadtxt(f'/ehome/bdecaro/xcmbneed/src/output_needlet_TG_OmL/Grid_spectra_30_old/TGsims_theoretical_OmL{oml}/beta_TS_galS_theoretical_OmL{oml}_B1.7422102429630426.dat') for oml in OmL])

cl_oml = np.array([np.loadtxt(f'spectra/Grid_spectra_{len(OmL)}_planck/CAMBSpectra_OmL{oml}_lmin0.dat') for oml in OmL])
cl_oml_no_mnu = np.array([np.loadtxt(f'spectra/Grid_spectra_{len(OmL)}_no_mnu_old/CAMBSpectra_OmL{oml}_lmin0.dat') for oml in OmL])

cov_matrix = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/TG_512_planck_2_lmin0/cov_TS_galS_jmax12_B1.7422102429630426_nside512.dat')
beta_fid_array = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/TG_512_planck_2_lmin0/betaj_sims_TS_galS_jmax12_B1.7422102429630426_nside512.dat')
cl_fid_array = np.loadtxt(f'spectra/Grid_spectra_{len(OmL)}_planck/CAMBSpectra_OmL_fiducial_lmin0.dat')

beta_fid =   np.mean(beta_fid_array, axis=0)
cl_fid_TG =   cl_fid_array[2]

print(cl_fid_TG.shape, beta_oml.shape)

## CL
fig1 = plt.figure(figsize=(17,10))
ax1 = fig1.add_subplot(1, 1, 1)

for p in range(len(OmL)):
    ax1.plot((cl_oml_no_mnu[p,2][2:500]-cl_oml[p,2][2:500])/cl_oml[p,2][2:500], label= f'OmL={OmL[p]}')
#ax1.plot((cl_oml_no_mnu[21,2][2:500]-cl_oml[21,2][2:500])/cl_oml[21,2][2:500], label= f'OmL={OmL[21]}')
ax1.set_xlabel(r'$\ell$')
ax1.set_ylabel(r'C$_{\ell~no~mnu}^{TG}$/C$_{\ell}^{TG}$-1')
plt.legend()
plt.savefig('plot_prova_neutrini/relative_difference_cl_oml_mnu_no_mnu.png')

fig2 = plt.figure(figsize=(17,10))
ax2 = fig2.add_subplot(1, 1, 1)

ax2.plot(cl_oml_no_mnu[21,2][:1000]-cl_oml[21,2][:1000], label= f'OmL={OmL[21]}')
ax2.set_xlabel(r'$\ell$')
ax2.set_ylabel(r'C$_{\ell~no~mnu}^{TG}$-C$_{\ell}^{TG}$')
plt.legend()
plt.savefig('plot_prova_neutrini/difference_cl_oml_mnu_no_mnu.png')

fig3 = plt.figure(figsize=(17,10))
ax3 = fig3.add_subplot(1, 1, 1)

for p in range(len(OmL)):
    ax3.plot((cl_oml_no_mnu[p,2][2:500]-cl_fid_TG[2:500])/cl_fid_TG[2:500], label= f'OmL={OmL[p]}')
#ax1.plot((cl_oml_no_mnu[21,2][2:500]-cl_oml[21,2][2:500])/cl_oml[21,2][2:500], label= f'OmL={OmL[21]}')
ax3.set_xlabel(r'$\ell$')
ax3.set_ylabel(r'C$_{\ell~no~mnu}^{TG}$/C$_{\ell}^{TG}$-1')
plt.legend()
plt.savefig('plot_prova_neutrini/relative_difference_cl_fid_mnu_no_mnu.png')

fig4 = plt.figure(figsize=(17,10))
ax4 = fig4.add_subplot(1, 1, 1)

ax4.plot(cl_oml_no_mnu[21,2][:1000]-cl_fid_TG[:1000], label= f'OmL={OmL[21]}')
ax4.set_xlabel(r'$\ell$')
ax4.set_ylabel(r'C$_{\ell~no~mnu}^{TG}$-C$_{\ell}^{TG}$')
plt.legend()
plt.savefig('plot_prova_neutrini/difference_cl_fid_mnu_no_mnu.png')

## NEEDLET
fig5 = plt.figure(figsize=(17,10))
ax5 = fig5.add_subplot(1, 1, 1)

for p in range(len(OmL)):
    ax5.plot((beta_oml_no_mnu[p]-beta_oml[p])/beta_oml[p], label= f'OmL={OmL[p]}')
#ax1.plot((cl_oml_no_mnu[21,2][2:500]-cl_oml[21,2][2:500])/cl_oml[21,2][2:500], label= f'OmL={OmL[21]}')
ax5.set_xlabel(r'j')
ax5.set_ylabel(r'$\beta_{j~no~mnu}^{TG}$/$\beta_{j}^{TG}$-1')
plt.legend()
plt.savefig('plot_prova_neutrini/relative_difference_beta_oml_mnu_no_mnu.png')

fig6 = plt.figure(figsize=(17,10))
ax6 = fig6.add_subplot(1, 1, 1)

ax6.plot(beta_oml_no_mnu[21]-beta_oml[21], label= f'OmL={OmL[21]}')
ax6.set_xlabel(r'j')
ax6.set_ylabel(r'$\beta_{j~no~mnu}^{TG}$-$\beta_{j}^{TG}$')
plt.legend()
plt.savefig('plot_prova_neutrini/difference_beta_oml_mnu_no_mnu.png')

fig7 = plt.figure(figsize=(17,10))
ax7 = fig7.add_subplot(1, 1, 1)

for p in range(len(OmL)):
    ax7.plot(np.arange(jmax+1)[1:jmax],(beta_oml_no_mnu[p][1:jmax]-beta_fid[1:jmax])/(beta_oml[p][1:jmax]-beta_fid[1:jmax]), label= f'OmL={OmL[p]}')
#ax1.plot((cl_oml_no_mnu[21,2][2:500]-cl_oml[21,2][2:500])/cl_oml[21,2][2:500], label= f'OmL={OmL[21]}')
ax7.set_xlabel(r'j')
ax7.set_ylabel(r'$\beta_{j~no~mnu}^{TG}$/$\beta_{j}^{TG}$-1')
plt.legend()
plt.savefig('plot_prova_neutrini/relative_difference_beta_fid_mnu_no_mnu.png')

fig8 = plt.figure(figsize=(17,10))
ax8 = fig8.add_subplot(1, 1, 1)

ax8.plot(np.arange(jmax+1)[1:jmax], (beta_oml_no_mnu[21][1:jmax]-beta_fid[1:jmax])/(beta_oml[21][1:jmax]-beta_fid[1:jmax]), label= f'OmL={OmL[21]}')
ax8.set_xlabel(r'j')
ax8.set_ylabel(r'$\beta_{j~no~mnu}^{TG}$-$\beta_{j}^{TG}$')
plt.legend()
plt.savefig('plot_prova_neutrini/difference_beta_fid_mnu_no_mnu.png')
