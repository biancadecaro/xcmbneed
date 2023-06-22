import numpy as np
import analysis, utils, spectra, sims
from scipy.stats import chi2
import matplotlib.pyplot as plt


cov = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/TG_512/cov_TS_galS_jmax10_B1.9467970312828855_nside512.dat', unpack=True)#np.loadtxt('output_needlet_TG/Planck/TG_256_mask_noise_nonoisePlanck/cov_TT_galT_jmax12_B = 1.74 $_nside256_mask.dat', unpack=True)
beta_j_sims = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/TG_512/betaj_sims_TS_galS_jmax10_B1.9467970312828855_nside512.dat', unpack=True)#np.loadtxt('output_needlet_TG/Planck/TG_256_mask_noise_nonoisePlanck/betaj_sims_TT_galT_jmax12_B_1.7422102429630426_nside256_fsky_0.7004634737968445.dat', unpack=True)

beta_j_mean = np.mean(beta_j_sims, axis=1)

print(cov.shape, beta_j_sims.shape, beta_j_mean.shape)

beta_j_sim_400 = beta_j_sims[:, 400]

#print(beta_j_sim_400)

cov_inv = np.linalg.inv(cov)

chi_squared = 0

temp = np.zeros(len(cov[0]))
for i in range(1,len(cov[0])):
    for j in range(1,len(beta_j_sim_400)):
        temp[i] += cov_inv[i][j]*(beta_j_sim_400[j]-beta_j_mean[j])
    chi_squared += (beta_j_sim_400[i]-beta_j_mean[i]).T*temp[i]


print('chi squared=', chi_squared, chi2.cdf(chi_squared, beta_j_mean.shape[0]-1))

#Null hypothesis

#beta_j_sims_null = np.loadtxt('output_needlet_TG_Null/betaj_sims_TS_galS_jmax12_B1.7422102429630426_nside512.dat', unpack=True)
#beta_j_sim_400_null = beta_j_sims_null[:, 400]
#
#chi_squared_null = 0
#
#temp_n = np.zeros(len(cov[0]))
#for i in range(1,len(cov[0])):
#    for j in range(1,len(beta_j_sim_400)):
#        temp_n[i] += cov_inv[i][j]*(beta_j_sim_400_null[j]-beta_j_sim_400[j])
#    chi_squared_null += (beta_j_sim_400_null[i]-beta_j_sim_400[i]).T*temp_n[i]
#print('chi squared null =' ,chi_squared_null, chi2.cdf(chi_squared_null, 12), chi2.cdf(29.8, 12))


#simparams = {'nside'   : 512,
#             'ngal'    : 5.76e5,
# 	     	 'ngal_dim': 'ster',
#	     	 'pixwin'  : False}
#sims_dir        = 'sims/Needlet/TGsims_'+str(simparams['nside'])+'/'
#out_dir         = 'output_needlet_TG/'
#
#jmax = 12
#lmax = 782
#nsims = 500
#
#fname_xcspectra = 'spectra/CAMBSpectra.dat'
#xcspectra = spectra.XCSpectraFile(fname_xcspectra, WantTG = True)
#
#simulations = sims.KGsimulations(xcspectra, sims_dir, simparams, WantTG = True)
##simulations.Run(nsim, WantTG = True)
#
#myanalysis = analysis.NeedAnalysis(jmax, lmax, out_dir, simulations)
#need_theory = spectra.NeedletTheory(myanalysis.B)
#
#delta = need_theory.delta_beta_j(jmax,  cltt = xcspectra.cltt, cltg = xcspectra.cltg, clgg = xcspectra.clg1g1)
#
#chi_squared_delta = 0#np.sum((beta_j_sim_400-beta_j_mean)**2/delta**2)
#for j in range(jmax+1):
#    chi_squared_delta += (beta_j_sim_400[j]-beta_j_mean[j])**2/delta[j]**2
#
#
#print(chi_squared, chi_squared_delta)
#
#
#perc_1 = chi2.cdf(chi_squared, 13)
#perc_2 = chi2.cdf(chi_squared_delta, 13)
#
#print(perc_1, perc_2)
#
#fig = plt.figure(figsize=(17,10))
#ax = fig.add_subplot(1, 1, 1)
#
#ax.errorbar(np.arange(jmax+1), beta_j_sim_400, yerr=delta,fmt='o')
#ax.errorbar(np.arange(jmax+1), beta_j_mean, yerr=delta/np.sqrt(nsims-1),fmt='ro')
##ax.plot(np.arange(jmax+1), need_theory.cl2betaj(jmax, xcspectra.cltg), 'ro')
##ax.errorbar(np.arange(jmax+1), beta_j_mean, yerr=np.sqrt(np.diag(cov))/np.sqrt(nsims-1),fmt='o')
#ax.set_ylabel(r'$\beta_j$')
#ax.set_xlabel(r'j')
#
#plt.savefig('betaj_mean_betaj_sim_plot'+str(jmax)+'_B_'+str(myanalysis.B)+'_nsim'+str(nsims)+'_nside'+str(simparams['nside'])+'.png')
##plt.savefig('betaj_mean_plot_cov_err.png')
#
#chi_squared_cov_diag = np.sum((beta_j_sim_400-beta_j_mean)**2/need_theory.delta_beta_j_cov(jmax, cltt = xcspectra.cltt, cltg = xcspectra.cltg, clgg = xcspectra.clg1g1, cov = cov)**2)
##
##chi_squared_delta = np.sum((betaj_TS_galS_mean-betaj_sims_TS_galS[400,:])[1:jmax+1]**2/delta[1:jmax+1]**2)
##
##chi_squared = np.dot((betaj_TS_galS_mean-betaj_sims_TS_galS[400,:]), np.matmul(cov_inv,(betaj_TS_galS_mean-betaj_sims_TS_galS[400,:])[1:jmax+1] ))
##
#print('chi squared='+str(chi_squared)+', chi_squared_cov_diag= '+str(chi_squared_cov_diag)+',chi_squared_delta=' + str(chi_squared_delta))