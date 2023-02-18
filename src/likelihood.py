import numpy as np
import matplotlib.pyplot as plt
import likelihood_analysis_module as liklh
from scipy.stats import chi2, mode

OmL = np.linspace(0.0,0.95,30)
jmax = 12
lmax = 782

beta_oml = np.array([np.loadtxt('output_needlet_TG_OmL/TGsims_512_OmL'+str(oml)+'/betaj_sims_TS_galS_Om'+str(oml)+'_jmax12_B = 1.74 _nside512.dat') for oml in OmL])

cov_matrix = np.loadtxt('output_needlet_TG/Planck/TG_512/cov_TS_galS_jmax12_B1.7422102429630426_nside512.dat')

#beta_fid = np.mean(np.loadtxt('output_needlet_TG/TG_512/betaj_sims_TS_galS_jmax12_B1.7422102429630426_nside512.dat'), axis=0)
#
beta_fid_array = np.loadtxt('output_needlet_TG/Planck/TG_512/betaj_sims_TS_galS_jmax12_B1.7422102429630426_nside512.dat')

num_sim = 158

beta_fid = beta_fid_array[num_sim]

#delta = [np.subtract(beta_fid, beta_oml[p]) for p in range(len(OmL))]

#chi_squared = [liklh.Calculate_chi2(delta[p], cov_matrix) for p in range(len(OmL))]#liklh.Calculate_chi2_grid(beta_fid, beta_oml, cov_matrix, jmax, OmL)

chi_squared = liklh.Calculate_chi2_grid(beta_fid=beta_fid, beta_grid=beta_oml, cov=cov_matrix, jmax=jmax, params = OmL )

#print(chi_squared)

perc = chi2.cdf(chi_squared, 12)

#print(perc*100)


posterior_distr = liklh.Sample_posterior(chi_squared, OmL)
#index_p, = np.where(posterior_distr == posterior_distr.max())


filename = 'Posterior_OmL_'+str(len(OmL))+'_sim'+str(num_sim)+'_best-fit'

percentile = liklh.Plot_posterior(posterior_distr, OmL,chi_squared, filename)


fig1 = plt.figure(figsize=(17,10))
ax1 = fig1.add_subplot(1, 1, 1)

ax1.plot(OmL, chi_squared, 'o')
ax1.set_title(r'Chi squared for $\Omega_{\Lambda}$ , grid = '+str(len(OmL))+' points')
ax1.set_xlabel(r'$\Omega_{\Lambda}$')
index, = np.where(chi_squared == chi_squared.min())
ax1.axvline(OmL[index], color = 'r', label = 'min chi squared=%1.2f' %OmL[index])
ax1.axvline(percentile[1],color ='k',linestyle='-.', label = 'percentile posterior=%1.2f' %percentile[1])#(0.6847,color ='k',linestyle='-.', label = 'Planck 2018')
ax1.axvline(mode(posterior_distr)[0][0],color ='k',linestyle=':', label = 'max posterior=%1.2f' %mode(posterior_distr)[0][0])#(0.6847,color ='k',linestyle='-.', label = 'Planck 2018')
ax1.axvline(np.mean(posterior_distr),color ='r',linestyle='-.', label = 'mean posterior=%1.2f' %np.mean(posterior_distr))#(0.6847,color ='k',linestyle='-.', label = 'Planck 2018')
plt.legend(loc = 'best')
plt.savefig('parameter_estimation/chi_squared_'+str(num_sim)+'_'+str(len(OmL))+'.png')


print(np.median(posterior_distr), mode(posterior_distr)[0][0] ,np.mean(posterior_distr), OmL[index] )

#plt.savefig('prova_posterior.png')
#plt.hist(posterior_distr, bins=30)