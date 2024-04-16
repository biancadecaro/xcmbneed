import numpy as np
import matplotlib.pyplot as plt
import likelihood_analysis_module as liklh
from scipy.stats import chi2, mode
import cython_mylibc as pippo

OmL = np.linspace(0.0,0.95,30)
#Aisw=np.linspace(0.0,2.,31)
jmax = 12
lmax = 782
B = pippo.mylibpy_jmax_lmax2B(jmax, lmax)
print(B)

idx = (np.abs(OmL - 0.6847)).argmin()
print(idx)

beta_oml = np.array([np.loadtxt(f'output_needlet_TG_OmL/Grid_spectra_{len(OmL)}_planck/TGsims_theoretical_OmL{oml}/beta_TS_galS_theoretical_OmL{len(OmL)}_B{B}.dat') for oml in OmL])

cov_matrix_variance = np.array([np.loadtxt(f'output_needlet_TG_OmL/Grid_spectra_{len(OmL)}_planck/TGsims_theoretical_OmL{oml}/variance_TS_galS_theoretical_OmL{len(OmL)}_B{B}.dat') for oml in OmL])
cov_matrix = np.loadtxt('output_needlet_TG/Planck/TG_512_planck/cov_TS_galS_jmax12_B1.7422102429630426_nside512.dat')
cov_inv=np.linalg.inv(cov_matrix)
#print('sim=',cov_matrix)
#cov_matrix=np.eye(jmax+1)

beta_fid_array=np.loadtxt('output_needlet_TG/Planck/TG_512_planck/betaj_sims_TS_galS_jmax12_B1.7422102429630426_nside512.dat')
beta_fid= np.mean(beta_fid_array, axis=0)#beta_fid_array[325]
#beta_fid = np.loadtxt(f'output_needlet_TG_OmL/Grid_spectra_{len(OmL)}_planck/beta_TS_galS_theoretical_OmL_fiducial_B{B}.dat')
print(beta_fid.shape, beta_oml.shape, cov_matrix.shape)

#print(beta_fid.shape, beta_oml.shape, cov_matrix.shape)
#print( 'grid=%d\n'%beta_oml.shape, 'cov=%d\n'%cov_matrix.shape)# ,'fid=%d\n'%beta_fid.shape,)

chi_squared = liklh.delta_chi2_grid(beta_fid=beta_fid, beta_grid=beta_oml, cov=(cov_matrix_variance**2), jmax=jmax, params = OmL )
#chi_squared = liklh.Calculate_chi2_grid(beta_fid=beta_fid, beta_grid=beta_oml, cov=cov_matrix, jmax=jmax, params = OmL )

delta = np.array([np.subtract(beta_fid, beta_oml[p]) for p in range(len(OmL))])
print(delta[21])
print(chi_squared)

perc = chi2.cdf(chi_squared, jmax)

#print(perc*100)

lik = liklh.Likelihood(chi_squared=chi_squared)
#print(lik)
#perc1 =np.percentile(lik, q=[16,50,84])
#print("perc1=",perc1)
#index1, = np.where(lik == perc1[1])
index1, = np.where(lik == lik.max())

print(index1,lik[index1], OmL[index1])

posterior_distr = liklh.Sample_posterior(chi_squared, OmL)
percentile = np.percentile(posterior_distr, q = [16,50,84])
print(percentile)

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
plt.savefig(f'parameter_estimation/chi_squared_theoretical_OmL{len(OmL)}_planck.png')


fig1 = plt.figure(figsize=(17,10))
ax1 = fig1.add_subplot(1, 1, 1)

ax1.plot(OmL, lik, 'o')
ax1.set_title(r'Likelihood for $\Omega_{\Lambda}$ , grid = '+str(len(OmL))+' points')
ax1.set_xlabel(r'$\Omega_{\Lambda}$')
index, = np.where(chi_squared == chi_squared.min())
ax1.axvline(OmL[index], color = 'r', label = 'min chi squared=%1.2f' %OmL[index])
ax1.axvline(percentile[1],color ='k',linestyle='-.', label = 'percentile posterior=%1.2f' %percentile[1])#(0.6847,color ='k',linestyle='-.', label = 'Planck 2018')
ax1.axvline(mode(posterior_distr)[0][0],color ='k',linestyle=':', label = 'max posterior=%1.2f' %mode(posterior_distr)[0][0])#(0.6847,color ='k',linestyle='-.', label = 'Planck 2018')
ax1.axvline(np.mean(posterior_distr),color ='r',linestyle='-.', label = 'mean posterior=%1.2f' %np.mean(posterior_distr))#(0.6847,color ='k',linestyle='-.', label = 'Planck 2018')
plt.legend(loc = 'best')
plt.savefig(f'parameter_estimation/likelihood_theoretical_OmL{len(OmL)}.png')


filename = f'Posterior_OmL_{len(OmL)}_theoretical_best-fit_planck'#_no_mnu'

percentile = liklh.Plot_posterior(posterior_distr, OmL,chi_squared, filename)



print(np.median(posterior_distr), mode(posterior_distr)[0][0] ,np.mean(posterior_distr), OmL[index] )
#
##plt.savefig('prova_posterior.png')
##plt.hist(posterior_distr, bins=30)