import numpy as np
import matplotlib.pyplot as plt
import likelihood_analysis_module as liklh
from scipy.stats import chi2, mode
import cython_mylibc as pippo
import seaborn as sns

OmL = np.linspace(0.0,0.95,30)

jmax = 12
lmax = 782
B = pippo.mylibpy_jmax_lmax2B(jmax, lmax)
print(B)

idx = (np.abs(OmL - 0.6847)).argmin()
print(idx, OmL[idx])

beta_fid_array = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/Mask_noise/TG_256_mask_shot_noise/betaj_sims_TT_galT_jmax12_B_1.7422102429630426_nside256_fsky_0.7004634737968445.dat')
beta_oml = np.array([np.loadtxt(f'/ehome/bdecaro/xcmbneed/src/output_needlet_TG_OmL/Grid_spectra_30_planck_2_lmin0/TGsims_theoretical_OmL{oml}/beta_TS_galS_theoretical_OmL{oml}_B1.7422102429630426.dat') for oml in OmL])[:,1:(jmax)]

nsim=beta_fid_array.shape[0]
print(nsim)

print(f'shape beta_oml = {beta_oml.shape}')
cov_matrix = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/Mask_noise/TG_256_mask_shot_noise/cov_TT_galT_jmax12_B = 1.74 $_nside256_mask.dat')[1:(jmax),1:(jmax)]
print(f'shape cov = {cov_matrix.shape}')



num_sim = 456

beta_fid =   np.mean(beta_fid_array, axis=0)[1:(jmax)]  #beta_fid_array[num_sim]


delta = np.array([np.subtract(beta_fid, beta_oml[p]) for p in range(len(OmL))])

nj=cov_matrix.shape[1]
    

chi_squared = liklh.Calculate_chi2_grid(beta_fid=beta_fid, beta_grid=beta_oml, cov=cov_matrix, jmax=jmax, params = OmL )

print(chi_squared)

perc = chi2.cdf(chi_squared, nj)

lik = liklh.Likelihood(chi_squared=chi_squared)



posterior_distr = liklh.Sample_posterior(chi_squared, OmL)


percentile = np.percentile(posterior_distr, q = [16,50,84])


filename = f'Posterior_OmL_{str(len(OmL))}_mean_best-fit_mask_noise_nsim{nsim}'#_no_mnu'


fig1 = plt.figure(figsize=(17,10))
ax1 = fig1.add_subplot(1, 1, 1)

ax1.plot(OmL, chi_squared, 'o')
ax1.set_title(r'Chi squared for $\Omega_{\Lambda}$ , grid = %d points, nsim=%d'%(len(OmL),nsim))
ax1.set_xlabel(r'$\Omega_{\Lambda}$')
index, = np.where(chi_squared == chi_squared.min())
ax1.axvline(OmL[index], color = 'r', label = 'min chi squared=%1.2f' %OmL[index])
ax1.axvline(percentile[1],color ='k',linestyle='-.', label = 'percentile posterior=%1.2f' %percentile[1])#(0.6847,color ='k',linestyle='-.', label = 'Planck 2018')
ax1.axvline(mode(posterior_distr)[0][0],color ='k',linestyle=':', label = 'max posterior=%1.2f' %mode(posterior_distr)[0][0])#(0.6847,color ='k',linestyle='-.', label = 'Planck 2018')
ax1.axvline(np.mean(posterior_distr),color ='r',linestyle='-.', label = 'mean posterior=%1.2f' %np.mean(posterior_distr))#(0.6847,color ='k',linestyle='-.', label = 'Planck 2018')
plt.legend(loc = 'best')
plt.savefig(f'parameter_estimation/chi_squared/Mask_noise/chi_squared_mean_{str(len(OmL))}_mask_noise_nsim{nsim}.png')


#print(np.median(posterior_distr), mode(posterior_distr)[0][0] ,np.mean(posterior_distr), OmL[index] )


#Confronto distribuzione vecchia e nuova

mean = np.mean(posterior_distr)

print(f'mean={mean}')
print(f'percetile={percentile}')
index, = np.where(chi_squared == chi_squared.min())

print(f'min chi squared={OmL[index[0]]}')
fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.set_title(r'Probability distribution for $\Omega_{\Lambda}$ , grid = '+str(len(OmL))+' points')
ax.set_xlabel(r'$\Omega_{\Lambda}$')


binwidth = (OmL[-1]-OmL[0])/(len(OmL)-1)
binrange = [OmL[0]+binwidth/2, OmL[-1]+binwidth/2]
sns.histplot(posterior_distr, stat='density',binwidth=binwidth,binrange=binrange,element='step',fill=True, ax=ax)

ax.set_xlim(binrange[0], binrange[1])
ax.axvline(percentile[0],color='b')

#ax.axvline(mean,color='r')
ax.axvline(percentile[2],color='b')
ax.axvline(OmL[index], color = 'b', linestyle='-')
ax.axvline(0.6847,color ='k',linestyle='-.', label = 'Planck 2018')

plt.legend(loc='best')
plt.tight_layout()

plt.savefig('parameter_estimation/Posterior_OmL/Mask_noise/'+filename +'.png')

