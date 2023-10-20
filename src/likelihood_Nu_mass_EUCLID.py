import numpy as np
import matplotlib.pyplot as plt
import likelihood_analysis_module as liklh
from scipy.stats import chi2, mode
import cython_mylibc as pippo
import seaborn as sns
sns.set()
sns.set(style = 'white')
sns.set_palette('husl')

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)

from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 

#plt.rcParams['axes.linewidth']  = 5.
plt.rcParams['axes.labelsize']  =18
plt.rcParams['xtick.labelsize'] =15
plt.rcParams['ytick.labelsize'] =15
#plt.rcParams['xtick.major.size'] = 20
#plt.rcParams['ytick.major.size'] = 20
#plt.rcParams['xtick.minor.size'] = 20
#plt.rcParams['ytick.minor.size'] = 20
plt.rcParams['legend.fontsize']  = 18
#plt.rcParams['legend.frameon']  = False
plt.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['axes.titlesize'] = '20'
plt.rcParams["errorbar.capsize"] = 5
##
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['font.size'] = 20
plt.rcParams['lines.linewidth']  = 3.

mnu = np.linspace(0.0,0.12,30)

jmax = 12
lmax = 256
B = pippo.mylibpy_jmax_lmax2B(jmax, lmax)
print(B)

idx = (np.abs(mnu - 0.06)).argmin()
print(idx, mnu[idx])

gamma_mnu = np.array([np.loadtxt(f'/ehome/bdecaro/xcmbneed/src/output_needlet_TG_Mnu/Grid_spectra_{len(mnu)}_Gammaj_Euclid/TGsims_theoretical_Nu_mass{m}/beta_TS_galS_theoretical_Nu_mass{m}_B1.5874010519681994.dat') for m in mnu])[:,1:(jmax)]
print(f'shape gamma_mnu = {gamma_mnu.shape}')

plt.plot(gamma_mnu[idx])
plt.savefig('gamma_mnu.png')

cov_matrix = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000_non_linear/cov_TS_galT_jmax12_B_1.59_nside128_fsky0.36.dat')[1:(jmax),1:(jmax)]
print(f'shape cov = {cov_matrix.shape}')

icov=np.linalg.inv(cov_matrix)


beta_fid_array = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000_non_linear/betaj_sims_TS_galT_jmax12_B_1.59_nside128_fsky0.36.dat')[:,1:(jmax)]
print(f'beta_fid={beta_fid_array.shape}')
num_sim = 789

beta_fid =  np.mean(beta_fid_array, axis=0)


delta = np.array([np.subtract(beta_fid, gamma_mnu[p]) for p in range(len(mnu))])#beta_fid-beta_oml#


nj=cov_matrix.shape[1]

chi_squared = liklh.Calculate_chi2_grid(beta_fid=beta_fid, beta_grid=gamma_mnu, cov=cov_matrix, jmax=jmax, params = mnu )


perc = chi2.cdf(chi_squared, nj)

lik = liklh.Likelihood(chi_squared=chi_squared)


posterior_distr = liklh.Sample_posterior(chi_squared, mnu)

percentile = np.percentile(posterior_distr, q = [16,50,84])


fig1 = plt.figure(figsize=(10,7))
ax1 = fig1.add_subplot(1, 1, 1)

ax1.plot(mnu, chi_squared, 'o')
ax1.set_title(r'Chi squared for $\sum m_{\nu}$ , grid = '+str(len(mnu))+' points')
ax1.set_xlabel(r'$\sum m_{\nu}$')
index, = np.where(chi_squared == chi_squared.min())
ax1.axvline(mnu[index], color = 'r', label = 'min chi squared=%1.2f' %mnu[index])
ax1.axvline(percentile[1],color ='k',linestyle='-.', label = 'percentile posterior=%1.2f' %percentile[1])#(0.6847,color ='k',linestyle='-.', label = 'Planck 2018')
ax1.axvline(mode(posterior_distr)[0][0],color ='k',linestyle=':', label = 'max posterior=%1.2f' %mode(posterior_distr)[0][0])#(0.6847,color ='k',linestyle='-.', label = 'Planck 2018')
ax1.axvline(np.mean(posterior_distr),color ='r',linestyle='-.', label = 'mean posterior=%1.2f' %np.mean(posterior_distr))#(0.6847,color ='k',linestyle='-.', label = 'Planck 2018')
plt.legend(loc = 'best')
plt.savefig(f'parameter_estimation/Mnu/chi_squared_mean_{len(mnu)}_EUCLID.png')

filename = f'Posterior_Mnu_{len(mnu)}_mean_sim_best-fit_EUCLID'


mean = np.mean(posterior_distr)

index, = np.where(chi_squared == chi_squared.min())
print(f'index where chi2 min={index}')

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(1, 1, 1)

ax.set_title(r'Probability distribution for $\sum m_{\nu}$ , grid = '+str(len(mnu))+' points')
ax.set_xlabel(r'$\sum m_{\nu}$')

textstr = '\n'.join((
    r'$\sum m_{\nu}=%.2f^{+ %.2f}_{-%.2f}$' % (mnu[index], percentile[2]-mnu[index], mnu[index]-percentile[0] ),
    #r'$-=%.2f$' % (percentile[0], ),
    #r'$+=%.2f$' % (percentile[2], )
    ))
 
ax.text(0.1, 1.5, textstr, 
    verticalalignment='top')#, bbox=props)


binwidth = (mnu[-1]-mnu[0])/(len(mnu)-1)
binrange = [mnu[0]+binwidth/2, mnu[-1]+binwidth/2]
sns.histplot(posterior_distr, stat='density',binwidth=binwidth,binrange=binrange,element='step',fill=True, alpha=0.5 ,color='#2b7bbc',ax=ax)


ax.set_xlim(binrange[0], binrange[1])
ax.axvline(percentile[0],color='b')

ax.axvline(percentile[2],color='b')

ax.axvline(mnu[index], color = 'b', linestyle='-')

ax.axvline(0.06,color ='grey',linestyle='--', label = r'$\sum m_{\nu}=0.06$')

plt.legend(loc='best')
plt.tight_layout()

plt.savefig('parameter_estimation/Mnu/' +filename +'.png')

####################################################################################################################
##################################### ALTRO PLOT ##################################################################
nsim = beta_fid_array.shape[0]

chi2_mnu =np.zeros((nsim,len(mnu)))
post_mnu= np.zeros((nsim,len(mnu)))
mnu_min = np.zeros(nsim)
for n in range(nsim):
    chi2_mnu[n] = liklh.Calculate_chi2_grid(beta_fid=beta_fid_array[n], beta_grid=gamma_mnu, cov=cov_matrix, jmax=jmax, params = mnu )
    post_mnu[n] = liklh.Likelihood(chi2_mnu[n])
    index,  = np.where(chi2_mnu[n]==chi2_mnu[n].min())
    mnu_min[n] = mnu[index]

mnu_mean= np.mean(mnu_min)
mnu_std = np.std(mnu_min)
percentile_sim = np.percentile(mnu_min, q = [16,50,84])
print(percentile_sim)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(1, 1, 1)

ax.set_title(r'Probability distribution for $\sum m_{\nu}$ , grid = '+str(len(mnu))+' points')
ax.set_xlabel(r'$\sum m_{\nu}$')

textstr1 = '\n'.join((
   r'$\sum m_{\nu}=%.2f^{+ %.2f}_{-%.2f}$' % (mnu_mean, percentile_sim[2]-mnu_mean, mnu_mean-percentile_sim[0] ),
    #r'$\pm=%.2f$' % (OmL_std, ),
#    r'$+=%.2f$' % (percentile[2], )
    ))
 
ax.text(0.1, 1.5, textstr1, 
    verticalalignment='top')#, bbox=props)


binwidth = (mnu[-1]-mnu[0])/(len(mnu)-1)
binrange = [mnu[0]+binwidth/2, mnu[-1]+binwidth/2]
sns.histplot(mnu_min, stat='density',binwidth=binwidth,binrange=binrange,element='step',fill=True, alpha=0.5 ,color='#2b7bbc',ax=ax)


ax.set_xlim(binrange[0], binrange[1])
ax.axvline(mnu_mean,color='b', linestyle='-')

ax.axvline(percentile_sim[2],color='b')

ax.axvline(percentile_sim[0], color = 'b' )

ax.axvline(0.06,color ='grey',linestyle='--', label = r'$\sum m_{\nu}=0.06$')

plt.legend(loc='best')
plt.tight_layout()
filename = f'Posterior_Mnu_{len(mnu)}_mean_sim_best-fit_from_sims_EUCLID'
plt.savefig('parameter_estimation/Mnu/' +filename +'.png')

