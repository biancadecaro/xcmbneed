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

OmL = np.linspace(0.0,0.95,30)

jmax = 12
lmax = 782
B = pippo.mylibpy_jmax_lmax2B(jmax, lmax)
print(B)

idx = (np.abs(OmL - 0.6847)).argmin()
print(idx, OmL[idx])

beta_oml = np.array([np.loadtxt(f'/ehome/bdecaro/xcmbneed/src/output_needlet_TG_OmL/Grid_spectra_30_planck_2_lmin0/TGsims_theoretical_OmL{oml}/beta_TS_galS_theoretical_OmL{oml}_B1.7422102429630426.dat') for oml in OmL])[:,1:(jmax)]
print(f'shape beta_oml = {beta_oml.shape}')

cov_matrix = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/TG_512/cov_TS_galS_jmax12_B1.74_nside512.dat')[1:(jmax),1:(jmax)]
print(f'shape cov = {cov_matrix.shape}')

icov=np.linalg.inv(cov_matrix)


beta_fid_array = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/TG_512/betaj_sims_TS_galS_jmax12_B1.74_nside512.dat')[:,1:(jmax)]
print(f'beta_fid={beta_fid_array.shape}')
num_sim = 123

beta_fid =   beta_fid_array[num_sim]


delta = np.array([np.subtract(beta_fid, beta_oml[p]) for p in range(len(OmL))])#beta_fid-beta_oml#


nj=cov_matrix.shape[1]

chi_squared = liklh.Calculate_chi2_grid(beta_fid=beta_fid, beta_grid=beta_oml, cov=cov_matrix, jmax=jmax, params = OmL )


perc = chi2.cdf(chi_squared, nj)

lik = liklh.Likelihood(chi_squared=chi_squared)


posterior_distr = liklh.Sample_posterior(chi_squared, OmL)

percentile = np.percentile(posterior_distr, q = [16,50,84])


fig1 = plt.figure(figsize=(10,7))
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
plt.savefig(f'plot_tesi/Parameter_estimation/chi_squared_mean_{len(OmL)}_theoretical_OmL.png')

filename = f'Posterior_OmL_{len(OmL)}_mean_sim_best-fit_theoretical_OmL'


mean = np.mean(posterior_distr)

index, = np.where(chi_squared == chi_squared.min())
print(f'index where chi2 min={index}')

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(1, 1, 1)

ax.set_title(r'Probability distribution for $\Omega_{\Lambda}$ , grid = '+str(len(OmL))+' points')
ax.set_xlabel(r'$\Omega_{\Lambda}$')

textstr = '\n'.join((
    r'$\Omega_{\Lambda}=%.2f^{+ %.2f}_{-%.2f}$' % (OmL[index], percentile[2]-OmL[index], OmL[index]-percentile[0] ),
    #r'$-=%.2f$' % (percentile[0], ),
    #r'$+=%.2f$' % (percentile[2], )
    ))
 
ax.text(0.3, 3, textstr, 
    verticalalignment='top')#, bbox=props)


binwidth = (OmL[-1]-OmL[0])/(len(OmL)-1)
binrange = [OmL[0]+binwidth/2, OmL[-1]+binwidth/2]
sns.histplot(posterior_distr, stat='density',binwidth=binwidth,binrange=binrange,element='step',fill=True, alpha=0.5 ,color='#2b7bbc',ax=ax)


ax.set_xlim(binrange[0], binrange[1])
ax.axvline(percentile[0],color='b')

ax.axvline(percentile[2],color='b')

ax.axvline(OmL[index], color = 'b', linestyle='-')

ax.axvline(0.6847,color ='grey',linestyle='--', label = 'Planck 2018')

plt.legend(loc='best')
plt.tight_layout()

plt.savefig('plot_tesi/Parameter_estimation/' +filename +'.png')
plt.savefig('plot_tesi/Parameter_estimation/' +filename +'.pdf')

####################################################################################################################
##################################### ALTRO PLOT ##################################################################
nsim = beta_fid_array.shape[0]

chi2_OmL =np.zeros((nsim,len(OmL)))
post_OmL= np.zeros((nsim,len(OmL)))
OmL_min = np.zeros(nsim)
for n in range(nsim):
    chi2_OmL[n] = liklh.Calculate_chi2_grid(beta_fid=beta_fid_array[n], beta_grid=beta_oml, cov=cov_matrix, jmax=jmax, params = OmL )
    post_OmL[n] = liklh.Likelihood(chi2_OmL[n])
    index,  = np.where(chi2_OmL[n]==chi2_OmL[n].min())
    OmL_min[n] = OmL[index]

OmL_mean= np.mean(OmL_min)
OmL_std = np.std(OmL_min)
percentile_sim = np.percentile(OmL_min, q = [16,50,84])
print(percentile_sim)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(1, 1, 1)

ax.set_title(r'Probability distribution for $\Omega_{\Lambda}$ , grid = '+str(len(OmL))+' points')
ax.set_xlabel(r'$\Omega_{\Lambda}$')

textstr1 = '\n'.join((
   r'$\Omega_{\Lambda}=%.2f^{+ %.2f}_{-%.2f}$' % (OmL_mean, percentile_sim[2]-OmL_mean, OmL_mean-percentile_sim[0] ),
    #r'$\pm=%.2f$' % (OmL_std, ),
#    r'$+=%.2f$' % (percentile[2], )
    ))
 
ax.text(0.3, 3, textstr1, 
    verticalalignment='top')#, bbox=props)


binwidth = (OmL[-1]-OmL[0])/(len(OmL)-1)
binrange = [OmL[0]+binwidth/2, OmL[-1]+binwidth/2]
sns.histplot(OmL_min, stat='density',binwidth=binwidth,binrange=binrange,element='step',fill=True, alpha=0.5 ,color='#2b7bbc',ax=ax)


ax.set_xlim(binrange[0], binrange[1])
ax.axvline(OmL_mean,color='b', linestyle='-')

ax.axvline(percentile_sim[2],color='b')

ax.axvline(percentile_sim[0], color = 'b' )

ax.axvline(0.6847,color ='grey',linestyle='--', label = 'Planck 2018')

plt.legend(loc='best')
plt.tight_layout()
filename = f'Posterior_OmL_{len(OmL)}_mean_sim_best-fit_theoretical_OmL_from_sims'
plt.savefig('plot_tesi/Parameter_estimation/' +filename +'.png')
plt.savefig('plot_tesi/Parameter_estimation/' +filename +'.pdf')
