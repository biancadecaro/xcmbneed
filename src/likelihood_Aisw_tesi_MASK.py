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


Aisw=np.linspace(0.0,2.,31)
jmax = 12
lmax = 782

cov_matrix = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/Mask_noise/TG_512_tesi_NO_fsky_approx/cov_TS_galT_jmax12_B = 1.74 $_nside512_mask.dat')[1:(jmax+1),1:(jmax+1)]
#cov_matrix_fsky =np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/Mask_noise/TG_512_tesi/cov_TS_galT_jmax12_B = 1.74 $_nside512_mask.dat')[1:(jmax+1),1:(jmax+1)]

variance_theory = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/Mask_noise/TG_512_tesi_NO_fsky_approx/variance_beta_TS_galS_theoretical_fiducial_B1.7422102429630426_noise.dat')[1:(jmax+1)]
print(f'shape cov = {cov_matrix.shape}')
nj=cov_matrix.shape[1]

beta_fid_array = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/Mask_noise/TG_512_tesi_NO_fsky_approx/betaj_sims_TS_galT_jmax12_B_1.7422102429630426_nside512_fsky_0.7794118722279867.dat')[:,1:(jmax+1)]
nsim = beta_fid_array.shape[0]

#beta_theory= np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/Mask_noise/TG_512_tesi/beta_TS_galS_theoretical_fiducial_B1.7422102429630426.dat')[1:(jmax+1)]
gamma_theory = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/Mask_noise/TG_512_tesi_NO_fsky_approx/gammaj_theory_B1.74_jmax12_nside512_lmax782_nsim500_fsky0.78.dat')[1:(jmax+1)]


fsky=0.7794118722279867

num_sim = 45

beta_fid =   beta_fid_array[num_sim]

beta_fid_mean = np.mean(beta_fid_array, axis=0)
beta_A =np.array([a*gamma_theory for a in Aisw])

#chi_squared = liklh.Calculate_chi2_grid(beta_fid=beta_fid_mean, beta_grid=beta_A, cov=cov_matrix, jmax=jmax, params = Aisw )
chi_squared = liklh.delta_chi2_grid(beta_fid=beta_fid, beta_grid=beta_A, cov=variance_theory**2, jmax=jmax, params = Aisw )
perc = chi2.cdf(chi_squared, nj)

lik = liklh.Likelihood(chi_squared=chi_squared)

posterior_distr = liklh.Sample_posterior(chi_squared, Aisw)

percentile = np.percentile(posterior_distr, q = [16,50,84])


#Aisw
fig1 = plt.figure(figsize=(10,7))
ax1 = fig1.add_subplot(1, 1, 1)

ax1.plot(Aisw, chi_squared, 'o')
ax1.set_title(r'Chi squared for $A_{ISW}$ , grid = '+str(len(Aisw))+' points')
ax1.set_xlabel(r'$A_{ISW}$')
index, = np.where(chi_squared == chi_squared.min())
ax1.axvline(Aisw[index], color = 'r', label = 'min chi squared=%1.2f' %Aisw[index])
ax1.axvline(percentile[1],color ='k',linestyle='-.', label = 'percentile posterior=%1.2f' %percentile[1])#(0.6847,color ='k',linestyle='-.', label = 'Planck 2018')
ax1.axvline(mode(posterior_distr)[0][0],color ='k',linestyle=':', label = 'max posterior=%1.2f' %mode(posterior_distr)[0][0])#(0.6847,color ='k',linestyle='-.', label = 'Planck 2018')
ax1.axvline(np.mean(posterior_distr),color ='r',linestyle='-.', label = 'mean posterior=%1.2f' %np.mean(posterior_distr))#(0.6847,color ='k',linestyle='-.', label = 'Planck 2018')
plt.legend(loc = 'best')
plt.savefig(f'plot_tesi/Parameter_estimation/chi_squared_Aisw_{len(Aisw)}_MASK.png')


index, = np.where(chi_squared == chi_squared.min())

filename = 'Posterior_Aisw_best-fit_MASK'

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(1, 1, 1)

ax.set_title(r'Probability distribution for $A_{iSW}$ , grid = '+str(len(Aisw))+' points')

ax.set_xlabel(r'$A_{iSW}$')
textstr = '\n'.join((
    r'$A_{iSW}=%.2f \pm %.2f$' % (Aisw[index], Aisw[index]-percentile[0]),
    #r'$\pm%.2f$' % (Aisw[index]-percentile[0], ),
    #r'$+%.2f$' % (percentile[2]-Aisw[index], )
    ))


ax.text(1.40,0.75, textstr, 
    verticalalignment='top')#, bbox=props)
binwidth = (Aisw[-1]-Aisw[0])/(len(Aisw)-1)
binrange = [Aisw[0]+binwidth/2, Aisw[-1]+binwidth/2]

sns.histplot(posterior_distr, stat='density',binwidth=binwidth,binrange=binrange,element='step',fill=True,alpha=0.5 ,color='#2b7bbc', ax=ax)

ax.set_xlim(binrange[0], binrange[1])
ax.axvline(percentile[0],color='b')

#ax.axvline(mean,color='r')
ax.axvline(percentile[2],color='b')
ax.axvline(Aisw[index], color = 'b', linestyle='-')

ax.axvline(1.,color ='grey',linestyle='--', label = 'Fiducial Aisw')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('plot_tesi/Parameter_estimation/' +filename +'.png')
plt.savefig('plot_tesi/Parameter_estimation/' +filename +'.pdf')

################################################################################
############################## ALRO PLOT #######################################



chi2_Aisw =np.zeros((nsim,len(Aisw)))
post_Aisw= np.zeros((nsim,len(Aisw)))
Aisw_min = np.zeros(nsim)
for n in range(nsim):
    chi2_Aisw[n] = liklh.Calculate_chi2_grid(beta_fid=beta_fid_array[n], beta_grid=beta_A, cov=cov_matrix, jmax=jmax, params = Aisw )
    post_Aisw[n] = liklh.Likelihood(chi2_Aisw[n])
    index,  = np.where(chi2_Aisw[n]==chi2_Aisw[n].min())
    Aisw_min[n] = Aisw[index]

Aisw_mean= np.mean(Aisw_min)
Aisw_std = np.std(Aisw_min)
print(Aisw_mean, Aisw_std)


percentile_sim = np.percentile(Aisw_min, q = [16,50,84])
print(percentile_sim)

filename = 'Posterior_Aisw_best-fit_from_sims_MASK'

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(1, 1, 1)

ax.set_title(r'Probability distribution for $A_{iSW}$ , grid = '+str(len(Aisw))+' points')

ax.set_xlabel(r'$A_{iSW}$')
textstr = '\n'.join((
    r'$A_{iSW}=%.2f \pm %.2f$' % (Aisw_mean, Aisw_std),
    #r'$\pm%.2f$' % Aisw_std, ),
    #r'$%.2f$' % (percentile[2]-Aisw[index], 
    ))


ax.text(1.40, 0.75, textstr, 
    verticalalignment='top')#, bbox=props)
binwidth = (Aisw[-1]-Aisw[0])/(len(Aisw)-1)
binrange = [Aisw[0]+binwidth/2, Aisw[-1]+binwidth/2]

sns.histplot(Aisw_min, stat='density',binwidth=binwidth,binrange=binrange,element='step',fill=True,alpha=0.5 ,color='#2b7bbc', ax=ax)

ax.set_xlim(binrange[0], binrange[1])
#ax.axvline(percentile[0],color='b')

#ax.axvline(mean,color='r')
#ax.axvline(percentile[2],color='b')
ax.axvline(Aisw_mean, color = 'b', linestyle='-')

ax.axvline(1.,color ='grey',linestyle='--', label = 'Fiducial Aisw')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('plot_tesi/Parameter_estimation/' +filename +'.png')
plt.savefig('plot_tesi/Parameter_estimation/' +filename +'.pdf')


