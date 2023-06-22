import numpy as np
import matplotlib.pyplot as plt
import likelihood_analysis_module as liklh
from scipy.stats import chi2, mode
import seaborn as sns


Aisw=np.linspace(0.0,2.,31)
jmax = 12
lmax = 782


cov_matrix_old = np.loadtxt('output_needlet_TG/Planck/old_beta/TG_512_new/cov_TS_galS_jmax12_B1.7422102429630426_nside512.dat')[1:(jmax),1:(jmax)]
cov_matrix = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/TG_512_planck_2_lmin0/cov_TS_galS_jmax12_B1.7422102429630426_nside512.dat')[1:(jmax),1:(jmax)]
print(f'shape cov = {cov_matrix.shape}')
nj=cov_matrix.shape[1]

beta_fid_array = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/TG_512_planck_2_lmin0/betaj_sims_TS_galS_jmax12_B1.7422102429630426_nside512.dat')
beta_fid_array_old = np.loadtxt('output_needlet_TG/Planck/old_beta/TG_512_new/betaj_sims_TS_galS_jmax12_B1.7422102429630426_nside512.dat')



num_sim = 456

beta_fid =   np.mean(beta_fid_array, axis=0)[1:(jmax)]  #beta_fid_array[num_sim]
beta_fid_old =  np.mean(beta_fid_array_old, axis=0)[1:(jmax)]  #beta_fid_array_old[num_sim]
print(beta_fid-beta_fid_old)

beta_A =np.array([a*beta_fid for a in Aisw])
beta_A_old =np.array([a*beta_fid_old for a in Aisw])

chi_squared = liklh.Calculate_chi2_grid(beta_fid=beta_fid, beta_grid=beta_A, cov=cov_matrix, jmax=jmax, params = Aisw )
chi_squared_old = liklh.Calculate_chi2_grid(beta_fid=beta_fid_old, beta_grid=beta_A_old, cov=cov_matrix_old, jmax=jmax, params = Aisw )

perc = chi2.cdf(chi_squared, nj)
perc_old = chi2.cdf(chi_squared_old, nj)
#print(f'perc={perc}, perc_old={perc_old}')
lik = liklh.Likelihood(chi_squared=chi_squared)
lik_old = liklh.Likelihood(chi_squared=chi_squared_old)

posterior_distr = liklh.Sample_posterior(chi_squared, Aisw)
posterior_distr_old = liklh.Sample_posterior(chi_squared_old, Aisw)

percentile = np.percentile(posterior_distr, q = [16,50,84])
percentile_old = np.percentile(posterior_distr_old, q = [16,50,84])

#filename = 'Posterior_Aisw_sim'+str(num_sim)+'_best-fit'
filename = 'Posterior_Aisw_mean_best-fit'


#Aisw
fig1 = plt.figure(figsize=(17,10))
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
plt.savefig('parameter_estimation/chi_squared_Aisw_mean_'+str(len(Aisw))+'.png')


index, = np.where(chi_squared == chi_squared.min())
index_old, = np.where(chi_squared_old == chi_squared_old.min())

print(f'min chi squared={Aisw[index[0]]}', f'min chi squared_old={Aisw[index_old[0]]}')
print(f'percetile={percentile}', f'percentile old={percentile_old}')

fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.set_title(r'Probability distribution for $A_{ISW}$ , grid = '+str(len(Aisw))+' points')

ax.set_xlabel(r'$A_{ISW}$')
textstr = '\n'.join((
    r'$A_{ISW}=%.4f$' % (Aisw[index], ),
    r'$-=%.4f$' % (percentile[0], ),
    r'$+=%.4f$' % (percentile[2], )))


ax.text(0.25, 1, textstr, 
    verticalalignment='top')#, bbox=props)
binwidth = (Aisw[-1]-Aisw[0])/(len(Aisw)-1)
binrange = [Aisw[0]+binwidth/2, Aisw[-1]+binwidth/2]

sns.histplot(posterior_distr, stat='density',binwidth=binwidth,binrange=binrange,element='step',fill=True, ax=ax)
sns.histplot(posterior_distr_old, stat='density',binwidth=binwidth,binrange=binrange,element='step',fill=True, color='r',ax=ax)

ax.set_xlim(binrange[0], binrange[1])
ax.axvline(percentile[0],color='b')
ax.axvline(percentile_old[0],color='r')

#ax.axvline(mean,color='r')
ax.axvline(percentile[2],color='b')
ax.axvline(percentile_old[2],color='r')
ax.axvline(Aisw[index], color = 'b', linestyle='-')
ax.axvline(Aisw[index_old], color = 'r', linestyle='-',label='old')

ax.axvline(1.,color ='k',linestyle='-.', label = 'Aisw')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('parameter_estimation/' +filename +'.png')







