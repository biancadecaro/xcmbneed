import numpy as np
import matplotlib.pyplot as plt
import likelihood_analysis_module as liklh
from scipy.stats import chi2, mode
import cython_mylibc as pippo
import seaborn as sns

OmL = np.linspace(0.0,0.95,30)
#Aisw=np.linspace(0.0,2.,31)
jmax = 12
lmax = 800
delta_ell = 50
lbins = 15
idx = (np.abs(OmL - 0.6847)).argmin()
print(idx, OmL[idx])

cl_oml = np.array([np.loadtxt(f'/ehome/bdecaro/xcmbneed/src/output_cl_TG_OmL/Grid_spectra_30_planck_lmin0/TGsims_512_OmL{oml}/cl_sims_TS_galS_Om{oml}_delta_ell50_lmax = 800_nside512.dat') for oml in OmL])

cl_oml_old = np.array([np.loadtxt(f'/ehome/bdecaro/xcmbneed/src/output_cl_TG_OmL/Grid_spectra_30_old/TGsims_512_OmL{oml}/cl_sims_TS_galS_Om{oml}_delta_ell50_lmax = 800_nside512.dat') for oml in OmL])


cov_matrix = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_Cl_TG/Planck/TG_512_planck_2_lmin0/cov_TS_galS_lmax800_nside512.dat')
icov = np.linalg.inv(cov_matrix)

cov_matrix_old = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_Cl_TG/Planck/TG_512_old/cov_TS_galS_lmax800_nside512.dat')
icov_old = np.linalg.inv(cov_matrix_old)

cl_fid_array = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_Cl_TG/Planck/TG_512_planck_2_lmin0/cl_sims_TS_galS_lmax800_nside512.dat')
cl_fid_array_old = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_Cl_TG/Planck/TG_512_old/cl_sims_TS_galS_lmax800_nside512.dat')

num_sim = 123

cl_fid = np.array([cl_fid_array[num_sim] for _ in range(len(OmL))])
cl_fid_old = np.array([cl_fid_array_old[123] for _ in range(len(OmL))])

delta = cl_fid-cl_oml
delta_old =cl_fid_old-cl_oml_old


chi_squared=np.zeros(len(OmL))
chi_squared_old =np.zeros(len(OmL))
nj=cov_matrix.shape[1]
for om in range(len(OmL)):
    chi_squared[om]  += np.matmul(delta[om], np.matmul(icov, delta[om]))
    chi_squared_old[om]  += np.matmul(delta_old[om], np.matmul(icov_old, delta_old[om]))

print(chi_squared)
print(chi_squared_old)
#chi_squared_old=np.sum(delta_old**2/np.diag(cov_matrix_old)/500, axis=1)


perc = chi2.cdf(chi_squared, lbins)
perc_old = chi2.cdf(chi_squared_old, lbins)

print(f'perc={perc}, perc_old={perc_old}')

lik = liklh.Likelihood(chi_squared=chi_squared)
print(f'likelihood={lik}') 

posterior_distr = liklh.Sample_posterior(chi_squared, OmL)
percentile = np.percentile(posterior_distr, q = [16,50,84])
print(f'percentile={percentile}')

lik_old = liklh.Likelihood(chi_squared=chi_squared_old)

#posterior_distr_old = liklh.Sample_posterior(chi_squared_old, OmL)


#percentile_old = np.percentile(posterior_distr_old, q = [16,50,84])
print(f'percentile={percentile}' )#, percentile_old={percentile_old}' )
#print(f'percentile={percentile}')

fig1 = plt.figure(figsize=(17,10))
ax1 = fig1.add_subplot(1, 1, 1)

ax1.plot(OmL, chi_squared, 'o')
ax1.set_title(r'Chi squared for $\Omega_{\Lambda}$ , grid = '+str(len(OmL))+' points')
ax1.set_xlabel(r'$\Omega_{\Lambda}$')

index, = np.where(chi_squared == chi_squared.min())
index_old, = np.where(chi_squared_old == chi_squared_old.min())

ax1.axvline(OmL[index], color = 'r', label = 'min chi squared=%1.2f' %OmL[index])
ax1.axvline(percentile[1],color ='k',linestyle='-.', label = 'percentile posterior=%1.2f' %percentile[1])#(0.6847,color ='k',linestyle='-.', label = 'Planck 2018')
ax1.axvline(mode(posterior_distr)[0][0],color ='k',linestyle=':', label = 'max posterior=%1.2f' %mode(posterior_distr)[0][0])#(0.6847,color ='k',linestyle='-.', label = 'Planck 2018')
ax1.axvline(np.mean(posterior_distr),color ='r',linestyle='-.', label = 'mean posterior=%1.2f' %np.mean(posterior_distr))#(0.6847,color ='k',linestyle='-.', label = 'Planck 2018')
plt.legend(loc = 'best')
#plt.savefig('parameter_estimation/cl_chi_squared_mean_'+str(len(OmL))+'_planck2_lmin0.png')

#print(np.median(posterior_distr), mode(posterior_distr)[0][0] ,np.mean(posterior_distr), OmL[index] )

filename = 'cl_parameter_estimation_'+str(num_sim)+'_'+str(len(OmL))+'_new.png'
fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.set_title(r'Probability distribution for $\Omega_{\Lambda}$ , grid = '+str(len(OmL))+' points')
ax.set_xlabel(r'$\Omega_{\Lambda}$')

#textstr = '\n'.join((
#    r'$\Omega_{\Lambda}=%.2f$' % (OmL[index], ),
#    r'$-=%.2f$' % (percentile[0], ),
#    r'$+=%.2f$' % (percentile[2], )))
# 
#ax.text(0.3, 3, textstr, 
#    verticalalignment='top')#, bbox=props)

binwidth = (OmL[-1]-OmL[0])/(len(OmL)-1)  
binrange = [OmL[0]+binwidth/2, OmL[-1]+binwidth/2]
sns.histplot(posterior_distr, stat='density',binwidth=binwidth,binrange=binrange,element='step',fill=True, ax=ax)
#sns.histplot(posterior_distr_old, stat='density',binwidth=binwidth,binrange=binrange,element='step',color='r',fill=True, ax=ax)

ax.set_xlim(binrange[0], binrange[1])
ax.axvline(percentile[0],color='b')
#ax.axvline(percentile_old[0],color='r')

ax.axvline(percentile[2],color='b')
#ax.axvline(percentile_old[2],color='r')
ax.axvline(OmL[index], color = 'b', linestyle='-.')
ax.axvline(OmL[index_old], color = 'r', linestyle='-.', label='old')

ax.axvline(0.6847,color ='k',linestyle='-.', label = 'Planck 2018')

plt.legend(loc='best')
plt.tight_layout()
plt.savefig('parameter_estimation/' +filename +'.png')



#
##plt.savefig('prova_posterior.png')
##plt.hist(posterior_distr, bins=30)