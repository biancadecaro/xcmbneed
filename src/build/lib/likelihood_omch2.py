import numpy as np
import matplotlib.pyplot as plt
import likelihood_analysis_module as liklh
from scipy.stats import chi2, mode
import seaborn as sns

omch2 = np.linspace(0.0,0.6736**2,30)
#Aisw=np.linspace(0.0,2.,31)
jmax = 12
lmax = 782

idx = (np.abs(omch2 - 0.122)).argmin()
print(idx, omch2[idx])

beta_omch2 = np.array([np.loadtxt(f'/ehome/bdecaro/xcmbneed/src/output_needlet_TG_omch2/Grid_spectra_30_planck/TGsims_512_omch2{om}/betaj_sims_TS_galS_Om{om}_jmax12_B = 1.74 _nside512.dat') for om in omch2])

cov_matrix = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/TG_512_planck/cov_TS_galS_jmax12_B1.7422102429630426_nside512.dat')
#cov_matrix_diag=np.diag(cov_matrix)*np.eye(jmax+1)
cov_inv=np.linalg.inv(cov_matrix)

#print(cov_matrix)
#beta_fid = np.mean(np.loadtxt('output_needlet_TG/TG_512/betaj_sims_TS_galS_jmax12_B1.7422102429630426_nside512.dat'), axis=0)
#
#beta_fid_array = np.loadtxt('output_needlet_TG/Planck/TG_512/betaj_sims_TS_galS_jmax10_B1.9467970312828855_nside512.dat')
beta_fid_array = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/TG_512_planck/betaj_sims_TS_galS_jmax12_B1.7422102429630426_nside512.dat')

num_sim = 123

beta_fid = np.mean(beta_fid_array, axis=0) #
plt.errorbar(x=np.arange(0,13),y=beta_fid, yerr=np.sqrt(np.diag(cov_matrix)/500), fmt='o')
plt.savefig('betaj_mean.png')

delta = np.array([np.subtract(beta_fid, beta_omch2[p]) for p in range(len(omch2))])

chi_squared = liklh.Calculate_chi2_grid(beta_fid=beta_fid, beta_grid=omch2, cov=cov_matrix, jmax=jmax, params = omch2 )


perc = chi2.cdf(chi_squared, jmax)

#print(perc*100)

lik = liklh.Likelihood(chi_squared=chi_squared)
#print(chi_squared)
perc1 =np.percentile(lik, q=[16,50,84])
#print("perc1=",perc1)
#index1, = np.where(lik == perc1[1])
index1, = np.where(lik == lik.max())

#print(index1,lik[index1], perc1[1],omch2[index1])
#print(index1,lik[index1], perc1[1],Aisw[index1])
posterior_distr = liklh.Sample_posterior(chi_squared, omch2)
#index_p, = np.where(posterior_distr == posterior_distr.max())
percentile = np.percentile(posterior_distr, q = [16,50,84])
#print(percentile)

filename = 'Posterior_omch2_'+str(len(omch2))+'_sim'+str(num_sim)+'_best-fit'#_no_mnu'

index, = np.where(chi_squared == chi_squared.min())

#percentile = liklh.Plot_posterior(posterior_distr, omch2,chi_squared, filename)
fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.set_title(r'Probability distribution for $\omega_{c}h^{2}$ , grid = '+str(len(omch2))+' points')
ax.set_xlabel(r'$\omega_{c}h^{2}$')
#ax.set_xlabel(r'$A_{ISW}$')
#textstr = '\n'.join((
#    r'$A_{ISW}=%.2f$' % (omch2[index], ),
#    r'$-=%.2f$' % (percentile[0], ),
#    r'$+=%.2f$' % (percentile[2], )))
# place a text box in upper left in axes coords
textstr = '\n'.join((
    r'$\omega_{c}h^{2}=%.2f$' % (omch2[index], ),
    r'$-=%.2f$' % (percentile[0], ),
    r'$+=%.2f$' % (percentile[2], )))
 
ax.text(0.3, 3, textstr, 
    verticalalignment='top')#, bbox=props)
#sns.displot(data=posterior,kind= 'kde',bw_adjust=3)#,ax=ax)
binwidth = (omch2[-1]-omch2[0])/(len(omch2)-1)
binrange = [omch2[0]+binwidth/2, omch2[-1]+binwidth/2]
sns.histplot(posterior_distr, stat='density',binwidth=binwidth,binrange=binrange,element='step',fill=True, ax=ax)
ax.set_xlim(binrange[0], binrange[1])
ax.axvline(percentile[0],color='r')
#ax.axvline(mean,color='r')
ax.axvline(percentile[2],color='r')
ax.axvline(omch2[index], color = 'r')
ax.axvline(0.122,color ='k',linestyle='-.', label = 'Planck 2018')

plt.legend(loc='best')
plt.tight_layout()
plt.savefig('parameter_estimation/' +filename +'.png')



fig1 = plt.figure(figsize=(17,10))
ax1 = fig1.add_subplot(1, 1, 1)

ax1.plot(omch2, chi_squared, 'o')
ax1.set_title(r'Chi squared for $\omega_{c}h^{2}$ , grid = '+str(len(omch2))+' points')
ax1.set_xlabel(r'$\omega_{c}h^{2}$')

ax1.axvline(omch2[index], color = 'r', label = 'min chi squared=%1.2f' %omch2[index])
ax1.axvline(percentile[1],color ='k',linestyle='-.', label = 'percentile posterior=%1.2f' %percentile[1])#(0.6847,color ='k',linestyle='-.', label = 'Planck 2018')
ax1.axvline(mode(posterior_distr)[0][0],color ='k',linestyle=':', label = 'max posterior=%1.2f' %mode(posterior_distr)[0][0])#(0.6847,color ='k',linestyle='-.', label = 'Planck 2018')
ax1.axvline(np.mean(posterior_distr),color ='r',linestyle='-.', label = 'mean posterior=%1.2f' %np.mean(posterior_distr))#(0.6847,color ='k',linestyle='-.', label = 'Planck 2018')
plt.legend(loc = 'best')
plt.savefig('parameter_estimation/chi_squared_omch2_'+str(num_sim)+'_'+str(len(omch2))+'.png')


print(np.median(posterior_distr), mode(posterior_distr)[0][0] ,np.mean(posterior_distr), omch2[index] )
#
##plt.savefig('prova_posterior.png')
##plt.hist(posterior_distr, bins=30)