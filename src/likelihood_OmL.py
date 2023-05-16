import numpy as np
import matplotlib.pyplot as plt
import likelihood_analysis_module as liklh
from scipy.stats import chi2, mode
import cython_mylibc as pippo
import seaborn as sns

OmL = np.linspace(0.0,0.95,30)
#Aisw=np.linspace(0.0,2.,31)
jmax = 12
lmax = 782
B = pippo.mylibpy_jmax_lmax2B(jmax, lmax)
print(B)

idx = (np.abs(OmL - 0.6847)).argmin()
print(idx, OmL[idx])

beta_oml = np.array([np.loadtxt(f'/ehome/bdecaro/xcmbneed/src/output_needlet_TG_OmL/Grid_spectra_30_planck_2_lmin0/TGsims_theoretical_OmL{oml}/beta_TS_galS_theoretical_OmL{oml}_B1.7422102429630426.dat') for oml in OmL])[:,1:(jmax)]#np.array([np.loadtxt(f'/ehome/bdecaro/xcmbneed/src/output_needlet_TG_OmL/Grid_spectra_30_planck_2_lmin0/TGsims_512_OmL{oml}/betaj_sims_TS_galS_Om{oml}_jmax12_B = 1.74 _nside512.dat') for oml in OmL])[:,1:(jmax)]
beta_oml_old = np.array([np.loadtxt(f'/ehome/bdecaro/xcmbneed/src/output_needlet_TG_OmL/Grid_spectra_30_old/TGsims_theoretical_OmL{oml}/beta_TS_galS_theoretical_OmL{oml}_B1.7422102429630426.dat') for oml in OmL])[:,1:(jmax)]
#beta_oml_old = np.array([np.loadtxt(f'/ehome/bdecaro/xcmbneed/src/output_needlet_TG_OmL/Grid_spectra_30/TGsims_512_OmL{oml}/betaj_sims_TS_galS_Om{oml}_jmax12_B = 1.74 _nside512.dat') for oml in OmL])[:,1:(jmax)]
print(beta_oml.shape)

print(f'shape beta_oml = {beta_oml.shape}')
cov_matrix_old = np.loadtxt('output_needlet_TG/Planck/TG_512_new/cov_TS_galS_jmax12_B1.7422102429630426_nside512.dat')[1:(jmax),1:(jmax)]
cov_matrix = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/TG_512_planck_2_lmin0/cov_TS_galS_jmax12_B1.7422102429630426_nside512.dat')[1:(jmax),1:(jmax)]
print(f'shape cov = {cov_matrix.shape}')
#cov_matrix_oldcov_matrix_diag=np.diag(cov_matrix)*np.eye(jmax+1)
#cov_matrix_diag_old=np.diag(cov_matrix_old)*np.eye(jmax+1)

#cov_matrix_variance = np.array([np.loadtxt(f'/ehome/bdecaro/xcmbneed/src/output_needlet_TG_OmL/Grid_spectra_30_old/TGsims_theoretical_OmL{oml}/variance_TS_galS_theoretical_OmL{oml}_B1.7422102429630426.dat') for oml in OmL])[:,1:(jmax)]

icov=np.linalg.inv(cov_matrix)
icov_old=np.linalg.inv(cov_matrix_old)

#print(cov_matrix)
#beta_fid = np.mean(np.loadtxt('output_needlet_TG/TG_512/betaj_sims_TS_galS_jmax12_B1.7422102429630426_nside512.dat'), axis=0)
#
beta_fid_array = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/TG_512_planck_2_lmin0/betaj_sims_TS_galS_jmax12_B1.7422102429630426_nside512.dat')
beta_fid_array_old = np.loadtxt('output_needlet_TG/Planck/TG_512_new/betaj_sims_TS_galS_jmax12_B1.7422102429630426_nside512.dat')

num_sim = 456

beta_fid =   np.mean(beta_fid_array, axis=0)[1:(jmax)]  #beta_fid_array[num_sim]
beta_fid_old =  np.mean(beta_fid_array_old, axis=0)[1:(jmax)]  #beta_fid_array_old[num_sim]

#plt.errorbar(x=np.arange(0,13),y=beta_fid, yerr=np.sqrt(np.diag(cov_matrix)), fmt='o')
#plt.savefig('betaj_mean.png')
#beta_fid = np.array([beta_fid for _ in range(len(OmL))])
#beta_fid_old = np.array([beta_fid_old for _ in range(len(OmL))])

#delta = beta_fid-beta_oml#np.array([np.subtract(beta_fid, beta_oml[p]) for p in range(len(OmL))])
#delta_old =beta_fid_old-beta_oml_old#np.array([np.subtract(beta_fid_old, beta_oml_old[p]) for p in range(len(OmL))])
#print(f'delta={delta}', f'delta old={delta_old}')

#chi_2 = np.array([liklh.Calculate_chi2(delta[p], cov_matrix) for p in range(len(OmL))])#liklh.Calculate_chi2_grid(beta_fid, beta_oml, cov_matrix, jmax, OmL)
#print(np.where(chi_2==chi_2.min()))
#chi_squared=np.zeros(len(OmL))
#chi_squared_old =np.zeros(len(OmL))
nj=cov_matrix.shape[1]
#print(nj)
#for om in range(len(OmL)):
#    for j in range(nj):
#        chi_squared[om]  += np.matmul(delta[om], np.matmul(icov, delta[om]))
#        chi_squared_old[om]  += np.matmul(delta_old[om], np.matmul(icov_old, delta_old[om]))
    

chi_squared = liklh.Calculate_chi2_grid(beta_fid=beta_fid, beta_grid=beta_oml, cov=cov_matrix, jmax=jmax, params = OmL )
chi_squared_old = liklh.Calculate_chi2_grid(beta_fid=beta_fid_old, beta_grid=beta_oml_old, cov=cov_matrix_old, jmax=jmax, params = OmL ) #liklh.delta_chi2_grid(beta_fid=beta_fid_old, beta_grid=beta_oml_old,cov=cov_matrix_variance**2, jmax=jmax, params=OmL)   #
#chi_squared = liklh.Calculate_chi2_grid(beta_fid=beta_fid, beta_grid=beta_oml, cov=cov_matrix_diag, jmax=jmax, params = OmL )
print(chi_squared, chi_squared_old)

perc = chi2.cdf(chi_squared, nj)
perc_old = chi2.cdf(chi_squared_old, nj)
#print(f'perc={perc}, perc_old={perc_old}')
lik = liklh.Likelihood(chi_squared=chi_squared)
lik_old = liklh.Likelihood(chi_squared=chi_squared_old)


#perc1 =np.percentile(lik, q=[16,50,84])

#index1, = np.where(lik == lik.max())

#print(index1,lik[index1], perc1[1],OmL[index1])

posterior_distr = liklh.Sample_posterior(chi_squared, OmL)
posterior_distr_old = liklh.Sample_posterior(chi_squared_old, OmL)

percentile_old = np.percentile(posterior_distr_old, q = [16,50,84])

percentile = np.percentile(posterior_distr, q = [16,50,84])


filename = 'Posterior_OmL_'+str(len(OmL))+'_mean_old_newbest-fit_theoretical_OmL'#_no_mnu'
#filename = 'Posterior_OmL_'+str(len(OmL))+'_sim'+str(num_sim)+'old_new_best-fit_theoretical_OmL'#_no_mnu'

#percentile = liklh.Plot_posterior(posterior_distr, OmL,chi_squared, filename)


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
plt.savefig('parameter_estimation/chi_squared_mean_'+str(len(OmL))+'_theoretical_OmL.png')


#print(np.median(posterior_distr), mode(posterior_distr)[0][0] ,np.mean(posterior_distr), OmL[index] )


#Confronto distribuzione vecchia e nuova

mean = np.mean(posterior_distr)
mean_old = np.mean(posterior_distr_old)

print(f'mean={mean}', f'mean old={mean_old}')
print(f'percetile={percentile}', f'percentile old={percentile_old}')
index, = np.where(chi_squared == chi_squared.min())
index_old, = np.where(chi_squared_old == chi_squared_old.min())

print(f'min chi squared={OmL[index[0]]}', f'min chi squared_old={OmL[index_old[0]]}')
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
sns.histplot(posterior_distr_old, stat='density',binwidth=binwidth,binrange=binrange,element='step',fill=True, color='r',ax=ax)

ax.set_xlim(binrange[0], binrange[1])
ax.axvline(percentile[0],color='b')
ax.axvline(percentile_old[0],color='r')

#ax.axvline(mean,color='r')
ax.axvline(percentile[2],color='b')
ax.axvline(percentile_old[2],color='r')
ax.axvline(OmL[index], color = 'b', linestyle='-')
ax.axvline(OmL[index_old], color = 'r', linestyle='-',label='old')
ax.axvline(0.6847,color ='k',linestyle='-.', label = 'Planck 2018')

plt.legend(loc='best')
plt.tight_layout()

plt.savefig('parameter_estimation/' +filename +'.png')

