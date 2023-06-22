import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import likelihood_analysis_module as liklh

def cov(a, b):

    if len(a) != len(b):
        return

    a_mean = np.mean(a)
    b_mean = np.mean(b)

    sum = 0

    for i in range(0, len(a)):
        sum += ((a[i] - a_mean) * (b[i] - b_mean))

    return sum/(len(a)-1)

OmL = np.linspace(0.0,0.95,30)

jmax = 12
lmax = 800
delta_ell = 50
lbins = 15
idx = (np.abs(OmL - 0.6847)).argmin()

cl_sims = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_Cl_TG/Planck/TG_512_planck_2_lmin0/cl_sims_TS_galS_lmax800_nside512.dat')


cov_calc = np.zeros((lbins,lbins))

for ell in range(lbins):
    for ell1 in range(lbins):
        cov_calc[ell,ell1] = cov(cl_sims.T[ell], cl_sims.T[ell1])

#print(cov_calc)

cl_oml = np.array([np.loadtxt(f'/ehome/bdecaro/xcmbneed/src/output_cl_TG_OmL/Grid_spectra_30_planck_lmin0/TGsims_512_OmL{oml}/cl_sims_TS_galS_Om{oml}_delta_ell50_lmax = 800_nside512.dat') for oml in OmL])

cov_matrix = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_Cl_TG/Planck/TG_512_planck_2_lmin0/cov_TS_galS_lmax800_nside512.dat')
icov = np.linalg.inv(cov_matrix)

#print(cov_calc-cov_matrix)

cl_fid_array = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_Cl_TG/Planck/TG_512_planck_2_lmin0/cl_sims_TS_galS_lmax800_nside512.dat')

cl_fid = np.array([np.mean(cl_fid_array, axis=0)  for _ in range(len(OmL))])

delta = cl_fid-cl_oml

chi_squared=np.zeros(len(OmL))
nj=cov_matrix.shape[1]

#print(sum(np.dot(delta[0,:],np.dot(icov[0,:],delta[0,:]))))


chi_squared=np.zeros(len(OmL))

nj=cov_matrix.shape[1]
for om in range(len(OmL)):
    chi_squared[om]  += np.matmul(delta[om], np.matmul(icov, delta[om]))
        

print(f'chi2={chi_squared}')

lik = liklh.Likelihood(chi_squared=chi_squared)
print(f'likelihood={lik}') 