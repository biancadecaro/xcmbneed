import numpy as np
import analysis, utils, spectra, sims
from scipy.stats import chi2
import matplotlib.pyplot as plt
#from tqdm import tqdm
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
from matplotlib import rc, rcParams
#plt.rcParams['axes.linewidth']  = 5.
plt.rcParams['axes.labelsize']  =18
plt.rcParams['xtick.labelsize'] =15
plt.rcParams['ytick.labelsize'] =15
#plt.rcParams['xtick.major.size'] = 20
#plt.rcParams['ytick.major.size'] = 20
#plt.rcParams['xtick.minor.size'] = 20
#plt.rcParams['ytick.minor.size'] = 20
plt.rcParams['legend.fontsize']  = 'large'
#plt.rcParams['legend.frameon']  = False
plt.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['axes.titlesize'] = '20'
rcParams["errorbar.capsize"] = 5
##
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['font.size'] = 20
plt.rcParams['lines.linewidth']  = 3.

def chi_s2(delta, cov):
    cov_inv = np.linalg.inv(cov)

    chi_squared = 0

    temp = np.zeros(len(cov[0]))
    for i in range(1,len(cov[0])):
        for j in range(1,len(cov[0])):
            temp[i] += cov_inv[i][j]*(delta[j])
        chi_squared += (delta[i]).T*temp[i]
    return chi_squared

def chi2_theory(diff, variance):
    chi_squared_delta = 0#
    for j in range(len(diff)):
        chi_squared_delta += (diff[j])**2/variance[j]**2
    return chi_squared_delta

if __name__ == "__main__":

    cov_null = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG_Null/TGNull_512_Planck/cov_TS_galS_jmax12_B1.7422102429630426_nside512.dat', unpack=True)#np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/TG_512/cov_TS_galS_jmax10_B1.9467970312828855_nside512.dat', unpack=True)#np.loadtxt('output_needlet_TG/Planck/TG_256_mask_noise_nonoisePlanck/cov_TT_galT_jmax12_B = 1.74 $_nside256_mask.dat', unpack=True)
    beta_j_sims_null = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG_Null/TGNull_512_Planck/betaj_sims_TS_galS_jmax12_B1.74_nside512.dat', unpack=True)#np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/TG_512/betaj_sims_TS_galS_jmax10_B1.9467970312828855_nside512.dat', unpack=True)#np.loadtxt('output_needlet_TG/Planck/TG_256_mask_noise_nonoisePlanck/betaj_sims_TT_galT_jmax12_B_1.7422102429630426_nside256_fsky_0.7004634737968445.dat', unpack=True)

    beta_j_sims = np.loadtxt('/ehome/bdecaro/xcmbneed/src/output_needlet_TG/Planck/TG_512/betaj_sims_TS_galS_jmax12_B1.74_nside512.dat', unpack=True)
    beta_j_mean = np.mean(beta_j_sims, axis=1)

    beta_j_mean_null = np.mean(beta_j_sims_null, axis=1)

    print(beta_j_sims.shape)

    simparams = {'nside'   : 512,
                 'ngal'    : 5.76e5,
     	     	 'ngal_dim': 'ster',
    	     	 'pixwin'  : False}
    sims_dir        = 'sims/Needlet/TGsims_'+str(simparams['nside'])+'/'
    out_dir         = 'output_needlet_TG/'

    jmax = 12
    lmax = 782
    nsim = 500

    fname_xcspectra = f'/ehome/bdecaro/xcmbneed/src/spectra/inifiles/CAMBNull_planck.dat'#'spectra/CAMBSpectra.dat'
    xcspectra = spectra.XCSpectraFile(fname_xcspectra, WantTG = True)

    simulations = sims.KGsimulations(xcspectra, sims_dir, simparams, WantTG = True)
    #simulations.Run(nsim, WantTG = True)

    myanalysis = analysis.NeedAnalysis(jmax, lmax, out_dir, simulations)
    need_theory = spectra.NeedletTheory(myanalysis.B)

    delta = need_theory.delta_beta_j(jmax,  cltt = xcspectra.cltt, cltg = xcspectra.cltg, clgg = xcspectra.clg1g1)

    chi_2 = np.zeros(nsim)
    percentile =np.zeros(nsim)
    for n in range(beta_j_sims_null.shape[1]):
        #print(f'simulation n={n}')
        chi_2[n] = chi2_theory(beta_j_sims_null[:,n]-beta_j_mean, delta)
        percentile[n] = chi2.cdf(chi_2[n], 13)


    fig = plt.figure(figsize=(17,10))
    ax = fig.add_subplot(1, 1, 1)
    sns.histplot(data=chi_2, element="step")
    plt.tight_layout()
    plt.savefig('plot_tesi/hist_chi2_null_test.png')

    fig = plt.figure(figsize=(17,10))
    ax = fig.add_subplot(1, 1, 1)
    sns.histplot(data=percentile)
    plt.savefig('plot_tesi/hist_percentili_null_test.png')


    prova_chi2 = chi2_theory(beta_j_sims_null[:, 78]-beta_j_mean,delta )
    print(prova_chi2,chi2.cdf(prova_chi2, 13))




