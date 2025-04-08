import numpy as np
import matplotlib.pyplot as plt
import likelihood_analysis_module as liklh
from scipy.stats import chi2, mode
import cython_mylibc as pippo
from scipy.interpolate import CubicSpline, interp1d
from scipy.integrate import quad
import seaborn as sns
import os
sns.set_style(style = 'white')


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
###########################################################################################################################
def Compute_TG_covmat(ell, TT, TG, GG, fsky: float = 1.0, noise_flag=False, analytic_margi=False):

# Computes covariance matrix from fiducials (TT, TG, GG)
# INPUT:
#        - ell: array with desired multipoles
#        - TT, TG, GG: fiducial spectra
#        - fsky: observed sky fraction, set to 1.0 by default
#        - noise_flag: if True shot noise of galaxies is considered
# OUTPUT:
#        - cov[bin1, bin2, l]

    cov   = np.zeros_like(GG)
    nbins = TG.shape[0]
    nell  = TG.shape[1]

    noise_vec = np.zeros_like(GG)
    if noise_flag==True:
        gal_per_sqarcmn = 30.
        noise = 1./((3600.*gal_per_sqarcmn*(180./np.pi)**2)/nbins)
        for i in range(nbins):
            noise_vec[i,i,:] = noise
            
    for il in range(nell):
        for bin1 in range(nbins):
            for bin2 in range(nbins):
                cov[bin1, bin2, il] = (TG[bin1, il] * TG[bin2, il] \
                                       + TT[il] * (GG[bin1, bin2, il] + noise_vec[bin1,bin2,il]) ) \
                                    / ( fsky*(2*ell[il] + 1) )
    
    if analytic_margi==True:
        fisherIST = np.loadtxt('EuclidISTF_GCph_WL_XC_w0wa_flat_pessimistic.txt')
        covIST = np.linalg.inv(fisherIST)
        reduced_covIST = covIST[10:,10:]
        for il in range(nell):
            for bin1 in range(nbins):
                for bin2 in range(nbins):
                    cov[bin1,bin2,il] += TG[bin1,il]*TG[bin2,il]*reduced_covIST[bin1,bin2]
    
    return cov


def Invert_TG_covmat(cov):

# Computes inverse covariance
# INPUT:
#        - cov: covariance matrix computed from fiducials (TT, TG, GG)
# OUTPUT:
#        - icov[bin1, bin2, l]

    icov = np.zeros_like(cov)
    nell = cov.shape[2]
    
    for il in range(nell):
        icov[:, :, il] = np.linalg.inv(cov[:, :, il])

    return icov


def Compute_TG_chi2(TGobs, TGth, icov, Aisw: float=1.0):

# Computes chi2 between given set of theoretical and observed TGs
# INPUT:
#        - TGobs[bin, l]: values for observed TG
#        - TGth[bin, l] : values for theoretical TG, from desired input model
#        - icov: inverse covariance
#        - Aisw: value for the amplitude of ISW-effect, if present
#                (default is set to no effect; Aisw = 1.0)
# OUTPUT:
#        - chi2: value of the chi2 for the inputs

    nell  = icov.shape[2]
    if Aisw == 1.0:
        delta = TGobs - TGth
    else:
        delta = TGobs - TGth*Aisw
        
    chi2  = 0.0
    for il in range(nell):
        chi2 += np.dot(delta[:, il], np.dot(icov[:, :, il], delta[:, il]))

    return chi2


def Compute_TG_loglike(chi2):
    
    return -0.5 * chi2


def Get_icov_grid(ell, pklfile, fid, fsky: float=1.0, noise_flag=False, analytic_margi=False):

# Computes grid (as a list of dictionaries) for inverse covaraince
# INPUT:
#        - ell: array with desired multipoles
#        - pklfile: pickle file with data, used to recover analysis settings
#        - fid: fiducials for TT, TG and GG (list of dictionaries-like)
#        - fsky: observed sky fraction, set to 1.0 by default
#        - noise_flag: if True shot noise of galaxies is considered
# OUTPUT:
#        - icovGrid[settings][bin1, bin2, l]: grid for inverse covariance matrix
    
    icovGrid = {}
    for setting in pklfile.keys()  - ['grid']:
        
        TTfid = fid[setting]['TT']
        TGfid = fid[setting]['TG']
        GGfid = fid[setting]['GG']
        
        cov = Compute_TG_covmat(ell, TTfid, TGfid, GGfid, fsky, noise_flag, analytic_margi)
        icovGrid[setting] = Invert_TG_covmat(cov)
        
    return icovGrid


def Get_TG_chi2_grid(ell, pklfile, fid, icovGrid, AiswGrid=None, testing=False):

# Computes grid (as a list of dictionaries) for chi2
# INPUT:
#        - ell: array with desired multipoles
#        - pklfile: pickle file containing data for fiducials and grid spectra
#                   (TG values for the chosen model);
#                   TGth = pklfile[setting]['cls_grid'][v1][v2(if present)][v3(if present)][bin][l],
#                          where v1,2,3 are the varying parameters of the chosen
#                          model (NOTE: in case of Aisw validation
#                          TGth = pklfile[setting]['cls_grid'][bin][l], as there
#                          is no model parameter to construct a grid on)
#        - fid: fiducials for TT, TG and GG (list of dictionaries-like)
#        - icovGrid[setting]: grid for inverse covaraince (list of dictionaries)
#        - AiswGrid: numpy.ndarray-like object containing desired values for the
#                    amplitude of ISW-effect (needed only for Aisw validation =>
#                    TGth has no dependence on variations of model parameters)
#        - testing: for Aisw validation only; if False and AiswGrid is provided
#                   the standard likelihood for a set of fiducials and grid
#                   spectra is performed, if True and AiswGrid is provided
#                   the likelihood is computed using only fiducials
#                   (TGobs = TGfid and TGth = TGfid)
# OUTPUT:
#        - chi2Grid[settings][v1][v2(if present)][v3(if present)]: chi2 for each
#                                 setting and each variation of the chosen model
    
    chi2Grid = {}
    for setting in pklfile.keys()  - ['grid']:
        
        TGobs = fid[setting]['TG']
        icov  = icovGrid[setting]
        
        if 'grid' in pklfile.keys() and testing == False:
        
            pkldata = pklfile[setting]
        
            if len(pkldata.shape) == 3:
            
                nval1 = pkldata.shape[0]
                chi2Grid[setting] = np.zeros(nval1)
            
                for val1 in range(nval1):
                   
                    TGth = pkldata[val1, :, :]#ell
                    chi2Grid[setting][val1] = Compute_TG_chi2(TGobs, TGth, icov)
        
            elif len(pkldata.shape) == 4:
            
                nval1 = pkldata.shape[0]
                nval2 = pkldata.shape[1]
                chi2Grid[setting] = np.zeros([nval1, nval2])
            
                for val1 in range(nval1):
                    for val2 in range(nval2):
                        TGth = pkldata[val1, val2, :, ell].T
                        chi2Grid[setting][val1, val2] = Compute_TG_chi2(TGobs, TGth, icov)
                    
            elif len(pkldata.shape) == 5:
            
                nval1 = pkldata.shape[0]
                nval2 = pkldata.shape[1]
                nval3 = pkldata.shape[1]
                chi2Grid[setting] = np.zeros([nval1, nval2, nval3])
            
                for val1 in range(nval1):
                    for val2 in range(nval2):
                        for val3 in range(nval3):
                            TGth = pkldata[val1, val2, val3, :, ell].T
                            chi2Grid[setting][val1, val2] = Compute_TG_chi2(TGobs, TGth, icov)
                
        elif type(AiswGrid) == np.ndarray and testing == True:
        
            nval1 = len(AiswGrid)
            chi2Grid[setting] = np.zeros_like(AiswGrid)
            
            for val1 in range(nval1):
                chi2Grid[setting][val1] = Compute_TG_chi2(TGobs, TGobs, icov, AiswGrid[val1])
                
        elif type(AiswGrid) == np.ndarray and testing == False:
        
            print('ERROR: you are trying to perform Aisw analysis but set testing=False, try setting testing=True')
            break
        
        else:
        
            print('ERROR: Either the input grid depends on a number of parameters != (1,2,3) or AiswGrid is not of type np.ndarray')
            break
                        
    return chi2Grid


def Get_TG_loglike_grid(ell, pklfile, fid, icovGrid, AiswGrid=None, testing=False):

# Computes grid (as a list of dictionaries) for the likelihood
# INPUT:
#        - ell: array with desired multipoles
#        - pklfile: pickle file containing data for fiducials and grid spectra
#                   (TG values for the chosen model);
#                   TGth = pklfile[setting]['cls_grid'][v1][v2(if present)][v3(if present)][bin][l],
#                          where v1,2,3 are the varying parameters of the chosen
#                          model (NOTE: in case of Aisw validation
#                          TGth = pklfile[setting]['cls_grid'][bin][l], as there
#                          is no model parameter to construct a grid on)
#        - fid: fiducials for TT, TG and GG (list of dictionaries-like)
#        - icovGrid[setting]: grid for inverse covaraince (list of dictionaries)
#        - AiswGrid: numpy.ndarray-like object containing desired values for the
#                    amplitude of ISW-effect (needed only for Aisw validation =>
#                    TGth has no dependence on variations of model parameters)
#        - testing: for Aisw validation only; if False and AiswGrid is provided
#                   the standard likelihood for a set of fiducials and grid
#                   spectra is performed, if True and AiswGrid is provided
#                   the likelihood is computed using only fiducials
#                   (TGobs = TGfid and TGth = TGfid)
# OUTPUT:
#        - loglikeGrid[setting][v1][v2(if present)][v3(if present)]: loglike for
#                            each setting and each variation of the chosen model
    
    chi2Grid = Get_TG_chi2_grid(ell, pklfile, fid, icovGrid, AiswGrid, testing)
    
    loglikeGrid = {}
    for setting in pklfile.keys() - ['grid']:
        loglikeGrid[setting] = -0.5 * chi2Grid[setting]
                        
    return loglikeGrid
    
    
def Get_TG_like_grid(ell, pklfile, fid, icovGrid, AiswGrid=None, testing=False, norm=False):

# Computes grid (as a list of dictionaries) for the likelihood
# INPUT:
#        - ell: array with desired multipoles
#        - pklfile: pickle file containing data for fiducials and grid spectra
#                   (TG values for the chosen model);
#                   TGth = pklfile[setting]['cls_grid'][v1][v2(if present)][v3(if present)][bin][l],
#                          where v1,2,3 are the varying parameters of the chosen
#                          model (NOTE: in case of Aisw validation
#                          TGth = pklfile[setting]['cls_grid'][bin][l], as there
#                          is no model parameter to construct a grid on)
#        - fid: fiducials for TT, TG and GG (list of dictionaries-like)
#        - icovGrid[setting]: grid for inverse covaraince (list of dictionaries)
#        - AiswGrid: numpy.ndarray-like object containing desired values for the
#                    amplitude of ISW-effect (needed only for Aisw validation =>
#                    TGth has no dependence on variations of model parameters)
#        - testing: for Aisw validation only; if False and AiswGrid is provided
#                   the standard likelihood for a set of fiducials and grid
#                   spectra is performed, if True and AiswGrid is provided
#                   the likelihood is computed using only fiducials
#                   (TGobs = TGfid and TGth = TGfid)
#        - norm: if True, the likelihood is normalized
# OUTPUT:
#        - likeGrid[setting][v1][v2(if present)][v3(if present)]: like for each
#                                setting and each variation of the chosen model
    
    chi2Grid = Get_TG_chi2_grid(ell, pklfile, fid, icovGrid, AiswGrid, testing)
    
    likeGrid = {}
    for setting in pklfile.keys() - ['grid']:
        likeGrid[setting] = np.exp(-0.5 * chi2Grid[setting])

        if norm == True:
            print('normalizing likelihood for setting -> %s'%(setting))
            likeGrid[setting] = likeGrid[setting] /np.sum(likeGrid[setting])
            
    return likeGrid

###################################################################################################################################

out_dir_plot = 'parameter_estimation/Posterior_Aisw/Mask_noise/Planck_Euclid_Mask/'
if not os.path.exists(out_dir_plot):
		os.makedirs(out_dir_plot)
                
Aisw=np.linspace(0.0,2.,50)
Aisw_int=np.linspace(0.0,2.,1000)
jmax = 12
lmax = 782

cov_matrix = np.loadtxt('output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000_2masks_1/cov_TS_galT_jmax12_B_1.59_nside128_fsky0.36.dat')[1:(jmax),1:(jmax)]

#variance_theory = np.loadtxt('output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000_2masks_1/variance_gammaj_theory_B1.59_jmax12_nside128_lmax256_nsim1000_fsky0.36.dat')[1:(jmax+1)]
print(f'shape cov = {cov_matrix.shape}')
nj=cov_matrix.shape[1]

beta_fid_array = np.loadtxt('output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000_2masks_1/gamma_sims_TS_galT_jmax12_B_1.59_nside128_fsky0.36.dat')[:,1:(jmax)]
nsim = beta_fid_array.shape[0]

gamma_theory = np.loadtxt('output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000_2masks_1/gammaj_tg_jmax12_lmax256.dat')[1:(jmax)]
delta_gammaj = np.loadtxt('output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000_2masks_1/covariance_gammaj_tg_jmax12_lmax256.dat')[1:(jmax), 1:(jmax)]


num_sim = 79

beta_fid =   beta_fid_array[num_sim]

beta_fid_mean = np.mean(beta_fid_array, axis=0)
beta_A =np.array([a*gamma_theory for a in Aisw])

chi_squared = liklh.Calculate_chi2_grid(beta_fid=beta_fid_mean, beta_grid=beta_A, cov=cov_matrix, jmax=jmax, params = Aisw )
chi_squared_theory = liklh.Calculate_chi2_grid(beta_fid=beta_fid_mean, beta_grid=beta_A, cov=delta_gammaj, jmax=jmax, params = Aisw )

#chi_squared = liklh.delta_chi2_grid(beta_fid=beta_fid, beta_grid=beta_A, cov=variance_theory**2, jmax=jmax, params = Aisw )
perc = chi2.cdf(chi_squared, nj)

lik = liklh.Likelihood(chi_squared=chi_squared)
interp_lik= CubicSpline(x=Aisw,y=lik)
lik_int= interp_lik(Aisw_int)

lik_theory = liklh.Likelihood(chi_squared=chi_squared_theory)
interp_lik_theory= CubicSpline(x=Aisw,y=lik_theory)
lik_theory_int= interp_lik_theory(Aisw_int)

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
plt.savefig(out_dir_plot+f'chi_squared_Aisw_{len(Aisw)}_EUCLID.png')


index, = np.where(chi_squared == chi_squared.min())

filename = 'Posterior_Aisw_best-fit_EUCLID'

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(1, 1, 1)

ax.set_title(r'Probability distribution for $A_{iSW}$ , grid = '+str(len(Aisw))+' points')

ax.set_xlabel(r'$A_{iSW}$')
textstr = '\n'.join((
    r'$A_{iSW}=%.2f \pm %.2f$' % (Aisw[index], Aisw[index]-percentile[0]),
    #r'$\pm%.2f$' % (Aisw[index]-percentile[0], ),
    #r'$+%.2f$' % (percentile[2]-Aisw[index], )
    ))


ax.text(1.37,0.02, textstr, 
    verticalalignment='top')#, bbox=props)
binwidth = (Aisw[-1]-Aisw[0])/(len(Aisw)-1)
binrange = [Aisw[0]+binwidth/2, Aisw[-1]+binwidth/2]

sns.lineplot(x=Aisw, y=lik)
sns.histplot(posterior_distr, stat='probability',binwidth=binwidth,binrange=binrange,element='step',fill=True,alpha=0.5 ,color='#2b7bbc', ax=ax)

ax.set_xlim(binrange[0], binrange[1])
ax.axvline(percentile[0],color='b')

#ax.axvline(mean,color='r')
ax.axvline(percentile[2],color='b')
ax.axvline(Aisw[index], color = 'b', linestyle='-')

ax.axvline(1.,color ='grey',linestyle='--', label = 'Fiducial Aisw=1')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(out_dir_plot +filename +'_100grid_like.png')
#plt.savefig('plot_tesi/Parameter_estimation/' +filename +'_100grid_like.pdf')

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(1, 1, 1)

def integrand(x):
    return interp_lik(x)

y_area = np.array([quad(integrand, Aisw_int.min(), i)[0] for i in Aisw_int])

total_area = y_area[-1]

pdf = lik_int/total_area

area = np.array([quad(integrand, Aisw_int.min(), i)[0] for i in Aisw_int])/total_area

f = interp1d(area, Aisw_int)

sigma_left = f(0.16)
sigma_right = f(0.84)
plt.plot(Aisw_int, lik_int)
plt.fill_between(Aisw_int, 0, lik_int, where=(Aisw_int >= sigma_left) & (Aisw_int <= sigma_right), alpha=0.5)
ax.set_ylim(bottom=0)
ax.set_title(r'Probability distribution for $A_{iSW}$ , grid = '+str(len(Aisw))+' points')
textstr = '\n'.join((
       r'$A_{\mathrm{iSW}}=%.2f^{+ %.2f}_{-%.2f}$' % (Aisw[index], sigma_right-Aisw[index], Aisw[index]-sigma_left ),

    #r'$\pm%.2f$' % (Aisw[index]-percentile[0], ),
    #r'$+%.2f$' % (percentile[2]-Aisw[index], )
    ))
ax.text(1.37,0.02, textstr, 
    verticalalignment='top')#, bbox=props)
ax.axvline(1.,color ='grey',linestyle='--', label = 'Fiducial Aisw')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(out_dir_plot+'Likelihood_Aisw_EUCLID_sim_100grid.png')
#################################################################################################
############################### FIDUCIAL GAUSSIAN ##############################################
#lmin = 2
#lmax = 500
#filename_fiducial = 'spectra/inifiles/EUCLID_fiducial_lmin0.dat'
#nbins=1
#lmin=2
#lmax=500
#ell=np.arange(lmin, lmax+1)
#cl_fiducial={}
#cl_fiducial['nbins1']={}
#cl_fiducial['nbins1']['TT'] = np.loadtxt(filename_fiducial)[1][2:]
#cl_fiducial['nbins1']['TG'] = np.zeros((nbins, ell.shape[0]))
#cl_fiducial['nbins1']['TG'][0] = np.loadtxt(filename_fiducial)[2][2:]
#cl_fiducial['nbins1']['GG'] = np.zeros((nbins, nbins, ell.shape[0]))
#cl_fiducial['nbins1']['GG'][0][0] = np.loadtxt(filename_fiducial)[3][2:]
#
#iCov_grid = Get_icov_grid(ell, cl_fiducial, cl_fiducial, fsky=0.36, noise_flag=True)
#like_fid_grid = Get_TG_like_grid(ell, cl_fiducial, cl_fiducial, iCov_grid, AiswGrid=Aisw, testing=True)
#plt.figure()
#plt.suptitle('gaussian cov')
#plt.plot(Aisw, like_fid_grid['nbins1'])
#plt.show()
################################################################################################
############################## ALRO PLOT #######################################################À



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

filename = 'Posterior_Aisw_best-fit_from_sims_EUCLID'

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(1, 1, 1)

ax.set_title(r'Probability distribution for $A_{iSW}$ , grid = '+str(len(Aisw))+' points')

ax.set_xlabel(r'$A_{iSW}$')
textstr = '\n'.join((
    r'$A_{iSW}=%.2f \pm %.2f$' % (Aisw_mean, Aisw_std),
    #r'$\pm%.2f$' % Aisw_std, ),
    #r'$%.2f$' % (percentile[2]-Aisw[index], 
    ))


ax.text(1.37,0.6, textstr, 
    verticalalignment='top')#, bbox=props)
binwidth = (Aisw[-1]-Aisw[0])/(len(Aisw)-1)
binrange = [Aisw[0]+binwidth/2, Aisw[-1]+binwidth/2]
sns.lineplot(x=Aisw, y=lik_theory/lik_theory.max())
#sns.histplot(Aisw_min, stat='probability',binwidth=binwidth,binrange=binrange,element='step',fill=True,alpha=0.5 ,color='#2b7bbc', ax=ax)
bins=15
counts = np.histogram(Aisw_min, bins=bins)
weights = np.ones(Aisw_min.shape) / max(counts[0]).max()
plt.hist(Aisw_min,density=False, bins=bins, weights=weights,color='#2b7bbc',alpha=0.5, histtype='stepfilled' )

ax.set_xlim(binrange[0], binrange[1])
#ax.axvline(percentile[0],color='b')

#ax.axvline(mean,color='r')
#ax.axvline(percentile[2],color='b')
ax.axvline(Aisw_mean, color='#2b7bbc', linestyle='-')
ax.axvline(Aisw_mean+Aisw_std,color='#2b7bbc' )
ax.axvline(Aisw_mean-Aisw_std, color='#2b7bbc')

ax.axvline(1.,color ='grey',linestyle='--', label = 'Fiducial Aisw=1')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(out_dir_plot +filename +'.png')

plt.show()
