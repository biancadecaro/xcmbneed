import numpy as np
import matplotlib.pyplot as plt
import likelihood_analysis_module as liklh
from scipy.stats import chi2, mode
import cython_mylibc as pippo
from scipy.interpolate import CubicSpline, interp1d
from scipy.integrate import quad
import seaborn as sns
import os
import spectra

sns.set_theme(style = 'white')
#sns.set_palette('husl')
#plt.style.use("dark_background")
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

out_dir_plot = 'parameter_estimation/Posterior_OmL/Mask_noise/Planck_Euclid_Mask/'
if not os.path.exists(out_dir_plot):
		os.makedirs(out_dir_plot)

OmL = np.linspace(0.0,0.95,30)
OmL_int = np.linspace(0.0, 0.95, 1000)

jmax = 12
lmax = 256
B = pippo.mylibpy_jmax_lmax2B(jmax, lmax)

###########################################################################################################################
Mll_pl_eu=np.loadtxt(f'mask/EUCLID/kernel_Euclid_Planck_TTGG_lmax{lmax}.dat')
Mll_comb=np.loadtxt(f'mask/EUCLID/kernel_Euclid_Planck_TGTG_lmax{lmax}.dat')

cl_theory = np.loadtxt('spectra/inifiles/EUCLID_fiducial_lmin0.dat')
ell_theory = cl_theory[0]
cl_theory_tt = cl_theory[1]
cl_theory_tg = cl_theory[2]
cl_theory_gg = cl_theory[3]
Nll = np.ones(cl_theory_gg.shape[0])/354543085.80126834
need_theory = spectra.NeedletTheory(B)
b_need = need_theory.get_bneed(jmax, lmax)
gammaJ_tg = need_theory.gammaJ(cl_theory_tg, Mll_pl_eu, lmax)
delta_gammaj = need_theory.variance_gammaj(cltg=cl_theory_tg,cltt=cl_theory_tt, clgg=cl_theory_gg, Mll_1x2=Mll_comb, Mll=Mll_pl_eu, jmax=jmax, lmax=lmax, noise_gal_l=Nll)[1:jmax,1:jmax]


########################################################################################################################

idx = (np.abs(OmL - 0.6847)).argmin()
#print(idx, OmL[idx])

gamma_oml = np.array([np.loadtxt(f'output_needlet_TG_OmL/Grid_spectra_30_Gammaj_Euclid/TGsims_theoretical_OmL{oml}/beta_TS_galS_theoretical_OmL{oml}_B1.5874010519681994.dat') for oml in OmL])[:,1:(jmax)]
print(f'shape beta_oml = {gamma_oml.shape}')

#plt.plot(gamma_oml[21])
#plt.savefig('gamma_oml.png')

cov_matrix = np.loadtxt('output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000_2masks_1/cov_TS_galT_jmax12_B_1.59_nside128_fsky0.36.dat')[1:(jmax),1:(jmax)]
#print(f'shape cov = {cov_matrix.shape}')

icov=np.linalg.inv(cov_matrix)


beta_fid_array = np.loadtxt('output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000_2masks_1/gamma_sims_TS_galT_jmax12_B_1.59_nside128_fsky0.36.dat')[:,1:(jmax)]
#print(f'beta_fid={beta_fid_array.shape}')
num_sim = 372

beta_fid =  beta_fid_array[num_sim] #
beta_fig_theory=np.mean(beta_fid_array, axis=0)


delta = np.array([np.subtract(beta_fid, gamma_oml[p]) for p in range(len(OmL))])#beta_fid-beta_oml#
delta_theory = np.array([np.subtract(beta_fig_theory, gamma_oml[p]) for p in range(len(OmL))])#beta_fid-beta_oml#


nj=cov_matrix.shape[1]

chi_squared = liklh.Calculate_chi2_grid(beta_fid=beta_fig_theory, beta_grid=gamma_oml, cov=cov_matrix, jmax=jmax, params = OmL )
chi_squared_theory = liklh.Calculate_chi2_grid(beta_fid=beta_fig_theory, beta_grid=gamma_oml, cov=delta_gammaj, jmax=jmax, params = OmL )


perc = chi2.cdf(chi_squared, nj)
perc_theory = chi2.cdf(chi_squared_theory, nj)

lik = liklh.Likelihood(chi_squared=chi_squared)
lik_theory = liklh.Likelihood(chi_squared=chi_squared)
interp_lik= CubicSpline(x=OmL,y=lik)
lik_int= interp_lik(OmL_int)
interp_lik_theory= CubicSpline(x=OmL,y=lik_theory)
lik_theory_int= interp_lik_theory(OmL_int)

posterior_distr = liklh.Sample_posterior(chi_squared, OmL)
posterior_distr_theory = liklh.Sample_posterior(chi_squared_theory, OmL)

percentile = np.percentile(posterior_distr, q = [16,50,84])
percentile_theory = np.percentile(posterior_distr_theory, q = [16,50,84])
print(f'percentile theory:{percentile_theory}')

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
plt.savefig(out_dir_plot+f'chi_squared_mean_{len(OmL)}_theoretical_OmL_EUCLID.png')

filename = f'Posterior_OmL_{len(OmL)}_mean_sim_best-fit_theoretical_OmL_EUCLID'
filename_lik = f'Like_OmL_{len(OmL)}_sim_best-fit_EUCLID'


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
 
ax.text(0.2, 0.1, textstr, 
    verticalalignment='top')#, bbox=props)


binwidth = (OmL[-1]-OmL[0])/(len(OmL)-1)
binrange = [OmL[0]+binwidth/2, OmL[-1]+binwidth/2]
sns.histplot(posterior_distr, stat='probability',binwidth=binwidth,binrange=binrange,element='step',fill=True, alpha=0.5 ,color='#2b7bbc',ax=ax)


ax.set_xlim(binrange[0], binrange[1])
ax.axvline(percentile[0],color='b')

ax.axvline(percentile[2],color='b')

ax.axvline(OmL[index], color = 'b', linestyle='-')

ax.axvline(0.68,color ='grey',linestyle='--', label = r'$\Omega_{\Lambda}=0.68$')

plt.legend(loc='best')
plt.tight_layout()

plt.savefig(out_dir_plot +filename +'_like.png')
#plt.savefig(out_dir_plot +filename +'_like.pdf')


fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(1, 1, 1)

ax.set_title(r'Probability distribution for $\Omega_{\Lambda}$ , grid = '+str(len(OmL))+' points')
ax.set_xlabel(r'$\Omega_{\Lambda}$')

def integrand(x):
    return interp_lik(x)

y_area = np.array([quad(integrand, OmL_int.min(), i)[0] for i in OmL_int])

total_area = y_area[-1]

pdf = lik_int/total_area

area = np.array([quad(integrand, OmL_int.min(), i)[0] for i in OmL_int])/total_area

# interpolate the area array
f = interp1d(area, OmL_int)

sigma_left = f(0.16)
sigma_right = f(0.84)

plt.plot(OmL_int, lik_int)
plt.fill_between(OmL_int, 0, lik_int, where=(OmL_int >= sigma_left) & (OmL_int <= sigma_right), alpha=0.5)
# draw a vertical line at tx08 and tx92
#plt.axvline(sigma_left, color='k', linestyle='--')
#plt.axvline(sigma_right, color='k', linestyle='--')
ax.axvline(0.6847,color ='grey',linestyle='--', label = 'Planck 2018')
ax.set_ylim(bottom=0)
textstr = '\n'.join((
    r'$\Omega_{\Lambda}=%.2f^{+ %.2f}_{-%.2f}$' % (OmL[index], sigma_right-OmL[index], OmL[index]-sigma_left ),
    #r'$-=%.2f$' % (percentile[0], ),
    #r'$+=%.2f$' % (percentile[2], )
    ))
 
ax.text(0.2, 0.06, textstr, 
    verticalalignment='top')#, bbox=props)


plt.legend(loc='best')
plt.tight_layout()

plt.savefig(out_dir_plot+filename_lik +'.png')
#plt.savefig(out_dir_plot+filename_lik +'.pdf')
###################################################################################################################
#################################### FIDUCIAL GAUSSIAN #########################################################
#filename_fiducial = f'spectra/Grid_spectra_{len(OmL)}_EUCLID/EUCLID_cl_OmL_fiducial.dat'
#
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

#print((cl_fiducial['nbins1']['TG'][0]-cl_theory_tg[2:])/cl_theory_tg[2:])
#
#factor=ell*(ell+1)/(2*np.pi)
#cl_grid ={}
#cl_grid['nbins1'] = np.zeros((len(OmL), nbins, ell.shape[0]))
#for p, oml in enumerate(OmL):
#    filename = f'spectra/Grid_spectra_{len(OmL)}_EUCLID/EUCLID_cl_OmL{oml}.dat'#.replace('.', '')+'.dat'
#    cl_grid['nbins1'][p,0] = np.loadtxt(filename)[1][2:]
#cl_grid['grid'] = OmL
#iCov_grid = Get_icov_grid(ell, cl_grid, cl_fiducial, fsky=0.36, noise_flag=True)
#like_fid_grid = Get_TG_like_grid(ell, cl_grid, cl_fiducial, iCov_grid)
#
#plt.figure()
#plt.suptitle('gaussian cov')
#plt.plot(OmL, like_fid_grid['nbins1'])
#plt.show()
#####################################################################################################################
##################################### ALTRO PLOT ##################################################################
nsim = beta_fid_array.shape[0]

chi2_OmL =np.zeros((nsim,len(OmL)))
post_OmL= np.zeros((nsim,len(OmL)))
OmL_min = np.zeros(nsim)
for n in range(nsim):
    chi2_OmL[n] = liklh.Calculate_chi2_grid(beta_fid=beta_fid_array[n], beta_grid=gamma_oml, cov=cov_matrix, jmax=jmax, params = OmL )
    post_OmL[n] = liklh.Likelihood(chi2_OmL[n])
    index,  = np.where(chi2_OmL[n]==chi2_OmL[n].min())
    OmL_min[n] = OmL[index]

OmL_mean= np.mean(OmL_min)
OmL_std = np.std(OmL_min)
percentile_sim = np.percentile(OmL_min, q = [16,50,84])
print(percentile_sim, OmL_mean)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(1, 1, 1)

ax.set_title(r'Probability distribution for $\Omega_{\Lambda}$ , grid = '+str(len(OmL))+' points')
ax.set_xlabel(r'$\Omega_{\Lambda}$')

textstr1 = '\n'.join((
   r'$\Omega_{\Lambda}=%.2f^{+ %.2f}_{-%.2f}$' % (OmL_mean, percentile_sim[2]-OmL_mean, OmL_mean-percentile_sim[0] ),
    #r'$\pm=%.2f$' % (OmL_std, ),
#    r'$+=%.2f$' % (percentile[2], )
    ))
 
ax.text(0.2, 0.4, textstr1, 
    verticalalignment='top')#, bbox=props)


binwidth = (OmL[-1]-OmL[0])/(len(OmL)-1)
binrange = [OmL[0]+binwidth/2, OmL[-1]+binwidth/2]

sns.lineplot(x=OmL_int, y=lik_theory_int/lik_theory_int.max())
bins=12
counts = np.histogram(OmL_min, bins=bins)
print(OmL_int[np.where(lik_theory_int==lik_theory_int.max())])
weights = np.ones(OmL_min.shape) / max(counts[0]).max()
#sns.histplot(OmL_min, stat='counts', binwidth=binwidth,binrange=binrange,element='step',fill=True, alpha=0.5 ,color='#2b7bbc',ax=ax)
plt.hist(OmL_min,density=False, bins=bins, weights=weights,color='#2b7bbc',alpha=0.5, histtype='stepfilled' )

ax.set_xlim(binrange[0], binrange[1])
ax.axvline(OmL_mean,color='#2b7bbc', linestyle='-')

ax.axvline(percentile_sim[2],color='#2b7bbc')

ax.axvline(percentile_sim[0], color='#2b7bbc')

ax.axvline(0.68,color ='grey',linestyle='--', label = r'$\Omega_{\Lambda}=0.68$')

plt.legend(loc='best')
plt.tight_layout()
filename = f'Posterior_OmL_{len(OmL)}_mean_sim_best-fit_theoretical_OmL_from_sims_EUCLID'
plt.savefig(out_dir_plot+filename +'.png')
#plt.savefig(out_dir_plot+filename +'.pdf')


plt.show()