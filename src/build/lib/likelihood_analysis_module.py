import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import analysis
plt.rcParams['font.size'] = '25'

#per ogni punto devo calcolare chi^2 = (beta_fid - beta_grid).T*cov^-1*(cls_fid - cls_the)
#poi passare alla likelihood
#come beta_fid devo prendere una simulazione sola e come beta 

def Calculate_chi2(delta, cov):#, jmax ):
    #delta = np.diff(beta_fid, beta_grid) 
    cov_inv = np.linalg.inv(cov)
    
    chi_squared = 0
    #print('delta=',delta)
    temp = np.zeros(len(cov[0]))
    for i in range(len(cov[0])):
        for j in range(len(delta)):
            temp[i] += cov_inv[i][j]*delta[j]
        chi_squared += delta[i].T*temp[i]

    return chi_squared

def Calculate_chi2_grid(beta_fid, beta_grid, cov, jmax, params):
    delta =np.zeros((len(params), len(beta_fid)))
    chi2 = np.zeros(len(params))

    for p in range(len(params)):
        delta[p] = np.subtract(beta_fid, beta_grid[p])
        chi2[p] = Calculate_chi2(delta[p], cov)
    return chi2

def Calculate_chi2_grid_cl(cl_fid, cl_grid, cov, lbins, params):
    delta =np.zeros((len(params), lbins))
    chi2 = np.zeros(len(params))

    for p in range(len(params)):
        delta[p] = np.subtract(cl_fid, cl_grid[p])
        chi2[p] = Calculate_chi2(delta[p], cov)
    return chi2

def delta_chi2(delta, cov):
    #print(delta)
    chi_squared = np.divide((delta)**2, cov)
    return np.sum(chi_squared)

def delta_chi2_grid(beta_fid, beta_grid,cov, jmax, params):
    delta =np.zeros((len(params), len(beta_fid)))
    chi2 = np.zeros(len(params))

    for i in range(len(params)):
        delta[i] = np.subtract(beta_fid, beta_grid[i])
        #print(cov[i].shape, delta[i].shape)
<<<<<<< HEAD
        chi2[i] = delta_chi2(delta[i], cov=cov[i])
=======
        chi2[i] = delta_chi2(delta[i], cov=cov)
>>>>>>> euclid_implementation

    return chi2

def Likelihood(chi_squared):
    likelihood = np.exp(-0.5*chi_squared)
    #print(chi_squared,likelihood)
    #normalizzare
    posterior = [float(i)/sum(likelihood) for i in likelihood]
    return np.asarray(posterior)

def Sample_posterior(chi_squared, params, Nsample = 10000):
    
    likelihood = Likelihood(chi_squared)
    print(np.sum(likelihood))
    var = []
    for s in range(Nsample):
        rand = np.random.choice(params, p=likelihood)
        var.append(rand)
    #var = [i/sum(var) for i in var]
    return np.array(var)

def Plot_posterior(posterior, param, chi_squared,filename):#, params):
    #posterior = Sample_posterior(likelihood, params )
    percentile = np.percentile(posterior, q = [16,50,84])
    #print(percentile)
    mean = np.mean(posterior)
    print(f'mean={mean}')
    index, = np.where(chi_squared == chi_squared.min())
    print(f'min chi squared={param[index[0]]}')

    fig = plt.figure(figsize=(17,10))
    ax = fig.add_subplot(1, 1, 1)
    
    ax.set_title(r'Probability distribution for $\Omega_{\Lambda}$ , grid = '+str(len(param))+' points')
    ax.set_xlabel(r'$\Omega_{\Lambda}$')
    #ax.set_xlabel(r'$A_{ISW}$')
    #textstr = '\n'.join((
    #    r'$A_{ISW}=%.2f$' % (param[index], ),
    #    r'$-=%.2f$' % (percentile[0], ),
    #    r'$+=%.2f$' % (percentile[2], )))
    # place a text box in upper left in axes coords

    textstr = '\n'.join((
        r'$\Omega_{\Lambda}=%.2f$' % (param[index], ),
        r'$-=%.2f$' % (percentile[0], ),
        r'$+=%.2f$' % (percentile[2], )))
     
    ax.text(0.3, 3, textstr, 
        verticalalignment='top')#, bbox=props)
    #sns.displot(data=posterior,kind= 'kde',bw_adjust=3)#,ax=ax)
    binwidth = (param[-1]-param[0])/(len(param)-1)
    binrange = [param[0]+binwidth/2, param[-1]+binwidth/2]
    sns.histplot(posterior, stat='density',binwidth=binwidth,binrange=binrange,element='step',fill=True, ax=ax)
    ax.set_xlim(binrange[0], binrange[1])
    ax.axvline(percentile[0],color='r')
    #ax.axvline(mean,color='r')
    ax.axvline(percentile[2],color='r')
    ax.axvline(param[index], color = 'r')
    ax.axvline(0.6847,color ='k',linestyle='-.', label = 'Planck 2018')
    
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('parameter_estimation/' +filename +'.png')

    return percentile






