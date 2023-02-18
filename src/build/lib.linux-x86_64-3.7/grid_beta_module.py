import numpy as np
import matplotlib.pyplot as plt
import analysis, utils, spectra, sims

def Initialize_dir(params, nside):
    fname_xcspectra = []
    sim_dir         = []
    out_dir         = []
    cov_dir         = []

    for par in params:
        fname_xcspectra.append('spectra/Grid_spectra_'+str(len(params))+'/CAMBSpectra_OmL'+str(par)+'.dat')#.replace('.', '') np.chararray(len(OmL))
        sim_dir.append('sims/Needlet/Grid_spectra_'+str(len(params))+'/TGsims_'+str(nside)+'_OmL'+str(par)+'/' )#np.chararray(len(OmL))
        out_dir.append('output_needlet_TG_OmL/Grid_spectra_'+str(len(params))+'/TGsims_'+str(nside)+'_OmL'+str(par)+'/' )#np.chararray(len(OmL))
        cov_dir.append('covariance_TG_OmL/Grid_spectra_'+str(len(params))+'/TGsims_'+str(nside)+'_OmL'+str(par)+'/' )#np.chararray(len(OmL))
    
    return {'fname_xcspectra':fname_xcspectra, 'sim_dir':sim_dir, 'out_dir':out_dir, 'cov_dir':cov_dir}


def Analysis_sims_grid(params, dir, simparams,  lmax, jmax):
    xcspectra       = [spectra.XCSpectraFile(clfname= str(dir['fname_xcspectra'][xc]), WantTG = True)  for xc in range(len(params))]
    simulations     = [sims.KGsimulations(xcspectra[xc], dir['sim_dir'][xc], simparams, WantTG = True) for xc in range(len(params))]
    myanalysis      = []

    for xc, om in enumerate(params):
        print(xc,om)
        simulations[xc].Run(nsim=1, WantTG = True)

        # Needlet Analysis
        myanalysis.append(analysis.NeedAnalysis(jmax, lmax, dir['out_dir'][xc], simulations[xc]))

    return myanalysis

def Compute_beta_grid(params, jmax, myanalysis, nside):

   
    fname_betaj_sims_TS_galS = []
    betaj_sims_TS_galS = np.empty((len(params), jmax+1))
    #betatg = np.zeros((len(params), jmax+1))

    for xc, par in enumerate(params):    

        fname_betaj_sims_TS_galS.append('betaj_sims_TS_galS_Om'+str(par)+'_jmax'+str(jmax)+'_B = %1.2f ' %myanalysis[xc].B+'_nside'+str(nside)+'.dat')
    
    for xc, om in enumerate(params):
    # Computing simulated Cls 
        print("...computing Betajs for simulations...")

        betaj_sims_TS_galS[xc] = myanalysis[xc].GetBetajSimsFromMaps('TS', nsim=1, field2='galS', fname=fname_betaj_sims_TS_galS[xc])


    return betaj_sims_TS_galS

def Make_plot(betaj_sims_TS_galS, params, jmax, myanalysis, dir, nside):
    print(dir['fname_xcspectra'])
    need_theory     = []
    delta = np.zeros((len(params), jmax+1))
    xcspectra       = [spectra.XCSpectraFile(clfname= str(dir['fname_xcspectra'][xc]), WantTG = True)  for xc in range(len(params))]

    for xc, par in enumerate(params): 
        # Theory Needlet spectra
        need_theory.append(spectra.NeedletTheory(myanalysis[xc].B))

    for xc, om in enumerate(params):
        #Error
        delta[xc] = need_theory[xc].delta_beta_j(jmax, cltt = xcspectra[xc].cltt, cltg = xcspectra[xc].cltg, clgg = xcspectra[xc].clg1g1)

        fig = plt.figure(figsize=(17,10))
        ax = fig.add_subplot(1, 1, 1)
        #ax.plot(myanalysis[xc].jvec, betaj_sims_TS_galS[xc,:], 'o')
        ax.errorbar(myanalysis[xc].jvec[1:jmax+1]-0.15, betaj_sims_TS_galS[xc,1:jmax+1], yerr=delta[xc,1:jmax+1],fmt='o')
        ax.set_ylabel(r'$\beta_j$')

        ax.set_xlabel(r'j')
        plt.savefig(dir['out_dir'][xc]+'betaj_Om'+str(om)+'_jmax'+str(jmax)+'_B = %1.2f ' %myanalysis[xc].B+'_nside'+str(nside)+'.png')
    
    return