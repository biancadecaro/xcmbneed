#!/usr/bin/env python
import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import healpy as hp
import argparse, os, sys, warnings, glob
import cython_mylibc as mylibc
import analysis, utils, spectra, sims
from IPython import embed
import seaborn as sns
sns.set()
sns.set(style = 'white')
sns.set_palette('husl', n_colors=10)

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

b_need_1p59 = np.loadtxt('/home/bdecaro/xcmbneed/src/output_needlet_TG/EUCLID/Tomography/TG_128_lmax256_nbins10_nsim1000_nuovo/b2_lmax256_jmax12_B1.59.dat')

print(b_need_1p59, b_need_1p59.shape[0]) #dovrebbe essere j righe e l colonne

fig, ax1  = plt.subplots(1,1,figsize=(17,10)) 

for i in range(1,b_need_1p59.shape[0]):
    ax1.plot(b_need_1p59[i], label = 'j='+str(i) )
ax1.set_xscale('log')

ax1.set_xlabel(r'$\ell$')#, fontsize = 25)
ax1.set_ylabel(r'$w^{2}(\frac{\ell}{D^{j}})$')#, fontsize = 25)
ax1.legend(loc='right')#, fontsize = 25)
ax1.set_title('D = 1.59')#, fontsize = 25)

plt.tight_layout()


plt.savefig('/home/bdecaro/xcmbneed/src/output_needlet_TG/EUCLID/Tomography/TG_128_lmax256_nbins10_nsim1000_nuovo/b_need_D1.59.png')