import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
plt.rcParams['axes.linewidth']  = 5.
plt.rcParams['axes.labelsize']  =30
plt.rcParams['xtick.labelsize'] =30
plt.rcParams['ytick.labelsize'] =30
plt.rcParams['xtick.major.size'] = 30
plt.rcParams['ytick.major.size'] = 30
plt.rcParams['xtick.minor.size'] = 30
plt.rcParams['ytick.minor.size'] = 30
plt.rcParams['legend.fontsize']  = 32#'medium'
#plt.rcParams['legend.frameon']  = False
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['axes.titlesize'] = 'large'
rcParams["errorbar.capsize"] = 15
#
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['font.size'] = 40
plt.rcParams['lines.linewidth']  = 5.


b_need_1p5 = np.loadtxt('/ehome/bdecaro/xcmbneed/src/b_values_B=1.50.txt')

print(b_need_1p5, b_need_1p5.shape[0]) #dovrebbe essere j righe e l colonne

fig, ax  = plt.subplots(1,1,figsize=(22,15)) 

for i in range(1,b_need_1p5.shape[0]):
    ax.plot(b_need_1p5[i], label = 'j='+str(i) )

#ax.plot(b_need_1p5[6], label = 'j='+str(6) )

ax.set_xlabel(r'$\ell$')#, fontsize = 25)
ax.set_ylabel(r'$b^{2}(\frac{\ell}{B^{j}})$')#, fontsize = 25)
ax.legend(loc='best')#, fontsize = 25)
ax.set_title('B = 1.50')#, fontsize = 25)
ax.set_xlim(0,130)
plt.tight_layout()
#plt.plot(b_need[6])

plt.savefig('b_need_B1.47_nuovo.png')
#plt.savefig('b_need_B1.47_j6_nuovo.png')

b_need_1p7 = np.loadtxt('/ehome/bdecaro/xcmbneed/src/b_values_B=1.70.txt')

print(b_need_1p7, b_need_1p7.shape[0]) #dovrebbe essere j righe e l colonne

fig, ax1  = plt.subplots(1,1,figsize=(22,15)) 

for i in range(1,b_need_1p7.shape[0]):
    ax1.plot(b_need_1p7[i], label = 'j='+str(i) )

#ax1.plot(b_need_1p7[6], label = 'j='+str(6) )

ax1.set_xlabel(r'$\ell$')#, fontsize = 25)
ax1.set_ylabel(r'$b^{2}(\frac{\ell}{B^{j}})$')#, fontsize = 25)
ax1.legend(loc='right')#, fontsize = 25)
ax1.set_title('B = 1.70')#, fontsize = 25)
ax1.set_xlim(0,300)
plt.tight_layout()
#plt.plot(b_need[6])

plt.savefig('b_need_B1.70_nuovo.png')
#plt.savefig('b_need_B1.70_j6_nuovo.png')
