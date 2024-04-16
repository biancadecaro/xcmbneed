import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

sns.set(style = 'white')
sns.set_palette('husl')

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)


filename_D2 = 'bneed_lmax256_jmax8_B2.00.dat'
b2_D2 = np.loadtxt(filename_D2)

print(b2_D2.shape)

fig, ax1  = plt.subplots(1,1,figsize=(7,5)) 

for i in range(1,b2_D2.shape[0]):
    ax1.plot(b2_D2[i], label = 'j='+str(i) )
ax1.set_xscale('log')
ax1.set_xlabel(r'$\ell$')
ax1.set_ylabel(r'$w^{2}(\frac{\ell}{D^{j}})$')
ax1.legend(loc='right')
ax1.set_title('D = 2.')
plt.tight_layout()
plt.show()

filename_D2_a = 'alessandro_b_values_B=2.00.txt'
b2_D2_a = np.loadtxt(filename_D2_a)

print(b2_D2_a.shape)

fig, ax1  = plt.subplots(1,1,figsize=(7,5)) 

for i in range(1,b2_D2.shape[0]):
    ax1.plot(b2_D2_a[i]*b2_D2_a[i], label = 'j='+str(i) )
ax1.set_xscale('log')
ax1.set_xlabel(r'$\ell$')
ax1.set_ylabel(r'$w^{2}(\frac{\ell}{D^{j}})$')
ax1.legend(loc='right')
ax1.set_title('Alessandro D = 2.')
plt.tight_layout()
plt.show()

fig, ax1  = plt.subplots(1,1,figsize=(7,5)) 

for i in range(1,b2_D2.shape[0]):
    ax1.plot(b2_D2[i]-b2_D2_a[i]*b2_D2_a[i], label = 'j='+str(i) )
#ax1.set_xscale('log')
ax1.set_xlabel(r'$\ell$')
ax1.set_ylabel(r'$w^{2}(\frac{\ell}{D^{j}})$')
ax1.legend(loc='right')
ax1.set_title('D = 2.')
plt.tight_layout()
plt.show()

print(b2_D2-b2_D2_a*b2_D2_a)

############################################

filename_D1p59 = 'bneed_lmax256_jmax12_B1.59.dat'
b2_D1p59 = np.loadtxt(filename_D1p59)

print(b2_D1p59.shape)

fig, ax1  = plt.subplots(1,1,figsize=(7,5)) 

for i in range(1,b2_D1p59.shape[0]):
    ax1.plot(b2_D1p59[i], label = 'j='+str(i) )
ax1.set_xscale('log')
ax1.set_xlabel(r'$\ell$')
ax1.set_ylabel(r'$w^{2}(\frac{\ell}{D^{j}})$')
ax1.legend(loc='right')
ax1.set_title('D = 1.59')
plt.tight_layout()
plt.show()

filename_D1p59_a = 'alessandro_b_values_B=1.59.txt'
b2_D1p59_a = np.loadtxt(filename_D1p59_a)

print(b2_D2_a.shape)

fig, ax1  = plt.subplots(1,1,figsize=(7,5)) 

for i in range(1,b2_D1p59.shape[0]):
    ax1.plot(b2_D1p59_a[i]*b2_D1p59_a[i], label = 'j='+str(i) )
ax1.set_xscale('log')
ax1.set_xlabel(r'$\ell$')
ax1.set_ylabel(r'$w^{2}(\frac{\ell}{D^{j}})$')
ax1.legend(loc='right')
ax1.set_title('Alessandro D = 1.59')
plt.tight_layout()
plt.show()

fig, ax1  = plt.subplots(1,1,figsize=(7,5)) 

for i in range(1,b2_D1p59.shape[0]):
    ax1.plot(b2_D1p59[i]-b2_D1p59_a[i]*b2_D1p59_a[i], label = 'j='+str(i) )
#ax1.set_xscale('log')
ax1.set_xlabel(r'$\ell$')
ax1.set_ylabel(r'$w^{2}(\frac{\ell}{D^{j}})$')
ax1.legend(loc='right')
ax1.set_title('D = 1.59')
plt.tight_layout()
plt.show()

print(b2_D1p59-b2_D1p59_a*b2_D1p59_a)