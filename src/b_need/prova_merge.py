import numpy as np
import matplotlib.pyplot as plt

import cython_mylibc as pippo

jmax=12
lmax=256

jvec = np.arange(jmax+1)
B = pippo.mylibpy_jmax_lmax2B(jmax,lmax)

b_values = pippo.mylibpy_needlets_std_init_b_values(B, jmax, lmax)

fig, ax1  = plt.subplots(1,1) 
plt.suptitle(r'$D = %1.2f $' %B+r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax))

for i in range(b_values.shape[0]):
    ax1.plot(b_values[i]*b_values[i], label = 'j='+str(i) )
ax1.set_xscale('log')
#ax1.set_xlim(-1,10)
ax1.set_xlabel(r'$\ell$')
ax1.set_ylabel(r'$w^{2}(\frac{\ell}{D^{j}})$')
ax1.legend(loc='right')
plt.tight_layout()

b_values_new = np.copy(b_values)

mergej=[1,2]
jmin_merge = mergej[0]
jmax_merge = mergej[-1]
jvec_new = np.arange(len(jvec) -len(mergej)+1)
print(jvec, jvec_new, jmin_merge, jmax_merge)

temp = np.zeros(lmax+1)

for ell in range(b_values.shape[1]):
    for j in mergej:
        temp[ell] += b_values[j][ell]**2
    temp[ell] = np.sqrt(temp[ell])
b_values_new[jmin_merge] =  temp#np.sqrt(temp)


idx_left=np.where(jvec<=jmin_merge)
idx_right=np.where(jvec>jmax_merge)
b_values_left = b_values_new[idx_left] 
b_values_right = b_values_new[idx_right] 

b_values_new_2 = np.concatenate([b_values_left, b_values_right], axis=0)
print(b_values[jmin_merge], b_values[jmax_merge], b_values_new_2[1])

for ell in range(b_values_new_2.shape[1]):
    s = 0
    for jj in range(b_values_new_2.shape[0]):
        s += b_values_new_2[jj][ell]**2
    if s-1 > np.finfo(float).eps*100:
        print('WRONG WINDOW FUNCTIONS')


fig, ax1  = plt.subplots(1,1) 
plt.suptitle(r'MERGE $D = %1.2f $' %B+r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax))

for i in range(b_values_new.shape[0]):
    ax1.plot(b_values_new[i]*b_values_new[i], label = 'j='+str(i) )
ax1.set_xscale('log')
#ax1.set_xlim(-1,10)
ax1.set_xlabel(r'$\ell$')
ax1.set_ylabel(r'$w^{2}(\frac{\ell}{D^{j}})$')
ax1.legend(loc='right')
plt.tight_layout()

fig, ax1  = plt.subplots(1,1) 
plt.suptitle(r'MERGE FINAL $D = %1.2f $' %B+r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax))

for i in range(b_values_new_2.shape[0]):
    ax1.plot(b_values_new_2[i]*b_values_new_2[i], label = 'j='+str(i) )
ax1.set_xscale('log')
#ax1.set_xlim(-1,10)
ax1.set_xlabel(r'$\ell$')
ax1.set_ylabel(r'$w^{2}(\frac{\ell}{D^{j}})$')
ax1.legend(loc='right')
plt.tight_layout()




plt.show()