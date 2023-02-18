import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = '25'

b_need = np.loadtxt('/ehome/bdecaro/xcmbneed/src/b_values_B=1.50.txt')

print(b_need, b_need.shape[0]) #dovrebbe essere j righe e l colonne

fig, ax  = plt.subplots(1,1,figsize=(17,10)) 

for i in range(1,b_need.shape[0]):
    ax.plot(b_need[i], label = 'j='+str(i) )

#ax.plot(b_need[6], label = 'j='+str(6) )

ax.set_xlabel(r'$\ell$', fontsize = 25)
ax.set_ylabel(r'$b^{2}(\frac{\ell}{B^{j}})$', fontsize = 25)
ax.legend(loc='best', fontsize = 25)
ax.set_title('B = 1.50', fontsize = 25)
ax.set_xlim(0,130)
plt.tight_layout()
#plt.plot(b_need[6])

plt.savefig('b_need_B1.47.png')