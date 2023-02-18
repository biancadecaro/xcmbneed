import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

corr_100 = np.loadtxt('covariance/corr_100_jmax12_B1.7422102429630426.dat')
corr_200 = np.loadtxt('covariance/corr_200_jmax12_B1.7422102429630426.dat')
corr_300 = np.loadtxt('covariance/corr_300_jmax12_B1.7422102429630426.dat')
corr_400 = np.loadtxt('covariance/corr_400_jmax12_B1.7422102429630426.dat')

#print(corr_10,corr_10.shape)

fig, axs = plt.subplots(2,2,figsize=(25,18))   
#fig.suptitle('nsim = 10')

mask_ = np.tri(corr_100.shape[0],corr_100.shape[1],0)

#plt.subplot(131)
axs[0,0].set_title('nsim = 100')
sns.heatmap(corr_100, annot=True, fmt='.2f', mask=mask_, ax=axs[0,0])

axs[0,1].set_title('nsim = 200')
sns.heatmap(corr_200, annot=True, fmt='.2f', mask = mask_, ax = axs[0,1])

axs[1,0].set_title('nsim = 300')
sns.heatmap(corr_300, annot=True, fmt='.2f', mask=mask_, ax=axs[1,0])

axs[1,1].set_title('nsim = 400')
sns.heatmap(corr_400, annot=True, fmt='.2f', mask=mask_, ax=axs[1,1])

plt.suptitle(r'$B = %1.2f $' %1.7422102429630426 + r'$  N_{side} =$'+str(512) + r' $N_{sim} = $'+str(500))

plt.savefig('prova_corr_B = %1.2f ' %1.7422102429630426 +'.png')

#plt.show()
