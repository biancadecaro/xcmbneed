import numpy as np
import matplotlib.pyplot as plt

cl = np.loadtxt('cl_camb.dat')

ell = cl[0]
tt = cl[1]
tg = cl[2]
gg = cl[3]

fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.plot(ell[:500], ell[:500]*(ell[:500]+1)*tg[:500])
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$C_{\ell}^{TG}$')

plt.savefig('cl_tg.png')


cl_camb = np.loadtxt('CAMBPietrobon.dat')

ell_camb = cl_camb[0]
tt_camb = cl_camb[1]
tg_camb = cl_camb[2]
gg_camb = cl_camb[3]

fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(1, 1, 1)

ax.plot(ell_camb[:500], tg_camb[:500])
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\ell(\ell+1)C_{\ell}^{TG}$')

plt.savefig('cl_tg_pietrobon.png')

