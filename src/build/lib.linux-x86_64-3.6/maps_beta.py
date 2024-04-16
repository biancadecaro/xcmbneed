import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

jmax = 12
nsim = 5
nside = 256
npix = 12*nside*nside

map = np.zeros((nsim, jmax+1, npix))

for n in range(nsim):
    for j in range(jmax):
        map[n, j, :] = hp.read_map('maps_beta/map_beta_'+str(j)+'_nsim_'+str(j))

        hp.mollview(map[n,j, :], title = 'map_beta_j'+str(j)+'_nsim'+str(n))
        plt.savefig('./maps_beta/map_beta_j'+str(j)+'_nsim'+str(n)+'.png')
        
