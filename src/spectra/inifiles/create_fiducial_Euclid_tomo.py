import numpy as np
import grid_spectra as spectra
import matplotlib.pyplot as plt

filename ='fiducial_IST_nbins3_lmax2050_lmin0'
pkl = spectra.Read_spectra(filename)

lmax=2050

ell =  np.arange(lmax+1)

header = 'L, TxT, TxW1, TxW2, TxW3, W1xW1, W1xW2, W1xW3, W2xW2, W2xW3, W3xW3'

filename_inifiles='fiducial_EUCLID_tomo_nbins3.dat'
print(len(pkl['nbins3']['cls_fid']))

data = np.zeros((len(pkl['nbins3']['cls_fid'])+1,ell.shape[0]))
data[0] = ell
for k, key in enumerate(list(pkl['nbins3']['cls_fid'].keys())):
    data[k+1] = pkl['nbins3']['cls_fid'][key][ell]

print(ell.shape, data.shape)

np.savetxt(filename_inifiles,data, header=header)

plt.figure()
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[1])
plt.savefig('prova_TT_tomo.png')

plt.figure()
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[2], label='TxW1')
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[3], label='TxW2')
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[4], label='TxW3')
plt.legend()
plt.savefig('prova_TG_tomo')

plt.figure()
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[5], label='W1xW1')
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[8], label='W2xW2')
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[10], label='W3xW3')
plt.legend()
plt.savefig('prova_GG_tomo')