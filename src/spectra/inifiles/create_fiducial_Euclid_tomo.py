import numpy as np
import grid_spectra as spectra
import matplotlib.pyplot as plt

filename ='fiducial_IST_nbins3_lmax2050_lmin0'
pkl = spectra.Read_spectra(filename)

lmax=2050

ell =  np.arange(lmax+1)

header = 'L TxT TxW1 TxW2 TxW3 W1xW1 W1xW2 W1xW3 W2xW2 W2xW3 W3xW3'

filename_inifiles='fiducial_EUCLID_tomo_nbins3.dat'
print(len(pkl['nbins3']['cls_fid']))

data = np.zeros((len(pkl['nbins3']['cls_fid'])+1,ell.shape[0]), dtype=object)
#data[0,0]=header[0]
data[0,:] = ell
for k, key in enumerate(list(pkl['nbins3']['cls_fid'].keys())):
    data[k+1,:] = pkl['nbins3']['cls_fid'][key][ell]#data[k+1] = 

#output = open(filename_inifiles, "w")
#for k, v in pkl['nbins3']['cls_fid'].items():
#    output.write(f'{k} {v}\n')

np.savetxt(filename_inifiles,data.T,header=header, delimiter=' ')

plt.figure()
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[1])
plt.savefig('prova_TT_tomo.png')

plt.figure()
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[2], label='TxW1')
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[3], label='TxW2')
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[4], label='TxW3')
plt.legend()
plt.savefig('prova_TG_tomo.png')

plt.figure()
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[5], label='W1xW1')
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[6], label='W1xW2')
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[7], label='W1xW3')
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[8], label='W2xW2')
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[9], label='W2xW3')
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[10], label='W3xW3')
plt.legend()
plt.savefig('prova_GG_tomo.png')

data_read = np.genfromtxt(filename_inifiles, names=True)
nbins=3

clgg = np.zeros((nbins,nbins,ell.shape[0]))
for bin1 in range(nbins):
    for bin2 in range(nbins):
        ibin1 = min(bin1+1,bin2+1)
        ibin2 = max(bin1+1,bin2+1)
        print(bin1,bin2,ibin1,ibin2)
        clgg[bin1, bin2,:] = data_read[f'W{ibin1}xW{ibin2}']
plt.figure()
plt.plot(data[5]-clgg[0,0])
plt.plot(data[6]-clgg[0,1])
plt.plot(data[7]-clgg[0,2])
plt.plot(data[8]-clgg[1,1])
plt.plot(data[9]-clgg[1,2])
plt.plot(data[10]-clgg[2,2])
plt.plot(data[6]-clgg[1,0])
plt.plot(data[7]-clgg[2,0])
plt.plot(data[9]-clgg[2,1])
plt.savefig('prova_gg.png')
