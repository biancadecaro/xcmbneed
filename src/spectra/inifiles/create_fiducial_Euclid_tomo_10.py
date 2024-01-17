import numpy as np
import grid_spectra as spectra
import matplotlib.pyplot as plt

nbins=10

filename =f'fiducial_IST_nbins{nbins}_lmax2050_lmin0'
pkl = spectra.Read_spectra(filename)

lmax=2050

ell =  np.arange(lmax+1)

header = 'L TxT TxW1 TxW2 TxW3 TxW4 TxW5 TxW6 TxW7 TxW8 TxW9 TxW10 W1xW1 W1xW2 W1xW3 W1xW4 W1xW5 W1xW6 W1xW7 W1xW8 W1xW9 W1xW10 W2xW2 W2xW3 W2xW4 W2xW5 W2xW6 W2xW7 W2xW8 W2xW9 W2xW10 W3xW3 W3xW4 W3xW5 W3xW6 W3xW7 W3xW8 W3xW9 W3xW10 W4xW4 W4xW5 W4xW6 W4xW7 W4xW8 W4xW9 W4xW10 W5xW5 W5xW6 W5xW7 W5xW8 W5xW9 W5xW10 W6xW6 W6xW7 W6xW8 W6xW9 W6xW10 W7xW7 W7xW8 W7xW9 W7xW10 W8xW8 W8xW9 W8xW10 W9xW9 W9xW10 W10xW10'
print(header)

filename_inifiles=f'fiducial_EUCLID_tomo_nbins{nbins}.dat'
print(len(pkl[f'nbins{nbins}']['cls_fid']))

data = np.zeros((len(pkl[f'nbins{nbins}']['cls_fid'])+1,ell.shape[0]), dtype=object)
#data[0,0]=header[0]
data[0,:] = ell
for k, key in enumerate(list(pkl[f'nbins{nbins}']['cls_fid'].keys())):
    data[k+1,:] = pkl[f'nbins{nbins}']['cls_fid'][key][ell]#data[k+1] = 

#output = open(filename_inifiles, "w")
#for k, v in pkl['nbins3']['cls_fid'].items():
#    output.write(f'{k} {v}\n')

np.savetxt(filename_inifiles,data.T,header=header, delimiter=' ')

plt.figure()
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[1])
plt.savefig('prova_TT_tomo_10.png')

plt.figure()
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[2], label='TxW1')
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[3], label='TxW2')
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[4], label='TxW3')
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[5], label='TxW4')
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[6], label='TxW5')
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[7], label='TxW6')
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[8], label='TxW7')
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[9], label='TxW8')
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[10], label='TxW9')
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[11], label='TxW10')
plt.legend()
plt.savefig('prova_TG_tomo_10.png')

plt.figure()
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[5], label='W1xW1')
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[6], label='W1xW2')
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[7], label='W1xW3')
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[8], label='W2xW2')
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[9], label='W2xW3')
plt.plot(ell, ell*(ell+1)/(2*np.pi)*data[10], label='W3xW3')
plt.legend()
plt.savefig('prova_GG_tomo_10.png')

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
plt.savefig('prova_gg_10.png')
