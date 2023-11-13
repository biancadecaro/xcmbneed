import numpy as np

#data = np.genfromtxt('inifiles/EUCLID_fiducial_1.dat', names = True)
#
#ell = data['L']
#tw1 = data['TxW1']
#w1w1 = data['W1xW1']
#tt = data['TxT']
#
#header = 'L ,TxT, TxW1 ,W1xW1'
#camb = np.array([ell,tt,tw1,w1w1])
#
#filename = 'inifiles/EUCLID_fiducial.dat'
#np.savetxt(filename,camb, header = header)

#null test
data = np.loadtxt('inifiles/CAMBSpectra_planck_fiducial_lmin0_2050.dat')

ell = data[0]
tw1 = data[2]
w1w1 = data[3]
tt = data[1]

print(data.shape)

tw1_null = np.zeros(len(ell))
camb_null = np.array([ell,tt,tw1_null,w1w1])
header = 'L ,TxT, TxW1 ,W1xW1'
filename_null = 'inifiles/CAMBNull_planck.dat'
np.savetxt(filename_null,camb_null, header = header)