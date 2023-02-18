import numpy as np

data = np.genfromtxt('./CAMB_01_planck_scalarCovCls.dat', names = True)

ell = data['L']
tw1 = data['TxW1']
w1w1 = data['W1xW1']
tt = data['TxT']

header = 'L ,TxT, TxW1 ,W1xW1'
camb = np.array([ell,tt,tw1,w1w1])

filename = 'CAMBSpectra_planck.dat'
np.savetxt(filename,camb, header = header)

#null test
tw1_null = np.zeros(len(ell))
camb_null = np.array([ell,tt,tw1_null,w1w1])

filename_null = 'CAMBNull.dat'
np.savetxt(filename_null,camb_null, header = header)