import numpy as np
import matplotlib.pyplot as plt
import spectra, utils
import healpy as hp
import cython_mylibc as mylibc
from IPython import embed
from matplotlib import rc, rcParams
import seaborn as sns
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
import master_needlets

# Matplotlib defaults ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#rc('text',usetex=True)
#rc('font',**{'family':'serif','serif':['Computer Modern']})
plt.rcParams['axes.linewidth']  = 5.
plt.rcParams['axes.labelsize']  =30
plt.rcParams['xtick.labelsize'] =30
plt.rcParams['ytick.labelsize'] =30
plt.rcParams['xtick.major.size'] = 30
plt.rcParams['ytick.major.size'] = 30
plt.rcParams['xtick.minor.size'] = 30
plt.rcParams['ytick.minor.size'] = 30
plt.rcParams['legend.fontsize']  = 'large'
plt.rcParams['legend.frameon']  = False
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['axes.titlesize'] = 'large'
plt.rcParams["errorbar.capsize"] = 15
#
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['font.size'] = 40
plt.rcParams['lines.linewidth']  = 5.
#plt.rcParams['backend'] = 'WX'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# embed()
nside=128
jmax = 12
lmax = 256
nsim = 100
B    = mylibc.mylibpy_jmax_lmax2B(jmax, lmax)
jvec = np.arange(jmax+1)
lmin = 1
print(f'B={B:0.2f}')

mask_EP = hp.read_map(f'/ehome/bdecaro/xcmbneed/src/mask/EUCLID/mask_planck_comm_2018_x_euclid_fsky0p35_nside={nside}.fits')
fsky_EP = np.mean(mask_EP)
wl_EP = hp.anafast(mask_EP, lmax=lmax)

cl_theory = np.loadtxt('/ehome/bdecaro/xcmbneed/src/spectra/inifiles/EUCLID_fiducial_lmin0.dat')
ell_theory = cl_theory[0]
cl_theory_tt = cl_theory[1]
cl_theory_tg = cl_theory[2]
cl_theory_gg = cl_theory[3]

fname_xcspectra = '/ehome/bdecaro/xcmbneed/src/spectra/inifiles/EUCLID_fiducial_lmin0.dat'
xcspectra = spectra.XCSpectraFile(fname_xcspectra,   WantTG = True)

# Theory Needlet spectra
need_theory = spectra.NeedletTheory(B)
betatg    = need_theory.cl2betaj(jmax=jmax, cl=cl_theory_tg)
M_ll = need_theory.get_Mll(wl_EP, lmax=lmax)

P_jl = np.zeros((jmax, lmax+1 ))
Q_lj = np.zeros((lmax+1 , jmax))

j_binned = np.arange(start=1,stop=jmax+1) # ! Starting from 1

for _j, j in enumerate(j_binned):
    lminj = np.ceil(B**(j-1))
    lmaxj = np.floor(B**(j+1))
    if j ==  jmax:
        ellj  = np.arange(lminj,  lmax+1, dtype=np.int)
    else:
        ellj  = np.arange(lminj, lmaxj+1, dtype=np.int)
    print(f'j={j}. lminj={ellj[0]}, lmaxj={ellj[-1]}, ellj.shape={ellj.shape[0]}')
    b = need_theory.b_need(ellj/ B**j)
    b2   = b**2
    P_jl[_j,  ellj] =b2*(2. * ellj+1)/(4.*np.pi)#b2*(2. * ellj+1) / (4.*np.pi *(ellj[0]-ellj[-1]))
    Q_lj[ellj, _j] = 1.#4.*np.pi / ((2. * ellj+1))
    #for _l,l in enumerate(ellj):
    #    P_jl[_j,  _l] =b2[_l]*(2. * l+1)/(4.*np.pi)#b2*(2. * ellj+1) / (4.*np.pi *(ellj[0]-ellj[-1])) #b2*(1. * flat[ellj] / (lminj - lmaxj))
    #    Q_lj[_l, _j] = 4.*np.pi / (b2[_l]*(2. * l+1))    #b2*(1. / flat[ellj]



M_jj      = np.dot(np.dot(P_jl, M_ll), Q_lj)
M_jj_inv  = np.linalg.inv(M_jj)

pseudo_cl = np.dot(M_ll, cl_theory_gg[:lmax+1])
print(pseudo_cl.shape)
beta_recover = np.dot(M_jj_inv, np.dot(P_jl, pseudo_cl))

print(f'beta_recover={beta_recover}\nbetatg={betatg[1:]}')


lmin=0
delta_ell= 10

nbins = (lmax - lmin + 1) // delta_ell
start = lmin + np.arange(nbins) * delta_ell
stop  = start + delta_ell
ell_binned = (start + stop - 1) // 2

print(f'start={start}, stop={stop}, ell_binned={ell_binned}')

flat = np.arange(lmax + 1)
flat = flat * (flat + 1) / (2 * np.pi)
P_bl = np.zeros((nbins, lmax + 1))
Q_lb = np.zeros((lmax + 1, nbins))
for b, (a, z) in enumerate(zip(start, stop)):
    P_bl[b, a:z] = 1. * flat[a:z] / (z - a)
    Q_lb[a:z, b] = 1. / flat[a:z]

M_bb      = np.dot(np.dot(P_bl, M_ll), Q_lb)
M_bb_inv  = np.linalg.inv(M_bb)

cl_recover = np.dot(np.dot(M_bb_inv,P_bl), pseudo_cl)
cl_theory_binned = np.dot(P_bl, cl_theory_tg[:lmax+1])

print(f'cl_recover={cl_recover}\ncl_theory_binned={cl_theory_binned}')
print(cl_recover/cl_theory_binned-1)