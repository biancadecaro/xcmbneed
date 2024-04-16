import numpy as np
import spectra
import cython_mylibc as mylibc 
import matplotlib.pyplot as plt


jmax=12
lmax=256
B    = mylibc.mylibpy_jmax_lmax2B(jmax, lmax)
need_theory = spectra.NeedletTheory(B)
j_binned = np.arange(start=1,stop=jmax)


#vediamo quanto variano in un bin 

cl_theory = np.loadtxt('/ehome/bdecaro/xcmbneed/src/spectra/inifiles/EUCLID_fiducial_lmin0.dat')
ell_theory = cl_theory[0]
cl_theory_tt = cl_theory[1]
cl_theory_tg = cl_theory[2]
cl_theory_gg = cl_theory[3]

#needlet_machine_limit_lower = 1. + 8.e-15
for _j, j in enumerate(j_binned):
    #print(_j, j)
    lminj = np.floor(B**(j-1))
    lmaxj = np.floor(B**(j+1))
    ellj  = np.arange(lminj, lmaxj+1, dtype=np.int)
    b2   = need_theory.b_need(ellj/B**j)**2
    #b2[0] = b2[0] + 8.e-15
    #print(lminj, b2[int(lminj)])
    index, = np.where(b2==0)[0]
    print(f'j={j}')
    print(f'lminj={lminj}, lmaxj={lmaxj}, index={index}, b2[index]={b2[index]}')
    print()


b_need_ale= mylibc.mylibpy_needlets_std_init_b_values(B, jmax, lmax)
for _j, j in enumerate(j_binned):
#print(_j, j)
    lminj = np.ceil(B**(j-1))
    lmaxj = np.floor(B**(j+1))
    ellj  = np.arange(lminj, lmaxj+1, dtype=np.int)
    b2=b_need_ale[j,ellj]**2
    print(f'j={j}')
    print(f'b2={b2}')
    print()

b_jl = (2.*lmax+1)*b_need_ale**2/(4*np.pi)*cl_theory_tg[:lmax+1]

fig = plt.figure(figsize=(27,20))
ax = fig.add_subplot(1, 1, 1)
for j in range(jmax+1):
    ax.plot(b_jl[j,:], label=f'j={j}')
ax.set_xlim(0, 100)
plt.legend()
plt.savefig('beta_need_ell.png')


