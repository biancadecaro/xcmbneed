import numpy as np
import healpy as h
from ctypes import cdll, POINTER, c_double, c_uint32, byref
from time import time

def diffusive_inpaint(mappa,mask,iterations):
  """Wrapper for inpainting module :D
  """

  mappa*=mask
  mappa=mappa.astype(np.float64,copy=False)
  mask=mask.astype(np.float64,copy=False)
  nside = h.get_nside(mappa)
  Npix = 12*nside**2
  mask_nzeropix=len(np.where(mask < 1.)[0])
  #print "mask_nzeropix = ", mask_nzeropix
  lib=cdll.LoadLibrary("/ehome/bdecaro/xcmbneed/src/inpainting/libpainting.so")
  start_time = time()
  lib.paint_(mappa.ctypes.data_as(POINTER(c_double)),mask.ctypes.data_as(POINTER(c_double)),byref(c_uint32(mask_nzeropix)),byref(c_uint32(Npix)),byref(c_uint32(nside)),byref(c_uint32(iterations)))
  end_time = time()
  time_taken = end_time - start_time # time_taken is in seconds

  print("Diffusive Inpainting Time = ", time_taken, " seconds, for",iterations,"iterations:")
  hours, rest = divmod(time_taken,3600)
  minutes, seconds = divmod(rest, 60)
  print("Diffusive Inpainting Time = %02d:%02d:%02d" % (hours,minutes, seconds))

  return mappa

def diffusive_inpaint2(mappa,mask,iterations):
  """Wrapper for inpainting module :D
  """

  mappa*=mask
  mappa=mappa.astype(np.float64,copy=False)
  mask=mask.astype(np.float64,copy=False)
  nside = h.get_nside(mappa)
  Npix = 12*nside**2
  mask_nzeropix=len(np.where(mask < 1.)[0])
  #print "mask_nzeropix = ", mask_nzeropix
  lib=cdll.LoadLibrary("/ehome/bdecaro/xcmbneed/src/inpainting/libpainting.so")
  #start_time = time()
  lib.paint2_(mappa.ctypes.data_as(POINTER(c_double)),mask.ctypes.data_as(POINTER(c_double)),byref(c_uint32(mask_nzeropix)),byref(c_uint32(Npix)),byref(c_uint32(nside)),byref(c_uint32(iterations)))
  #end_time = time()
  #time_taken = end_time - start_time # time_taken is in seconds

  #print("Diffusive Inpainting Time = ", time_taken, " seconds, for",iterations,"iterations:")
  #hours, rest = divmod(time_taken,3600)
  #minutes, seconds = divmod(rest, 60)
  #print("Diffusive Inpainting Time = %02d:%02d:%02d" % (hours,minutes, seconds))

  return mappa

#********** INPAINTING - VERO *****************

#def linear_decrease(n,lambda_vect,lambda_min,iter_max):
#  for j in np.arange(len(lambda_vect)):
#    lambda_vect[j] -= ((n*(lambda_vect[j]-lambda_min))/(iter_max-1))
#  return lambda_vect
#
#def linear_decrease_single(n,lambda_max,lambda_min,iter_max):
#  return (lambda_max-((n*(lambda_max-lambda_min))/(iter_max-1)))
#
#def inpaint_feeney(map_in,mask,lmax,nside,cl00):
#
#  c=(12.*nside**2)/4./np.pi
#
#  obs_pixels=np.where(mask > 0.5)
#  alm0=c*h.map2alm(map_in,lmax=lmax,use_weights=True,regression=False)
#
#  alm_n=alm0.copy()
#
#  #lambda_max=l1norm_joint(alm_n)
#  #lambda_max=np.amax(l1prod_joint(alm_n))
#  #lambda_max=np.amax(np.abs(alm_n))
#  lambda_max=1.e-18
#  #lambda_max=np.linalg.norm(alm_n,1)
#  #lambda_max=l2norm(alm_n)
#  print("lambda_max = ", lambda_max)
#
#  #cl00=h.alm2cl(alm0,lmax=lmax)
#  conv_n=10000.
#
#  itermax=100
#  for n in np.arange(itermax):
#    
#    #Joint inpainting
#    lambdan=lambda_max
#    #lambdan=linear_decrease_single(n,lambda_max,0.,itermax)
#    #lambdan=erf_decrease_single(n,lambda_max,0.,itermax)
#    #FIRST step:
#    appmap=h.alm2map(alm_n,lmax=lmax,nside=nside)
#    appmap[obs_pixels]=map_in[obs_pixels].copy()
#    alm_n12=h.map2alm(appmap,lmax=lmax,use_weights=True,regression=False)
#    #h.mollview(appmap,title="Iteration = "+str(n))
#    #plt.show()
#
#    #SECOND step:
#    appalm=(2.*alm_n12)-alm_n
#    abs_appalm=np.abs(appalm)
#    sign=appalm/abs_appalm
#
#    print("abs_appalm = ",abs_appalm)
#    maximum=np.maximum(0.,abs_appalm-lambdan)
#    print("Maximum(number of elements diffrent from zero) =",np.linalg.norm(maximum,0))
#    update=(sign*maximum) - alm_n12
#    alm_np1=(alm_n + update)
#
#    convergence1=np.linalg.norm(alm_np1-alm_n,2)
#    convergence2=np.linalg.norm(alm0,2)
#
#    alm_n=alm_np1.copy()
#    
#    #cl_inp0=h.alm2cl(alm_n,lmax=lmax)
#    #cl_inp0=np.sqrt(cl00[2:]/cl_inp0[2:])
#    #cl_inp0[0:2]=0.
#    #alm_n=h.almxfl(alm_n,cl_inp0)
#
#    #conv_np1 = convergence1/convergence2
#    #print "conv = ", np.abs(conv_np1), "iteration = ", n
#    #if (np.abs(conv_np1) < 0.001):
#    #  break
#
#    #conv_np1 = convergence1/convergence2
#    #print "conv = ", conv_np1-1., "iteration = ", n
#    #if (np.abs(conv_np1-1.) < 0.001) or (conv_np1 > conv_n):
#    #  break
#    #else:
#    #  conv_n=conv_np1
#
#  appmap=h.alm2map(alm_n/c,lmax=lmax,nside=nside)
#  appmap[obs_pixels]=map_in[obs_pixels]
#  return appmap
 


if __name__=='__main__':

  import sys
  from matplotlib.pylab import *

  mappa=h.read_map(sys.argv[1],dtype=np.float64)
  mask=h.read_map(sys.argv[2],dtype=np.float64)
  diffusive_inpaint2(mappa,mask,1000)
  h.write_map(sys.argv[3],mappa)
  h.mollview(mappa)
  show()



