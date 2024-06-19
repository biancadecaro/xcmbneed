import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

nside = 128

#mask_map_planck_comm_2048 = hp.read_map('COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits')
mask_map_euclid_fsky0p35_128 = hp.read_map('EUCLID/mask_rsd2022g-wide-footprint-year-6-equ-order-13-moc_ns0128_G_filled_2deg2.fits')
#mask_map_20 = hp.read_map('./HFI_Mask_GalPlane-apo0_2048_R2.00.fits', field= 0)
#
#mask_map_40 = hp.read_map('./HFI_Mask_GalPlane-apo0_2048_R2.00.fits', field= 1)
#
#mask_map_60 = hp.read_map('./HFI_Mask_GalPlane-apo0_2048_R2.00.fits', field= 2)
#
#mask_map_70 = hp.read_map('./HFI_Mask_GalPlane-apo0_2048_R2.00.fits', field= 3)
#
#mask_map_80 = hp.read_map('./HFI_Mask_GalPlane-apo0_2048_R2.00.fits', field= 4)
#
#mask_map_90 = hp.read_map('./HFI_Mask_GalPlane-apo0_2048_R2.00.fits', field= 5)
#
#mask_map_97 = hp.read_map('./HFI_Mask_GalPlane-apo0_2048_R2.00.fits', field= 6)
#
#mask_map_99 = hp.read_map('./HFI_Mask_GalPlane-apo0_2048_R2.00.fits', field= 7)
#
#mask_map_20_u = hp.ud_grade(mask_map_20, nside)
#mask_map_40_u = hp.ud_grade(mask_map_40, nside)
#mask_map_60_u = hp.ud_grade(mask_map_60, nside)
#mask_map_70_u = hp.ud_grade(mask_map_70, nside)
#mask_map_80_u = hp.ud_grade(mask_map_80, nside)
#mask_map_90_u = hp.ud_grade(mask_map_90, nside)
#mask_map_97_u = hp.ud_grade(mask_map_97, nside)
#mask_map_99_u = hp.ud_grade(mask_map_99, nside)

#mask_map_planck_comm_128_u = hp.ud_grade(mask_map_planck_comm_2048, nside)
#mask_map_planck_comm_256_u = hp.ud_grade(mask_map_planck_comm_2048, 256)

#hp.write_map(f'mask20_gal_nside={nside}.fits', mask_map_20_u )
#hp.write_map(f'mask40_gal_nside={nside}.fits', mask_map_40_u )
#hp.write_map(f'mask60_gal_nside={nside}.fits', mask_map_60_u )
#hp.write_map(f'mask70_gal_nside={nside}.fits', mask_map_70_u )
#hp.write_map(f'mask80_gal_nside={nside}.fits', mask_map_80_u )
#hp.write_map(f'mask90_gal_nside={nside}.fits', mask_map_90_u )
#hp.write_map(f'mask97_gal_nside={nside}.fits', mask_map_97_u )
#hp.write_map(f'mask99_gal_nside={nside}.fits', mask_map_99_u )

#mask_new = mask_map_planck_comm_128_u*mask_map_euclid_fsky0p35_128
#mask_new[np.where(mask_new>=0.5 )]=1
#mask_new[np.where(mask_new<0.5 )]=0

#hp.write_map(f'mask_planck_comm_2018_nside={nside}.fits', mask_map_planck_comm_128_u, overwrite=True )
#hp.write_map(f'mask_planck_comm_2018_nside={256}.fits', mask_map_planck_comm_256_u, overwrite=True )
#hp.write_map(f'EUCLID/mask_planck_comm_2018_x_euclid_fsky0p35_nside={nside}.fits', mask_new )


mask_pl_binary = hp.read_map(f'mask_temp_ns{nside}.fits')
mask_new_binary = mask_pl_binary*mask_map_euclid_fsky0p35_128
fsky = np.mean(mask_new_binary)
hp.mollview(mask_pl_binary, title ='Planck binary ', cmap = 'viridis')
hp.mollview(mask_map_euclid_fsky0p35_128, title='Euclid', cmap='viridis')
hp.mollview(mask_new_binary, title='EuclidxPlanck binary', cmap='viridis')
plt.show()

hp.write_map(f'EUCLID/mask_planck_comm_2018_x_euclid_binary_fsky{fsky}_nside={nside}.fits', mask_new_binary )
