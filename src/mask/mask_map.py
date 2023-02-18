import healpy as hp

nside = 256

mask_map_20 = hp.read_map('HFI_Mask_GalPlane-apo0_2048_R2.00.fits', field= 0)

mask_map_40 = hp.read_map('HFI_Mask_GalPlane-apo0_2048_R2.00.fits', field= 1)

mask_map_60 = hp.read_map('HFI_Mask_GalPlane-apo0_2048_R2.00.fits', field= 2)

mask_map_70 = hp.read_map('HFI_Mask_GalPlane-apo0_2048_R2.00.fits', field= 3)

mask_map_80 = hp.read_map('HFI_Mask_GalPlane-apo0_2048_R2.00.fits', field= 4)

mask_map_90 = hp.read_map('HFI_Mask_GalPlane-apo0_2048_R2.00.fits', field= 5)

mask_map_97 = hp.read_map('HFI_Mask_GalPlane-apo0_2048_R2.00.fits', field= 6)

mask_map_99 = hp.read_map('HFI_Mask_GalPlane-apo0_2048_R2.00.fits', field= 7)

mask_map_20_u = hp.ud_grade(mask_map_20, nside)
mask_map_40_u = hp.ud_grade(mask_map_40, nside)
mask_map_60_u = hp.ud_grade(mask_map_60, nside)
mask_map_70_u = hp.ud_grade(mask_map_70, nside)
mask_map_80_u = hp.ud_grade(mask_map_80, nside)
mask_map_90_u = hp.ud_grade(mask_map_90, nside)
mask_map_97_u = hp.ud_grade(mask_map_97, nside)
mask_map_99_u = hp.ud_grade(mask_map_99, nside)

hp.write_map(f'mask20_gal_nside={nside}.fits', mask_map_20_u )
hp.write_map(f'mask40_gal_nside={nside}.fits', mask_map_40_u )
hp.write_map(f'mask60_gal_nside={nside}.fits', mask_map_60_u )
hp.write_map(f'mask70_gal_nside={nside}.fits', mask_map_70_u )
hp.write_map(f'mask80_gal_nside={nside}.fits', mask_map_80_u )
hp.write_map(f'mask90_gal_nside={nside}.fits', mask_map_90_u )
hp.write_map(f'mask97_gal_nside={nside}.fits', mask_map_97_u )
hp.write_map(f'mask99_gal_nside={nside}.fits', mask_map_99_u )

