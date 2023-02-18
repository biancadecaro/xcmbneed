!MODULE inpainting

!  USE healpix_types
!  USE pix_tools
  
!  IMPLICIT NONE

!CONTAINS

  SUBROUTINE paint(map,mask,mask_nzeropix,Npix,nsmax,iterations)
  
    USE healpix_types
    USE pix_tools

    IMPLICIT NONE

    !...Takes array of alm, including all maps in the MC sample. Gets pixel space maps
    !...then inpaint them and output back alms from inpainted MC maps
    
    integer(i4b), intent(in) :: mask_nzeropix, nsmax, Npix, iterations
    real(dp), dimension(0:Npix-1), intent(inout) :: map
    real(dp), dimension(0:Npix-1), intent(in) :: mask
   
    integer(i4b), dimension(0:mask_nzeropix-1) :: vec_positions
    real(dp) :: inv_nneigh

    integer(i4b) :: i,j,nneigh, iterate, inest
    integer(i4b), dimension(1:8) :: applist

    !print*, "*** Paint Routine ***"
    !print*, "Npix = ", Npix
    !print*, "nside = ", nsmax
    !print*, "mask_nzeropix = ", mask_nzeropix
    !...Getting positions of masked pixels 
    map = map*mask

    inv_nneigh = 1.0d0/8.0d0

    !print*, "*** Conversion start ***"
    CALL convert_ring2nest(nsmax,map)
    !print*, "*** Conversion end ***"

    !print*, "*** Vec_positions filling start ***"
    j=0
    do i=0,Npix-1
      if (mask(i) < 1.) then
        !if (j > mask_nzeropix) then
        !  print*, "problemi!!!! j = ",j
        !endif
        call ring2nest(nsmax, i, inest)
        vec_positions(j) = inest
        j = j + 1
        !print*, "j= ", j
      endif
    enddo
    !print*, "*** Vec_positions filling finish ***"
    
    !print*, "*** Iterate start ***"
    do iterate=1,iterations

      do i=0,mask_nzeropix-1

        CALL neighbours_nest(nsmax, vec_positions(i), applist, nneigh)
        map(vec_positions(i)) = SUM(map(applist(1:nneigh)))*inv_nneigh

      enddo

    enddo
    !print*, "*** Iterate end ***"

    !print*, "*** Conversion start ***"
    CALL convert_nest2ring(nsmax,map)
    !print*, "*** Conversion end ***"

    print*, "INPAINTING FINISHED"

    RETURN

  END SUBROUTINE paint

  SUBROUTINE paint2(map,mask,mask_nzeropix,Npix,nsmax,iterations)
  
    USE healpix_types
    USE pix_tools

    IMPLICIT NONE

    !...Takes array of alm, including all maps in the MC sample. Gets pixel space maps
    !...then inpaint them and output back alms from inpainted MC maps
    
    integer(i4b), intent(in) :: mask_nzeropix, nsmax, Npix, iterations
    real(dp), dimension(0:Npix-1), intent(inout) :: map
    real(dp), dimension(0:Npix-1), intent(in) :: mask
   
    integer(i4b), dimension(0:mask_nzeropix-1) :: vec_positions
    real(dp) :: inv_nneigh

    integer(i4b) :: i,j,nneigh, iterate, inest, app_position
    integer(i4b), dimension(:), allocatable :: applist

    !print*, "*** Paint Routine ***"
    !print*, "Npix = ", Npix
    !print*, "nside = ", nsmax
    !print*, "mask_nzeropix = ", mask_nzeropix
    !...Getting positions of masked pixels 
    map = map*mask

    inv_nneigh = 1.0d0/8.0d0

    !print*, "*** Conversion start ***"
    CALL convert_ring2nest(nsmax,map)
    !print*, "*** Conversion end ***"

    !print*, "*** Vec_positions filling start ***"
    j=0
    do i=0,Npix-1
      if (mask(i) < 1.0d0) then
        !if (j > mask_nzeropix) then
        !  print*, "problemi!!!! j = ",j
        !endif
        call ring2nest(nsmax, i, inest)
        vec_positions(j) = inest
        j = j + 1
        !print*, "j= ", j
      endif
    enddo
    !print*, "*** Vec_positions filling finish ***"

    !print*, "*** Iterate start ***"

    !$OMP PARALLEL &
    !$OMP DEFAULT(SHARED) &
    !$OMP PRIVATE(i,iterate,app_position,applist,nneigh)
    ALLOCATE(applist(1:8))
    !$OMP DO SCHEDULE(static,1)
    do iterate=1,iterations

      do i=0,mask_nzeropix-1

        app_position=vec_positions(i)
        CALL neighbours_nest(nsmax,app_position,applist, nneigh)
        map(app_position) = SUM(map(applist(1:nneigh)))*inv_nneigh

      enddo

    enddo
    !$OMP END DO
    DEALLOCATE(applist)
    !$OMP END PARALLEL

    !print*, "*** Iterate end ***"

    !print*, "*** Conversion start ***"
    CALL convert_nest2ring(nsmax,map)
    !print*, "*** Conversion end ***"

    print*, "INPAINTING FINISHED"

    RETURN

  END SUBROUTINE paint2

!END MODULE inpainting
