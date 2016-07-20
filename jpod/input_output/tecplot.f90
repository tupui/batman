module ascii

    use io_tools, only : new_unit, stop

    implicit none

    private
    integer :: io_unit, io_stat, i, j, k

    public :: open_file, close_file, skip_header, read_array, write_array

    contains

!===================================================================
! Open a file and return unit index
!===================================================================
subroutine open_file(file, form, action, position, unit)
    character(len=*), intent(in)  :: file, form, action, position
    integer         , intent(out) :: unit

    unit = new_unit()

    open ( &
      UNIT     = unit, &
      ACTION   = action, &
      FORM     = form, &
      FILE     = file, &
      POSITION = position, &
      IOSTAT   = io_stat)

    if (io_stat /= 0) call stop('problem in open_file')
end subroutine open_file
!===================================================================
! Close a file.
!===================================================================
subroutine close_file(unit)
    integer, intent(in) :: unit

    ! local variables
    logical opened

    inquire(unit, OPENED = opened)
    if (.not. opened) call stop('problem in close_file')

    close(unit)
end subroutine close_file
!===================================================================
! Go to the line after the header.
!===================================================================
subroutine skip_header(unit)
    integer, intent(in) :: unit

    ! local variables
    character(len=4) :: str

    do
        read(unit, *, iostat=io_stat) str
        if (io_stat /= 0) call stop('problem in skip_header')
        if (str == 'ZONE' .or. str == 'zone') exit
    enddo
end subroutine skip_header
!===================================================================
! Read an array from a file.
! Use tecplot data order.
!===================================================================
subroutine read_array(unit, ni, nj, nk, array)
    !f2py depend(ni, nj, nk) array
    integer, intent(in)  :: unit, ni, nj, nk
    real*8 , intent(out) :: array(ni, nj, nk)
    read(unit, *) (((array(i,j,k), i=1,ni), j=1,nj), k=1,nk)
end subroutine read_array
!===================================================================
! Write an array with format to a file.
! Use tecplot data order.
!===================================================================
subroutine write_array(unit, format, ni, nj, nk, array)
    !f2py integer intent(hide), depend(array) :: ni = shape(array, 0)
    !f2py integer intent(hide), depend(array) :: nj = shape(array, 1)
    !f2py integer intent(hide), depend(array) :: nk = shape(array, 2)
    character(len=*), intent(in) :: format
    integer         , intent(in) :: unit, ni, nj, nk
    real*8          , intent(in) :: array(ni, nj, nk)
    write(unit, format) (((array(i,j,k), i=1,ni), j=1,nj), k=1,nk)
end subroutine write_array

end module ascii
