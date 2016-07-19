module io_tools

    implicit none

    contains

!===================================================================
! Cherche une chaine dans un fichier ouvert et se positionne
! a la ligne d apres
!===================================================================
subroutine find_string(unit, string)

    integer         , intent(in) :: unit
    character(len=*), intent(in) :: string

    ! local
    character(len=30) temp

    rewind(unit)

    do
      read(unit,'(a)') temp
      if ( trim(adjustl(temp)) == string ) exit
    end do

end subroutine find_string
!===================================================================
! Cherche une unité libre pour creer un fichier 
!===================================================================
integer function new_unit()

    ! local
    integer :: i = 10 ! on commence a l'unité 10
    logical :: not_available = .true.

    do while ( not_available )
      inquire( i , OPENED = not_available )
      i = i+1
    enddo

    new_unit = i

end function new_unit
!=====================================================================
! Tests a section name in input file
!=====================================================================
logical function test_msg(unit, msg)
  
    integer     , intent(in) :: unit
    character(*), intent(in) :: msg

    ! local
    character(len(msg)) :: msg_read

    read( unit , '(A)' , advance = 'no' ) msg_read
    test_msg = msg_read == msg

end function test_msg
!===================================================================
! ecrit un message d erreur sur la console et stop l execution
!===================================================================
subroutine stop(message)

    character(len=*), intent(in) :: message

    write(*,'(a)') message

    write(*,'(a)') '>>>>>>>>>>>>>>> STOP <<<<<<<<<<<<<<<'

    stop

end subroutine stop
!===================================================================
! Traduction entier vers chaine
!===================================================================
function int2char(i)

    integer, intent(in) :: i

    ! local
    character(len=20) :: int2char

    write(int2char,'(i20)') i

    int2char = adjustl( int2char )

    end function int2char

end module io_tools
 
