! echo ""
! echo "****************************************************************"
! echo "*  Cost Function for delay coordinate reconstruction           *"
! echo "*  of Mackey-Glass time series                                 *"
! echo "****************************************************************"
! echo ""
! costfunc mcgl17.dat -W1000 -omcgl17_dc
! costfunc mcgl17.dat -W1000 -e2 -omcgl17_fdc
pause 1

set style data lines
set style line 3 lt 1 lc rgb "blue" lw 4
set style line 10 lt 2 lw 10 lc rgb "grey" 

set logscale x
set ylabel "Cost Function"
set xlabel "t_w" offset 0.00,0.5
set tics front

set key left reverse
#set key bottom

plot [1:1000][-2:0]	"mcgl17_fdc.amp" u 1:2 t "fdc" ls 10, \
					"mcgl17_dc.amp" u 1:2 t "m = 2" ls 1, \
					"mcgl17_dc.amp" u ($1*2):3 t "m = 3" ls 2, \
					"mcgl17_dc.amp" u ($1*3):4 t "m = 4" ls 3, \
					"mcgl17_dc.amp" u ($1*4):5 t "m = 5" ls 4, \
					"mcgl17_dc.amp" u ($1*5):6 t "m = 6" ls 5, \
					"mcgl17_dc.amp" u ($1*3):4 not ls 3
pause -1 "Press <return> when finished"

