! echo ""
! echo "****************************************************************"
! echo "*  Cost Function for Legendre coordinate reconstruction        *"
! echo "*  of Chua's circuit time series                               *"
! echo "****************************************************************"
! echo ""
! costfunc chua.dat -W1000 -e3 -N8000 -ochua_lc
# ! costfunc chua.dat -W1000 -e2 -N8000 -ochua_fdc
pause 1

set style data lines
set style line 2 lt 1 lc rgb "red" lw 4
set style line 10 lt 2 lw 10 lc rgb "grey" 

set logscale x
set ylabel "Cost Function"
set xlabel "t_w" offset 0.00,0.5
set tics front

set key left reverse
#set key bottom

plot [1:1000][-1.5:0]	"chua_fdc.amp" u 1:2 t "fdc" ls 10, \
					"chua_lc.amp" u 1:2 t "m = 2" ls 1, \
					"chua_lc.amp" u 1:3 t "m = 3" ls 2, \
					"chua_lc.amp" u 1:4 t "m = 4" ls 3, \
					"chua_lc.amp" u 1:5 t "m = 5" ls 4, \
					"chua_lc.amp" u 1:6 t "m = 6" ls 5, \
					"chua_lc.amp" u 1:3 not ls 2
pause -1 "Press <return> when finished"

