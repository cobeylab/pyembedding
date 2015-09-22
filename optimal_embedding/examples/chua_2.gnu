! echo ""
! echo "*****************************************************************"
! echo "*  Local Cost Function for Legendre coordinate reconstruction   *"
! echo "*  of Chua's circuit time series                                *"
! echo "*****************************************************************"
! echo ""
pause 1

set style data lines
set palette defined ( 0 "black", 2 "blue", 4 "red", 6 "orange", 8 "yellow", 10 "white" )
set hidden3d nooffset
#
rgb(r,g,b) = int(r)*65536 + int(g)*256 + int(b)
norma(x)=log(sqrt(x))/log(10)
#
unset tics
set cbrange [-2.25:0]
set xlabel "y0"
set ylabel "y1"
set zlabel "y2"

unset key
! costfunc chua.dat -L -e3 -w81 -m3 -ochua.leg
set label 10 "m=3" at screen 0.1,0.9
splot 'chua.leg.loc' u 3:4:5:(norma($1)) lw .5 lc palette
pause -1 "Press <return> when finished"


! rm chua.leg.loc
