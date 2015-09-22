! echo ""
! echo "**************************************************************"
! echo "*  Local Cost Function for delay coordinate reconstruction   *"
! echo "*  of Mackey-Glass time series                               *"
! echo "**************************************************************"
! echo ""
pause 1

set style data lines
set palette defined ( 0 "black", 2 "blue", 4 "red", 6 "orange", 8 "yellow", 10 "white" )
set hidden3d nooffset
#
rgb(r,g,b) = int(r)*65536 + int(g)*256 + int(b)
norma(x)=log(sqrt(x))/log(10)
#
unset xtics
unset ytics
set cbrange [-2.25:0]

set xlabel "x(t)"
set ylabel "x(t+tw)"
unset key
! costfunc mcgl17.dat -L -w30 -m2 -omcgl17.del
set label 10 "m=2" at screen 0.1,0.9
plot [-0.05:1.05][-0.05:1.05] 'mcgl17.del.loc' u 4:3:(norma($1)) lw .5 lc palette
pause -1 "Press <return> when finished"

! costfunc mcgl17.dat -L -w30 -m3 -omcgl17.del
set label 10 "m=3"
plot [-0.05:1.05][-0.05:1.05] 'mcgl17.del.loc' u 5:3:(norma($1)) lw .5 lc palette
pause -1 "Press <return> when finished"

! costfunc mcgl17.dat -L -w30 -m4 -omcgl17.del
set label 10 "m=4"
plot [-0.05:1.05][-0.05:1.05] 'mcgl17.del.loc' u 6:3:(norma($1)) lw .5 lc palette
pause -1 "Press <return> when finished"

! rm mcgl17.del.loc
