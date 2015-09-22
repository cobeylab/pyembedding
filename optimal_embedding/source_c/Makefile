SHELL = /bin/sh

prefix = /Users/ebaskerv
exec_prefix = ${prefix}
BINDIR = ${exec_prefix}/bin

CC = cc
CFLAGS = -O3
AR = ar
ARFLAGS = r
INSTALL = /usr/bin/install -c

LOADLIBS = routines/libddtsa.a -lm

# list of executables we want to produce
   ALL =   costfunc

all: $(ALL) 

routines/libddtsa.a:
	(cd routines && $(MAKE))

$(ALL): routines/libddtsa.a *.c
	-$(CC) $(CFLAGS) $(COPTS) -o $@ $@.c $(LOADLIBS)

install: all
	-for bin in $(ALL); do $(INSTALL) $$bin $(BINDIR); done

clean:
	@rm -f *.o *~ #*#
	@rm -f $(ALL)
	-(cd routines && $(MAKE) clean)

missing:
	-@for bin in $(ALL); do \
	   test -z "`$$bin -h 2>&1 | grep Usage`" \
	      && echo   $$bin "(Uzal, L.C.)" >> ../missing.log; \
	   $$bin -h 2>&1 | cat >>  ../install.log; \
	  done; :

uninstall:
	-@for bin in $(ALL); do rm -f $(BINDIR)/$$bin; done
