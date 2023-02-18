# Compiler:
F90C = gfortran

# Set up appropriate suffix list
.SUFFIXES:
.SUFFIXES: .o .f .for .f90 .c

# Compile-time options
# F90 options

##FFLAGS = -ladebug -check all -check nounderflow -warn all -C -assume dummy_aliases -automatic 
##FFLAGS  = -fast -O5 -arch host -tune host -inline speed -assume dummy_aliases -automatic 
##FFLAGS = -p -g -gen_feedback


##FFLAGS = --ap --chk --co --f95 -g --pca --sav --trace --warn --xref --info --wo
#FFLAGS = -O2 -I/planck/alexnino/Healpix_2.20a/include_serial -heap-arrays -limf -traceback -vec-report0
#FFLAGS = -O2 -fopenmp -shared -fPIC -I/scratch/alexnino/Healpix_3.30/include_arch
FFLAGS  = -O2 -fopenmp -shared -fPIC -I$(HOME)/Healpix_3.81/include_f90

# CC options:
#CFLAGS	= -g -Wall
CFLAGS	= -O2 -fomit-frame-pointer -fforce-addr -fforce-mem

# Link libraries and options
#LDFLAGS	= -L/scratch/alexnino/Healpix_3.30/lib_arch -L/scratch/alexnino/cfitsio/3370/arch/lib -lhealpix -lcfitsio
LDFLAGS = -L$(HOME)/Healpix_3.81/lib_f90 -L$(HOME)/cfitsio-4.1.0 -lhealpix -lcfitsio 

# Obj list:
MODS	= inpainting.f90
OBJ	=

#+
# Make some default rules
#-

# Double suffix rules:
#.f90.o .f.o .for.o :
#	$(F90C) $(FFLAGS) -c $<
####	$(F90C) $(FFLAGS) -c -o $@ $<
#
#.c.o	:
#	$(CC) $(CFLAGS) -c $<
#
## Single suffix rules:
#.f90 .f .for :
#	$(F90C) $(FFLAGS) -o $@ $<
#
#.o :
#	$(F90C) $(LDFLAGS) -o $@ $<

# Build all programs

all		:	libpainting.so

libpainting.so	:	${MODS}
		$(F90C) $(FFLAGS) -g -o $@ ${MODS} ${LDFLAGS}

# Remove all files created from building 

clean	:
	-$(RM) libpainting.so
	-$(RM) *.mod
	-$(RM) *.o
	-$(RM) *.lst

tidy:	clean

#+
# Dependencies
#-

