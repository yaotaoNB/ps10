#
# This file is part of the course materials for CSE P 524 at the University of Washington,
# Autumn 2018
#
# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
# https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Author: Andrew Lumsdaine
# 

DEFS		= # -D__USE_CIMG
XINCLUDES	= # -I/opt/X11/include
XLIBS		= # -L/opt/X11/lib -lX11 
OPTS		+= 

LIBS		+= 

CXXFLAGS += 
#	    -fsanitize=address -fno-omit-frame-pointer 
TESTS	 += nbody.exe

include ../include/Make_mpi.inc

CXX = mpic++


# The following 4 (yes 4) lines must be in all the subdirectory makefiles
#-----------------------------------------------------------------
# DO NOT DELETE THIS LINE -- make depend uses it
# DEPENDENCIES MUST END AT END OF FILE
nbody.o: nbody.cpp ../include/Timer.hpp coordinate.hpp \
 /sw/contrib/niac-src/ompi/4.1.1/include/mpi.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/mpi_portable_platform.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/mpicxx.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/constants.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/functions.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/datatype.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/exception.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/op.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/status.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/request.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/group.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/comm.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/win.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/file.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/errhandler.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/intracomm.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/topology.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/intercomm.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/info.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/datatype_inln.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/functions_inln.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/request_inln.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/comm_inln.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/intracomm_inln.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/topology_inln.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/intercomm_inln.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/group_inln.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/op_inln.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/errhandler_inln.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/status_inln.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/info_inln.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/win_inln.h \
 /sw/contrib/niac-src/ompi/4.1.1/include/openmpi/ompi/mpi/cxx/file_inln.h
# DEPENDENCIES MUST END AT END OF FILE
# IF YOU PUT STUFF HERE IT WILL GO AWAY
# see make depend above
