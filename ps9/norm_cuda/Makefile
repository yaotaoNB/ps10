#
# This file is part of the course materials for AMATH483/583 at the University of Washington,
# Spring 2020
#
# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
# https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Author: Andrew Lumsdaine
#


CUTARGETS	:= norm_thrust.exe cu_norm_0.exe cu_norm_1.exe cu_norm_2.exe cu_norm_3.exe cu_norm_4.exe

CUTESTS		:= 

TARGETS		:= norm_parfor.exe norm_seq.exe
TESTS		:= 

include ../include/Make_cu.inc  # DRY -- logic moved to include file

norm_parfor.exe: norm_parfor.o $(LOBJ)
		  $(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

norm_seq.exe: norm_seq.o $(LOBJ)
		  $(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

norm_thrust.exe: norm_thrust.o $(LOBJ)
		  $(NVCC) $(NVCCFLAGS) $^ -o $@ $(LIBS)


# The following 4 (yes 4) lines must be in all the subdirectory makefiles
#-----------------------------------------------------------------
# DO NOT DELETE THIS LINE -- make depend uses it
# DEPENDENCIES MUST END AT END OF FILE
norm_parfor.o: norm_parfor.cpp norm_utils.hpp ../include/Timer.hpp \
 norms.hpp
norm_seq.o: norm_seq.cpp norms.hpp ../include/Timer.hpp norm_utils.hpp
norm_thrust.o: norm_thrust.cu ../include/Timer.hpp
cu_norm_0.o: cu_norm_0.cu ../include/Timer.hpp ../include/helper_cuda.h \
 ../include/helper_string.h
cu_norm_1.o: cu_norm_1.cu ../include/Timer.hpp ../include/helper_cuda.h \
 ../include/helper_string.h
cu_norm_2.o: cu_norm_2.cu ../include/Timer.hpp ../include/helper_cuda.h \
 ../include/helper_string.h
cu_norm_3.o: cu_norm_3.cu ../include/Timer.hpp ../include/helper_cuda.h \
 ../include/helper_string.h
cu_norm_4.o: cu_norm_4.cu ../include/Timer.hpp ../include/helper_cuda.h \
 ../include/helper_string.h
# DEPENDENCIES MUST END AT END OF FILE
# IF YOU PUT STUFF HERE IT WILL GO AWAY
# see make depend above
