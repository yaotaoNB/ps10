#!/bin/bash

make clean 
make all

for a in aos soa; do
  exe=nbody_${a}_simple
  for num_bodies in 256 1024 4096; do
    for num_threads in 1 2 4 8; do
      # Execute program -- add problem size and number of trips before the first pipe
	echo -n "${exe} / ${num_threads} threads / ${bodies} bodies "
      OMP_NUM_THREADS=${num_threads} ./${exe}.exe -n ${num_bodies} 
    done
  done
done


