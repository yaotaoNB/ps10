#!/bin/bash

echo "Strong scaling tests"
make clean
make nbody.exe

for size in 256 1024 4096;
do
    
    /bin/rm -f strong${size}.out.txt
    touch strong${size}.out.txt
    
    printf "${size}\n" | tee -a strong${size}.out.txt
    printf "size\tprocs\ttime\n" | tee -a strong${size}.out.txt
    
    for nodes in 1 2 4 8 16 32 
    do
	echo $nodes nodes 
	job=`sbatch --nodes=${nodes} nbody.bash -t 1024 -n ${size} | awk '{ print $4 }'`
	echo job ${job}
	
	while [ `squeue -h -j${job} -r | wc -l` == "1" ]
	do
	    printf "."
	    sleep 2
	done
	printf "\n"

# elapsed time [run]: 3939 ms

	time=`fgrep elapsed slurm-${job}.out | awk '{print $5}'`
#	gflops=`fgrep gflops slurm-${job}.out | awk '{print $6}'`
#	ms_per=`fgrep msec_per slurm-${job}.out | awk '{print $5}'`
	printf "${size}\t${nodes}\t${time}\n" | tee -a strong${size}.out.txt
	
    done
done

python3 plot.py strong256.out.txt strong1024.out.txt strong4096.out.txt
mv time.pdf strong.pdf
