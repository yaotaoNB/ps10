//
// This file is part of the course materials for CSE P 524 at the University of Washington
// Winter 2022
//
// Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
// https://creativecommons.org/licenses/by-nc-sa/4.0/
//
// Author: Andrew Lumsdaine
//

#include <vector>
#include <numeric>
#include <cmath>
#include <functional>
#include <iostream>
#include <mpi.h>
#include <random>
#include <string>
#include <vector>
#include "Timer.hpp"
#include "norms.hpp"

#include <algorithm>
#include <stdlib.h>
#include <time.h>


// this function instead will sum up the square of each element in the current process's local_x
double mpi_norm(const std::vector<double>& local_x) {

  double sum = 0.0;

  for(auto ptr = local_x.begin(); ptr != local_x.end(); ++ptr){
    sum += (*ptr) * (*ptr);
  }

  return sum;
}


template <class F>
size_t find_10ms_size(F&& f) {
  Timer t;
  size_t sz = 128;

  for (; sz < 1024 * 1024 * 1024; sz *= 1.414) {
    std::vector<double> x(sz);
    t.start();
    f(x);
    t.stop();
    if (t.elapsed() >= 10.0) {
      break;
    }
  }

  return sz;
}


size_t num_trials(size_t base_size, size_t nnz) {
  double N_1k = std::ceil((static_cast<double>(base_size) * 50.0) / static_cast<double>(nnz));

  return 5 + static_cast<size_t>(N_1k);
}

// this function use current time as seed to randomize elements in the array of size num_elements
std::vector<double> gen_rand_vec(size_t num_elements){

  std::vector<double> res(num_elements, 0);
  srand(time(0));
  std::generate(res.begin(), res.end(), rand);
  return res;

}

// for computing the sum of the global gather array after we get the local sigma result from each process
double compute_sum(std::vector<double> sigmas){
  double sum = 0.0;
  for(auto ptr = sigmas.begin(); ptr != sigmas.end(); ++ptr)
    sum += *ptr;

  return sum;
}


int main(int argc, char* argv[]) {
  MPI::Init();

  size_t myrank = MPI::COMM_WORLD.Get_rank();
  size_t mysize = MPI::COMM_WORLD.Get_size();

  size_t exponent           = 24;
  size_t num_trips          = 32;
  size_t num_elements 	    = 0;

  // this is the original big global array where we'll randomize each element
  std::vector<double> global_x;

  if (0 == myrank) {

    if (argc >= 2) exponent   = std::stol(argv[1]);

    size_t total_elements = 1 << exponent; // num of elements for the global array
    num_elements = total_elements / mysize; // num of elements per process

    size_t base_size = find_10ms_size(static_cast<double(*)(const std::vector<double>&)>(two_norm_sequential));
    num_trips = num_trials(base_size, num_elements);

    global_x = gen_rand_vec(total_elements);

  } 

  MPI::COMM_WORLD.Bcast(&num_elements, 1, MPI::UNSIGNED_LONG, 0);
  MPI::COMM_WORLD.Bcast(&num_trips, 1, MPI::UNSIGNED_LONG, 0);

  // each process will receive their own protion of local_x by Scatter
  std::vector<double> local_x(num_elements);

  // here we use scatter to distribute the big array 
  MPI_Scatter(global_x.data(), num_elements, MPI_DOUBLE, local_x.data(), num_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double sigma = 0.0;
  
  // init timer, but only start timer at the root rank
  DEF_TIMER(mpi_norm);

  if (myrank == 0) {
    START_TIMER(mpi_norm);
  }
  
  // compute the sigma (square sum) of the local array
  for (size_t i = 0; i < num_trips; ++i)
    sigma = mpi_norm(local_x);

  if(myrank == 0)
    STOP_TIMER(mpi_norm);

  // this is the array where we store all sum results from each process
  auto sub_sigmas = std::vector<double>(mysize, 0);

  // gathter the sum result before square root and store them into sub_sigmas at rank 0
  MPI_Gather(&sigma, 1, MPI_DOUBLE, sub_sigmas.data(), 1, MPI_DOUBLE, 0,
           MPI_COMM_WORLD);

  if (myrank == 0) {

    double rho = std::sqrt(std::inner_product(global_x.begin(), global_x.end(), global_x.begin(), 0.0)); // get sequential result on Rank 0

    // sum the global sum array and square root the final result at rank 0 then compare with ground truth
    double sigma_global = std::sqrt(compute_sum(sub_sigmas));

    double ms_per = t_mpi_norm.elapsed() / static_cast<double>(num_trips);
    std::cout << "# msec_per norm [mpi_norm]: " << ms_per << std::endl;
    double gflops = 2.0 * num_trips * num_elements * mysize / 1.e9;
    double gflops_sec = gflops / (t_mpi_norm.elapsed() * 1.e-3);
    std::cout << "# gflops / sec [mpi_norm]: " << gflops_sec << std::endl;
    std::cout << "# | rho - sigma | = " << std::abs((rho-sigma_global)/sigma_global) << std::endl;

  } 
  
  MPI::Finalize();
  
  return 0;
}
