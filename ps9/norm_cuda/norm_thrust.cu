// -*- c++ -*-
//
// This file is part of the course materials for AMATH483/583 at the University of Washington,
// Spring 2020
//
// Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
// https://creativecommons.org/licenses/by-nc-sa/4.0/
//
// Author: Andrew Lumsdaine
//

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "Timer.hpp"


template<typename T>
void randomize(std::vector<T>& v) {
  static std::default_random_engine             generator;
  static std::uniform_real_distribution<T> distribution(-1.0, 1.0);
  static auto                                   dice = std::bind(distribution, generator);

  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = dice();
  }
}


template<typename T>
T two_norm_sequential(const std::vector<T>& v) {
  T sum = 0.0;

  for (size_t i = 0; i < v.size(); ++i) {
    sum += v[i] * v[i];
  }

  return std::sqrt(sum);
}



template<typename T>
T norm_thrust(const thrust::device_vector<T>& x) {
  T sum = thrust::reduce(x.begin(), x.end(), 0);
  return std::sqrt(sum);
}


void header(const std::string& msg = "") {
  auto& os_ = std::cout;
  if (msg != "") {
    os_ << std::setw(12) << std::left << msg << std::endl;
  }
  os_ << std::setw(12) << std::right << "N";
  os_ << std::setw(12) << std::right << "Sequential";

  os_ << std::setw(12) << std::right << "First";
  os_ << std::setw(12) << std::right << "Second";

  os_ << std::setw(12) << std::right << "First";
  os_ << std::setw(12) << std::right << "Second";

  os_ << std::endl;
}

double Gflops_sec(size_t nnz, size_t trials, double msec) {
  double Gflops = static_cast<double>(trials) * (2.0 * nnz) / 1.e9;
  double sec    = msec / 1.e3;
  if (sec == 0) {
    return 0;
  }
  return Gflops / sec;
}

void log(size_t nnz, size_t ntrials, const std::vector<double>& ms_times, const std::vector<double>& norms) {
  auto& os_ = std::cout;
  os_ << std::setw(12) << std::right << nnz;

  for (size_t i = 0; i < ms_times.size(); ++i) {
    os_ << std::setw(12) << std::right << Gflops_sec(nnz, ntrials, ms_times[i]);
  }
  for (size_t i = 1; i < ms_times.size(); ++i) {
    os_ << std::setw(14) << std::right << std::abs(norms[i] - norms[0]) / norms[0];
  }
  os_ << std::endl;
}

size_t num_trials(size_t nnz) {
  double N_1k = std::ceil(2E9 / static_cast<double>(nnz));
  return 5 + static_cast<size_t>(N_1k);
}

template <class T, typename Function>
void run_cu(Function&& f, size_t N_min, size_t N_max) {
  header(sizeof(T) == 4 ? "\nFloat" : "\nDouble");
  Timer t;

  for (size_t size = N_min; size <= N_max; size *= 2) {
    std::vector<double> ms_times;
    std::vector<double> norms;

    std::vector<T> x(size);
      
    randomize(x);

    double norm0 = two_norm_sequential(x);
    double norm1 = 0.0;

    size_t ntrials = num_trials(size);

    t.start();
    for (size_t i = 0; i < ntrials; ++i) {
      norm0 = two_norm_sequential(x);
    }
    t.stop();
    ms_times.push_back(t.elapsed());
    norms.push_back(norm0);

    thrust::device_vector<T> X(size);
    thrust::copy(x.begin(), x.end(), X.begin());

    for (size_t trip = 0; trip < 2; ++trip) {

      t.start();
      cudaDeviceSynchronize();
      for (size_t i = 0; i < ntrials; ++i) {
        norm1 = f(X);
        cudaDeviceSynchronize();
      }
      t.stop();
      ms_times.push_back(t.elapsed());
      norms.push_back(norm1);
    }

    log(size, ntrials, ms_times, norms);
  }
}




int main(int argc, char* argv[]) {
  size_t N_min = 1024 * 1024;
  size_t N_max = 128 * 1024 * 1024;

  if (argc >= 2) {
    N_min = std::stol(argv[1]);
  }
  if (argc >= 3) {
    N_max = std::stol(argv[2]);
  }

  run_cu<float>(norm_thrust<float>, N_min, N_max);
  run_cu<double>(norm_thrust<double>, N_min, N_max);

  return 0;
}
