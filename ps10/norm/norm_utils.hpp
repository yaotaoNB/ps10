//
// This file is part of the course materials for CSE P 524 at the University of Washington,
// Winter 2022
//
// Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
// https://creativecommons.org/licenses/by-nc-sa/4.0/
//
// Author: Andrew Lumsdaine
//

#ifndef CSEP524_NORM_UTILS_HPP
#define CSEP524_NORM_UTILS_HPP

#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "norms.hpp"
#include "Timer.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif


void header(size_t n_threads, const std::string& msg = "") {
  auto& os_ = std::cout;
  if (msg != "") {
    os_ << std::setw(12) << std::left << msg << std::endl;
  }
  os_ << std::setw(12) << std::right << "N";
  os_ << std::setw(12) << std::right << "Sequential";
  os_ << std::setw(12) << std::right << "1_thread";
  for (size_t i = 2; i <= n_threads; i *= 2) {
    os_ << std::setw(4) << std::right << i << "_threads";
  }
  os_ << std::setw(14) << std::right << "1_thread";
  for (size_t i = 2; i <= n_threads; i *= 2) {
    os_ << std::setw(6) << std::right << i << "_threads";
  }
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


template <class F, class... Args>
size_t find_10ms_size(F&& f, Args&&... args) {
  Timer t;
  size_t sz = 128;

  for (; sz < 1024 * 1024 * 1024; sz *= 1.414) {
    std::vector<double> x(sz);
    t.start();
    f(x, std::forward<Args>(args)...);
    t.stop();
    if (t.elapsed() >= 10.0) {
      break;
    }
  }

  return sz;
}

size_t num_trials(size_t base_size, size_t nnz) {
  // trials*nnz/1G = 1S => trials = 1G/nnz

  double N_1k = std::ceil((static_cast<double>(base_size) * 100.0) / static_cast<double>(nnz));

  return 5 + static_cast<size_t>(N_1k);
}

template <typename Function, class... Args >
void run(Function&& f, size_t N_min, size_t N_max, size_t n_parts, Args&&... args) {
  header(n_parts);
  Timer t;

  size_t base_size = find_10ms_size(f, std::forward<Args>(args)...);

  for (size_t size = N_min; size <= N_max; size *= 2) {
    std::vector<double> ms_times;
    std::vector<double> norms;
    std::vector<double> x(size);
    randomize(x);
    double norm0 = two_norm_sequential(x);
    double norm1 = 0.0;

    size_t ntrials = num_trials(base_size, size);

    t.start();
    for (size_t i = 0; i < ntrials; ++i) {
      norm0 = two_norm_sequential(x);
    }
    t.stop();
    ms_times.push_back(t.elapsed());
    norms.push_back(norm0);

    for (size_t nthreads = 1; nthreads <= n_parts; nthreads *= 2) {
#ifdef _OPENMP
      omp_set_num_threads(nthreads);
#endif
      t.start();
      for (size_t i = 0; i < ntrials; ++i) {
        norm1 = std::max(norm1, f(x, std::forward<Args>(args)...));
      }
      t.stop();
      ms_times.push_back(t.elapsed());
      norms.push_back(norm1);
    }

    log(size, ntrials, ms_times, norms);
  }
}

#endif    // CSEP524_NORM_UTILS_HPP
