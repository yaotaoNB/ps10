//
// This file is part of the course materials for AMATH483/583 at the University of Washington,
// Spring 2020
//
// Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
// https://creativecommons.org/licenses/by-nc-sa/4.0/
//
// Author: Andrew Lumsdaine
//

#ifndef CSEP524_NORMS_HPP
#define CSEP524_NORMS_HPP

#include <cmath>
#include <iostream>

#include "Timer.hpp"

#include <functional>
#include <future>
#include <random>
#include <vector>

#ifdef _OPENMP
#include "omp.h"
#endif

void randomize(std::vector<double>& x) {
  std::default_random_engine             generator;
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  static auto                            dice = std::bind(distribution, generator);
for (size_t i = 0; i < x.size(); ++i) {
    x[i] = dice();
  }
}

double two_norm_sequential(const std::vector<double>& x) {
  double sum = 0;

  for (size_t i = 0; i < x.size(); ++i) {
    sum += x[i] * x[i];
  }
  return std::sqrt(sum);
}

double two_norm_parfor(const std::vector<double>& x) {
  double sum = 0;

#pragma omp parallel for reduction(+:sum)
  for (size_t i = 0; i < x.size(); ++i) {
    sum += x[i] * x[i];
  }

  return std::sqrt(sum);
}

#endif // CSEP524_NORMS_HPP
