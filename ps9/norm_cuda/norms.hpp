//
// This file is part of the course materials for CSE P 524 at the University of Washington,
// Winter 2022
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
#include <functional>
#include <random>
#include <vector>

#include "Timer.hpp"
#include "omp.h"


template <class T>
void randomize(std::vector<T>& v) {
  static std::default_random_engine        generator;
  static std::uniform_real_distribution<T> distribution(2.0, 32.0);
  static auto                              dice = std::bind(distribution, generator);

  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = dice();
  }
}


template <class T>
T norm_seq(const std::vector<T>& x) {
  T sum = 0;

  for (size_t i = 0; i < x.size(); ++i) {
    sum += x[i] * x[i];
  }
  return std::sqrt(sum);
}

template <class T>
T norm_parfor(const std::vector<T>& x) {
  T sum = 0;

#pragma omp parallel for simd reduction(+:sum)
  for (size_t i = 0; i < x.size(); ++i) {
    sum += x[i] * x[i];
  }

  return std::sqrt(sum);
}

#endif // CSEP524_NORMS_HPP
