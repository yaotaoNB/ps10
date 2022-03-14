//
// This file is part of the course materials for CSE P 524 at the University of Washington,
// Autumn 2018
//
// Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
// https://creativecommons.org/licenses/by-nc-sa/4.0/
//
// Author: Andrew Lumsdaine
//

#include <cmath>

template <typename T>
struct scalar_proxy;

template <typename T>
struct coordinate {
  T x;
  T y;
  T z;
};

template <typename T>
struct scalar_proxy {
  T                    the_scalar;
  const coordinate<T>& the_coordinate;
                       operator coordinate<T>() {
    return {the_scalar * the_coordinate.x, the_scalar * the_coordinate.y,
            the_scalar * the_coordinate.z};
  }
};

template <typename T>
coordinate<T> operator+(const coordinate<T>& u, const coordinate<T>& v) {
  return {u.x + v.x, u.y + v.y, u.z + v.z};
}

template <typename T>
coordinate<T> operator-(const coordinate<T>& u, const coordinate<T>& v) {
  return {u.x - v.x, u.y - v.y, u.z - v.z};
}

template <typename T>
coordinate<T>& operator+=(coordinate<T>& u, const coordinate<T>& v) {
  u.x += v.x;
  u.y += v.y;
  u.z += v.z;
  return u;
}

template <typename T>
coordinate<T>& operator+=(coordinate<T>& u, const scalar_proxy<T>& v) {
  u.x += v.the_scalar * v.the_coordinate.x;
  u.y += v.the_scalar * v.the_coordinate.y;
  u.z += v.the_scalar * v.the_coordinate.z;
  return u;
}

template <typename T>
coordinate<T>& operator-=(coordinate<T>& u, const coordinate<T>& v) {
  u.x -= v.x;
  u.y -= v.y;
  u.z -= v.z;
  return u;
}

template <typename T>
coordinate<T>& operator-=(coordinate<T>& u, const scalar_proxy<T>& v) {
  u.x -= v.the_scalar * v.the_coordinate.x;
  u.y -= v.the_scalar * v.the_coordinate.y;
  u.z -= v.the_scalar * v.the_coordinate.z;
  return u;
}

template <typename T>
coordinate<T>& abs(coordinate<T>& u) {
  return {std::abs(u.x), std::abs(u.y), std::abs(u.z)};
}

template <typename T>
T dot(const coordinate<T>& u, const coordinate<T>& v) {
  return (u.x * v.x + u.y * v.y + u.z * v.z);
}

template <typename T>
T magnitude(const coordinate<T>& u) {
  return std::sqrt(dot(u, u));
}

#if 0
template<typename T>
auto operator*(T a, const coordinate<T>& u) {
  return scalar_proxy<T>{a, u};
}
#else
template <typename T>
coordinate<T> operator*(T a, const coordinate<T>& u) {
  return {a * u.x, a * u.y, a * u.z};
}
#endif

template <typename T>
coordinate<T>& operator*=(coordinate<T>& u, T a) {
  u.x *= a;
  u.y *= a;
  u.z *= a;
  return u;
}

template <typename T>
coordinate<T> ror3(const coordinate<T>& u, const coordinate<T>& v, T epsilon = 0.0) {
  coordinate<T> dx = u - v;
  T             r  = magnitude(dx);
  r += epsilon;

  coordinate<T> res = (T{1.0} / (r * r * r)) * dx;

  return res;
}
