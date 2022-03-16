//
// This file is part of the course materials for CSE P 524 at the University of Washington,
// Winter 2022
//
// Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
// https://creativecommons.org/licenses/by-nc-sa/4.0/
//
// Author: Andrew Lumsdaine
//

#include "../include/Timer.hpp"
#include "coordinate.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <mpi.h>
#include <mutex>
#include <random>

#ifdef __USE_CIMG
#define cimg_display 1
#include <CImg.h>
#endif

// Gravitational constant

// MKS
// const double G = 6.6372E-8;

// Normalized (Henon)
const float G = 1;

template <typename T>
struct body {
  coordinate<T> pos;
  coordinate<T> vel;
  coordinate<T> field = {0, 0, 0};
  T             mass  = 1;
};

template <class T>
void print_body(body<T>& b) {
  std::cout << "  {" << std::endl;
  std::cout << "    [ " << b.pos.x << " " << b.pos.y << " " << b.pos.z << " ]" << std::endl;
  std::cout << "    [ " << b.vel.x << " " << b.vel.y << " " << b.vel.z << " ]" << std::endl;
  std::cout << "    [ " << b.field.x << " " << b.field.y << " " << b.field.z << " ]"
            << std::endl;
  std::cout << "    " << b.mass << std::endl;
  std::cout << "  }" << std::endl;
};

template <class T>
void print_bodies(std::vector<body<T>>& bodies) {
  std::cout << "{" << std::endl;
  for (auto& b : bodies) {
    print_body(b);
  }
  std::cout << "}" << std::endl;
}

double lb(size_t id, size_t num_threads) {
  if (id == 0) {
    return 0;
  }
  double b = 1.0 - 2 * lb(id - 1, num_threads) +
             lb(id - 1, num_threads) * lb(id - 1, num_threads) - 1.0 / ((double)num_threads);
  b = std::abs(b);
  return 1.0 - std::sqrt(b);
}

auto make_lbs(size_t num_threads) {
  std::vector<double> lb(num_threads + 1);
  lb[0] = 0;
  for (size_t i = 1; i < num_threads + 1; ++i) {
    double b = 1.0 - 2 * lb[i - 1] + lb[i - 1] * lb[i - 1] - 1.0 / ((double)num_threads);
    b        = std::abs(b);
    lb[i]    = 1.0 - std::sqrt(b);
  }
  return lb;
}

template <typename It, typename Function>
void for_each_pair(It begin, It end, Function fcn) {
  for (It i = begin; i != end; /* */) {
    auto& obj = *i++;
    for (It j = i; j != end; ++j) {
      fcn(obj, *j);
    }
  }
}

auto make_body_type() {

  static bool          called = false;
  static MPI::Datatype body_type;

  if (called == false) {

    int           coordinate_block_lengths[] = {3};
    MPI::Aint     coordinate_displacements[] = {0};
    MPI::Datatype coordinate_types[]         = {MPI::FLOAT};

    auto coordinate_type = MPI::Datatype::Create_struct(
        1, coordinate_block_lengths, coordinate_displacements, coordinate_types);
    coordinate_type.Commit();

    body<float> foo;
    MPI::Aint   disp = MPI::Get_address(&foo.mass) - MPI::Get_address(&foo);

    int           body_block_lengths[] = {3, 1};
    MPI::Aint     body_displacements[] = {0, disp};
    MPI::Datatype body_types[]         = {coordinate_type, MPI::FLOAT};

    body_type =
        MPI::Datatype::Create_struct(2, body_block_lengths, body_displacements, body_types);
    body_type.Commit();

    called = true;
  }

  return body_type;
}

template <typename T>
void accumulate_fields(std::vector<body<T>>& bodies) {
  for (auto& b : bodies) {
    b.field = {0, 0, 0};
  }

  size_t id          = MPI::COMM_WORLD.Get_rank();
  size_t num_threads = MPI::COMM_WORLD.Get_size();

  static std::vector<double> lb = make_lbs(num_threads);

  auto   begin = bodies.begin();
  auto   end   = bodies.end();
  size_t len   = end - begin;

  auto local_begin = begin + len * lb[id];
  auto local_end   = begin + len * lb[id + 1];

  for_each_pair(local_begin, local_end, [](body<T>& pi, body<T>& pj) {
    auto u = ror3(pj.pos, pi.pos, 1.e-3f);
    pi.field += pj.mass * u;
    pj.field -= pi.mass * u;
  });

  std::vector<int> recvcounts(num_threads);
  std::vector<int> displs(num_threads);
  for (size_t i = 0; i < recvcounts.size(); ++i) {
    recvcounts[i] = len * (lb[i + 1] - lb[i]);
  }
  std::partial_sum(recvcounts.begin(), recvcounts.end(), displs.begin() + 1);

  static auto body_type = make_body_type();

  // Every process send their bodies to every other
  /* Write me */
  MPI::COMM_WORLD.Bcast(bodies.data(), (int)bodies.size(), make_body_type(), 0);

}

template <typename T, typename TimeType>
void update_state(std::vector<body<T>>& bodies, TimeType dt) {
  for (auto& b : bodies) {
    b.vel += (G * dt) * b.field;
    b.pos += (dt * b.vel);
  }
}

template <typename T>
void randomize(std::vector<body<T>>& bodies) {
  static std::default_random_engine  generator;
  static std::normal_distribution<T> distribution(0, 2.0);
  static auto                        position_dice = std::bind(distribution, generator);
  static std::normal_distribution<T> n_distribution(1.0, 0.5);
  static auto                        normal_dice = std::bind(n_distribution, generator);

  int i          = 0;
  T   total_mass = 0;
  for (auto& b : bodies) {

    if (i == 0) {
      i = 1;

      // A big mass at the center of the universe
      b.pos.x = 0;
      b.pos.y = 0;
      b.pos.z = 0;
      b.vel.x = 0;
      b.vel.y = 0;
      b.vel.z = 0;
      b.mass  = bodies.size();
      total_mass += b.mass;
    } else {

      // Other masses
      b.pos.x = position_dice();
      b.pos.y = position_dice();
      b.pos.z = 0. * position_dice() / 10;
      b.vel.x = 1. * position_dice();
      b.vel.y = 1. * position_dice();
      b.vel.z = 0. * position_dice() / 10;

      b.mass = std::abs(normal_dice());
      total_mass += b.mass;
    }
  }
  for (auto& b : bodies) {
    b.mass /= total_mass;
  }
  i = 0;

  // Set the masses in circular orbit around the center
  for (auto& b : bodies) {
    if (i++ == 0) continue;

    auto r      = std::sqrt(dot(b.pos, b.pos));
    auto target = std::sqrt(G * bodies[0].mass / r);

    if (dot(b.vel, b.vel) > 1.e-12 * dot(b.pos, b.pos)) {

      if (dot(b.pos, b.pos) > 1.e-12) {
        auto a  = dot(b.pos, b.vel) / dot(b.pos, b.pos);
        auto w  = b.vel - a * b.pos;
        auto wn = std::sqrt(dot(w, w));

        auto z   = b.pos + w;
        auto dir = b.pos.x * z.y - z.x * b.pos.y;
        dir      = dir / std::abs(dir);
        target   = std::copysign(target, dir);

        w *= target / wn;
        b.vel = w;
      }
    }
  }
}

template <typename T, typename TimeType>
void offset_velocities(std::vector<body<T>>& bodies, TimeType offset) {
  // Offset the velocities (used for leapfrog integration)
  for (auto& b : bodies) {
    b.vel += (G * offset) * b.field;
  }
}

template <typename T, typename TimeType>
void run(std::vector<body<T>>& bodies, TimeType dt, TimeType end_time) {
  accumulate_fields(bodies);
  offset_velocities(bodies, -dt / 2);

  for (TimeType t = 0; t < end_time - dt / 2; t += dt) {
    update_state(bodies, dt);
    accumulate_fields(bodies);
  }
}

void usage(const std::string& msg) {
  std::cout << "Usage: " + msg << " [-d delta_t] [-t num_timesteps] [-n num_bodies] [-v]"
            << std::endl;
}

#ifdef __USE_CIMG
class visualizer {

public:
  visualizer() : white{255, 255, 255} {}
  visualizer(const std::string& str)
      : visu(900, 600, 1, 3, 0), main_disp(visu, str.c_str()), white{255, 255, 255} {
    visu.fill(0);
  }
  visualizer(const visualizer&) : white{255, 255, 255} {}

  void operator()(const std::vector<body<float>>& bodies) { visualize(bodies); }

  void visualize(const std::vector<body<float>>& bodies) {
    visu.fill(0);
    for (auto& b : bodies) {
      visu.draw_circle((int)(b.pos.x / 2.5e-2) + 450, (int)(b.pos.y / 2.5e-2) + 300, 2, white);
    }
    visu.display(main_disp);
  }

private:
  cimg_library::CImg<unsigned char> visu;
  cimg_library::CImgDisplay         main_disp;
  const unsigned char               white[3];
};

template <typename T, typename TimeType>
void run(std::vector<body<T>>& bodies, TimeType dt, TimeType end_time, visualizer& pvis) {
  accumulate_fields(bodies);
  offset_velocities(bodies, -dt / 2);

  for (TimeType t = 0; t < end_time - dt / 2; t += dt) {
    update_state(bodies, dt);
    accumulate_fields(bodies);
    pvis.visualize(bodies);
  }
}
#endif

int main(int argc, char* argv[]) {
  MPI::Init();

  size_t myrank = MPI::COMM_WORLD.Get_rank();
  size_t mysize = MPI::COMM_WORLD.Get_size();

  bool   visualize     = false;
  float  dt            = 0.001;
  size_t num_bodies    = 1024;
  size_t num_timesteps = 1024;

  if (0 == myrank) {

    try {
      for (int arg = 1; arg < argc; ++arg) {

        if (std::string(argv[arg]) == "-d") {
          if (argc == ++arg) usage(argv[0]);
          dt = std::stod(argv[arg]);
        }

        else if (std::string(argv[arg]) == "-t") {
          if (argc == ++arg) usage(argv[0]);
          num_timesteps = std::stol(argv[arg]);
        }

        else if (std::string(argv[arg]) == "-n") {
          if (argc == ++arg) usage(argv[0]);
          num_bodies = std::stol(argv[arg]);

        } else if (std::string(argv[arg]) == "-v") {
          visualize = true;
        } else {
          usage(argv[0]);
          return -1;
        }
      }
    } catch (int) {
      usage(argv[0]);
      return -1;
    }
  }

  MPI::COMM_WORLD.Bcast(&dt, 1, MPI::FLOAT, 0);
  MPI::COMM_WORLD.Bcast(&num_timesteps, 1, MPI::UNSIGNED_LONG, 0);
  MPI::COMM_WORLD.Bcast(&num_bodies, 1, MPI::UNSIGNED_LONG, 0);

  std::vector<body<float>> bodies(num_bodies);
  if (0 == myrank) {
    randomize(bodies);
  }
  MPI::COMM_WORLD.Bcast(bodies.data(), (int)bodies.size(), make_body_type(), 0);

  if (visualize) {
#ifdef __USE_CIMG
    visualizer pvis("Nbody abstract");
    run(bodies, dt, dt * num_timesteps, pvis);
#else
    ;
#endif
  } else {
    DEF_TIMER(run);
    START_TIMER(run);
    run(bodies, dt, dt * num_timesteps);

    if (0 == myrank) {
      STOP_TIMER(run);
    }
  }

  MPI::Finalize();

  return 0;
}
