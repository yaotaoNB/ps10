//
// This file is part of the course materials for CSE P 524 at the University of Washington,
// Winter 2022
//
// Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
// https://creativecommons.org/licenses/by-nc-sa/4.0/
//
// Author: Andrew Lumsdaine
//

#include <iostream>
#include <string>
#include <mpi.h>

int main(int argc, char* argv[]) {
  MPI::Init();

  int token     = 0;
  size_t rounds = 1;
  if (argc >= 2) rounds = std::stol(argv[1]);

  int myrank = MPI::COMM_WORLD.Get_rank();
  int mysize = MPI::COMM_WORLD.Get_size();

  
  // get the rank id of next and previous node, ensure they are in range
  int rank_next  = (myrank + 1) % mysize;  
  int rank_prev = myrank == 0 ? mysize - 1 : myrank - 1;  

  while (rounds--) {
    // when currrent rank is 0, we send first then receive
    if (0 == myrank) { 
      std::cout << myrank << ": sending  " << token << std::endl;
      MPI::COMM_WORLD.Send(&token, 1, MPI::INT, rank_next, 321);
      MPI::COMM_WORLD.Recv(&token, 1, MPI::INT, rank_prev, 321);
      std::cout << myrank << ": received " << token << std::endl;
      ++token;
    } else { // we receive first then send for all other ranks
      MPI::COMM_WORLD.Recv(&token, 1, MPI::INT, rank_prev, 321);
      std::cout << myrank << ": received " << token << std::endl;
      ++token;
      std::cout << myrank << ": sending  " << token << std::endl;
      MPI::COMM_WORLD.Send(&token, 1, MPI::INT, rank_next, 321);
    } 
  }

  MPI::Finalize();

  return 0;
}
