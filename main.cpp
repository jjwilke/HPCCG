
//@HEADER
// ************************************************************************
// 
//               HPCCG: Simple Conjugate Gradient Benchmark Code
//                 Copyright (2006) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
// 
// BSD 3-Clause License
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// 
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
// 
// ************************************************************************
//@HEADER

// Main routine of a program that reads a sparse matrix, right side
// vector, solution vector and initial guess from a file  in HPC
// format.  This program then calls the HPCCG conjugate gradient
// solver to solve the problem, and then prints results.

// Calling sequence:

// test_HPCCG linear_system_file

// Routines called:

// read_HPC_row - Reads in linear system

// mytimer - Timing routine (compile with -DWALL to get wall clock
//           times

// HPCCG - CG Solver

// compute_residual - Compares HPCCG solution to known solution.

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cassert>
#include <string>
#include <cmath>
#include <mpi.h> // If this routine is compiled with -DUSING_MPI
                 // then include mpi.h
#include "make_local_matrix.hpp" // Also include this function
#ifdef _OPENMP
#include <omp.h>
#endif
#include "generate_matrix.hpp"
#include "mytimer.hpp"
#include "HPC_sparsemv.hpp"
#include "compute_residual.hpp"
#include "HPCCG.hpp"
#include "HPC_Sparse_Matrix.hpp"

void init(HPC_Sparse_Matrix*& A, double*& x, double*& b, double*& xexact, MPI_Comm comm,
          int xGrid, int yGrid, int zGrid,
          int nx, int ny, int nz)
{
  int size, rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);


  int myZ = rank / (xGrid*yGrid);
  int rem = rank % (xGrid*yGrid);
  int myY = rem / xGrid;
  int myX = rem % xGrid;
  generate_matrix(nx, ny, nz,
                  myX, myY, myZ,
                  xGrid, yGrid, zGrid,
                  &A, &x, &b, &xexact, comm);
  make_local_matrix(A, comm);
}

#define sstmac_app_name hpccg
int main(int argc, char *argv[])
{

  HPC_Sparse_Matrix *A;
  double *x, *b, *xexact;
  int ierr = 0;
  int nx,ny,nz;


  MPI_Init(&argc, &argv);
  int size, rank; // Number of MPI processes, My process ID
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(argc!=7) {
    if (rank==0)
      cerr << "Usage:" << endl
     << "Mode 1: " << argv[0] << " nx ny nz procX procY procZ" << endl
     << "     where nx, ny and nz are the local sub-block dimensions, or" << endl;
    exit(1);
  }

  nx = atoi(argv[1]);
  ny = atoi(argv[2]);
  nz = atoi(argv[3]);
  int xGrid = atoi(argv[4]);
  int yGrid = atoi(argv[5]);
  int zGrid = atoi(argv[6]);

  init(A, x, b, xexact, MPI_COMM_WORLD,
       xGrid, yGrid, zGrid, nx, ny, nz);

  int niters = 0;
  double normr = 0.0;
  int max_iter = 150;
  double tolerance = 0.0; // Set tolerance to zero to make all runs do max_iter iterations
  ierr = HPCCG(A, b, x, max_iter, tolerance, niters, normr, MPI_COMM_WORLD);

  int max_err;
  MPI_Reduce(&ierr, &max_err, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  if (max_err && rank == 0) cerr << "Error in call to CG: " << max_err << "\n" << endl;
  else if (rank == 0) cout << "CG Succeeded!" << std::endl;

  // Finish up

  MPI_Finalize();

  return 0 ;
} 
