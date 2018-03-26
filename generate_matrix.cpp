
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

/////////////////////////////////////////////////////////////////////////

// Routine to read a sparse matrix, right hand side, initial guess, 
// and exact solution (as computed by a direct solver).

/////////////////////////////////////////////////////////////////////////

// nrow - number of rows of matrix (on this processor)

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include "generate_matrix.hpp"

void generate_matrix(int nx, int ny, int nz,
                     int myX, int myY, int myZ,
                     int xGrid, int yGrid, int zGrid,
                     HPC_Sparse_Matrix **Aptr,
                     double **xPtr, double **bPtr, double **xexactPtr)

{
#ifdef DEBUG
  int debug = 1;
#else
  int debug = 0;
#endif

  int size, rank; // Number of MPI processes, My process ID
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto A = new HPC_Sparse_Matrix; // Allocate matrix struct and fill it
  A->title = 0;
  *Aptr = A;

  int local_nrow = nx*ny*nz; // This is the size of our subblock
  assert(local_nrow>0); // Must have something to work with
  int local_nnz = 27*local_nrow; // Approximately 27 nonzeros per row (except for boundary nodes)

  auto total_nrow = global_t(local_nrow)*size; // Total number of grid points in mesh
  auto total_nnz = 27*global_t(total_nrow); // Approximately 27 nonzeros per row (except for boundary nodes)


  // Allocate arrays that are of length local_nrow
  A->nnz_in_row = new int[local_nrow];
  A->ptr_to_vals_in_row = new double*[local_nrow];
  A->ptr_to_inds_in_row = new global_t*[local_nrow];
  A->ptr_to_diags       = new double*[local_nrow];

  auto x = new double[local_nrow]; *xPtr = x;
  auto b = new double[local_nrow]; *bPtr = b;
  auto xexact = new double[local_nrow]; *xexactPtr = xexact;


  // Allocate arrays that are of length local_nnz
  A->list_of_vals = new double[local_nnz];
  A->list_of_inds = new global_t[local_nnz];

  double * curvalptr = A->list_of_vals;
  auto curindptr = A->list_of_inds;


  auto Nx = global_t(nx) * global_t(xGrid);
  auto Ny = global_t(ny) * global_t(yGrid);
  auto Nz = global_t(nz) * global_t(zGrid);


  A->Nx = Nx;
  A->Ny = Ny;
  A->Nz = Nz;
  A->myXgrid = myX;
  A->myYgrid = myY;
  A->myZgrid = myZ;
  A->nx = nx;
  A->ny = ny;
  A->nz = nz;
  A->myXstart = myX*nx;
  A->myXstop = A->myXstart + nx;
  A->myYstart = myY*ny;
  A->myYstop = A->myYstart + ny;
  A->myZstart = myZ*nz;
  A->myZstop = A->myZstart + nz;
  A->xGrid = xGrid;
  A->yGrid = yGrid;
  A->zGrid = zGrid;

  global_t nnzglobal = 0;
  auto xOffset = global_t(myX) * global_t(nx);
  auto yOffset = global_t(myY) * global_t(ny);
  auto zOffset = global_t(myZ) * global_t(nz);
  for (int iz=0; iz<nz; iz++) {
    auto gz = iz + zOffset;
    for (int iy=0; iy<ny; iy++) {
      auto gy = iy + yOffset;
      for (int ix=0; ix<nx; ix++) {
        auto gx = ix + xOffset;
        auto curlocalrow = A->getLocalIndex(ix,iy,iz);
        auto currow = A->getGlobalIndex(gx,gy,gz);
        int nnzrow = 0;
        A->ptr_to_vals_in_row[curlocalrow] = curvalptr;
        A->ptr_to_inds_in_row[curlocalrow] = curindptr;


        for (int sz=-1; sz<=1; sz++) {
          auto gsz = sz+gz;
          for (int sy=-1; sy<=1; sy++) {
            auto gsy = sy+gy;
            for (int sx=-1; sx<=1; sx++) {
              auto gsx = sx+gx;
              bool check = A->boundsCheck(gx,gy,gz,sx,sy,sz);
              if (check){
                auto curcol = A->getGlobalIndex(gsx,gsy,gsz);
                if (curcol == currow){
                  A->ptr_to_diags[curlocalrow] = curvalptr;
                  *curvalptr++ = 27.0;
                } else {
                  *curvalptr++ = -1.0;
                }
                //printf("Rank %d setting A(%d,%d) to %f\n",
                //       rank, currow, curcol, *(curvalptr-1));
                *curindptr++ = curcol;
                nnzrow++;
              }
            }
          }
        }
        A->nnz_in_row[curlocalrow] = nnzrow;
        nnzglobal += nnzrow;
        x[curlocalrow] = 0.0;
        b[curlocalrow] = 27.0 - ((double) (nnzrow-1));
        xexact[curlocalrow] = 1.0;
      } // end ix loop
     } // end iy loop
  } // end iz loop  
  if (debug) cout << "Process "<<rank<<" of "<<size<<" has "<<local_nrow;
  
  if (debug) cout << "Process "<<rank<<" of "<<size
		  <<" has "<<local_nnz<<" nonzeros."<<endl;

  A->total_nrow = total_nrow;
  A->total_nnz = total_nnz;
  A->local_nrow = local_nrow;
  A->local_ncol = local_nrow;
  A->local_nnz = local_nnz;

  return;
}
