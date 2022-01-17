#ifndef   HEADS_H
#define   HEADS_H
#include <cuda_runtime.h> 
//TODO: include NCCL headers
#include <nccl.h>
#include <string>
#include <algorithm>

#ifdef USE_DOUBLE
typedef double real;
#define MPI_REAL_TYPE MPI_DOUBLE
#define NCCL_REAL_TYPE ncclDouble
#else
typedef float real;
#define MPI_REAL_TYPE MPI_FLOAT
#define NCCL_REAL_TYPE ncclFloat
#endif

#define NCCL_CALL(call)                                                                 \
{                                                                                       \
    ncclResult_t  ncclStatus = call;                                                    \
    if (ncclSuccess != ncclStatus)                                                      \
        fprintf(stderr,                                                                 \
                "ERROR: NCCL call \"%s\" in line %d of file %s failed "                 \
                "with "                                                                 \
                "%s (%d).\n",                                                           \
                #call, __LINE__, __FILE__, ncclGetErrorString(ncclStatus), ncclStatus); \
}

#define MPI_CALL(call)                                                            \
{                                                                                 \
    int mpi_status = call;                                                        \
    if (0 != mpi_status) {                                                        \
        char mpi_error_string[MPI_MAX_ERROR_STRING];                              \
        int mpi_error_string_length = 0;                                          \
        MPI_Error_string(mpi_status, mpi_error_string, &mpi_error_string_length); \
        if (NULL != mpi_error_string)                                             \
            fprintf(stderr,                                                       \
                    "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                    "with %s "                                                    \
                    "(%d).\n",                                                    \
                    #call, __LINE__, __FILE__, mpi_error_string, mpi_status);     \
        else                                                                      \
            fprintf(stderr,                                                       \
                    "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                    "with %d.\n",                                                 \
                    #call, __LINE__, __FILE__, mpi_status);                       \
    }                                                                             \
}

#define CUDA_RT_CALL(call)                                                              \
{                                                                                       \
    cudaError_t cudaStatus = call;                                                      \
    if (cudaSuccess != cudaStatus)                                                      \
        fprintf(stderr,                                                                 \
                "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                "with "                                                                 \
                "%s (%d).\n",                                                           \
                #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
}

#ifdef USE_DOUBLE
typedef double real;
#define MPI_REAL_TYPE MPI_DOUBLE
#else
typedef float real;
#define MPI_REAL_TYPE MPI_FLOAT
#endif

constexpr real tol = 1.0e-8;
const real PI = 2.0 * std::asin(1.0);

void diriclet_boundary(real* __restrict__ const a_new, real* __restrict__ const a,
                                  const real pi, const int offset, const int nx, const int my_ny,
                                  const int ny);

void stencil_kernel(real* __restrict__ const a_new, const real* __restrict__ const a,
                          real* __restrict__ const l2_norm, const int iy_start, const int iy_end,
                          const int nx, const bool calculate_norm, cudaStream_t stream);

double serial_impl(const int nx, const int ny, const int iter_max, real* const a_ref_h,
                  const int nccheck, const bool print);

double parallel_impl(const int rank, const int size, const int nx, const int ny, 
                const int iter_max, real* const a_h, const int nccheck, const bool print);

int check(const int rank, const int size, const int nx, const int ny, real* const a_ref_h, 
                real* const a_h,  const double s_elapse, const double p_elapse);

#endif