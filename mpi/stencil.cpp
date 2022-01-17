
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <mpi.h>
#include "heads.h"

bool get_arg(char** begin, char** end, const std::string& arg) {
    char** itr = std::find(begin, end, arg);
    if (itr != end) {
        return true;
    }
    return false;
}

double serial_impl(const int nx, const int ny, const int iter_max, real* const a_ref_h,
                  const int nccheck, const bool print) {
    real* a;
    real* a_new;

    cudaStream_t compute_stream;
    cudaStream_t push_top_stream;
    cudaStream_t push_bottom_stream;
    cudaEvent_t compute_done;
    cudaEvent_t push_top_done;
    cudaEvent_t push_bottom_done;

    real* l2_norm_d;
    real* l2_norm_h;

    int iy_start = 1;
    int iy_end = (ny - 1);

    CUDA_RT_CALL(cudaMalloc(&a, nx * ny * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc(&a_new, nx * ny * sizeof(real)));

    CUDA_RT_CALL(cudaMemset(a, 0, nx * ny * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(a_new, 0, nx * ny * sizeof(real)));

    // Set diriclet boundary conditions on left and right boarder
    diriclet_boundary(a, a_new, PI, 0, nx, ny, ny);
    CUDA_RT_CALL(cudaDeviceSynchronize());

    CUDA_RT_CALL(cudaStreamCreate(&compute_stream));
    CUDA_RT_CALL(cudaStreamCreate(&push_top_stream));
    CUDA_RT_CALL(cudaStreamCreate(&push_bottom_stream));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&compute_done, cudaEventDisableTiming));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&push_top_done, cudaEventDisableTiming));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&push_bottom_done, cudaEventDisableTiming));

    CUDA_RT_CALL(cudaMalloc(&l2_norm_d, sizeof(real)));
    CUDA_RT_CALL(cudaMallocHost(&l2_norm_h, sizeof(real)));

    CUDA_RT_CALL(cudaDeviceSynchronize());

    if (print)
        printf(
            "Single GPU jacobi relaxation: %d iterations on %d x %d mesh with "
            "norm "
            "check every %d iterations\n",
            iter_max, ny, nx, nccheck);

    int iter = 0;
    real l2_norm = 1.0;
    bool calculate_norm;

    double start = MPI_Wtime();
    while (l2_norm > tol && iter < iter_max) {
        CUDA_RT_CALL(cudaMemsetAsync(l2_norm_d, 0, sizeof(real), compute_stream));

        CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, push_top_done, 0));
        CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, push_bottom_done, 0));

        calculate_norm = (iter % nccheck) == 0 || (iter % 100) == 0;
        stencil_kernel(a_new, a, l2_norm_d, iy_start, iy_end, nx, calculate_norm,
                             compute_stream);
        CUDA_RT_CALL(cudaEventRecord(compute_done, compute_stream));

        if (calculate_norm) {
            CUDA_RT_CALL(cudaMemcpyAsync(l2_norm_h, l2_norm_d, sizeof(real), cudaMemcpyDeviceToHost,
                                         compute_stream));
        }

        // Apply periodic boundary conditions
        CUDA_RT_CALL(cudaStreamWaitEvent(push_top_stream, compute_done, 0));
        CUDA_RT_CALL(cudaMemcpyAsync(a_new, a_new + (iy_end - 1) * nx, nx * sizeof(real),
                                     cudaMemcpyDeviceToDevice, push_top_stream));
        CUDA_RT_CALL(cudaEventRecord(push_top_done, push_top_stream));

        CUDA_RT_CALL(cudaStreamWaitEvent(push_bottom_stream, compute_done, 0));
        CUDA_RT_CALL(cudaMemcpyAsync(a_new + iy_end * nx, a_new + iy_start * nx, nx * sizeof(real),
                                     cudaMemcpyDeviceToDevice, push_bottom_stream));
        CUDA_RT_CALL(cudaEventRecord(push_bottom_done, push_bottom_stream));

        if (calculate_norm) {
            CUDA_RT_CALL(cudaStreamSynchronize(compute_stream));
            l2_norm = *l2_norm_h;
            l2_norm = std::sqrt(l2_norm);
            if (print && (iter % 100) == 0) printf("%5d, %0.6f\n", iter, l2_norm);
        }

        std::swap(a_new, a);
        iter++;
    }
    double stop = MPI_Wtime();

    CUDA_RT_CALL(cudaMemcpy(a_ref_h, a, nx * ny * sizeof(real), cudaMemcpyDeviceToHost));

    CUDA_RT_CALL(cudaEventDestroy(push_bottom_done));
    CUDA_RT_CALL(cudaEventDestroy(push_top_done));
    CUDA_RT_CALL(cudaEventDestroy(compute_done));
    CUDA_RT_CALL(cudaStreamDestroy(push_bottom_stream));
    CUDA_RT_CALL(cudaStreamDestroy(push_top_stream));
    CUDA_RT_CALL(cudaStreamDestroy(compute_stream));

    CUDA_RT_CALL(cudaFreeHost(l2_norm_h));
    CUDA_RT_CALL(cudaFree(l2_norm_d));

    CUDA_RT_CALL(cudaFree(a_new));
    CUDA_RT_CALL(cudaFree(a));
    return (stop - start);
}

double parallel_impl(const int rank, const int size, const int nx, const int ny, 
                const int iter_max, real* const a_h, const int nccheck, const bool print){
    int chunk_size;
    int chunk_size_low = (ny - 2) / size;
    int chunk_size_high = chunk_size_low + 1;
    // num_ranks_low * chunk_size_low + (size - num_ranks_low) * (chunk_size_low + 1) = ny - 2
    int num_ranks_low = size * chunk_size_low + size - (ny - 2); 
    if (rank < num_ranks_low)
        chunk_size = chunk_size_low;
    else
        chunk_size = chunk_size_high;

    real* a;
    CUDA_RT_CALL(cudaMalloc(&a, nx * (chunk_size + 2) * sizeof(real)));
    real* a_new;
    CUDA_RT_CALL(cudaMalloc(&a_new, nx * (chunk_size + 2) * sizeof(real)));

    CUDA_RT_CALL(cudaMemset(a, 0, nx * (chunk_size + 2) * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(a_new, 0, nx * (chunk_size + 2) * sizeof(real)));

    // Calculate local domain boundaries
    int iy_start_global;  // My start index in the global array
    if (rank < num_ranks_low) {
        iy_start_global = rank * chunk_size_low + 1;
    } else {
        iy_start_global =
            num_ranks_low * chunk_size_low + (rank - num_ranks_low) * chunk_size_high + 1;
    }
    int iy_end_global = iy_start_global + chunk_size - 1;  // My last index in the global array

    int iy_start = 1;
    int iy_end = iy_start + chunk_size;

    
    // Set diriclet boundary conditions on left and right boarder
    diriclet_boundary(a, a_new, PI, iy_start_global - 1, nx, (chunk_size + 2), ny);
    CUDA_RT_CALL(cudaDeviceSynchronize());

    //Overlapped Optimization.
    int leastPriority = 0;
    int greatestPriority = leastPriority;
    CUDA_RT_CALL(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

    cudaStream_t push_top_stream;
    cudaEvent_t push_top_done;
    CUDA_RT_CALL(cudaEventCreateWithFlags(&push_top_done, cudaEventDisableTiming));
    cudaStream_t push_bottom_stream;
    cudaEvent_t push_bottom_done;
    CUDA_RT_CALL(cudaEventCreateWithFlags(&push_bottom_done, cudaEventDisableTiming));
    cudaStream_t compute_stream;
    cudaEvent_t compute_done;
    CUDA_RT_CALL(cudaEventCreateWithFlags(&compute_done, cudaEventDisableTiming));
    cudaEvent_t reset_l2norm_done;
    CUDA_RT_CALL(cudaEventCreateWithFlags(&reset_l2norm_done, cudaEventDisableTiming));

    CUDA_RT_CALL(cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault, leastPriority));
    CUDA_RT_CALL(cudaStreamCreateWithPriority(&push_top_stream, cudaStreamDefault, greatestPriority));
    CUDA_RT_CALL(cudaStreamCreateWithPriority(&push_bottom_stream, cudaStreamDefault, greatestPriority));

    real* l2_norm_d;
    CUDA_RT_CALL(cudaMalloc(&l2_norm_d, sizeof(real)));
    real* l2_norm_h;
    CUDA_RT_CALL(cudaMallocHost(&l2_norm_h, sizeof(real)));


    if (0 == rank) {
        printf(
            "Jacobi relaxation: %d iterations on %d x %d mesh with norm check "
            "every %d iterations\n",
            iter_max, ny, nx, nccheck);
    }

    int iter = 0;
    real l2_norm = 1.0;
    bool calculate_norm;  

    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();
    printf("Num GPUs: %d.\n", size);
    while (l2_norm > tol && iter < iter_max) {
        CUDA_RT_CALL(cudaMemsetAsync(l2_norm_d, 0, sizeof(real), compute_stream));

        //Overlapped Optimization.
        CUDA_RT_CALL(cudaEventRecord(reset_l2norm_done, compute_stream));

        calculate_norm = (iter % nccheck) == 0 || ((iter % 100) == 0);

        //Overlapped Optimization.
        stencil_kernel(a_new, a, l2_norm_d, (iy_start + 1), (iy_end - 1), nx, calculate_norm,
                             compute_stream);
        CUDA_RT_CALL(cudaEventRecord(compute_done, compute_stream));

        CUDA_RT_CALL(cudaStreamWaitEvent(push_top_stream, reset_l2norm_done, 0));
        stencil_kernel(a_new, a, l2_norm_d, iy_start, (iy_start + 1), nx, calculate_norm,
                             push_top_stream);
        CUDA_RT_CALL(cudaEventRecord(push_top_done, push_top_stream));

        CUDA_RT_CALL(cudaStreamWaitEvent(push_bottom_stream, reset_l2norm_done, 0));
        stencil_kernel(a_new, a, l2_norm_d, (iy_end - 1), iy_end, nx, calculate_norm,
                             push_bottom_stream);
        CUDA_RT_CALL(cudaEventRecord(push_bottom_done, push_bottom_stream));

        if (calculate_norm) {
            //Overlapped Optimization.
            CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, push_top_done, 0));
            CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, push_bottom_done, 0));

            CUDA_RT_CALL(cudaMemcpyAsync(l2_norm_h, l2_norm_d, sizeof(real), cudaMemcpyDeviceToHost,
                                         compute_stream));
        }

        //TODO: Compute top and bottom neighbor, use reflecting/periodic boundaries.
        const int top = rank > 0 ? rank - 1 : (size - 1);
        const int bottom = (rank + 1) % size;

        // TODO: Use MPI_Sendrecv to exchange the data with the top and bottom neighbors by CUDA-aware MPI.
        //Overlapped Optimization.
        CUDA_RT_CALL(cudaStreamSynchronize(push_top_stream));
        MPI_CALL(MPI_Sendrecv(a_new + iy_start * nx, nx, MPI_REAL_TYPE, top, 0,
                              a_new + (iy_end * nx), nx, MPI_REAL_TYPE, bottom, 0, MPI_COMM_WORLD,
                              MPI_STATUS_IGNORE));

        //Overlapped Optimization.                   
        CUDA_RT_CALL(cudaStreamSynchronize(push_bottom_stream));
        MPI_CALL(MPI_Sendrecv(a_new + (iy_end - 1) * nx, nx, MPI_REAL_TYPE, bottom, 0, a_new, nx,
                              MPI_REAL_TYPE, top, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

        if (calculate_norm) {
            CUDA_RT_CALL(cudaStreamSynchronize(compute_stream));
            MPI_CALL(MPI_Allreduce(l2_norm_h, &l2_norm, 1, MPI_REAL_TYPE, MPI_SUM, MPI_COMM_WORLD));
            l2_norm = std::sqrt(l2_norm);

            if (0 == rank && (iter % 100) == 0) {
                printf("%5d, %0.6f\n", iter, l2_norm);
            }
        }
        
        std::swap(a_new, a);
        iter++;
    }
    double stop = MPI_Wtime();


    CUDA_RT_CALL(cudaMemcpy(a_h + iy_start_global * nx, a + nx,
                            std::min((ny - iy_start_global) * nx, chunk_size * nx) * sizeof(real),
                            cudaMemcpyDeviceToHost));

    
    CUDA_RT_CALL(cudaEventDestroy(compute_done));
    CUDA_RT_CALL(cudaStreamDestroy(compute_stream));

    CUDA_RT_CALL(cudaFreeHost(l2_norm_h));
    CUDA_RT_CALL(cudaFree(l2_norm_d));

    CUDA_RT_CALL(cudaFree(a_new));
    CUDA_RT_CALL(cudaFree(a));
    return (stop - start);
}

int check(const int rank, const int size, const int nx, const int ny, real* const a_ref_h, 
                real* const a_h,  const double s_elapse, const double p_elapse){
    int result_correct = 1;
    int chunk_size;
    int chunk_size_low = (ny - 2) / size;
    int chunk_size_high = chunk_size_low + 1;
    int num_ranks_low = size * chunk_size_low + size - (ny - 2); 
    if (rank < num_ranks_low)
        chunk_size = chunk_size_low;
    else
        chunk_size = chunk_size_high;

    // Calculate local domain boundaries
    int iy_start_global;  // My start index in the global array
    if (rank < num_ranks_low) {
        iy_start_global = rank * chunk_size_low + 1;
    } else {
        iy_start_global =
            num_ranks_low * chunk_size_low + (rank - num_ranks_low) * chunk_size_high + 1;
    }
    int iy_end_global = iy_start_global + chunk_size - 1;  // My last index in the global array


    for (int iy = iy_start_global; result_correct && (iy < iy_end_global); ++iy) {
        for (int ix = 1; result_correct && (ix < (nx - 1)); ++ix) {
            if (std::fabs(a_ref_h[iy * nx + ix] - a_h[iy * nx + ix]) > tol) {
                fprintf(stderr,
                        "ERROR on rank %d: a[%d * %d + %d] = %f does not match %f "
                        "(reference)\n",
                        rank, iy, nx, ix, a_h[iy * nx + ix], a_ref_h[iy * nx + ix]);
                result_correct = 0;
            }
        }
    }

    int global_result_correct = 1;
    MPI_CALL(MPI_Allreduce(&result_correct, &global_result_correct, 1, MPI_INT, MPI_MIN,
                           MPI_COMM_WORLD));
    result_correct = global_result_correct;

    if (rank == 0 && result_correct) {
        printf("Num GPUs: %d.\n", size);
        printf( 
            "%dx%d: 1 GPU: %8.4f s, %d GPUs: %8.4f s, speedup: %8.2f, "
            "efficiency: %8.2f \n",
            ny, nx, s_elapse, size, p_elapse, s_elapse / p_elapse,
            s_elapse / (size * p_elapse) * 100);   
    }
    return result_correct;
}

int main(int argc, char* argv[]) {

    int size, rank;

    MPI_CALL(MPI_Init(&argc, &argv));
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));


    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 200);
    const int nccheck = get_argval<int>(argv, argv + argc, "-nccheck", 1);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 4096);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 4096);

    int local_rank = -1;
    int iy_start_global = rank * ((ny-2)/size)+1;
    {
        MPI_Comm local_comm;
        MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                     &local_comm));

        MPI_CALL(MPI_Comm_rank(local_comm, &local_rank));

        MPI_CALL(MPI_Comm_free(&local_comm));
    }

    int num_devices = 0;
    // TODO: Get the available GPU devices into `num_devices` and use it and the current rank to set the active GPU.
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));
    CUDA_RT_CALL(cudaSetDevice(local_rank%num_devices));
    //TODO: Init device.
    CUDA_RT_CALL(cudaFree(0));

    printf("My rank is %d of %d, local: %d\n",rank,size,local_rank);

    real* a_ref_h;
    CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * sizeof(real)));
    real* a_h;
    CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * sizeof(real)));
    
    double runtime_serial = serial_impl(nx, ny, iter_max, a_ref_h, nccheck, (0 == rank));
    double runtime_parallel = parallel_impl(rank, size, nx, ny, iter_max, a_h, nccheck, (0 == rank));

    int result_correct = check(rank, size, nx, ny, a_ref_h, a_h, runtime_serial,runtime_parallel);
    

    CUDA_RT_CALL(cudaFreeHost(a_h));
    CUDA_RT_CALL(cudaFreeHost(a_ref_h));

    MPI_CALL(MPI_Finalize());
    return (result_correct == 1) ? 0 : 1;
}



