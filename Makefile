# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
NP ?= 4
NVCC=/usr/local/cuda-11.3/bin/nvcc
JSC_SUBMIT_CMD ?= /home/v-liku/openmpi/bin/mpirun --gres=gpu:4 --ntasks-per-node 4
CUDA_HOME ?= /usr/local/cuda-11.3
MPI_HOME = /home/v-liku/openmpi
MPICXX= /home/v-liku/openmpi/bin/mpicxx
ifndef MPI_HOME
$(error MPI_HOME is not set)
endif
GENCODE_SM30	:= -gencode arch=compute_30,code=sm_30
GENCODE_SM35	:= -gencode arch=compute_35,code=sm_35
GENCODE_SM37	:= -gencode arch=compute_37,code=sm_37
GENCODE_SM50	:= -gencode arch=compute_50,code=sm_50
GENCODE_SM52	:= -gencode arch=compute_52,code=sm_52
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM60) $(GENCODE_SM70) $(GENCODE_SM80)

NVCC_FLAGS = -Xptxas --optimize-float-atomics
NVCC_FLAGS += -dc -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -lnvToolsExt $(GENCODE_FLAGS) -std=c++14  -I$(MPI_HOME)/include
NVCC_LDFLAGS = -ccbin=mpic++ -L$(NVSHMEM_HOME)  -L$(MPI_HOME)/lib -lmpi -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lnvToolsExt
LD_FLAGS = -L$(CUDA_HOME)/lib64 -lcudart -lnvToolsExt -lnccl
MPICXX_FLAGS = -DUSE_NVTX -I$(CUDA_HOME)/include -std=c++14
jacobi: Makefile jacobi.cpp jacobi_kernels.o
	$(MPICXX) $(MPICXX_FLAGS) jacobi.cpp jacobi_kernels.o $(LD_FLAGS) -o jacobi

jacobi_kernels.o: Makefile jacobi_kernels.cu
	$(NVCC) $(NVCC_FLAGS) jacobi_kernels.cu -c

.PHONY.: clean
clean:
	rm -f jacobi jacobi_kernels.o *.nsys-rep jacobi.*.compute-sanitizer.log

run: jacobi
	$(JSC_SUBMIT_CMD) -n $(NP) ./jacobi
