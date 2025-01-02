# Makefile for vecadd
CUDA_HOME ?= /usr/local/cuda-13.0
NVCC      ?= $(CUDA_HOME)/bin/nvcc
ARCH      ?= sm_89
CXXFLAGS  ?= -O3 -Xcompiler -fopenmp
LDFLAGS   ?=

all: vecadd

vecadd: vecadd.cu
	$(NVCC) $(CXXFLAGS) -arch=$(ARCH) $< -o $@ $(LDFLAGS)

clean:
	rm -f vecadd
