objects = build/histograms.o build/entropies.o build/utilities.o build/TreeBuilder.o
NVCC ?= nvcc
CUDA_ARCH ?= sm_86
OPENMP_FLAGS ?= -Xcompiler -fopenmp
NVCCFLAGS ?= -std=c++11 -arch=$(CUDA_ARCH) $(OPENMP_FLAGS)
PYTHON ?= python3

pre-setup:
	mkdir -p build

Tea: pre-setup $(objects)
	${NVCC} $(NVCCFLAGS) $(objects) -o Tea

build/%.o: %.cu 
	${NVCC} $(NVCCFLAGS) -x cu -I. -dc $< -o $@

smoke: Tea
	${PYTHON} tests/run_smoke_tests.py --tea ./Tea

help:
	@echo "Targets:"
	@echo "  make Tea                 Build the Tea binary"
	@echo "  make smoke               Run the depth-0 and depth-1 smoke tests"
	@echo ""
	@echo "Variables:"
	@echo "  NVCC=<path>              CUDA compiler to use"
	@echo "  CUDA_ARCH=sm_86          CUDA architecture to target"
	@echo "  OPENMP_FLAGS=...         Host OpenMP flags, or empty to disable"

clean:
	rm -f build/*.o Tea

.PHONY: Tea clean smoke help
