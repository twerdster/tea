objects = build/histograms.o build/entropies.o build/utilities.o build/TreeBuilder.o
NVCC=/usr/local/cuda/bin/nvcc

pre-setup:
	mkdir -p build

Tea: pre-setup $(objects)
	${NVCC} -std=c++11 -arch=sm_30 $(objects) -lboost_thread -lboost_system -o Tea

build/%.o: %.cu 
	${NVCC} -std=c++11 -x cu -arch=sm_30 -I. -dc $< -o $@

clean:
	rm -f build/*.o Tea

