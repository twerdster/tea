objects = build/errors.o build/histograms.o build/entropies.o build/utilities.o build/TreeBuilder.o


pre-setup:
	mkdir -p build

Tea: pre-setup $(objects)
	nvcc -arch=sm_30 $(objects) -lboost_thread -lboost_system -o Tea

build/%.o: %.cu
	nvcc -x cu -arch=sm_30 -I. -dc $< -o $@

clean:
	rm -f build/*.o Tea