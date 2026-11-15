// Independent count/scatter/partition checks against straightforward CPU loops.
#include "GPUDevice.h"
#include "DoubleSort.h"
#include "errors.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
static void partitions() {
    for (int n = 0; n <= 12; ++n) {
        for (unsigned mask = 0; mask < (1u << n); ++mask) {
            std::vector<Sample> samples(n + 4);
            std::vector<uint> indices(n + 4);
            std::iota(indices.begin(), indices.end(), 0);
            int expected = 2;
            for (int i = 0; i < n; ++i) {
                samples[i+2].bestTraj = (mask >> i) & 1;
                samples[i+2].label = i;
                if (!samples[i+2].bestTraj) ++expected;
            }
            int split = partition(samples.data(), indices.data(), 2, n+1);
            require(split == expected, "CPU partition has wrong boundary");
            for (int i = 2; i < n+2; ++i) {
                require(samples[i].bestTraj == (i >= split), "CPU partition misroutes sample");
                require(samples[i].label + 2 == indices[i], "CPU partition loses paired index");
            }
        }
    }
}

template<class T> static void histograms(int classes, HistType method) {
    const int n = 101;
    GPUDevice device;
    device.init(0, n, 7, 2, classes, 2);
    std::vector<Sample> samples(n);
    std::vector<T> values(n);
    std::vector<uint> permutation(n);
    for (int i = 0; i < n; ++i) {
        samples[i].label = (i*13 + 1) % classes;
        values[i] = T(i % 7 - 3);
        permutation[i] = n-i-1;
    }
    device.pushSamples(samples.data());
    std::vector<RFTrainNode> nodes(7);
    for (int i = 0; i < 7; ++i) { nodes[i].idxBegin = 0; nodes[i].idxEnd = n-1; }
    const int begins[] = {0, 3, 3, 54};
    const int ends[] = {2, 2, 53, 100};
    for (int i = 0; i < 4; ++i) {
        nodes[i+3].idxBegin = begins[i]; nodes[i+3].idxEnd = ends[i];
    }
    device.pushNodes(nodes.data(), 0, 7);
    T *gpu_values, *gpu_input, *gpu_gather;
    uint *gpu_indices;
    teaCudaCheck(cudaMalloc(&gpu_values, n*sizeof(T)), "test values");
    teaCudaCheck(cudaMalloc(&gpu_input, n*sizeof(T)), "test input");
    teaCudaCheck(cudaMalloc(&gpu_gather, n*sizeof(T)), "test gather");
    teaCudaCheck(cudaMalloc(&gpu_indices, n*sizeof(uint)), "test indices");
    teaCudaCheck(cudaMemcpy(gpu_input, values.data(), n*sizeof(T), cudaMemcpyHostToDevice), "test upload");
    teaCudaCheck(cudaMemcpy(gpu_indices, permutation.data(), n*sizeof(uint), cudaMemcpyHostToDevice), "test upload indices");
    gpuScatteredWrite(gpu_values, gpu_input, gpu_indices, n, 0);
    gpuScatteredRead(gpu_gather, gpu_values, gpu_indices, n, 0);
    std::vector<T> got(n);
    teaCudaCheck(cudaMemcpy(got.data(), gpu_gather, n*sizeof(T), cudaMemcpyDeviceToHost), "test gather read");
    require(got == values, "scatter/gather loses odd tail");
    teaCudaCheck(cudaMemcpy(gpu_values, values.data(), n*sizeof(T), cudaMemcpyHostToDevice), "test upload values");
    for (NodeType type : {PARENT_NODE, CHILD_NODE}) {
        for (float threshold : {-1.125f, 0.0f, 4.0f}) {
            buildHistogram(type, method, threshold, gpu_values, &device, nodes.data(), 3, 6, 0);
            std::vector<uint> cpu(4*classes), gpu(4*classes);
            for (int node = 0; node < 4; ++node)
                for (int i = begins[node]; i <= ends[node]; ++i)
                    if (type == PARENT_NODE || float(values[i]) > threshold)
                        ++cpu[node*classes + samples[i].label];
            uint *src = device.histograms() + (type == CHILD_NODE ? device.histWords()/2 : 0);
            teaCudaCheck(cudaMemcpy(gpu.data(), src, gpu.size()*sizeof(uint), cudaMemcpyDeviceToHost), "test histogram read");
            require(cpu == gpu, "GPU histogram differs from CPU count on odd/empty nodes");
        }
        // Empty folded batch follows a nonempty batch: stale counts are forbidden.
        buildHistogram(type, method, 0.0f, gpu_values, &device, nodes.data(), 4, 4, 0);
        std::vector<uint> gpu(classes);
        uint *src = device.histograms() + (type == CHILD_NODE ? device.histWords()/2 : 0);
        teaCudaCheck(cudaMemcpy(gpu.data(), src, gpu.size()*sizeof(uint), cudaMemcpyDeviceToHost), "empty histogram read");
        require(std::accumulate(gpu.begin(), gpu.end(), 0u) == 0, "empty batch retains stale counts");
    }
    cudaFree(gpu_values); cudaFree(gpu_input); cudaFree(gpu_gather); cudaFree(gpu_indices);
}
int main() {
    try {
        FILELog::ReportingLevel() = logERROR;
        partitions();
        histograms<char>(8, HIST_64); histograms<short>(8, HIST_64);
        histograms<int>(8, HIST_64); histograms<float>(8, HIST_64);
        histograms<char>(70, HIST_256); histograms<short>(70, HIST_256);
        histograms<int>(70, HIST_256); histograms<float>(70, HIST_256);
        histograms<char>(8, HIST_REGULAR); histograms<short>(257, HIST_REGULAR);
        std::cout << "PASS: exhaustive partitions, odd scatter tails, CPU/GPU histograms (64/256/regular, all feature types)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
