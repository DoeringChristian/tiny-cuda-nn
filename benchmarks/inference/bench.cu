
#include "pcg32/pcg32.h"
#include "tiny-cuda-nn/object.h"
#include <tiny-cuda-nn/common_device.h>

#include <tiny-cuda-nn/config.h>

#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/random.h>

#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/networks/fully_fused_mlp.h>
#include <tiny-cuda-nn/optimizer.h>

#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/trainer.h>

#include <fmt/core.h>

#include <stbi/stbi_wrapper.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace tcnn;
using precision_t = network_precision_t;

int main() {

  using namespace tcnn;

  const auto width = 64;
  auto in_width = 64;
  auto out_width = 64;
  auto hidden_layers = 2;
  auto n_iters = 1000;

  std::vector<double> throughput_log;
  std::vector<uint32_t> batch_size_log;

  std::vector<uint32_t> batch_sizes = {1 << 14, 1 << 15, 1 << 16, 1 << 17,
                                       1 << 18, 1 << 19, 1 << 20, 1 << 21};
  // std::vector<uint32_t> batch_sizes = {1 << 21};

  for (uint32_t batch_size : batch_sizes) {
    GPUMatrixDynamic<network_precision_t> input(in_width, batch_size, CM);
    GPUMatrixDynamic<network_precision_t> output(out_width, batch_size, RM);

    auto network =
        std::make_shared<tcnn::FullyFusedMLP<network_precision_t, width>>(
            in_width, out_width, hidden_layers, Activation::ReLU,
            Activation::None);

    pcg32 rng{0};
    // network->initialize_params(rng, 1.0);

    // Initialize parameters
    auto n_params = network->n_params();

    GPUMemory<char> params;

    params.resize(sizeof(network_precision_t) * n_params * 2 +
                  sizeof(float) * n_params * 1);
    // network->initialize_params(rng, (float *)params.data());
    network->set_params((network_precision_t *)params.data(),
                        (network_precision_t *)params.data(), nullptr);

    // Benchmark:
    auto start = std::chrono::steady_clock::now();
    for (uint i = 0; i < n_iters; i++) {
      network->inference_mixed_precision(input, output, false);
      // network->inference(&input, &output, true);
    }
    auto end = std::chrono::steady_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    double throughput = n_iters * batch_size / ((double)duration / 1'000'000.0);
    std::cout << " Batch Size: " << batch_size << " duration: " << duration
              << " us"
              << " throughput" << throughput << "/s" << std::endl;

    batch_size_log.push_back(batch_size);
    throughput_log.push_back(throughput);
  }

  json j;
  j["throughputs"] = throughput_log;
  j["batch_sizes"] = batch_size_log;
  std::ofstream o("data/out/tcnn.json");
  o << j << std::endl;

  return 0;
}
