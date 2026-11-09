#include "multi_bench.h"

namespace multi_bench {

Experiment::Experiment(const Config& c)
    : config(c), owner("res", {c.dimension, c.dimension}, Format{Dense, Dense}) {
  const int n = c.dimension;
  Tensor<double> tensor("A", {n, n, n}, Format{Dense, Dense, Dense});
  Tensor<double> vector("B", {n}, Format{Dense});
  util::fillTensor(tensor, util::FillMethod::Random, c.density);
  util::fillTensor(vector, util::FillMethod::Random);
  A = castToTypeZero<float>("A", tensor);
  B = castToTypeZero<float>("B", vector);
  size_t nonzeros = 0;
  for (const auto& entry : A) nonzeros += entry.second != 0;
  actualDensity = double(nonzeros) / (double(n) * n * n);
  expression = A(i, j, k) * B(k);
}

}

int main(int argc, char** argv) {
  using namespace taco;
  try {
    const auto catalog = multi_bench::interfaceCatalog();
    if (multi_bench::printHelp(argc, argv, catalog, false)) return 0;
    for (int index = 1; index < argc; ++index) {
      if (std::string(argv[index]) == "--benchmark_help") {
        argv[index] = const_cast<char*>("--help");
        benchmark::Initialize(&argc, argv);
        return 0;
      }
    }
    const auto config = multi_bench::readConfig("ttv", argc, argv);
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    const auto interfaces = multi_bench::selectInterfaces(config.interfaces, catalog);
    benchmark::AddCustomContext("input_generator", "taco::util::fillTensor");
    benchmark::AddCustomContext("timing_scope", "host end-to-end packed call");
    auto data = std::make_shared<multi_bench::Experiment>(config);
    data->owner(data->resultIndices) = data->expression;
    data->original = makeReductionNotation(data->owner.getAssignment());
    data->schedules = data->original.autoAccelerate(data->original, interfaces, config.constantFolding);
    return multi_bench::runBenchmarks(data);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
