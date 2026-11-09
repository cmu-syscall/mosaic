#include "multi_bench.h"

namespace multi_bench {

Experiment::Experiment(const Config& c)
    : config(c), owner("res", {c.dimension, c.dimension, c.dimension, c.dimension},
                       Format{Dense, Dense, Dense, Dense}) {
  const int n = c.dimension;
  A = Tensor<float>("A", {n, n, n, n}, Format{Sparse, Dense, Sparse, Dense});
  B = Tensor<float>("B", {n, n, n, n}, Format{Dense, Dense, Dense, Dense});
  const auto blocks = randomMatrix("blocks", n, CSR, c.density);
  size_t nonzeroBlocks = 0;
  for (const auto& block : blocks) {
    if (block.second == 0) continue;
    ++nonzeroBlocks;
    for (int row = 0; row < n; ++row)
      for (int column = 0; column < n; ++column)
        A.insert({block.first[0], row, block.first[1], column}, 100.0f);
  }
  for (int row = 0; row < n; ++row)
    for (int innerRow = 0; innerRow < n; ++innerRow)
      for (int column = 0; column < n; ++column)
        for (int innerColumn = 0; innerColumn < n; ++innerColumn)
          B.insert({row, innerRow, column, innerColumn}, 100.0f);
  A.pack();
  B.pack();
  actualDensity = double(nonzeroBlocks) / (double(n) * n);
  IndexVar l("l"), m("m"), p("p");
  resultIndices = {i, k, m, p};
  expression = A(i, k, j, l) * B(j, l, m, p);
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
    const auto config = multi_bench::readConfig("blockspmm", argc, argv);
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    const auto interfaces = multi_bench::selectInterfaces(config.interfaces, catalog);
    benchmark::AddCustomContext("input_generator", "taco block mask with uniform dense blocks");
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
