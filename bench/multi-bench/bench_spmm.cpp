#include <set>
#include <memory>
#include <string>
#include <vector>
#include "benchmark/benchmark.h"
#include "taco/tensor.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/mkl_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/storage/file_io_mtx.h"
#include <fstream>
#include "taco/util/env.h"

using namespace taco;

namespace multi_bench {

struct Config {
  std::string kernel, input, matrix, backends;
  int dimension, seed;
  double density;
};

struct Experiment {
  Config config;
  taco::Tensor<float> A, B, mask, owner;
  taco::IndexVar i{"i"}, j{"j"}, k{"k"};
  taco::IndexExpr expression;
  taco::IndexStmt original;
  std::vector<taco::IndexStmt> schedules;
  std::vector<bool> printed;
  std::vector<double> expected;
  double actualDensity;

  Experiment(const Config& config, bool sparseA);
  void enumerate();
};

bool failed = false;

Experiment::Experiment(const Config& c, bool sparseA)
    : config(c), A("A", {c.dimension, c.dimension}, sparseA ? CSR : Format{Dense, Dense}),
      B("B", {c.dimension, c.dimension}, Format{Dense, Dense}),
      mask("mask", {c.dimension, c.dimension}, CSR),
      owner("res", {c.dimension, c.dimension}, Format{Dense, Dense}),
      expected(size_t(c.dimension) * c.dimension, 0.0) {
  const int n = c.dimension;
  std::mt19937 random(c.seed);
  std::uniform_real_distribution<float> values(0.1f, 1.0f);
  std::bernoulli_distribution populated(c.density);
  std::vector<float> selected(size_t(n) * n, 0.0f);
  if (c.input == "matrix-market") {
    auto loaded = readMTX(c.matrix, CSR);
    if (loaded.getDimensions() != std::vector<int>{n, n})
      throw std::runtime_error("Matrix dimensions do not match configuration");
    for (const auto& entry : Tensor<double>(loaded)) {
      const float value = static_cast<float>(entry.second);
      if (!std::isfinite(value)) throw std::runtime_error("Input contains non-finite values");
      selected[size_t(entry.first[0]) * n + entry.first[1]] = value;
    }
  } else {
    for (auto& value : selected) if (populated(random)) value = values(random);
  }
  size_t nonzeros = 0;
  std::vector<float> left(selected.size()), right(selected.size());
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      const size_t p = size_t(row) * n + col;
      nonzeros += selected[p] != 0;
      left[p] = c.kernel == "sddmm" ? values(random) : selected[p];
      right[p] = values(random);
      if (!sparseA || left[p] != 0) A.insert({row, col}, left[p]);
      B.insert({row, col}, right[p]);
      if (c.kernel == "sddmm" && selected[p] != 0) mask.insert({row, col}, selected[p]);
    }
  }
  actualDensity = double(nonzeros) / selected.size();
  A.pack(); B.pack(); mask.pack();
  for (int row = 0; row < n; ++row) {
    for (int inner = 0; inner < n; ++inner) {
      const double a = left[size_t(row) * n + inner];
      if (a == 0) continue;
      for (int col = 0; col < n; ++col)
        expected[size_t(row) * n + col] += a * right[size_t(inner) * n + col];
    }
  }
  if (c.kernel == "sddmm")
    for (size_t p = 0; p < expected.size(); ++p) expected[p] *= selected[p];
}

void Experiment::enumerate() {
  owner(i, j) = expression;
  original = makeReductionNotation(owner.getAssignment());
  std::vector<FunctionInterface> interfaces;
  std::istringstream names(config.backends);
  std::string name;
  std::set<std::string> selected;
  while (std::getline(names, name, ',')) {
    if (!selected.insert(name).second) throw std::runtime_error("Duplicate backend");
    if (name != "taco" && name != "mkl") throw std::runtime_error("SpMM supports taco,mkl");
    if (name == "tblis") interfaces.push_back(new TblisMultiply());
    else if (name == "mkl") {
      if (config.kernel == "spmm") interfaces.push_back(new SparseMklMM());
      else interfaces.push_back(new MatrixMultiply());
    } else if (name == "blas") interfaces.push_back(new MatrixMultiply());
    else if (name != "taco") throw std::runtime_error("Unsupported backend: " + name);
  }
  if (selected.empty() || config.backends.back() == ',') throw std::runtime_error("Empty backend");
  if (selected.count("mkl") && selected.count("blas")) throw std::runtime_error("Select only one BLAS provider");
  schedules = original.autoAccelerate(original, interfaces);
  const auto baseline = makeConcreteNotation(original);
  if (std::none_of(schedules.begin(), schedules.end(), [&](IndexStmt s) { return equals(s, baseline); }))
    schedules.insert(schedules.begin(), baseline);
  printed.assign(schedules.size(), false);
  std::cout << "Kernel: " << config.kernel << "\nDimension: " << config.dimension
            << "\nActual density: " << actualDensity << "\nSchedules: " << schedules.size() << "\n";
}

bool saveSource(size_t index, const std::string& source) {
  if (source.empty()) return true;
  std::ofstream file("code/schedule" + std::to_string(index) + ".c");
  file << source;
  file.close();
  return bool(file);
}

void run(benchmark::State& state, Experiment& data) {
  const size_t index = static_cast<size_t>(state.range(0));
  const auto schedule = data.schedules.at(index);
  std::ostringstream label;
  label << schedule;
  state.SetLabel(label.str());
  state.counters["dimension"] = data.config.dimension;
  state.counters["density"] = data.actualDensity;
  Tensor<float> result("res", {data.config.dimension, data.config.dimension}, Format{Dense, Dense});
  const bool print = !data.printed[index];
  if (print) {
    std::cout << "\n\n========== SCHEDULE " << index << " ==========\n\n"
              << "----- Original Schedule -----\n\n" << data.original
              << "\n\n----- Bound Schedule -----\n\n" << schedule << "\n\n";
    data.printed[index] = true;
  }
  try {
    result(data.i, data.j) = data.expression;
    const auto bound = replace(schedule, std::map<TensorVar, TensorVar>{
        {data.owner.getTensorVar(), result.getTensorVar()}});
    result.compile(bound);
    if (print && !saveSource(index, result.getSource()))
      throw std::runtime_error("Could not write generated C to code/");
    result.assemble();
    auto arguments = result.compute_split();
    auto function = result.returnFuncPackedRaw(arguments);
    if (print) {
      std::cout << "----- Generated C -----\n\n" << result.getSource()
                << "\n\n========== END SCHEDULE " << index << " ==========\n";
    }
    int status = 0;
    for (auto _ : state) status |= function.first(arguments.data());
    if (status != 0) { failed = true; state.SkipWithError("Kernel returned a nonzero status"); return; }
    size_t count = 0;
    for (const auto& entry : result) {
      const double expected = data.expected[size_t(entry.first[0]) * data.config.dimension + entry.first[1]];
      if (!std::isfinite(entry.second) || std::abs(entry.second - expected) > 1e-4 * std::max(1.0, std::abs(expected))) {
        failed = true; state.SkipWithError("Numerical validation failed"); return;
      }
      ++count;
    }
    if (count != data.expected.size()) { failed = true; state.SkipWithError("Incomplete output"); }
  } catch (const std::exception& e) {
    if (print && !saveSource(index, result.getSource()))
      std::cerr << "Could not write generated C to code/\n";
    if (print) std::cout << "----- Generated C (if available) -----\n\n" << result.getSource()
                        << "\n\n----- Failure -----\n\n" << e.what()
                        << "\n\n========== END SCHEDULE " << index << " ==========\n";
    failed = true; state.SkipWithError(e.what());
  }
}
}

int main(int argc, char** argv) {
  try {
    using taco::util::getFromEnv;
    multi_bench::Config config{
        "spmm", getFromEnv("MULTI_INPUT", "synthetic"),
        getFromEnv("MULTI_MATRIX", ""), getFromEnv("MULTI_BACKENDS", "taco"),
        std::stoi(getFromEnv("MULTI_DIM", "250")), std::stoi(getFromEnv("MULTI_SEED", "1")),
        std::stod(getFromEnv("MULTI_DENSITY", "1"))};
    if (config.dimension <= 0 || !(config.density >= 0 && config.density <= 1))
      throw std::runtime_error("Invalid dimension or density");
    if (config.input != "synthetic" && config.input != "matrix-market")
      throw std::runtime_error("Unsupported input mode");
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    if (config.seed < 0) throw std::runtime_error("SEED must be nonnegative");
    if (config.input == "matrix-market") {
      std::ifstream file(config.matrix);
      std::string line, banner, object, storage, type, symmetry;
      std::getline(file, line);
      std::istringstream fields(line);
      fields >> banner >> object >> storage >> type >> symmetry;
      if (banner != "%%MatrixMarket" || object != "matrix" ||
          (storage != "coordinate" && storage != "array") || type != "real" ||
          (symmetry != "general" && symmetry != "symmetric"))
        throw std::runtime_error("Expected a real general/symmetric Matrix Market matrix");
      do { if (!std::getline(file, line)) throw std::runtime_error("Missing matrix dimensions"); }
      while (line.empty() || line[0] == '%');
      int rows = 0, cols = 0;
      std::istringstream(line) >> rows >> cols;
      if (rows <= 0 || rows != cols) throw std::runtime_error("Expected a nonempty square matrix");
      config.dimension = rows;
    }
    auto data = std::make_shared<multi_bench::Experiment>(config, true);
    data->expression = data->A(data->i, data->k) * data->B(data->k, data->j);
    data->enumerate();
    const std::string name = "bench_multi_" + config.kernel;
    benchmark::RegisterBenchmark(name.c_str(), [data](benchmark::State& state) { multi_bench::run(state, *data); })
        ->ArgName("schedule")->DenseRange(0, data->schedules.size() - 1)
        ->Iterations(1)->UseRealTime()->Unit(benchmark::kMillisecond);
    const auto matched = benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return matched && !multi_bench::failed ? 0 : 1;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
