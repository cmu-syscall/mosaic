#include <algorithm>
#include <cmath>
#include <exception>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <vector>

#include "bench.h"
#include "taco/storage/file_io_mtx.h"
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/mkl_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"

using namespace taco;

namespace {

constexpr int kDimension = 1806;

struct GemvExperiment {
  const int dimension = kDimension;
  Tensor<float> A{"A", {dimension, dimension}, Format{Dense, Dense}};
  Tensor<float> b{"b", {dimension}, Format{Dense}};
  Tensor<float> scheduleOwner{"res", {dimension}, Format{Dense}};
  IndexVar i{"i"}, j{"j"};
  IndexExpr expression = A(i, j) * b(j);
  IndexStmt original;
  std::vector<IndexStmt> schedules;
  std::vector<bool> printed;
  std::vector<double> expected;
  std::vector<double> magnitude;
  bool inputsReady = false;

  GemvExperiment() : expected(kDimension, 0.0), magnitude(kDimension, 0.0) {
    scheduleOwner(i) = expression;
    original = makeReductionNotation(scheduleOwner.getAssignment());
    std::vector<FunctionInterface> interfaces = {
        FunctionInterface(new TblisGemv()),
        FunctionInterface(new MklSgemv()),
        FunctionInterface(new CblasGemv()),
        FunctionInterface(new Sgemv())};
    schedules = original.autoAccelerate(original, interfaces);
    printed.assign(schedules.size(), false);
  }

  // Registration needs tensor shapes, but only selected cases need input data.
  void prepareInputs() {
    if (inputsReady) {
      return;
    }
    const char* root = std::getenv("PATH_TO_MOSAIC_ARTIFACT");
    if (!root) throw std::runtime_error("Source the benchmark library environment first");
    const std::string path = std::string(root) +
        "/local-tests/suitesparse/HB/bcsstk14/bcsstk14.mtx";
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Missing SuiteSparse matrix: " + path);
    Tensor<double> source = readMTX(input, CSR);
    if (source.getDimensions() != std::vector<int>{dimension, dimension}) {
      throw std::runtime_error("Unexpected SuiteSparse matrix dimensions");
    }
    // TACO expands Matrix Market symmetry; only the final timed A is dense.
    std::vector<float> vectorValues(dimension, 0.0f);
    for (const auto& entry : source) {
      if (entry.first[1] == 0) vectorValues[entry.first[0]] = float(entry.second);
    }
    bool nonzeroVector = false;
    for (int row = 0; row < dimension; ++row) {
      if (!std::isfinite(vectorValues[row])) throw std::runtime_error("Nonfinite vector value");
      nonzeroVector = nonzeroVector || vectorValues[row] != 0;
      b.insert({row}, vectorValues[row]);
    }
    if (!nonzeroVector) throw std::runtime_error("SuiteSparse first column is zero");
    for (const auto& entry : source) {
      const int row = entry.first[0], column = entry.first[1];
      const float value = float(entry.second);
      if (!std::isfinite(value)) throw std::runtime_error("Nonfinite matrix value");
      A.insert({row, column}, value);
      const double product = double(value) * double(vectorValues[column]);
      expected[row] += product;
      magnitude[row] += std::abs(product);
    }
    A.pack();
    b.pack();
    inputsReady = true;
  }
};

// Registration and every repetition share this one autoAccelerate result.
GemvExperiment& experiment() {
  static GemvExperiment instance;
  return instance;
}

void scheduleArguments(benchmark::internal::Benchmark* benchmark) {
  const auto count = experiment().schedules.size();
  benchmark->ArgName("schedule");
  if (count == 0) {
    // Keep an empty search visible as an error rather than registering no tests.
    benchmark->Arg(0);
  } else {
    benchmark->DenseRange(0, static_cast<int64_t>(count) - 1);
  }
}

bool resultIsCorrect(Tensor<float>& result, const GemvExperiment& data) {
  for (int row = 0; row < data.dimension; ++row) {
    const float actual = result.at({row});
    // A componentwise scale handles cancellation in real matrix data.
    const double tolerance = 1e-4 * std::max(1.0, data.magnitude[row]);
    if (!std::isfinite(actual) ||
        std::abs(actual - data.expected[row]) > tolerance) {
      return false;
    }
  }
  return true;
}

void bench_gemv_multi_bcsstk14_all_schedules(benchmark::State& state) {
  auto& data = experiment();
  if (data.schedules.empty()) {
    state.SkipWithError("autoAccelerate returned no schedules");
    return;
  }
  const auto index = static_cast<size_t>(state.range(0));
  const IndexStmt& schedule = data.schedules.at(index);
  std::ostringstream label;
  label << schedule;
  state.SetLabel(label.str());
  state.counters["dimension"] = data.dimension;

  try {
    data.prepareInputs();
    // A fresh output per repetition; the stored schedule is not rewritten.
    Tensor<float> result("res", {data.dimension}, Format{Dense});
    result(data.i) = data.expression;
    result.compile(schedule);
    result.assemble();
    auto arguments = result.compute_split();
    auto function = result.returnFuncPackedRaw(arguments);

    if (!data.printed[index]) {
      std::cout << "\n\n========== SCHEDULE " << index << " ==========\n\n"
                << "Dataset: SuiteSparse HB/bcsstk14; b = first column; dense float storage\n"
                << "Dimension: " << data.dimension << "\n\n"
                << "----- Original Schedule -----\n\n" << data.original << "\n\n\n"
                << "----- Bound Schedule -----\n\n" << schedule << "\n\n\n"
                << "----- Generated C -----\n\n" << result.getSource()
                << "\n\n========== END SCHEDULE " << index << " ==========\n\n";
      data.printed[index] = true;
    }

    // Google Benchmark times only this loop, not setup or validation.
    int status = 0;
    for (auto _ : state) {
      status = function.first(arguments.data());
    }

    if (status != 0) {
      state.SkipWithError("Generated schedule returned a nonzero status");
    } else if (!resultIsCorrect(result, data)) {
      state.SkipWithError("Generated schedule produced an incorrect GEMV result");
    }
  } catch (const std::exception& error) {
    state.SkipWithError(error.what());
  }
}

TACO_BENCH(bench_gemv_multi_bcsstk14_all_schedules)->Apply(scheduleArguments);

}  // namespace
