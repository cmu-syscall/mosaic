#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <sstream>
#include <vector>

#include "bench.h"
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/mkl_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"

using namespace taco;

namespace {

constexpr int kDimension = 250;

struct GemmExperiment {
  const int dimension = kDimension;
  Tensor<float> A{"A", {dimension, dimension}, Format{Dense, Dense}};
  Tensor<float> B{"B", {dimension, dimension}, Format{Dense, Dense}};
  Tensor<float> scheduleOwner{"res", {dimension, dimension}, Format{Dense, Dense}};
  IndexVar i{"i"}, j{"j"}, k{"k"};
  IndexExpr expression = A(i, k) * B(k, j);
  IndexStmt original;
  std::vector<IndexStmt> schedules;
  std::vector<bool> printed;
  std::vector<double> expected;
  bool inputsReady = false;

  GemmExperiment() : expected(kDimension * kDimension, 0.0) {
    scheduleOwner(i, j) = expression;
    original = makeReductionNotation(scheduleOwner.getAssignment());
    std::vector<FunctionInterface> interfaces = {
        FunctionInterface(new TblisMultiply()),
        FunctionInterface(new MatrixMultiply()),
        FunctionInterface(new Sgemm())};
    schedules = original.autoAccelerate(original, interfaces);
    printed.assign(schedules.size(), false);
  }

  // Registration needs tensor shapes, but only selected cases need input data.
  void prepareInputs() {
    if (inputsReady) {
      return;
    }
    for (int row = 0; row < dimension; ++row) {
      for (int column = 0; column < dimension; ++column) {
        A.insert({row, column}, static_cast<float>(row + 2 * column));
        B.insert({row, column}, static_cast<float>(3 * row + column));
      }
    }
    A.pack();
    B.pack();
    // Independent double-precision reference, outside the timed region.
    for (int row = 0; row < dimension; ++row) {
      for (int column = 0; column < dimension; ++column) {
        for (int inner = 0; inner < dimension; ++inner) {
          expected[row * dimension + column] +=
              double(row + 2 * inner) * double(3 * inner + column);
        }
      }
    }
    inputsReady = true;
  }
};

// Registration and every repetition share this one autoAccelerate result.
GemmExperiment& experiment() {
  static GemmExperiment instance;
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

bool resultIsCorrect(Tensor<float>& result, const GemmExperiment& data) {
  size_t count = 0;
  for (const auto& entry : result) {
    const float actual = entry.second;
    const double expected =
        data.expected[entry.first[0] * data.dimension + entry.first[1]];
    const double tolerance = 1e-4 * std::max(1.0, std::abs(expected));
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
      return false;
    }
    ++count;
  }
  return count == data.expected.size();
}

void bench_gemm_multi_250_all_schedules(benchmark::State& state) {
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
    Tensor<float> result("res", {data.dimension, data.dimension}, Format{Dense, Dense});
    result(data.i, data.j) = data.expression;
    result.compile(schedule);
    result.assemble();
    auto arguments = result.compute_split();
    auto function = result.returnFuncPackedRaw(arguments);

    if (!data.printed[index]) {
      std::cout << "\n\n========== SCHEDULE " << index << " ==========\n\n"
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
      state.SkipWithError("Generated schedule produced an incorrect GEMM result");
    }
  } catch (const std::exception& error) {
    state.SkipWithError(error.what());
  }
}

TACO_BENCH(bench_gemm_multi_250_all_schedules)->Apply(scheduleArguments);

}  // namespace
