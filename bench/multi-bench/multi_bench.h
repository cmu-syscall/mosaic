#ifndef TACO_MULTI_BENCH_H
#define TACO_MULTI_BENCH_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../bench.h"
#include "taco/util/fill.h"
#include "taco/util/strings.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/storage/file_io_mtx.h"

#include "taco/tensor.h"
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"
#include "taco/accelerator_interface/gsl_interface.h"
#include "taco/accelerator_interface/mkl_interface.h"
#include "taco/accelerator_interface/cuda_interface.h"
#include "taco/accelerator_interface/avx2_interface.h"

namespace multi_bench {

using namespace taco;

struct Config {
  std::string kernel, interfaces;
  int dimension;
  double density;
  bool constantFolding = true;
  std::string matrix;
  std::shared_ptr<Tensor<double>> matrixData;
};

struct Experiment {
  Config config;
  taco::Tensor<float> A, B, mask, owner;
  taco::IndexVar i{"i"}, j{"j"}, k{"k"};
  taco::IndexExpr expression;
  taco::IndexStmt original;
  std::vector<taco::IndexStmt> schedules;
  std::vector<bool> sourceSaved;
  double actualDensity;

  Experiment(const Config& config);
  bool failed = false;
};

class SilentReporter : public benchmark::BenchmarkReporter {
 public:
  bool ReportContext(const Context&) override { return true; }
  void ReportRuns(const std::vector<Run>&) override {}
};

inline Config readConfig(const std::string& kernel, int& argc, char** argv, bool fileInput = false) {
  Config config{kernel, "taco", 250, 1.0};
  int remaining = 1;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto separator = argument.find('=');
    const std::string name = argument.substr(0, separator);
    const std::string value = separator == std::string::npos ? "" : argument.substr(separator + 1);
    size_t parsed = value.size();
    if (name == "--interfaces" || name == "--backends") config.interfaces = value;
    else if (name == "--constant-folding") {
      if (value != "true" && value != "false")
        throw std::runtime_error("Expected --constant-folding=true or false");
      config.constantFolding = value == "true";
    }
    else if (fileInput && name == "--matrix") config.matrix = value;
    else if (!fileInput && name == "--dim") config.dimension = std::stoi(value, &parsed);
    else if (!fileInput && name == "--density") config.density = std::stod(value, &parsed);
    else { argv[remaining++] = argv[index]; continue; }
    if (separator == std::string::npos || value.empty() || parsed != value.size())
      throw std::runtime_error("Expected " + name + "=value");
  }
  argc = remaining;
  argv[argc] = nullptr;
  if (config.dimension <= 0 || !(config.density >= 0 && config.density <= 1))
    throw std::runtime_error("Invalid dimension or density");
  if (fileInput) {
    if (config.matrix.empty()) throw std::runtime_error("Specify --matrix=/path/to/input.mtx");
    config.matrixData = std::make_shared<Tensor<double>>(readMTX(config.matrix, CSR));
    const auto& dimensions = config.matrixData->getDimensions();
    if (dimensions.size() != 2 || dimensions[0] <= 0 || dimensions[0] != dimensions[1])
      throw std::runtime_error("Expected a nonempty square matrix");
    config.dimension = dimensions[0];
  }
  benchmark::AddCustomContext("constant_folding", config.constantFolding ? "true" : "false");
  if (fileInput) benchmark::AddCustomContext("matrix", config.matrix);
  return config;
}

using InterfaceCatalog = std::map<std::string, std::vector<FunctionInterface>>;

inline InterfaceCatalog interfaceCatalog() {
  InterfaceCatalog catalog;
  catalog["tblis"] = {new TblisMultiply(), new TblisTTM(),
                      new TblisSaxpy(), new TblisPlus(), new TblisGemv(), new TblisTTV()};
  catalog["gsl"] = {new GSLVecAdd(), new GSLDot(), new GSLSgemv(), new GSLGemv(), new GSLMM()};
  catalog["blas"] = {new Saxpy(), new Sdot(), new Sgemv(), new Sgemm(),
                     new MatrixMultiply(), new CblasGemv()};
  catalog["openblas"] = catalog["blas"];
  catalog["accelerate"] = catalog["blas"];
  catalog["mkl"] = catalog["blas"];
  for (FunctionInterface interface : std::vector<FunctionInterface>{
           new SparseMklMM(), new MklDot()})
    catalog["mkl"].push_back(interface);
  catalog["cublas"] = {new CudaGemm(), new CudaGemv()};
  catalog["cusparse"] = {new CudaSpmv(), new CudaSpmm()};
  catalog["cuda"] = catalog["cublas"];
  catalog["cuda"].insert(catalog["cuda"].end(), catalog["cusparse"].begin(), catalog["cusparse"].end());
  catalog["avx2"] = {new AVXSaxpy()};
  return catalog;
}

inline bool printHelp(int argc, char** argv, const InterfaceCatalog& catalog, bool fileInput) {
  bool help = false, list = false;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    help |= arg == "-h" || arg == "--h" || arg == "--help";
    list |= arg == "--list-interfaces";
  }
  if (!help && !list) return false;
  if (help) {
    std::cout << "Usage: " << argv[0];
    if (fileInput) std::cout << " --matrix=/path/input.mtx";
    else std::cout << " [--dim=250] [--density=1]";
    std::cout << " [--interfaces=taco] [--constant-folding=true] [Google Benchmark flags]\n"
                 "Select library.function or library.*; separate selections with commas.\n"
                 "Example: --interfaces='tblis.tblis_tensor_mult,mkl.cblas_sgemm'\n"
                 "taco selects no external interfaces. A TACO baseline is always included.\n"
                 "--backends remains an alias; bare library names mean library.*.\n"
                 "--list-interfaces lists registered functions without running a benchmark.\n"
                 "Use --benchmark_help for Google Benchmark flags.\n\n";
  }
  for (const auto& library : catalog) {
    std::cout << library.first << ":\n";
    std::set<std::string> names;
    for (const auto& interface : library.second)
      names.insert(interface.getNode()->getFunctionName());
    for (const auto& name : names) std::cout << "  " << name << '\n';
  }
  std::cout << "\nNames come from Mosaic's getFunctionName(), including wrapper names.\n"
               "One name may select several interface variants. Registration does not check installation.\n"
               "BLAS provider aliases use the library configured in your compile/link environment.\n"
               "Symmetric-only, missing-wrapper, and unsafe legacy interfaces are excluded.\n"
               "CUDA requires NVIDIA hardware/toolkit; AVX2 requires x86 AVX2 support and compile flags.\n"
               "Stardust uses the separate simulator workflow; tensor-extension/demo interfaces are excluded.\n";
  return true;
}

inline std::vector<FunctionInterface> selectInterfaces(const std::string& selectors,
                                               const InterfaceCatalog& catalog) {
  std::vector<FunctionInterface> result;
  std::set<const AbstractFunctionInterface*> selected;
  std::set<std::string> providers;
  std::set<std::string> resolved;
  std::istringstream input(selectors);
  std::string selector;
  if (selectors.empty() || selectors.back() == ',')
    throw std::runtime_error("Empty interface selector");
  while (std::getline(input, selector, ',')) {
    const auto first = selector.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) throw std::runtime_error("Empty interface selector");
    selector = selector.substr(first, selector.find_last_not_of(" \t\r\n") - first + 1);
    if (selector == "taco") continue;
    const auto dot = selector.find('.');
    const std::string library = selector.substr(0, dot);
    const std::string name = dot == std::string::npos ? "*" : selector.substr(dot + 1);
    const auto entry = catalog.find(library);
    if (entry == catalog.end()) throw std::runtime_error("Unknown library: " + library);
    bool matched = false;
    for (const auto& interface : entry->second) {
      const auto function = interface.getNode()->getFunctionName();
      if (name != "*" && name != function) continue;
      matched = true;
      if (function.compare(0, 6, "cblas_") == 0) providers.insert(library);
      if (selected.insert(interface.getNode()).second) result.push_back(interface);
      resolved.insert(library + "." + function);
    }
    if (!matched) throw std::runtime_error("Unknown interface: " + selector + "; use --list-interfaces");
  }
  if (providers.size() > 1) throw std::runtime_error("Select only one CBLAS provider");
  const auto names = util::join(resolved, ",");
  benchmark::AddCustomContext("interfaces", names.empty() ? "taco" : names);
  return result;
}

inline Tensor<float> randomMatrix(const std::string& name, int dimension, Format format, double density = 1.0) {
  Tensor<double> input(name, {dimension, dimension}, format);
  util::fillTensor(input, util::FillMethod::Random, density);
  return castToTypeZero<float>(name, input);
}

inline bool saveSource(size_t index, const std::string& source) {
  if (source.empty()) return true;
  std::ofstream file("code/schedule" + std::to_string(index) + ".c");
  file << source;
  file.close();
  return bool(file);
}

inline void run(benchmark::State& state, Experiment& data) {
  const size_t index = static_cast<size_t>(state.range(0));
  const auto schedule = data.schedules.at(index);
  state.SetLabel(util::toString(schedule));
  state.counters["dimension"] = data.config.dimension;
  state.counters["density"] = data.actualDensity;
  Tensor<float> result("res", {data.config.dimension, data.config.dimension}, Format{Dense, Dense});
  const bool save = !data.sourceSaved[index];
  try {
    result(data.i, data.j) = data.expression;
    const auto bound = replace(schedule, std::map<TensorVar, TensorVar>{
        {data.owner.getTensorVar(), result.getTensorVar()}});
    result.compile(bound);
    if (save) {
      if (!saveSource(index, result.getSource()))
        throw std::runtime_error("Could not write generated C to code/");
      data.sourceSaved[index] = true;
    }
    result.assemble();
    auto arguments = result.compute_split();
    auto function = result.returnFuncPackedRaw(arguments);
    for (auto _ : state) function.first(arguments.data());
  } catch (const std::exception& e) {
    data.failed = true;
    if (save && !saveSource(index, result.getSource()))
      state.SkipWithError("Could not write generated C to code/");
    else
      state.SkipWithError(e.what());
  }
}

inline int runBenchmarks(const std::shared_ptr<Experiment>& data) {
  const auto baseline = makeConcreteNotation(data->original);
  if (std::none_of(data->schedules.begin(), data->schedules.end(),
                  [&](IndexStmt s) { return equals(s, baseline); }))
    data->schedules.insert(data->schedules.begin(), baseline);
  data->sourceSaved.assign(data->schedules.size(), false);
  const std::string name = "bench_multi_" + data->config.kernel;
  benchmark::RegisterBenchmark(name.c_str(), [data](benchmark::State& state) { run(state, *data); })
      ->ArgName("schedule")->DenseRange(0, data->schedules.size() - 1)
      ->Iterations(1)->UseRealTime()->Unit(benchmark::kMillisecond);
  SilentReporter reporter;
  const auto matched = benchmark::RunSpecifiedBenchmarks(&reporter);
  benchmark::Shutdown();
  return matched && !data->failed ? 0 : 1;
}

}

#endif
