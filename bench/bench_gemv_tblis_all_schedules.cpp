#include <vector>
#include "bench.h"
#include "benchmark/benchmark.h"
#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/mkl_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"
#include "taco/accelerator_interface/gsl_interface.h"
#include "taco/accelerator_interface/tensor_interface.h"

using namespace taco;


static void bench_gemv_tblis_all_schedules(benchmark::State& state) {

     // actual computation
   int dim = state.range(0);

   Tensor<float> b("b", {dim}, Format{Dense});
   Tensor<float> A("A", {dim, dim}, Format{Dense, Dense});

   for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
         A.insert({i, j}, (float) i + j);
      }
   }

   A.pack();
   for (int i = 0; i < dim; i++) {
      b.insert({i}, (float) i);
   }

   IndexVar i("i");
   IndexVar j("j");
   IndexExpr accelerateExpr = A(i, j) * b(j);

   std::vector<FunctionInterface> availableInterfaces = {
       FunctionInterface(new TblisGemv())
   };

   Tensor<float> scheduleOwner("res", {dim}, Format{Dense});
   scheduleOwner(i) = accelerateExpr;

   IndexStmt stmt = makeReductionNotation(scheduleOwner.getAssignment());
   std::vector<IndexStmt> boundSchedules =
       stmt.autoAccelerate(stmt, availableInterfaces);

   for (auto _ : state) {
    // Setup.
    state.PauseTiming();

    for (size_t scheduleIndex = 0;
         scheduleIndex < boundSchedules.size();
         scheduleIndex++) {
      // Each returned schedule gets an independent Tensor lifecycle and module.
      Tensor<float> res("res", {dim}, Format{Dense});
      res(i) = accelerateExpr;

      const IndexStmt& boundSchedule = boundSchedules[scheduleIndex];

      std::cout << "\n========== FULL SCHEDULE "
                << (scheduleIndex + 1)
                << " ==========\n\n";
      std::cout << "----- Schedule -----\n";
      std::cout << stmt << "\n";

      std::cout << "----- Bound Schedule -----\n";
      std::cout << boundSchedule << "\n";

      res.compile(boundSchedule);
      std::cout << "\n\n----- Generated C -----\n";
      std::cout << res.getSource();
      std::cout << "\n========== END FULL SCHEDULE "
                << (scheduleIndex + 1)
                << " ==========\n\n";

      res.assemble();
      auto func = res.compute_split();
      auto pair = res.returnFuncPackedRaw(func);
      state.ResumeTiming();
      pair.first(func.data());
      state.PauseTiming();
    }

    state.ResumeTiming();
  }

}

TACO_BENCH(bench_gemv_tblis_all_schedules)->DenseRange(250, 5000, 250);
