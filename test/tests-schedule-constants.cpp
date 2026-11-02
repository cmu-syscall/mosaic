#include "test.h"
#include <limits>
#include <set>
#include "taco/tensor.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"
#include "taco/accelerator_notation/accelerate_search.h"
#include "taco/lower/lower.h"
#include "taco/ir/ir.h"

using namespace taco;

TEST(scheduleConstants, autoAcceleratePreservesDifferentLibraries) {
  Tensor<float> a("a", {3, 3}, Format{Dense, Dense});
  Tensor<float> b("b", {3, 3}, Format{Dense, Dense});
  Tensor<float> out("out", {3, 3}, Format{Dense, Dense});
  IndexVar i("i"), j("j"), k("k");
  auto stmt = makeReductionNotation(Assignment(out(i,j), a(i,k) * b(k,j)));
  std::set<std::string> names;
  for (auto schedule : stmt.autoAccelerate(stmt, {new TblisMultiply(), new MatrixMultiply()})) {
    match(schedule,
      std::function<void(const AccelerateNode*)>([&](const AccelerateNode* node) {
        names.insert(node->accelGen.getFunctionName());
      }),
      std::function<void(const InterfaceCallNode*)>([&](const InterfaceCallNode* node) {
        names.insert(node->codeGen.getFunctionName());
      }));
  }
  EXPECT_GE(names.size(), size_t(2));
}

TEST(acceleratorMatches, consistentIndices) {
  Tensor<float> a("a", {3, 3}, Format{Dense, Dense});
  Tensor<float> b("b", {3, 3}, Format{Dense, Dense});
  IndexVar i("i"), j("j"), k("k"), l("l");
  TblisMultiply gemm;
  auto rhs = to<AcceleratorAssignment>(gemm.getStmt()).getRhs();
  EXPECT_TRUE(hasPreciseMatch(a(i,k) * b(k,j), rhs).possible);
  EXPECT_TRUE(hasPreciseMatch(a(i,k) * a(k,j), rhs).possible);
  EXPECT_FALSE(hasPreciseMatch(a(i,k) * b(l,j), rhs).possible);
  EXPECT_FALSE(hasPreciseMatch(a(i,k) * b(k,i), rhs).possible);
  EXPECT_FALSE(hasPreciseMatch(a(i,i) * b(i,j), rhs).possible);
}

TEST(acceleratorMatches, repeatedTensorAndReduction) {
  Tensor<float> a("a", {3}, Dense), b("b", {3}, Dense);
  TensorObject x(Type(Float32, {Dimension()}), Dense);
  IndexVar i("i"), j("j");
  EXPECT_TRUE(hasPreciseMatch(a(i) * a(i), x(j) * x(j)).possible);
  EXPECT_FALSE(hasPreciseMatch(a(i) * b(i), x(j) * x(j)).possible);
  EXPECT_FALSE(hasPreciseMatch(sum(i, a(i)), sum(j, x(i))).possible);
}

TEST(acceleratorMatches, rectangularTblisOutputDimensions) {
  Tensor<float> a("a", {2, 3}, Format{Dense, Dense});
  Tensor<float> b("b", {3, 4}, Format{Dense, Dense});
  Tensor<float> out("out", {2, 4}, Format{Dense, Dense});
  IndexVar i("i"), j("j"), k("k");
  auto stmt = makeReductionNotation(Assignment(out(i,j), a(i,k) * b(k,j)));
  const auto schedules = stmt.helperCheckForMatches(stmt, {new TblisMultiply()});
  ASSERT_EQ(schedules.size(), size_t(1));
  EXPECT_NO_THROW(lower(schedules.front(), "compute"));
}
