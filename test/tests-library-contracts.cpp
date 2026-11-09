#include "test.h"
#include "taco/tensor.h"
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"
#include "taco/accelerator_interface/mkl_interface.h"
#include "taco/accelerator_interface/gsl_interface.h"
#include "taco/accelerator_interface/cuda_interface.h"

using namespace taco;

static void checkRectangularProduct(FunctionInterface interface, bool sparse = false,
                                    bool masked = false) {
  const int rows = 3, inner = 5, columns = 4;
  Tensor<float> a("a", {rows, inner}, sparse ? CSR : Format{Dense, Dense});
  Tensor<float> b("b", {inner, columns}, Format{Dense, Dense});
  Tensor<float> mask("mask", {rows, columns}, CSR);
  Tensor<float> result("result", {rows, columns}, Format{Dense, Dense});
  for (int row = 0; row < rows; ++row)
    for (int column = 0; column < inner; ++column)
      if (!sparse || (row + column) % 3 != 0)
        a.insert({row, column}, float(1 + row + column));
  for (int row = 0; row < inner; ++row)
    for (int column = 0; column < columns; ++column)
      b.insert({row, column}, float(1 + 2 * row + column));
  for (int row = 0; row < rows; ++row)
    for (int column = 0; column < columns; ++column)
      if ((row + column) % 2 == 0)
        mask.insert({row, column}, 2.0f);
  a.pack();
  b.pack();
  mask.pack();

  IndexVar i("i"), j("j"), k("k");
  IndexExpr product = a(i, k) * b(k, j);
  IndexExpr contraction = sum(k, product);
  result(i, j) = masked ? mask(i, j) * contraction : product;
  IndexStmt schedule = result.getAssignment().concretize();
  schedule = schedule.accelerate(interface, masked ? contraction : product);
  result.compile(schedule);
  result.assemble();
  auto arguments = result.compute_split();
  auto function = result.returnFuncPackedRaw(arguments);
  for (int repeat = 0; repeat < 2; ++repeat) {
    ASSERT_EQ(0, function.first(arguments.data()));
    for (int row = 0; row < rows; ++row) {
      for (int column = 0; column < columns; ++column) {
        float expected = 0;
        for (int index = 0; index < inner; ++index)
          if (!sparse || (row + index) % 3 != 0)
            expected += float(1 + row + index) * float(1 + 2 * index + column);
        if (masked) expected *= (row + column) % 2 == 0 ? 2.0f : 0.0f;
        EXPECT_FLOAT_EQ(expected, result.at({row, column}));
      }
    }
  }
}

TEST(DISABLED_libraryContracts, TblisGemm) { checkRectangularProduct(new TblisMultiply()); }
TEST(DISABLED_libraryContracts, TblisSddmm) { checkRectangularProduct(new TblisMultiply(), false, true); }
TEST(DISABLED_libraryContracts, CblasGemm) { checkRectangularProduct(new MatrixMultiply()); }
TEST(DISABLED_libraryContracts, MklGemm) { checkRectangularProduct(new MklMM()); }
TEST(DISABLED_libraryContracts, MklSpmm) { checkRectangularProduct(new SparseMklMM(), true); }
TEST(DISABLED_libraryContracts, GslGemm) { checkRectangularProduct(new GSLMM()); }
TEST(DISABLED_libraryContracts, CudaGemm) { checkRectangularProduct(new CudaGemm()); }
TEST(DISABLED_libraryContracts, CudaSpmm) { checkRectangularProduct(new CudaSpmm(), true); }
TEST(DISABLED_libraryContracts, CudaSddmm) { checkRectangularProduct(new CudaGemm(), false, true); }
