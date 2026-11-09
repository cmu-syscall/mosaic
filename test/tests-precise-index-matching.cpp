#include "test.h"
#include "taco/tensor.h"
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"
#include "taco/accelerator_notation/accelerate_search.h"

using namespace taco;

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

TEST(acceleratorMatches, operatorTreePositions) {
  Tensor<float> a("a", {3}, Dense), b("b", {3}, Dense), c("c", {3}, Dense);
  TensorObject x(Type(Float32, {Dimension()}), Dense);
  TensorObject y(Type(Float32, {Dimension()}), Dense);
  TensorObject z(Type(Float32, {Dimension()}), Dense);
  IndexVar i("i"), j("j");
  EXPECT_TRUE(hasPreciseMatch((a(i) + b(i)) * c(i), (x(j) + y(j)) * z(j)).possible);
  EXPECT_FALSE(hasPreciseMatch((a(i) + b(i)) * c(i), x(j) * (y(j) + z(j))).possible);
}

TEST(acceleratorMatches, literalValues) {
  Tensor<float> a("a", {3}, Dense);
  TensorObject x(Type(Float32, {Dimension()}), Dense);
  IndexVar i("i"), j("j");
  EXPECT_TRUE(hasPreciseMatch(a(i) * 2.0f, x(j) * AcceleratorLiteral(2.0f)).possible);
  EXPECT_FALSE(hasPreciseMatch(a(i) * 2.0f, x(j) * AcceleratorLiteral(3.0f)).possible);
}
