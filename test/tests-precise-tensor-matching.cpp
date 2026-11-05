#include "test.h"
#include "taco/accelerator_notation/accelerate_search.h"

using namespace taco;

TEST(preciseTensorMatching, rejectsConflictingRepeatedTensor) {
  Tensor<float> a("a", {3}, Dense), b("b", {3}, Dense);
  TensorObject x(Type(Float32, {Dimension()}), Dense);
  IndexVar i("i"), j("j");
  EXPECT_FALSE(hasPreciseMatch(a(i) * b(i), x(j) * x(j)).possible);
  EXPECT_FALSE(hasPreciseMatch(b(i) * a(i), x(j) * x(j)).possible);
}

TEST(preciseTensorMatching, preservesRepeatedTensorMapping) {
  Tensor<float> a("a", {3}, Dense);
  TensorObject x(Type(Float32, {Dimension()}), Dense);
  IndexVar i("i"), j("j");
  auto match = hasPreciseMatch(a(i) * a(i), x(j) * x(j));
  ASSERT_TRUE(match.possible);
  ASSERT_EQ(match.tensors.size(), size_t(1));
  EXPECT_TRUE(match.tensors.at(x) == a.getTensorVar());
}

TEST(preciseTensorMatching, allowsDistinctParametersToAlias) {
  Tensor<float> a("a", {3}, Dense);
  TensorObject x(Type(Float32, {Dimension()}), Dense);
  TensorObject y(Type(Float32, {Dimension()}), Dense);
  IndexVar i("i"), j("j");
  auto match = hasPreciseMatch(a(i) * a(i), x(j) * y(j));
  ASSERT_TRUE(match.possible);
  ASSERT_EQ(match.tensors.size(), size_t(2));
  EXPECT_TRUE(match.tensors.at(x) == a.getTensorVar());
  EXPECT_TRUE(match.tensors.at(y) == a.getTensorVar());
}

TEST(preciseTensorMatching, preservesDistinctInputs) {
  Tensor<float> a("a", {3}, Dense), b("b", {3}, Dense);
  TensorObject x(Type(Float32, {Dimension()}), Dense);
  TensorObject y(Type(Float32, {Dimension()}), Dense);
  IndexVar i("i"), j("j");
  auto match = hasPreciseMatch(a(i) * b(i), x(j) * y(j));
  ASSERT_TRUE(match.possible);
  EXPECT_TRUE(match.tensors.at(x) == a.getTensorVar());
  EXPECT_TRUE(match.tensors.at(y) == b.getTensorVar());
}
