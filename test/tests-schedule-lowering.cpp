#include "gtest/gtest.h"
#include "taco/tensor.h"

using namespace taco;

TEST(scheduleLowering, nestedReductionArgumentOrder) {
  Tensor<float> mask("mask", {2, 4}, CSR);
  Tensor<float> A("A", {2, 3}, Format{Dense, Dense});
  Tensor<float> B("B", {3, 4}, Format{Dense, Dense});
  Tensor<float> result("result", {2, 4}, Format{Dense, Dense});
  mask.insert({0, 1}, 2.0f);
  mask.insert({1, 3}, 3.0f);
  mask.pack();
  for (int row = 0; row < 2; ++row)
    for (int column = 0; column < 3; ++column)
      A.insert({row, column}, float(1 + row + column));
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 4; ++column)
      B.insert({row, column}, float(1 + 2 * row + column));
  A.pack();
  B.pack();

  IndexVar i("i"), j("j"), k("k");
  result(i, j) = mask(i, j) * sum(k, A(i, k) * B(k, j));
  result.compile(makeConcreteNotation(result.getAssignment()));
  result.assemble();
  result.compute();

  for (int row = 0; row < 2; ++row) {
    for (int column = 0; column < 4; ++column) {
      float expected = 0;
      for (int inner = 0; inner < 3; ++inner)
        expected += float(1 + row + inner) * float(1 + 2 * inner + column);
      const float weight = row == 0 && column == 1 ? 2.0f :
                           row == 1 && column == 3 ? 3.0f : 0.0f;
      EXPECT_FLOAT_EQ(weight * expected, result.at({row, column}));
    }
  }
}
