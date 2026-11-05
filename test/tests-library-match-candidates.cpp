#include "test.h"
#include <set>
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"

using namespace taco;

namespace {

std::vector<std::string> libraryCalls(IndexStmt stmt) {
  std::vector<std::string> names;
  match(stmt,
    std::function<void(const InterfaceCallNode*)>([&](const InterfaceCallNode* node) {
      names.push_back(node->codeGen.getFunctionName());
    }));
  return names;
}

class LibraryMatchCandidates : public ::testing::Test {
protected:
  Tensor<float> a{"a", {3, 3}, Format{Dense, Dense}};
  Tensor<float> b{"b", {3, 3}, Format{Dense, Dense}};
  Tensor<float> out{"out", {3, 3}, Format{Dense, Dense}};
  IndexVar i{"i"}, j{"j"}, k{"k"};

  IndexStmt gemm() {
    return makeReductionNotation(Assignment(out(i,j), a(i,k) * b(k,j)));
  }
};

TEST_F(LibraryMatchCandidates, returnsBothLibraries) {
  auto stmt = gemm();
  auto candidates = stmt.helperCheckForMatches(stmt, {new TblisMultiply(), new MatrixMultiply()});
  ASSERT_EQ(candidates.size(), size_t(2));
  EXPECT_EQ(libraryCalls(candidates[0]), std::vector<std::string>({"tblis_tensor_mult"}));
  EXPECT_EQ(libraryCalls(candidates[1]), std::vector<std::string>({"cblas_sgemm"}));
}

TEST_F(LibraryMatchCandidates, preservesAlternativesWhenOrderChanges) {
  auto stmt = gemm();
  auto candidates = stmt.helperCheckForMatches(stmt, {new MatrixMultiply(), new TblisMultiply()});
  ASSERT_EQ(candidates.size(), size_t(2));
  EXPECT_EQ(libraryCalls(candidates[0]), std::vector<std::string>({"cblas_sgemm"}));
  EXPECT_EQ(libraryCalls(candidates[1]), std::vector<std::string>({"tblis_tensor_mult"}));
}

TEST_F(LibraryMatchCandidates, ignoresRepeatedInterfaceObject) {
  auto stmt = gemm();
  FunctionInterface tblis = new TblisMultiply();
  auto candidates = stmt.helperCheckForMatches(stmt, {tblis, tblis, new MatrixMultiply()});
  EXPECT_EQ(candidates.size(), size_t(2));
}

TEST_F(LibraryMatchCandidates, keepsDistinctInterfaceObjects) {
  auto stmt = gemm();
  auto candidates = stmt.helperCheckForMatches(stmt, {new TblisMultiply(), new TblisMultiply()});
  EXPECT_EQ(candidates.size(), size_t(2));
}

TEST_F(LibraryMatchCandidates, returnsUnboundWhenNothingMatches) {
  auto stmt = gemm();
  auto candidates = stmt.helperCheckForMatches(stmt, {new TblisGemv()});
  ASSERT_EQ(candidates.size(), size_t(1));
  EXPECT_TRUE(equals(candidates.front(), makeConcreteNotation(stmt)));
}

TEST_F(LibraryMatchCandidates, returnsUnboundWithNoInterfaces) {
  auto stmt = gemm();
  auto candidates = stmt.helperCheckForMatches(stmt, {});
  ASSERT_EQ(candidates.size(), size_t(1));
  EXPECT_TRUE(equals(candidates.front(), makeConcreteNotation(stmt)));
}

TEST_F(LibraryMatchCandidates, considersEachDistinctSubexpression) {
  IndexStmt stmt = Assignment(out(i,j),
      sum(k, a(i,k) * b(k,j)) + sum(k, b(i,k) * a(k,j)));
  auto candidates = stmt.helperCheckForMatches(stmt, {new TblisMultiply(), new MatrixMultiply()});
  ASSERT_EQ(candidates.size(), size_t(4));
  for (auto candidate : candidates) EXPECT_EQ(libraryCalls(candidate).size(), size_t(1));
}

TEST_F(LibraryMatchCandidates, skipsDuplicateMatchedExpressions) {
  IndexExpr product = sum(k, a(i,k) * b(k,j));
  IndexStmt stmt = Assignment(out(i,j), product + product);
  auto candidates = stmt.helperCheckForMatches(stmt, {new TblisMultiply()});
  ASSERT_EQ(candidates.size(), size_t(1));
  EXPECT_EQ(libraryCalls(candidates.front()).size(), size_t(1));
}

TEST_F(LibraryMatchCandidates, autoAcceleratePreservesBaselineAndLibraries) {
  auto stmt = gemm();
  auto baseline = makeConcreteNotation(stmt);
  auto candidates = stmt.autoAccelerate(stmt, {new TblisMultiply(), new MatrixMultiply()});
  std::set<std::string> names;
  size_t baselines = 0;
  for (auto candidate : candidates) {
    if (equals(candidate, baseline)) ++baselines;
    for (auto name : libraryCalls(candidate)) names.insert(name);
  }
  EXPECT_EQ(baselines, size_t(1));
  EXPECT_EQ(names, std::set<std::string>({"cblas_sgemm", "tblis_tensor_mult"}));
}

} // namespace
