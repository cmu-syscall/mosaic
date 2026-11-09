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

TEST(scheduleConstants, nestedConstantsAndIdentities) {
  Tensor<int> a("a", {3}, Dense), out("out", {3}, Dense);
  IndexVar i("i");
  IndexExpr constant = (IndexExpr(2) + IndexExpr(3)) * IndexExpr(4);
  auto stmt = forall(i, Assignment(out(i), (a(i) + (constant - 20)) * 1));
  EXPECT_TRUE(equals(helperSimplifyConstants(stmt), forall(i, Assignment(out(i), a(i)))));
  auto folded = helperSimplifyConstants(Assignment(out(i), constant / 1));
  EXPECT_TRUE(equals(to<Assignment>(folded).getRhs(), IndexExpr(20)));
  EXPECT_TRUE(equals(folded, helperSimplifyConstants(folded)));
}

TEST(scheduleConstants, floatingPointRules) {
  Tensor<float> a("a", {3}, Dense), out("out", {3}, Dense);
  IndexVar i("i");
  auto one = Assignment(out(i), (a(i) * 1.0f) / 1.0f);
  EXPECT_TRUE(equals(helperSimplifyConstants(one), Assignment(out(i), a(i))));
  auto mixed = Assignment(out(i), (a(i) * 1) / 1);
  EXPECT_TRUE(equals(helperSimplifyConstants(mixed), Assignment(out(i), a(i))));
  auto zero = Assignment(out(i), a(i) * 0.0f);
  auto add = Assignment(out(i), a(i) + 0.0f);
  EXPECT_TRUE(equals(helperSimplifyConstants(zero), Assignment(out(i), IndexExpr(0.0f))));
  for (auto expression : {IndexExpr(0.0f) * a(i), a(i) * 0, 0 * a(i),
                          a(i) * -0.0f, IndexExpr(-0.0f) * a(i)}) {
    EXPECT_TRUE(equals(helperSimplifyConstants(Assignment(out(i), expression)),
                       Assignment(out(i), IndexExpr(0.0f))));
  }
  auto nested = Assignment(out(i), a(i) + IndexExpr(0.0f) * a(i));
  EXPECT_TRUE(equals(helperSimplifyConstants(nested), Assignment(out(i), a(i))));
  auto subtract = Assignment(out(i), a(i) - IndexExpr(0.0f) * a(i));
  EXPECT_TRUE(equals(helperSimplifyConstants(subtract), Assignment(out(i), a(i))));
  EXPECT_TRUE(equals(helperSimplifyConstants(add), Assignment(out(i), a(i))));
  for (auto expression : {IndexExpr(0.0f) + a(i), a(i) - 0.0f,
                          a(i) + 0, a(i) - 0, a(i) + -0.0f, a(i) - -0.0f}) {
    EXPECT_TRUE(equals(helperSimplifyConstants(Assignment(out(i), expression)),
                       Assignment(out(i), a(i))));
  }
  auto literal = helperSimplifyConstants(Assignment(out(i), IndexExpr(6.0f) / 1.0f));
  EXPECT_TRUE(equals(to<Assignment>(literal).getRhs(), IndexExpr(6.0f)));
  auto division = Assignment(out(i), IndexExpr(6.0f) / 4.0f);
  EXPECT_TRUE(equals(helperSimplifyConstants(division), division));
}

TEST(scheduleConstants, generalDivisionIsUnchanged) {
  Tensor<int64_t> out("out", {1}, Dense);
  IndexVar i("i");
  auto lo = std::numeric_limits<int64_t>::min();
  for (auto expression : {IndexExpr(int64_t(6)) / IndexExpr(int64_t(2)),
                          IndexExpr(lo) / IndexExpr(int64_t(-1)),
                          IndexExpr(int64_t(3)) / IndexExpr(int64_t(0))}) {
    auto stmt = Assignment(out(i), expression);
    EXPECT_TRUE(equals(helperSimplifyConstants(stmt), stmt));
  }
}

TEST(scheduleConstants, opaqueLibraryNodes) {
  Tensor<float> a("a", {3}, Dense), out("out", {3}, Dense);
  IndexVar i("i");
  auto assignment = Assignment(out(i), a(i) * 1.0f);
  ConcreteAccelerateCodeGenerator generator;
  IndexStmt accelerate = new AccelerateNode(assignment, assignment, generator);
  IndexStmt call = new InterfaceCallNode(assignment, generator, out.getTensorVar());
  EXPECT_EQ(helperSimplifyConstants(accelerate).ptr, accelerate.ptr);
  EXPECT_EQ(helperSimplifyConstants(call).ptr, call.ptr);
  IndexStmt mixed = new WhereNode(assignment, call);
  EXPECT_EQ(helperSimplifyConstants(mixed).ptr, mixed.ptr);
}

TEST(scheduleConstants, autoAccelerateDeduplicates) {
  Tensor<int> a("a", {3}, Dense), out("out", {3}, Dense);
  IndexVar i("i");
  auto stmt = makeReductionNotation(Assignment(out(i), a(i) * 1));
  auto schedules = stmt.autoAccelerate(stmt, {});
  ASSERT_FALSE(schedules.empty());
  bool found = false;
  for (size_t x = 0; x < schedules.size(); ++x) {
    EXPECT_TRUE(equals(helperSimplifyConstants(schedules[x]), schedules[x]));
    found |= equals(schedules[x], forall(i, Assignment(out(i), a(i))));
    for (size_t y = 0; y < x; ++y) EXPECT_FALSE(equals(schedules[x], schedules[y]));
  }
  EXPECT_TRUE(found);
}

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

TEST(scheduleConstants, matrixAndRankThree) {
  for (const auto& shape : {std::vector<int>{2, 3}, std::vector<int>{2, 3, 4}}) {
    Format format(std::vector<ModeFormatPack>(shape.size(), Dense));
    Tensor<float> a("a", shape, format), out("out", shape, format);
    std::vector<IndexVar> indices(shape.size());
    IndexExpr expression = (a(indices) + (IndexExpr(2.0f) * 3.0f - 6.0f)) * 1.0f;
    IndexStmt original = Assignment(out(indices), expression);
    IndexStmt expected = Assignment(out(indices), a(indices));
    for (auto index = indices.rbegin(); index != indices.rend(); ++index) {
      original = forall(*index, original);
      expected = forall(*index, expected);
    }
    auto simplified = helperSimplifyConstants(original);
    EXPECT_TRUE(equals(expected, simplified));
    EXPECT_TRUE(equals(simplified, helperSimplifyConstants(simplified)));
    EXPECT_EQ(getIndexVars(original), getIndexVars(simplified));
  }
}

TEST(scheduleConstants, matrixReductionPreservesLoops) {
  Tensor<int> a("a", {2, 3, 4}, Format{Dense, Dense, Dense});
  Tensor<int> out("out", {2, 3}, Format{Dense, Dense});
  IndexVar i("i"), j("j"), k("k");
  auto original = forall(i, forall(j, forall(k, Assignment(out(i,j), a(i,j,k) * 1 + 0, Add()))));
  auto expected = forall(i, forall(j, forall(k, Assignment(out(i,j), a(i,j,k), Add()))));
  EXPECT_TRUE(equals(helperSimplifyConstants(original), expected));
}

TEST(scheduleConstants, foldingCanBeDisabled) {
  Tensor<float> a("a", {2, 3}, Format{Dense, Dense});
  Tensor<float> out("out", {2, 3}, Format{Dense, Dense});
  IndexVar i("i"), j("j");
  auto original = makeReductionNotation(Assignment(out(i,j), a(i,j) + 0.0f));
  auto unmodified = makeConcreteNotation(original);
  auto raw = original.autoAccelerate(original, {}, false);
  EXPECT_TRUE(std::any_of(raw.begin(), raw.end(), [&](IndexStmt stmt) { return equals(stmt, unmodified); }));
  for (auto stmt : original.autoAccelerate(original, {}, true))
    EXPECT_TRUE(equals(stmt, helperSimplifyConstants(stmt)));
}

TEST(scheduleConstants, compiledRankThreeReduction) {
  Tensor<int> a("a", {2, 3, 4}, Format{Dense, Dense, Dense});
  Tensor<int> out("out", {2, 3}, Format{Dense, Dense});
  IndexVar i("i"), j("j"), k("k");
  for (int row = 0; row < 2; ++row)
    for (int column = 0; column < 3; ++column)
      for (int depth = 0; depth < 4; ++depth)
        a.insert({row, column, depth}, row + column + depth);
  a.pack();
  out(i,j) = a(i,j,k) * (IndexExpr(2) - 1) + 0;
  out.compile(helperSimplifyConstants(out.getAssignment().concretize()));
  out.assemble();
  out.compute();
  for (int row = 0; row < 2; ++row)
    for (int column = 0; column < 3; ++column)
      EXPECT_EQ(out.at({row, column}), 4 * (row + column) + 6);
}
