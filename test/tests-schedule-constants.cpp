#include "test.h"
#include <limits>
#include <set>
#include "taco/tensor.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"

using namespace taco;

TEST(scheduleConstants, nestedConstantsAndIdentities) {
  Tensor<int> a("a", {3}, Dense), out("out", {3}, Dense);
  IndexVar i("i");
  IndexExpr constant = (IndexExpr(2) + IndexExpr(3)) * IndexExpr(4);
  auto stmt = forall(i, Assignment(out(i), (a(i) + (constant - 20)) * 1));
  EXPECT_TRUE(equals(propagateConstants(stmt), forall(i, Assignment(out(i), a(i)))));
  auto folded = propagateConstants(Assignment(out(i), constant / 1));
  EXPECT_TRUE(equals(to<Assignment>(folded).getRhs(), IndexExpr(20)));
  EXPECT_TRUE(equals(folded, propagateConstants(folded)));
}

TEST(scheduleConstants, floatingPointRules) {
  Tensor<float> a("a", {3}, Dense), out("out", {3}, Dense);
  IndexVar i("i");
  auto one = Assignment(out(i), (a(i) * 1.0f) / 1.0f);
  EXPECT_TRUE(equals(propagateConstants(one), Assignment(out(i), a(i))));
  auto mixed = Assignment(out(i), (a(i) * 1) / 1);
  EXPECT_TRUE(equals(propagateConstants(mixed), Assignment(out(i), a(i))));
  auto zero = Assignment(out(i), a(i) * 0.0f);
  auto add = Assignment(out(i), a(i) + 0.0f);
  EXPECT_TRUE(equals(propagateConstants(zero), Assignment(out(i), IndexExpr(0.0f))));
  for (auto expression : {IndexExpr(0.0f) * a(i), a(i) * 0, 0 * a(i),
                          a(i) * -0.0f, IndexExpr(-0.0f) * a(i)}) {
    EXPECT_TRUE(equals(propagateConstants(Assignment(out(i), expression)),
                       Assignment(out(i), IndexExpr(0.0f))));
  }
  auto nested = Assignment(out(i), a(i) + IndexExpr(0.0f) * a(i));
  EXPECT_TRUE(equals(propagateConstants(nested), Assignment(out(i), a(i))));
  auto subtract = Assignment(out(i), a(i) - IndexExpr(0.0f) * a(i));
  EXPECT_TRUE(equals(propagateConstants(subtract), Assignment(out(i), a(i))));
  EXPECT_TRUE(equals(propagateConstants(add), Assignment(out(i), a(i))));
  for (auto expression : {IndexExpr(0.0f) + a(i), a(i) - 0.0f,
                          a(i) + 0, a(i) - 0, a(i) + -0.0f, a(i) - -0.0f}) {
    EXPECT_TRUE(equals(propagateConstants(Assignment(out(i), expression)),
                       Assignment(out(i), a(i))));
  }
  auto literal = propagateConstants(Assignment(out(i), IndexExpr(6.0f) / 1.0f));
  EXPECT_TRUE(equals(to<Assignment>(literal).getRhs(), IndexExpr(6.0f)));
  auto division = Assignment(out(i), IndexExpr(6.0f) / 4.0f);
  EXPECT_TRUE(equals(propagateConstants(division), division));
}

TEST(scheduleConstants, generalDivisionIsUnchanged) {
  Tensor<int64_t> out("out", {1}, Dense);
  IndexVar i("i");
  auto lo = std::numeric_limits<int64_t>::min();
  for (auto expression : {IndexExpr(int64_t(6)) / IndexExpr(int64_t(2)),
                          IndexExpr(lo) / IndexExpr(int64_t(-1)),
                          IndexExpr(int64_t(3)) / IndexExpr(int64_t(0))}) {
    auto stmt = Assignment(out(i), expression);
    EXPECT_TRUE(equals(propagateConstants(stmt), stmt));
  }
}

TEST(scheduleConstants, opaqueLibraryNodes) {
  Tensor<float> a("a", {3}, Dense), out("out", {3}, Dense);
  IndexVar i("i");
  auto assignment = Assignment(out(i), a(i) * 1.0f);
  ConcreteAccelerateCodeGenerator generator;
  IndexStmt accelerate = new AccelerateNode(assignment, assignment, generator);
  IndexStmt call = new InterfaceCallNode(assignment, generator, out.getTensorVar());
  EXPECT_EQ(propagateConstants(accelerate).ptr, accelerate.ptr);
  EXPECT_EQ(propagateConstants(call).ptr, call.ptr);
  IndexStmt mixed = new WhereNode(assignment, call);
  EXPECT_EQ(propagateConstants(mixed).ptr, mixed.ptr);
}

TEST(scheduleConstants, autoAccelerateDeduplicates) {
  Tensor<int> a("a", {3}, Dense), out("out", {3}, Dense);
  IndexVar i("i");
  auto stmt = makeReductionNotation(Assignment(out(i), a(i) * 1));
  auto schedules = stmt.autoAccelerate(stmt, {});
  ASSERT_FALSE(schedules.empty());
  bool found = false;
  for (size_t x = 0; x < schedules.size(); ++x) {
    EXPECT_TRUE(equals(propagateConstants(schedules[x]), schedules[x]));
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

