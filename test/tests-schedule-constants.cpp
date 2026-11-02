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

TEST(acceleratorMatches, rectangularTblisOutputDimensions) {
  Tensor<float> a("a", {2, 3}, Format{Dense, Dense});
  Tensor<float> b("b", {3, 4}, Format{Dense, Dense});
  Tensor<float> out("out", {2, 4}, Format{Dense, Dense});
  IndexVar i("i"), j("j"), k("k");
  auto stmt = makeReductionNotation(Assignment(out(i,j), a(i,k) * b(k,j)));
  const auto schedule = stmt.helperCheckForMatches(stmt, {new TblisMultiply()});
  EXPECT_NO_THROW(lower(schedule, "compute"));
}
