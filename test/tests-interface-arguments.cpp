#include "test.h"
#include "taco/accelerator_notation/accel_interface.h"

using namespace taco;

TEST(interfaceArguments, printsConcreteAndAbstractTensors) {
  TensorVar tensor("A", Type(Float32, {3, 4}), Format{Dense, Dense});
  TensorObject object("B", Type(Float32, {3, 4}), Format{Dense, Dense});
  EXPECT_EQ("A", util::toString(Argument(new TensorName(tensor))));
  EXPECT_EQ("B", util::toString(Argument(new TensorName(object))));
  EXPECT_EQ("&A", util::toString(Argument(new AddrTensorVar(tensor))));
  EXPECT_EQ("&B", util::toString(Argument(new AddrTensorVar(object))));
  EXPECT_EQ("A.Dim[]", util::toString(Argument(new DimList(tensor))));
  EXPECT_EQ("A.Data[]", util::toString(Argument(new DataArray(tensor))));
  EXPECT_EQ("(float*)A.Data[]", util::toString(Argument(
      new CastArg(new DataArray(tensor), "float*"))));
  EXPECT_EQ(util::toString(Literal(2.5f)),
            util::toString(Argument(new LiteralArg(Float32, 2.5f))));
}
