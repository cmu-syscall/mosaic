#ifndef CUDA_INTERFACE_H
#define CUDA_INTERFACE_H

#include "taco/index_notation/index_notation.h"
#include "taco/type.h"
#include "taco/accelerator_notation/accel_interface.h"
#include "taco/accelerator_notation/accelerator_notation.h"

using namespace taco; 

class CudaSpmv : public AbstractFunctionInterface{
    public: 
        CudaSpmv() : x(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}), Format{Dense, Sparse})), 
                  y(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  s(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  i(IndexVar()),
                  j(IndexVar()) {};
        AcceleratorStmt getStmt() const override {return y(i) = x(i, j)*s(j);}
        std::vector<Argument> getArguments() const override {return 
                                                {
                                                    new TensorName(x),
                                                    new TensorObjectArg(s),
                                                    new TensorObjectArg(y),
                                                    new DimArg(i),
                                                };}
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override {return "cuda_spmv";}
    private: 
        TensorObject x;
        TensorObject y;
        TensorObject s;
        IndexVar i;
        IndexVar j;
};

class CudaGemm : public AbstractFunctionInterface {
public:
    CudaGemm()
        : a(Type(Float32, {Dimension(), Dimension()}), Format{Dense, Dense}),
          b(Type(Float32, {Dimension(), Dimension()}), Format{Dense, Dense}),
          c(Type(Float32, {Dimension(), Dimension()}), Format{Dense, Dense}) {}

    AcceleratorStmt getStmt() const override { return c(i, j) = a(i, k) * b(k, j); }
    std::vector<Argument> getArguments() const override {
        return {new DimArg(i), new DimArg(j), new DimArg(k),
                new TensorObjectArg(a), new TensorObjectArg(b), new TensorObjectArg(c)};
    }
    std::string getReturnType() const override { return "void"; }
    std::string getFunctionName() const override { return "mosaic_cuda_sgemm"; }

private:
    TensorObject a, b, c;
    IndexVar i, j, k;
};

class CudaGemv : public AbstractFunctionInterface {
public:
    CudaGemv()
        : a(Type(Float32, {Dimension(), Dimension()}), Format{Dense, Dense}),
          x(Type(Float32, {Dimension()}), dense),
          y(Type(Float32, {Dimension()}), dense) {}

    AcceleratorStmt getStmt() const override { return y(i) = a(i, j) * x(j); }
    std::vector<Argument> getArguments() const override {
        return {new DimArg(i), new DimArg(j), new TensorObjectArg(a),
                new TensorObjectArg(x), new TensorObjectArg(y)};
    }
    std::string getReturnType() const override { return "void"; }
    std::string getFunctionName() const override { return "mosaic_cuda_sgemv"; }

private:
    TensorObject a, x, y;
    IndexVar i, j;
};

class CudaSpmm : public AbstractFunctionInterface {
public:
    CudaSpmm()
        : a(Type(Float32, {Dimension(), Dimension()}), CSR),
          b(Type(Float32, {Dimension(), Dimension()}), Format{Dense, Dense}),
          c(Type(Float32, {Dimension(), Dimension()}), Format{Dense, Dense}) {}

    AcceleratorStmt getStmt() const override { return c(i, j) = a(i, k) * b(k, j); }
    std::vector<Argument> getArguments() const override {
        return {new TensorName(a), new DimArg(j), new TensorObjectArg(b), new TensorObjectArg(c)};
    }
    std::string getReturnType() const override { return "void"; }
    std::string getFunctionName() const override { return "mosaic_cuda_spmm"; }

private:
    TensorObject a, b, c;
    IndexVar i, j, k;
};


#endif
