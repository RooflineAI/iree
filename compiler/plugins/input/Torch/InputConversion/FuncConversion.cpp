// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/Torch/InputConversion/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

namespace Torch = mlir::torch::Torch;
namespace TorchConversion = mlir::torch::TorchConversion;

namespace mlir::iree_compiler::TorchInput {

#define GEN_PASS_DEF_FUNCCONVERSIONPASS
#include "compiler/plugins/input/Torch/InputConversion/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Overall Approach
// ----------------
// This pass converts from the "torch" programming model to the "iree"
// programming model by rewriting all functions and calls to operate on native
// IREE types.
//===----------------------------------------------------------------------===//

Value convertToBuiltinTensor(OpBuilder &builder, Value possibleTorchTensor) {
  Type ty = possibleTorchTensor.getType();
  if (isa<TensorType>(ty))
    return possibleTorchTensor;

  if (auto defining = dyn_cast_or_null<TorchConversion::FromBuiltinTensorOp>(
          possibleTorchTensor.getDefiningOp())) {
    return defining.getOperand();
  }

  Torch::ValueTensorType vtensorType = cast<Torch::ValueTensorType>(ty);
  TensorType builtinTy = vtensorType.toBuiltinTensor();
  if (auto intTy = dyn_cast<IntegerType>(builtinTy.getElementType())) {
    builtinTy =
        builtinTy.clone(builder.getIntegerType(intTy.getIntOrFloatBitWidth()));
  }

  return builder.create<TorchConversion::ToBuiltinTensorOp>(
      possibleTorchTensor.getLoc(), builtinTy, possibleTorchTensor);
}

enum class TypeDisposition {
  IMMUTABLE_TENSOR,
  TORCH_PRIMITIVE,
  PASSTHROUGH,
};

struct ConvertedFunctionInfo {
  IREE::Util::FuncOp funcOp;
  SmallVector<IREE::Util::ReturnOp> returnOps;
  SmallVector<DictionaryAttr> torchArgAttrs;
  SmallVector<DictionaryAttr> torchResultAttrs;
  SmallVector<Type> torchInputTypes;
  SmallVector<Type> torchResultTypes;
  SmallVector<TypeDisposition> inputDispositions;
  SmallVector<TypeDisposition> resultDispositions;

  LogicalResult postProcess();
  LogicalResult convertImmutableTensorArg(BlockArgument argValue,
                                          Type torchType, OpBuilder &builder);

  Attribute getTorchArgAttr(BlockArgument argValue, StringRef attrName) {
    return torchArgAttrs.empty()
               ? Attribute{}
               : torchArgAttrs[argValue.getArgNumber()].get(attrName);
  }
  Attribute getTorchResultAttr(int returnIndex, StringRef attrName) {
    return torchResultAttrs.empty()
               ? Attribute{}
               : torchResultAttrs[returnIndex].get(attrName);
  }
};

LogicalResult ConvertedFunctionInfo::postProcess() {
  if (funcOp.isExternal())
    return success();

  if (returnOps.size() != 1) {
    // Multi-exit/CFG could be supported but requires more complicated dominance
    // analysis with respect to where the exit happens relative to mutated
    // buffers.
    return emitError(funcOp.getLoc())
           << "currently only single exit torch funcs are supported";
  }

  Block *entryBlock = &funcOp.getBlocks().front();

  // Materialize argument conversions.
  OpBuilder preambleBuilder = OpBuilder::atBlockBegin(entryBlock);
  auto entryArgs = entryBlock->getArguments();
  for (auto [disp, argValue, torchType] :
       llvm::zip_equal(inputDispositions, entryArgs, torchInputTypes)) {
    switch (disp) {
    case TypeDisposition::IMMUTABLE_TENSOR: {
      if (failed(
              convertImmutableTensorArg(argValue, torchType, preambleBuilder)))
        return failure();
      break;
    }
    case TypeDisposition::TORCH_PRIMITIVE: {
      Location loc = argValue.getLoc();
      Operation *convertUser = nullptr;
      Value convertResult;
      if (isa<Torch::BoolType>(torchType)) {
        convertUser =
            preambleBuilder.create<TorchConversion::FromI1Op>(loc, argValue);
        convertResult = convertUser->getResult(0);
      } else if (isa<Torch::FloatType>(torchType)) {
        convertUser =
            preambleBuilder.create<TorchConversion::FromF64Op>(loc, argValue);
        convertResult = convertUser->getResult(0);
      } else if (isa<Torch::IntType>(torchType)) {
        convertUser =
            preambleBuilder.create<TorchConversion::FromI64Op>(loc, argValue);
        convertResult = convertUser->getResult(0);
      } else {
        emitError(loc) << "unhandled torch primitive materialization: "
                       << torchType;
        return failure();
      }
      argValue.replaceAllUsesExcept(convertResult, convertUser);
      break;
    }
    case TypeDisposition::PASSTHROUGH:
      // Do nothing.
      break;
    }
  }

  // Materialize return conversions.
  IREE::Util::ReturnOp returnOp = returnOps.front();
  SmallVector<Value> newReturnOperands;
  OpBuilder postambleBuilder(returnOp);
  for (auto [disp, returnValue, torchType] : llvm::zip_equal(
           resultDispositions, returnOp.getOperands(), torchResultTypes)) {
    switch (disp) {
    case TypeDisposition::IMMUTABLE_TENSOR: {
      // Convert back to builtin tensor if needed
      Value toReturn = convertToBuiltinTensor(postambleBuilder, returnValue);
      newReturnOperands.push_back(toReturn);
      break;
    }
    case TypeDisposition::TORCH_PRIMITIVE: {
      Location loc = returnValue.getLoc();
      if (isa<Torch::BoolType>(torchType)) {
        newReturnOperands.push_back(
            postambleBuilder.create<TorchConversion::ToI1Op>(loc, returnValue));
      } else if (isa<Torch::FloatType>(torchType)) {
        newReturnOperands.push_back(
            postambleBuilder.create<TorchConversion::ToF64Op>(loc,
                                                              returnValue));
      } else if (isa<Torch::IntType>(torchType)) {
        newReturnOperands.push_back(
            postambleBuilder.create<TorchConversion::ToI64Op>(loc,
                                                              returnValue));
      } else if (isa<Torch::GeneratorType>(torchType)) {
        newReturnOperands.push_back(
            postambleBuilder.create<TorchConversion::GeneratorToI64Op>(
                loc, returnValue));
      } else {
        emitError(loc) << "unhandled torch primitive materialization: "
                       << torchType;
        return failure();
      }
      break;
    }
    default: {
      // Non-tensor/converting. Just preserve.
      newReturnOperands.push_back(returnValue);
    }
    }
  }

  // New return operands are all collected.
  returnOp->setOperands(newReturnOperands);

  return success();
}

class OriginalUses {
public:
  OriginalUses(Value value) {
    for (auto &use : value.getUses()) {
      originalUses.push_back(&use);
    }
  }

  void assign(Value newValue) {
    for (OpOperand *originalUse : originalUses) {
      originalUse->assign(newValue);
    }
  }

private:
  SmallVector<OpOperand *> originalUses;
};

LogicalResult ConvertedFunctionInfo::convertImmutableTensorArg(
    BlockArgument argValue, Type torchType, OpBuilder &builder) {
  Location loc = argValue.getLoc();

  // If the arg is just directly returned, then don't do anything special with
  // it.
  bool hasNonTrivialUse = false;
  for (auto *userOp : argValue.getUsers()) {
    if (isa<IREE::Util::ReturnOp>(userOp))
      continue;
    hasNonTrivialUse = true;
  }
  if (!hasNonTrivialUse)
    return success();

  // The type can either be a builtin TensorType or a Torch::ValueTensorType.
  // If it's already a builtin tensor, nothing to do.
  if (isa<TensorType>(torchType)) {
    // Already a builtin tensor type, no conversion needed
    return success();
  }

  // Remember original uses so we can redirect them.
  OriginalUses originalUses(argValue);

  // Convert from builtin tensor to torch tensor type
  if (auto vtType = dyn_cast<Torch::ValueTensorType>(torchType)) {
    Value converted = builder.create<TorchConversion::FromBuiltinTensorOp>(
        loc, torchType, argValue);
    originalUses.assign(converted);
    return success();
  }

  return emitError(loc) << "unsupported immutable tensor argument: "
                        << torchType;
}

void retainFunctionAttributes(Operation *srcOp, IREE::Util::FuncOp destOp) {
  // Allowlist of function attributes to retain when importing funcs.
  constexpr const char *kRetainedAttributes[] = {
    "iree.reflection",
    "iree.abi.affinity",
    "noinline",
  };
  auto retainedAttributes = ArrayRef<const char *>(
      kRetainedAttributes,
      sizeof(kRetainedAttributes) / sizeof(kRetainedAttributes[0]));
  for (auto retainAttrName : retainedAttributes) {
    StringRef attrName(retainAttrName);
    Attribute attr = srcOp->getAttr(attrName);
    if (attr)
      destOp->setAttr(attrName, attr);
  }
}

} // namespace

class FuncConversionPass final
    : public impl::FuncConversionPassBase<FuncConversionPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::tensor::TensorDialect>();
    registry.insert<IREE::HAL::HALDialect>();
    registry.insert<IREE::Util::UtilDialect>();
    registry.insert<TorchConversion::TorchConversionDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Convert all functions in the module to IREE funcs. In this stage,
    // we convert contained return ops and argument/result types, but we have
    // not yet converted anything "on the inside". Therefore, it is pretty
    // likely the functions are still illegal.
    SmallVector<Operation *> eraseFuncOps;
    std::vector<ConvertedFunctionInfo> convertedFuncInfos;
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      if (!shouldConvertFunc(funcOp))
        continue;
      ConvertedFunctionInfo &convertedFuncInfo =
          convertedFuncInfos.emplace_back();
      if (failed(convertFuncOp(funcOp, convertedFuncInfo))) {
        signalPassFailure();
        return;
      }
      eraseFuncOps.push_back(funcOp);
    }
    for (auto op : eraseFuncOps) {
      op->erase();
    }

    // Now post-process async functions.
    for (auto &info : convertedFuncInfos) {
      if (failed(info.postProcess())) {
        signalPassFailure();
        return;
      }
    }
  }

  bool shouldConvertFunc(func::FuncOp torchFunc) {
    // For now, we don't touch externals and assume they are in the proper
    // calling convention. In the future, we may support "torch externals"
    // which we convert to mate up with a torch module. We can remove/adapt
    // this when that is elaborated.
    if (torchFunc.isExternal())
      return false;

    // Something has already converted this and told us not to touch it.
    if (torchFunc->hasAttr("iree.abi.stub"))
      return false;

    return true;
  }

  LogicalResult convertFuncOp(func::FuncOp torchFunc,
                              ConvertedFunctionInfo &convertedFuncInfo) {
    IRRewriter rewriter(torchFunc.getContext());
    rewriter.setInsertionPoint(torchFunc);
    Location loc = torchFunc.getLoc();
    StringRef originalName = torchFunc.getName();

    // Stash arg/result attrs so they can be referenced during conversion.
    torchFunc.getAllArgAttrs(convertedFuncInfo.torchArgAttrs);
    torchFunc.getAllResultAttrs(convertedFuncInfo.torchResultAttrs);

    // Convert function signature.
    FunctionType torchFuncType = torchFunc.getFunctionType();
    convertedFuncInfo.torchInputTypes.append(torchFuncType.getInputs().begin(),
                                             torchFuncType.getInputs().end());
    convertedFuncInfo.torchResultTypes.append(
        torchFuncType.getResults().begin(), torchFuncType.getResults().end());

    SmallVector<Type> ireeInputTypes(convertedFuncInfo.torchInputTypes);
    SmallVector<Type> ireeResultTypes(convertedFuncInfo.torchResultTypes);
    convertedFuncInfo.inputDispositions.resize(ireeInputTypes.size());
    convertedFuncInfo.resultDispositions.resize(ireeResultTypes.size());

    for (size_t i = 0; i < convertedFuncInfo.torchInputTypes.size(); ++i) {
      if (failed(convertType(loc, convertedFuncInfo.torchInputTypes[i],
                             ireeInputTypes[i],
                             convertedFuncInfo.inputDispositions[i])))
        return failure();
    }
    for (size_t i = 0; i < convertedFuncInfo.torchResultTypes.size(); ++i) {
      if (failed(convertType(loc, convertedFuncInfo.torchResultTypes[i],
                             ireeResultTypes[i],
                             convertedFuncInfo.resultDispositions[i])))
        return failure();
    }

        // Build tied operands index mapping results back to operands.
    SmallVector<int64_t> tiedOperands;
    bool anyTiedOperands = false;
    for (unsigned i = 0; i < torchFuncType.getNumResults(); ++i) {
      auto tiedAttr =
          torchFunc.getResultAttrOfType<IntegerAttr>(i, "iree.abi.tied");
      if (tiedAttr) {
        tiedOperands.push_back(tiedAttr.getInt());
        anyTiedOperands = true;
      } else {
        tiedOperands.push_back(-1);
      }
    }

    auto tiedOperandsAttr = anyTiedOperands
                            ? rewriter.getIndexArrayAttr(tiedOperands)
                            : ArrayAttr{};

    // Create new func.
    FunctionType syncFuncType =
        FunctionType::get(loc.getContext(), ireeInputTypes, ireeResultTypes);
    auto syncFuncOp = rewriter.create<IREE::Util::FuncOp>(
        torchFunc.getLoc(), originalName, syncFuncType, tiedOperandsAttr);
    convertedFuncInfo.funcOp = syncFuncOp;
    syncFuncOp.setSymVisibilityAttr(torchFunc.getSymVisibilityAttr());
    retainFunctionAttributes(torchFunc, syncFuncOp);

    // Copy argument attributes
    for (unsigned i = 0; i < torchFunc.getNumArguments(); ++i) {
      auto argAttrs = torchFunc.getArgAttrDict(i);
      if (argAttrs && !argAttrs.empty()) {
        syncFuncOp.setArgAttrs(i, argAttrs);
      }
    }

    // Copy result attributes (including iree.abi.tied)
    for (unsigned i = 0; i < torchFunc.getNumResults(); ++i) {
      auto resAttrs = torchFunc.getResultAttrDict(i);
      if (resAttrs && !resAttrs.empty()) {
        syncFuncOp.setResultAttrs(i, resAttrs);
      }
    }

    rewriter.inlineRegionBefore(torchFunc.getBody(),
                                syncFuncOp.getFunctionBody(), syncFuncOp.end());

    // Convert block arguments.
    Block *entryBlock = &syncFuncOp.getBlocks().front();
    for (size_t i = 0; i < ireeInputTypes.size(); ++i) {
      // Add if we have extended the list.
      if (i >= entryBlock->getNumArguments()) {
        entryBlock->addArgument(ireeInputTypes[i], loc);
        continue;
      }
      // Convert.
      entryBlock->getArgument(i).setType(ireeInputTypes[i]);
    }

    // Replace return ops.
    syncFuncOp->walk([&](func::ReturnOp returnOp) {
      rewriter.setInsertionPoint(returnOp);
      auto ireeReturnOp = rewriter.replaceOpWithNewOp<IREE::Util::ReturnOp>(
          returnOp, returnOp.getOperands());
      convertedFuncInfo.returnOps.push_back(ireeReturnOp);
    });

    return success();
  }

  LogicalResult convertType(Location loc, Type torchType, Type &ireeType,
                            TypeDisposition &disp) {
    if (isa<TensorType>(torchType)) {
      // Already a builtin tensor type, just pass through
      ireeType = torchType;
      disp = TypeDisposition::IMMUTABLE_TENSOR;
      return success();
    }

    if (isa<Torch::ValueTensorType>(torchType)) {
      // Convert to builtin tensor type
      auto vtType = cast<Torch::ValueTensorType>(torchType);
      ireeType = vtType.toBuiltinTensor();
      if (auto intTy = dyn_cast<IntegerType>(
              cast<TensorType>(ireeType).getElementType())) {
        ireeType = cast<TensorType>(ireeType).clone(IntegerType::get(
            torchType.getContext(), intTy.getIntOrFloatBitWidth()));
      }
      disp = TypeDisposition::IMMUTABLE_TENSOR;
      return success();
    }

    if (isa<Torch::NonValueTensorType>(torchType)) {
      // Mutable tensors are not supported
      return emitError(loc)
             << "mutable tensor arguments are not supported: " << torchType;
    }

    if (isa<Torch::BoolType>(torchType)) {
      ireeType = IntegerType::get(torchType.getContext(), 1);
      disp = TypeDisposition::TORCH_PRIMITIVE;
      return success();
    }

    if (isa<Torch::IntType, Torch::GeneratorType>(torchType)) {
      ireeType = IntegerType::get(torchType.getContext(), 64);
      disp = TypeDisposition::TORCH_PRIMITIVE;
      return success();
    }

    if (isa<Torch::FloatType>(torchType)) {
      ireeType = Float64Type::get(torchType.getContext());
      disp = TypeDisposition::TORCH_PRIMITIVE;
      return success();
    }

    if (isa<IntegerType, FloatType, IndexType>(torchType)) {
      ireeType = torchType;
      disp = TypeDisposition::PASSTHROUGH;
      return success();
    }

    return emitError(loc) << "unhandled torch type: " << torchType;
  }
};

} // namespace mlir::iree_compiler::TorchInput
