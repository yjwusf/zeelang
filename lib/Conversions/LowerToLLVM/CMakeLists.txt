add_mlir_library(MLIRZeeLowerToLLVM
  LowerToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Zee/Conversions

  DEPENDS
  MLIRZeeConversionsPassesIncGen

  LINK_LIBS PUBLIC
  MLIRZee
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRAffineToStandard
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRSCFToControlFlow
  MLIRLLVMDialect
)
