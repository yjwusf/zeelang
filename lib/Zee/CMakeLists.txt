add_mlir_dialect_library(MLIRZee
  ZeeTypes.cpp
  ZeeDialect.cpp
  ZeeOps.cpp
  ZeePasses.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Zee

  DEPENDS
  MLIRZeeOpsIncGen
  MLIRZeeTypesIncGen
  MLIRZeePassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRFuncDialect
)
