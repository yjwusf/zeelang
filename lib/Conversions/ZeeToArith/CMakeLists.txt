add_mlir_library(MLIRZeeToArith
  ZeeToArith.cpp

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
)
