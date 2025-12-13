add_mlir_library(MLIRZeeStableHLOToZee
  StableHLOToZee.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Zee/Conversions

  DEPENDS
  MLIRZeeConversionsPassesIncGen

  LINK_LIBS PUBLIC
  MLIRZee
  StablehloOps
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
)
