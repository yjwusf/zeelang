# ZeeToArith is always available
add_subdirectory(ZeeToArith)

if(ZEE_ENABLE_STABLEHLO)
  add_subdirectory(StableHLOToZee)
endif()
