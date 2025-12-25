# ZeeToArith is always available
add_subdirectory(ZeeToArith)

# LowerToLLVM for JIT execution
add_subdirectory(LowerToLLVM)

if(ZEE_ENABLE_STABLEHLO)
  add_subdirectory(StableHLOToZee)
endif()
