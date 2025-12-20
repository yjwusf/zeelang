# On macOS with Homebrew LLVM, use the dynamic MLIR library to avoid
# mixing static/dynamic libraries which causes parser compatibility issues
if(APPLE)
  set(LIBS
    MLIR
    MLIRZee
  )
else()
  set(LIBS
    MLIRArithDialect
    MLIRFuncDialect
    MLIRTensorDialect
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRTransformUtils
    MLIRSupport
    MLIRMlirOptMain
    MLIRZee
  )
endif()

if(ZEE_ENABLE_STABLEHLO)
  list(APPEND LIBS
    MLIRZeeStableHLOToZee
    StablehloRegister
    StablehloOps
    ChloOps
    VhloOps
  )
endif()

add_llvm_executable(zee-opt zee-opt.cpp)

llvm_update_compile_flags(zee-opt)

target_link_libraries(zee-opt PRIVATE ${LIBS})

if(NOT APPLE)
  mlir_check_all_link_libraries(zee-opt)
endif()
