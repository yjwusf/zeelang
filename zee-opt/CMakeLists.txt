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

mlir_check_all_link_libraries(zee-opt)
