set(LIBS
  MLIRArithDialect
  MLIRFuncDialect
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRMlirOptMain
  MLIRZee
)

add_llvm_executable(zee-opt zee-opt.cpp)

llvm_update_compile_flags(zee-opt)

target_link_libraries(zee-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(zee-opt)
