set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ZeeConversions)
add_public_tablegen_target(MLIRZeeConversionsPassesIncGen)
add_dependencies(mlir-headers MLIRZeeConversionsPassesIncGen)
