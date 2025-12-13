set(LLVM_TARGET_DEFINITIONS ZeeOps.td)
mlir_tablegen(ZeeOps.h.inc -gen-op-decls)
mlir_tablegen(ZeeOps.cpp.inc -gen-op-defs)
mlir_tablegen(ZeeOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(ZeeOpsDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRZeeOpsIncGen)
add_mlir_doc(ZeeOps ZeeOps Zee/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS ZeeTypes.td)
mlir_tablegen(ZeeOpsTypes.h.inc -gen-typedef-decls)
mlir_tablegen(ZeeOpsTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRZeeTypesIncGen)
add_mlir_doc(ZeeTypes ZeeTypes Zee/ -gen-typedef-doc)

set(LLVM_TARGET_DEFINITIONS ZeePasses.td)
mlir_tablegen(ZeePasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRZeePassesIncGen)
add_mlir_doc(ZeePasses ZeePasses Zee/ -gen-pass-doc)

# Conversion passes
add_subdirectory(Conversions)
