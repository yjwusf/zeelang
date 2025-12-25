# On macOS with Homebrew LLVM, use the dynamic MLIR library to avoid
# mixing static/dynamic libraries which causes parser compatibility issues
if(APPLE)
  set(LIBS
    MLIR
    MLIRExecutionEngineShared
    MLIRZee
    MLIRZeeToArith
    MLIRZeeLowerToLLVM
  )
else()
  set(LIBS
    # Core MLIR
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRTransformUtils
    MLIRSupport

    # Dialects
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRTensorDialect

    # Conversions
    MLIRAffineToStandard
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
    MLIRMemRefToLLVM
    MLIRSCFToControlFlow

    # Execution Engine
    MLIRExecutionEngine
    MLIRTargetLLVMIRExport
    MLIRLLVMToLLVMIRTranslation
    MLIRBuiltinToLLVMIRTranslation

    # Zee libraries
    MLIRZee
    MLIRZeeToArith
    MLIRZeeLowerToLLVM
  )
endif()

add_llvm_executable(zee-jit zee-jit.cpp)

llvm_update_compile_flags(zee-jit)

target_link_libraries(zee-jit PRIVATE ${LIBS})

if(NOT APPLE)
  mlir_check_all_link_libraries(zee-jit)
endif()
