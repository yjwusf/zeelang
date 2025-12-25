cmake_minimum_required(VERSION 3.20.0)
project(zee-dialect LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

# Option to enable StableHLO support
option(ZEE_ENABLE_STABLEHLO "Enable StableHLO dialect support" ON)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
else()
  # Build via external projects mechanism
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
endif()

set(ZEE_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(ZEE_BINARY_DIR ${PROJECT_BINARY_DIR})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

#-------------------------------------------------------------------------------
# StableHLO Configuration
#-------------------------------------------------------------------------------
if(ZEE_ENABLE_STABLEHLO)
  # StableHLO can be provided externally or built as a subdirectory
  if(NOT STABLEHLO_SOURCE_DIR)
    set(STABLEHLO_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../stablehlo")
  endif()

  if(EXISTS "${STABLEHLO_SOURCE_DIR}/CMakeLists.txt")
    message(STATUS "Building StableHLO from: ${STABLEHLO_SOURCE_DIR}")
    set(STABLEHLO_BUILD_EMBEDDED ON CACHE BOOL "" FORCE)
    add_subdirectory(${STABLEHLO_SOURCE_DIR} ${CMAKE_BINARY_DIR}/stablehlo EXCLUDE_FROM_ALL)

    # Add StableHLO include directories
    include_directories(${STABLEHLO_SOURCE_DIR})
    include_directories(${CMAKE_BINARY_DIR}/stablehlo)

    add_definitions(-DZEE_ENABLE_STABLEHLO)
  else()
    message(WARNING "StableHLO not found at ${STABLEHLO_SOURCE_DIR}, disabling StableHLO support")
    set(ZEE_ENABLE_STABLEHLO OFF)
  endif()
endif()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(zee-opt)
add_subdirectory(zee-jit)
