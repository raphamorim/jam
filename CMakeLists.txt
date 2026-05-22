# Jam CMakeList
cmake_minimum_required(VERSION 3.10)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

project(Jam)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_executable(jam
  src/abi.cpp
  src/analyzer.cpp
  src/astgen.cpp
  src/cabi.cpp
  src/codegen.cpp
  src/comptime.cpp
  src/decl.cpp
  src/diagnostics.cpp
  src/drop_registry.cpp
  src/init_analysis.cpp
  src/jam_llvm.cpp
  src/jir_codegen.cpp
  src/jir_verify.cpp
  src/lexer.cpp
  src/main.cpp
  src/module_resolver.cpp
  src/number_literal.cpp
  src/parser.cpp
  src/symbol_table.cpp
  src/target.cpp
)

llvm_map_components_to_libnames(llvm_libs
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  Core
  ExecutionEngine
  Interpreter
  MC
  MCJIT
  OrcJIT
  Passes
  Support
  native
  nativecodegen
)
target_link_libraries(jam ${llvm_libs})
include(GNUInstallDirs)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Default install prefix" FORCE)
endif()

install(TARGETS jam 
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT Runtime
)

install(FILES
  README.md
  LICENSE
  DESTINATION ${CMAKE_INSTALL_DOCDIR}
  COMPONENT Documentation
)

install(DIRECTORY std/
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/jam/std
  COMPONENT Runtime
)

install(DIRECTORY DESTINATION ${CMAKE_INSTALL_MANDIR}/man1)

message(STATUS "Jam will be installed to: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Executable will be installed to: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}")
message(STATUS "Documentation will be installed to: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DOCDIR}")

# Add uninstall target
# if(NOT TARGET uninstall)
#   configure_file(
#     "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
#     "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
#     IMMEDIATE @ONLY)
#
#   add_custom_target(uninstall
#     COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
# endif()
