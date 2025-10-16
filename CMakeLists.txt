cmake_minimum_required(VERSION 3.10)
project(Jam)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find LLVM package
find_package(LLVM REQUIRED CONFIG)

# Print LLVM information
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Include LLVM directories
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Add compiler executable
add_executable(jam 
  src/main.cpp
  src/lexer.cpp
  src/parser.cpp
  src/ast.cpp
  src/codegen.cpp
  src/target.cpp
  src/cabi.cpp
)

# Get proper link libraries for LLVM
llvm_map_components_to_libnames(llvm_libs
  Core
  ExecutionEngine
  Interpreter
  MC
  MCJIT
  Support
  nativecodegen
  OrcJIT
  native
)

# Link against LLVM libraries
target_link_libraries(jam ${llvm_libs})

# Installation rules
include(GNUInstallDirs)

# Set installation prefix if not specified
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Default install prefix" FORCE)
endif()

# Install the jam executable
install(TARGETS jam 
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT Runtime
)

# Install documentation files
install(FILES 
  README.md
  LICENSE
  LANGUAGE_SPEC.md
  DESTINATION ${CMAKE_INSTALL_DOCDIR}
  COMPONENT Documentation
)

# Create a simple man page directory structure (optional)
install(DIRECTORY DESTINATION ${CMAKE_INSTALL_MANDIR}/man1)

# Print installation info
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
