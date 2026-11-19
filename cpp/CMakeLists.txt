cmake_minimum_required(VERSION 3.20)
project(chaosys VERSION 1.0.0
        DESCRIPTION "Chaotic dynamics library (Lorenz, Rossler, Chen and custom chaotic ODEs)"
        LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(CHAOSYS_BUILD_CLI "Build the chaosys_cli executable" ON)
option(CHAOSYS_BUILD_BENCH "Build the benchmark executable" OFF)
option(CHAOSYS_NATIVE "Build with -mcpu/-march=native" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)

function(chaosys_enable_native target)
  if(NOT CHAOSYS_NATIVE)
    return()
  endif()
  if(APPLE)
    target_compile_options(${target} PRIVATE -mcpu=native)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    target_compile_options(${target} PRIVATE -mcpu=native)
  else()
    target_compile_options(${target} PRIVATE -march=native)
  endif()
endfunction()

function(chaosys_add_test target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE chaosys)
  set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX)
  add_test(NAME ${target} COMMAND ${target})
endfunction()

add_library(chaosys src/c_api.cpp src/kernels.cpp)
add_library(chaosys::chaosys ALIAS chaosys)

target_include_directories(chaosys PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(chaosys PUBLIC Threads::Threads)
target_compile_features(chaosys PUBLIC cxx_std_20)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(chaosys PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter)
endif()
chaosys_enable_native(chaosys)
set_target_properties(chaosys PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR})

if(CHAOSYS_BUILD_CLI)
  add_executable(chaosys_cli apps/chaosys_cli.cpp)
  target_link_libraries(chaosys_cli PRIVATE chaosys)
  target_compile_features(chaosys_cli PRIVATE cxx_std_20)
  if(APPLE)
    set_target_properties(chaosys_cli PROPERTIES
      INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
  else()
    set_target_properties(chaosys_cli PROPERTIES
      INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
  endif()
  install(TARGETS chaosys_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(BUILD_TESTING)
  chaosys_add_test(chaosys_tests tests/test_chaosys.cpp)
  chaosys_add_test(test_c_api tests/test_c_api.c)
  chaosys_add_test(test_exception_safety tests/test_exception_safety.cpp)
  chaosys_add_test(test_time_header tests/test_time_header.cpp)
endif()

if(CHAOSYS_BUILD_BENCH)
  add_executable(chaosys_bench src/bench.cpp src/baseline.cpp src/kernels_fast.cpp)
  target_link_libraries(chaosys_bench PRIVATE chaosys)
  set_source_files_properties(src/kernels_fast.cpp PROPERTIES
    COMPILE_OPTIONS "-ffast-math")
  chaosys_enable_native(chaosys_bench)
endif()

install(TARGETS chaosys
  EXPORT chaosysTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES include/chaosys_c.h include/chaosys.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY include/chaosys DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT chaosysTargets
  FILE chaosysTargets.cmake
  NAMESPACE chaosys::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/chaosys)

configure_package_config_file(
  cmake/chaosysConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/chaosysConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/chaosys)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/chaosysConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/chaosysConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/chaosysConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/chaosys)

# Relative path from <prefix>/<libdir>/pkgconfig back to <prefix>.  This works
# for plain lib/ and for multiarch libdirs such as lib/x86_64-linux-gnu.
file(RELATIVE_PATH CHAOSYS_PC_PREFIX_REL
     "/${CMAKE_INSTALL_LIBDIR}/pkgconfig" "/")
string(REGEX REPLACE "/$" "" CHAOSYS_PC_PREFIX_REL "${CHAOSYS_PC_PREFIX_REL}")
if(APPLE)
  set(CHAOSYS_PC_CXX_RUNTIME "-lc++")
else()
  set(CHAOSYS_PC_CXX_RUNTIME "-lstdc++")
endif()
configure_file(cmake/chaosys.pc.in ${CMAKE_CURRENT_BINARY_DIR}/chaosys.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/chaosys.pc
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/../docs/QUICKSTART.md
  ${CMAKE_CURRENT_SOURCE_DIR}/../docs/API.md
  ${CMAKE_CURRENT_SOURCE_DIR}/../docs/PURPOSE.md
  ${CMAKE_CURRENT_SOURCE_DIR}/../docs/EXTENDING.md
  ${CMAKE_CURRENT_SOURCE_DIR}/../docs/RIGOR.md
  DESTINATION ${CMAKE_INSTALL_DOCDIR})
