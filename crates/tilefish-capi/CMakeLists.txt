# The C and C++ host of tilefish as a CMake package: libtilefish_capi (built with cargo), tilefish.h and tilefish.hpp.
#
#   cmake -S crates/tilefish-capi -B build && cmake --build build && ctest --test-dir build      # build and test (no GPU needed)
#   cmake --install build --prefix /opt/tilefish                                               # install the package
#
# A consumer:
#
#   find_package(tilefish CONFIG REQUIRED)                       # or add_subdirectory(<repo>/crates/tilefish-capi)
#   target_link_libraries(app PRIVATE tilefish::tilefish)        # the shared library; tilefish::tilefish_static for the static one
#
# The libraries link nothing of CUDA at build time: the driver, cuBLASLt and cuDNN are opened at run time (docs/rfcs/0002-cpp-host.md, section 8).
cmake_minimum_required(VERSION 3.19)
project(tilefish_capi VERSION 0.0.1 LANGUAGES C CXX)

set(TILEFISH_ABI_MAJOR 1)  # TILEFISH_ABI_MAJOR of src/lib.rs and build.rs: the shared library's SONAME
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(tilefish_top_level ON)
else()
    set(tilefish_top_level OFF)
endif()
option(TILEFISH_BUILD_TESTS "Build the C and C++ tests of the C ABI (no GPU needed)" ${tilefish_top_level})
option(TILEFISH_BUILD_GPU_GATE "Build the C++ half of the GPU gate (python/tests/rust_gate.py; needs TILEFISH_GATE_DIR)" OFF)
set(TILEFISH_CARGO_PROFILE "release" CACHE STRING "The cargo profile of libtilefish_capi: release or dev")
set(TILEFISH_CARGO_FEATURES "" CACHE STRING
    "The cuda-* feature of the bindings, for a driver older than the default's (e.g. cuda-12080: built with --no-default-features)")

find_program(TILEFISH_CARGO cargo REQUIRED)
find_package(Threads REQUIRED)
include(GNUInstallDirs)

# -- libtilefish_capi, built by cargo --------------------------------------------------------------------------------------------------------
set(tilefish_cargo_dir "${CMAKE_CURRENT_BINARY_DIR}/cargo")
if(TILEFISH_CARGO_PROFILE STREQUAL "release")
    set(tilefish_profile_args --release)
    set(tilefish_profile_dir release)
else()
    set(tilefish_profile_args --profile ${TILEFISH_CARGO_PROFILE})
    if(TILEFISH_CARGO_PROFILE STREQUAL "dev")
        set(tilefish_profile_dir debug)
    else()
        set(tilefish_profile_dir ${TILEFISH_CARGO_PROFILE})
    endif()
endif()
set(tilefish_feature_args "")
if(TILEFISH_CARGO_FEATURES)
    set(tilefish_feature_args --no-default-features --features ${TILEFISH_CARGO_FEATURES})
endif()

set(tilefish_lib_dir "${CMAKE_CURRENT_BINARY_DIR}/lib")
if(APPLE)
    set(tilefish_cargo_shared libtilefish_capi.dylib)
    set(tilefish_shared_file libtilefish_capi.dylib)  # its install name is @rpath/libtilefish_capi.dylib (build.rs)
    set(tilefish_static_libs "-liconv")
elseif(WIN32)
    message(FATAL_ERROR "tilefish's C host is for Linux (and macOS, without CUDA)")
else()
    set(tilefish_cargo_shared libtilefish_capi.so)
    set(tilefish_shared_file libtilefish_capi.so.${TILEFISH_ABI_MAJOR})  # its SONAME (build.rs)
    set(tilefish_static_libs ${CMAKE_DL_LIBS} m rt)
endif()

file(GLOB_RECURSE tilefish_rust_sources CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.rs" "${CMAKE_CURRENT_SOURCE_DIR}/../tilefish/src/*")
set(tilefish_outputs "${tilefish_lib_dir}/${tilefish_shared_file}" "${tilefish_lib_dir}/libtilefish_capi.a")
set(tilefish_copy_commands
    COMMAND ${CMAKE_COMMAND} -E make_directory "${tilefish_lib_dir}"
    COMMAND ${CMAKE_COMMAND} -E copy "${tilefish_cargo_dir}/${tilefish_profile_dir}/${tilefish_cargo_shared}"
            "${tilefish_lib_dir}/${tilefish_shared_file}"
    COMMAND ${CMAKE_COMMAND} -E copy "${tilefish_cargo_dir}/${tilefish_profile_dir}/libtilefish_capi.a" "${tilefish_lib_dir}/libtilefish_capi.a")
if(NOT APPLE)
    list(APPEND tilefish_outputs "${tilefish_lib_dir}/libtilefish_capi.so")
    list(APPEND tilefish_copy_commands
        COMMAND ${CMAKE_COMMAND} -E create_symlink ${tilefish_shared_file} "${tilefish_lib_dir}/libtilefish_capi.so")
endif()
add_custom_command(
    OUTPUT ${tilefish_outputs}
    COMMAND ${TILEFISH_CARGO} build -p tilefish-capi --lib ${tilefish_profile_args} --target-dir "${tilefish_cargo_dir}" ${tilefish_feature_args}
    ${tilefish_copy_commands}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS ${tilefish_rust_sources} "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml" "${CMAKE_CURRENT_SOURCE_DIR}/build.rs"
            "${CMAKE_CURRENT_SOURCE_DIR}/../tilefish/Cargo.toml"
    COMMENT "cargo build -p tilefish-capi (${TILEFISH_CARGO_PROFILE})"
    USES_TERMINAL
    VERBATIM)
add_custom_target(tilefish_capi_cargo ALL DEPENDS ${tilefish_outputs})

# the targets of the build tree (the installed package makes the same names: cmake/tilefishConfig.cmake.in)
add_library(tilefish_capi_shared INTERFACE)
target_include_directories(tilefish_capi_shared INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_link_libraries(tilefish_capi_shared INTERFACE "${tilefish_lib_dir}/${tilefish_shared_file}")
add_dependencies(tilefish_capi_shared tilefish_capi_cargo)
add_library(tilefish::tilefish ALIAS tilefish_capi_shared)
add_library(tilefish::tilefish_shared ALIAS tilefish_capi_shared)

add_library(tilefish_capi_static INTERFACE)
target_include_directories(tilefish_capi_static INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_link_libraries(tilefish_capi_static INTERFACE "${tilefish_lib_dir}/libtilefish_capi.a" Threads::Threads ${tilefish_static_libs})
add_dependencies(tilefish_capi_static tilefish_capi_cargo)
add_library(tilefish::tilefish_static ALIAS tilefish_capi_static)

# -- install: headers, libraries, package config ---------------------------------------------------------------------------------------------
install(FILES include/tilefish.h include/tilefish.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${tilefish_outputs} DESTINATION ${CMAKE_INSTALL_LIBDIR})
include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/tilefishConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/tilefishConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tilefish PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/tilefishConfigVersion.cmake" VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/tilefishConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/tilefishConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tilefish)

# -- tests ---------------------------------------------------------------------------------------------------------------------------------
function(tilefish_strict target)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -Wall -Wextra -Werror)
    endif()
endfunction()

if(TILEFISH_BUILD_TESTS)
    enable_testing()
    set(tilefish_repo "${CMAKE_CURRENT_SOURCE_DIR}/../..")
    file(READ "${tilefish_repo}/crates/tilefish/tests/fixtures/expected.json" tilefish_fixtures_json)
    string(JSON tilefish_fixture_pin GET "${tilefish_fixtures_json}" fixtures sm_103 manifest_digest)
    file(READ "${tilefish_repo}/python/tests/golden/plans/expected.json" tilefish_plans_json)
    string(JSON tilefish_block_pin GET "${tilefish_plans_json}" block plan_digest)
    string(JSON tilefish_carried_pin GET "${tilefish_plans_json}" carried set_digest)
    set(tilefish_test_args
        "${tilefish_repo}/crates/tilefish/tests/fixtures/sm_103" ${tilefish_fixture_pin}
        "${tilefish_repo}/python/tests/golden/plans/block" ${tilefish_block_pin}
        "${tilefish_repo}/python/tests/golden/plans/carried" ${tilefish_carried_pin})

    foreach(kind shared static)
        add_executable(tilefish_c_abi_${kind} tests/c/abi.c)
        set_target_properties(tilefish_c_abi_${kind} PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
        target_link_libraries(tilefish_c_abi_${kind} PRIVATE tilefish::tilefish_${kind})
        tilefish_strict(tilefish_c_abi_${kind})
        add_test(NAME c_abi_${kind} COMMAND tilefish_c_abi_${kind} ${tilefish_test_args})

        add_executable(tilefish_cpp_wrapper_${kind} tests/cpp/wrapper.cpp)
        set_target_properties(tilefish_cpp_wrapper_${kind} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
        target_link_libraries(tilefish_cpp_wrapper_${kind} PRIVATE tilefish::tilefish_${kind})
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/generated.hpp")
            # the generated launchers and plan runner of the committed fixtures (python/tests/test_hosts_cpp.py keeps them current)
            target_include_directories(tilefish_cpp_wrapper_${kind} PRIVATE tests/fixtures)
            target_compile_definitions(tilefish_cpp_wrapper_${kind} PRIVATE TILEFISH_HAVE_GENERATED)
        endif()
        tilefish_strict(tilefish_cpp_wrapper_${kind})
        add_test(NAME cpp_wrapper_${kind} COMMAND tilefish_cpp_wrapper_${kind} ${tilefish_test_args})
    endforeach()
    # the installed package, as a consumer finds it
    add_test(NAME package_consumer COMMAND "${CMAKE_COMMAND}" "-DBUILD=${CMAKE_CURRENT_BINARY_DIR}"
        "-DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/tests/consumer" "-DWORK=${CMAKE_CURRENT_BINARY_DIR}/consumer"
        "-DMANIFEST=${tilefish_repo}/crates/tilefish/tests/fixtures/sm_103" "-DPIN=${tilefish_fixture_pin}"
        "-DGENERATOR=${CMAKE_GENERATOR}" "-DMAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}" "-DCXX=${CMAKE_CXX_COMPILER}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/consumer/run.cmake")
endif()

if(TILEFISH_BUILD_GPU_GATE)
    set(TILEFISH_GATE_DIR "" CACHE PATH "The directory python/tests/rust_gate.py prepare filled (generated.hpp, generated_embedded.cpp, cases.json)")
    if(NOT EXISTS "${TILEFISH_GATE_DIR}/generated.hpp")
        message(FATAL_ERROR "TILEFISH_GATE_DIR=${TILEFISH_GATE_DIR} has no generated.hpp: run python/tests/rust_gate.py prepare first")
    endif()
    add_executable(tilefish_cpp_gate tests/gpu/cpp_gate.cpp "${TILEFISH_GATE_DIR}/generated_embedded.cpp")
    set_target_properties(tilefish_cpp_gate PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    target_include_directories(tilefish_cpp_gate PRIVATE "${TILEFISH_GATE_DIR}" tests/gpu)
    target_link_libraries(tilefish_cpp_gate PRIVATE tilefish::tilefish Threads::Threads ${CMAKE_DL_LIBS})
    tilefish_strict(tilefish_cpp_gate)
endif()
