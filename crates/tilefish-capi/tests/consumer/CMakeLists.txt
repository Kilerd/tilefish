# A program that uses the installed package as any consumer would: find_package(tilefish) and the imported targets (tests/consumer/run.cmake
# installs the package to a scratch prefix and builds this against it).
cmake_minimum_required(VERSION 3.19)
project(tilefish_consumer LANGUAGES CXX)
find_package(tilefish 0.0 CONFIG REQUIRED)
foreach(kind shared static)
    add_executable(consumer_${kind} consumer.cpp)
    set_target_properties(consumer_${kind} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    target_link_libraries(consumer_${kind} PRIVATE tilefish::tilefish_${kind})
    if(kind STREQUAL "shared")
        set_target_properties(consumer_${kind} PROPERTIES BUILD_RPATH "${tilefish_LIB_DIR}")
    endif()
endforeach()
