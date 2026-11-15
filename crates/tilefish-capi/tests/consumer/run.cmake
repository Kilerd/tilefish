# ctest's package_consumer: install the build tree to a scratch prefix, build tests/consumer against it with find_package(tilefish), and run
# the consumer linked to the shared and to the static library.
#   cmake -DBUILD=<build dir> -DSOURCE=<tests/consumer> -DWORK=<scratch dir> -DMANIFEST=<dir> -DPIN=<digest>
#         -DGENERATOR=<generator> -DMAKE_PROGRAM=<its program> -DCXX=<C++ compiler> -P run.cmake      (those of the build that runs the test)
file(REMOVE_RECURSE "${WORK}")
execute_process(COMMAND "${CMAKE_COMMAND}" --install "${BUILD}" --prefix "${WORK}/prefix" RESULT_VARIABLE rc)
if(rc)
    message(FATAL_ERROR "install failed: ${rc}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${SOURCE}" -B "${WORK}/build" -G "${GENERATOR}" "-DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM}"
    "-DCMAKE_CXX_COMPILER=${CXX}" "-DCMAKE_PREFIX_PATH=${WORK}/prefix" RESULT_VARIABLE rc)
if(rc)
    message(FATAL_ERROR "configuring the consumer failed: ${rc}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${WORK}/build" RESULT_VARIABLE rc)
if(rc)
    message(FATAL_ERROR "building the consumer failed: ${rc}")
endif()
foreach(kind shared static)
    execute_process(COMMAND "${WORK}/build/consumer_${kind}" "${MANIFEST}" "${PIN}" RESULT_VARIABLE rc)
    if(rc)
        message(FATAL_ERROR "consumer_${kind} failed: ${rc}")
    endif()
endforeach()
