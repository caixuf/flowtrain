cmake_minimum_required(VERSION 3.16)
project(flowtrain VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_library(flowtrain INTERFACE)
target_include_directories(flowtrain INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../flowcoro/include>
    /home/caixuf/.local/lib/python3.12/site-packages/nvidia/cu13/include
)

find_package(Threads REQUIRED)
find_library(CUDA_DRIVER_LIB NAMES cuda PATHS /usr/lib/wsl/lib /usr/lib/x86_64-linux-gnu)

set(FLOWCORO_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../flowcoro/build/libflowcoro_net.a")

target_link_libraries(flowtrain INTERFACE
    Threads::Threads
    atomic
)
if(EXISTS ${FLOWCORO_LIB})
    target_link_libraries(flowtrain INTERFACE ${FLOWCORO_LIB})
endif()
if(CUDA_DRIVER_LIB)
    target_link_libraries(flowtrain INTERFACE ${CUDA_DRIVER_LIB})
endif()

enable_testing()

function(flowtrain_test name)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} PRIVATE flowtrain)
    target_compile_options(${name} PRIVATE -UNDEBUG)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

flowtrain_test(test_dp_bit_parity)
flowtrain_test(test_tp_gemm)
flowtrain_test(test_pp_1f1b)
flowtrain_test(test_tp_mla)
flowtrain_test(test_async_pipeline)

add_executable(bench_train tools/bench_train.cpp)
target_link_libraries(bench_train PRIVATE flowtrain)
