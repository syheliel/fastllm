cmake_minimum_required(VERSION 3.21)

project(fastllm LANGUAGES CXX)

option(USE_CUDA "use cuda" OFF)
option(USE_ROCM "use rocm" OFF)

option(PY_API "python api" OFF)
option(USE_MMAP "use mmap" OFF)

option(USE_SENTENCEPIECE "use sentencepiece" OFF)
option(CMAKE_CUDA_ARCHITECTURES "The CUDA arch: native or number" "native")

message(STATUS "USE_CUDA: ${USE_CUDA}")
message(STATUS "USE_ROCM: ${USE_ROCM}")
message(STATUS "PYTHON_API: ${PY_API}")

message(STATUS "USE_SENTENCEPIECE: ${USE_SENTENCEPIECE}")

link_directories($ENV{CONDA_PREFIX}/lib)
include_directories($ENV{CONDA_PREFIX}/include)
set(CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX} ${CMAKE_PREFIX_PATH})

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --std=c++17 -O3 -march=native")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX -O2 /std:c++17 /arch:AVX /source-charset:utf-8")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --std=c++17 -O2 -march=native")
endif()


message(STATUS "CMAKE_CXX_FLAGS" ${CMAKE_CXX_FLAGS})
set(FASTLLM_CXX_SOURCES src/fastllm.cpp src/device.cpp src/model.cpp src/executor.cpp
        src/devices/cpu/cpudevice.cpp src/devices/cpu/cpudevicebatch.cpp
        src/models/chatglm.cpp src/models/moss.cpp src/models/llama.cpp src/models/qwen.cpp src/models/basellm.cpp src/models/glm.cpp)

include_directories(include)
include_directories(include/utils)
include_directories(include/models)

if (USE_MMAP)
    add_compile_definitions(USE_MMAP)
endif()

if (USE_SENTENCEPIECE)
    set(CMAKE_CXX_STANDARD 17)
    add_compile_definitions(USE_SENTENCEPIECE)
    set(FASTLLM_LINKED_LIBS ${FASTLLM_LINKED_LIBS} sentencepiece)
endif()

if (USE_CUDA)
    enable_language(CUDA)
    add_compile_definitions(USE_CUDA)
    include_directories(include/devices/cuda)
    #message(${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    set(FASTLLM_CUDA_SOURCES src/devices/cuda/cudadevice.cpp src/devices/cuda/cudadevicebatch.cpp src/devices/cuda/fastllm-cuda.cu)
    set(FASTLLM_LINKED_LIBS ${FASTLLM_LINKED_LIBS} cublas)
  link_directories($ENV{CONDA_PREFIX}/lib)
  include_directories($ENV{CONDA_PREFIX}/include)
  set(CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX} ${CMAKE_PREFIX_PATH})

  set(CMAKE_CUDA_COMPILER $ENV{CONDA_PREFIX}/bin/nvcc)
  set(CUDAToolkit_ROOT $ENV{CONDA_PREFIX})
endif()

if (USE_ROCM)
    enable_language(HIP)
    add_compile_definitions(USE_ROCM)
    if (DEFINED $ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH})
    else()
        set(ROCM_PATH /opt/rocm)
    endif()
    set(CMAKE_HIP_COMPILER_ROCM_ROOT ${ROCM_PATH})
    find_package(rocprim REQUIRED CONFIG PATHS ${ROCM_PATH}"/rocprim")
    find_package(hipblas REQUIRED)
    set(FASTLLM_CUDA_SOURCES src/devices/rocm/rocmdevice.cpp src/devices/rocm/rocmdevicebatch.cpp src/devices/rocm/fastllm-rocm.hip)
    list(APPEND FASTLLM_LINKED_LIBS roc::hipblas roc::rocprim)
    set_property(GLOBAL PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(GLOBAL PROPERTY HIP_ARCHITECTURES gfx906 gfx908 gfx90a gfx1100) # Tested on MI50, MI100, MI210, W7900
    set_property(GLOBAL PROPERTY HIP_STANDARD 17)
    include_directories(include/devices/rocm)
endif()

if (PY_API)
    set(PYBIND third_party/pybind11)
    add_subdirectory(${PYBIND})
    add_compile_definitions(PY_API)
    
    find_package(Python3 REQUIRED)

    include_directories(third_party/pybind11/include)
    file(GLOB FASTLLM_CXX_HEADERS include/**/*.h)
    add_library(pyfastllm MODULE src/pybinding.cpp ${FASTLLM_CXX_SOURCES} ${FASTLLM_CXX_HEADERS} ${FASTLLM_CUDA_SOURCES} ${FASTLLM_ROCM_SOURCES})
    target_link_libraries(pyfastllm PUBLIC pybind11::module ${FASTLLM_LINKED_LIBS})
    pybind11_extension(pyfastllm)
else()
add_library(fastllm OBJECT
            ${FASTLLM_CXX_SOURCES}
            ${FASTLLM_CUDA_SOURCES}
            ${FASTLLM_ROCM_SOURCES}
            )
target_link_libraries(fastllm PUBLIC ${FASTLLM_LINKED_LIBS})

add_executable(main main.cpp)
target_link_libraries(main fastllm)

add_executable(quant tools/src/quant.cpp)
target_link_libraries(quant fastllm)

add_executable(testOps test/ops/cppOps.cpp)
target_link_libraries(testOps fastllm)

add_executable(webui example/webui/webui.cpp)
target_link_libraries(webui fastllm)
add_custom_command(
        TARGET webui
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory web
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/example/webui/web ${CMAKE_BINARY_DIR}/web
)

add_executable(benchmark example/benchmark/benchmark.cpp)
target_link_libraries(benchmark fastllm)

add_executable(apiserver example/apiserver/apiserver.cpp example/apiserver/json11.cpp)
target_link_libraries(apiserver fastllm)

add_library(fastllm_tools SHARED ${FASTLLM_CXX_SOURCES} ${FASTLLM_CUDA_SOURCES} ${FASTLLM_ROCM_SOURCES} tools/src/pytools.cpp)
target_link_libraries(fastllm_tools PUBLIC ${FASTLLM_LINKED_LIBS})

if (${CMAKE_HOST_WIN32})
    add_custom_command(
            TARGET fastllm_tools
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory tools
            COMMAND ${CMAKE_COMMAND} -E make_directory tools/fastllm_pytools
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/tools/fastllm_pytools ${CMAKE_BINARY_DIR}/tools/fastllm_pytools/.
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/tools/scripts ${CMAKE_BINARY_DIR}/tools/.
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/fastllm_tools.dll ${CMAKE_BINARY_DIR}/tools/fastllm_pytools/.
            COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/fastllm_tools.dll
    )
else()
    add_custom_command(
            TARGET fastllm_tools
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory tools
            COMMAND ${CMAKE_COMMAND} -E make_directory tools/fastllm_pytools
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/tools/fastllm_pytools ${CMAKE_BINARY_DIR}/tools/fastllm_pytools/.
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/tools/scripts ${CMAKE_BINARY_DIR}/tools/.
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/libfastllm_tools.* ${CMAKE_BINARY_DIR}/tools/fastllm_pytools/.
            COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/libfastllm_tools.*
    )
endif()

endif()
