cmake_minimum_required(VERSION 3.18)
project(fm_forward LANGUAGES CXX)

option(FM_ENABLE_CUDA "Build the CUDA forward backend" OFF)

if(FM_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        message(FATAL_ERROR "FM_ENABLE_CUDA=ON requires CMAKE_CUDA_ARCHITECTURES")
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile_commands.json so clangd-style analyzers (pi-lens) resolve
# include paths (system HDF5, project src) without per-file fallback.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# System HDF5 (serial) headers at /usr/include/hdf5/serial, libs at /usr/lib/x86_64-linux-gnu/hdf5/serial
set(HDF5_ROOT /usr/lib/x86_64-linux-gnu/hdf5/serial)
include_directories(/usr/include/hdf5/serial)
link_directories(/usr/lib/x86_64-linux-gnu/hdf5/serial/lib)

find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED)
include(CTest)

# ── Library: shared forward sources ──────────────────────────────────────
set(FORWARD_CORE_SOURCES
    src/mt_utils.cpp
    src/hdf5_io.cpp
    src/data_cache.cpp
    src/validation.cpp
    src/intermediates_transaction.cpp
    src/backends/cuda_batch.cpp
)

if(FM_ENABLE_CUDA)
    list(APPEND FORWARD_CORE_SOURCES
        src/backends/cuda_runtime.cu
        src/kernels/xcorr_kernel.cu
        src/kernels/waveform_kernel.cu
    )
else()
    list(APPEND FORWARD_CORE_SOURCES src/backends/cuda_runtime_stub.cpp)
endif()

add_library(forward_core STATIC ${FORWARD_CORE_SOURCES})
target_include_directories(forward_core PUBLIC src)
target_link_libraries(forward_core PUBLIC OpenMP::OpenMP_CXX hdf5)
target_compile_options(forward_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
)

if(FM_ENABLE_CUDA)
    set_target_properties(forward_core PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_link_libraries(forward_core PUBLIC CUDA::cudart)
endif()

add_executable(forward src/main.cpp)
target_link_libraries(forward PRIVATE forward_core)
target_compile_options(forward PRIVATE -ffp-contract=off)

if(BUILD_TESTING)
    add_executable(preflight_test tests/preflight_test.cpp)
    target_link_libraries(preflight_test PRIVATE forward_core)
    add_test(NAME preflight COMMAND preflight_test)

    add_executable(hdf5_commit_test tests/hdf5_commit_test.cpp)
    target_link_libraries(hdf5_commit_test PRIVATE forward_core)
    add_test(NAME hdf5_commit COMMAND hdf5_commit_test)

    add_executable(cuda_batch_test tests/cuda_batch_test.cpp)
    target_link_libraries(cuda_batch_test PRIVATE forward_core)
    add_test(NAME cuda_batch COMMAND cuda_batch_test)

    add_executable(xcorr_math_test tests/xcorr_math_test.cpp)
    target_link_libraries(xcorr_math_test PRIVATE forward_core)
    add_test(NAME xcorr_math COMMAND xcorr_math_test)

    add_executable(waveform_math_test tests/waveform_math_test.cpp)
    target_link_libraries(waveform_math_test PRIVATE forward_core)
    add_test(NAME waveform_math COMMAND waveform_math_test)

    if(FM_ENABLE_CUDA)
        add_executable(xcorr_cuda_test tests/xcorr_cuda_test.cu)
        target_link_libraries(xcorr_cuda_test PRIVATE forward_core)
        set_target_properties(xcorr_cuda_test PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
        add_test(NAME xcorr_cuda COMMAND xcorr_cuda_test)
    endif()
endif()
