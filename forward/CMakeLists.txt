cmake_minimum_required(VERSION 3.15)
project(fm_forward LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Kokkos REQUIRED)
find_package(HDF5 REQUIRED)

# ── Library: shared forward sources ──────────────────────────────────────
set(FORWARD_SOURCES
    src/main.cpp
    src/mt_utils.cpp
    src/hdf5_io.cpp
    src/data_cache.cpp
)

add_executable(forward ${FORWARD_SOURCES})
target_include_directories(forward PRIVATE src)
target_link_libraries(forward Kokkos::kokkos hdf5)

# ── Test: DataCache ──────────────────────────────────────────────────────
add_executable(test_data_cache
    tests/test_data_cache.cpp
    src/hdf5_io.cpp
    src/data_cache.cpp
)
target_include_directories(test_data_cache PRIVATE src)
target_link_libraries(test_data_cache Kokkos::kokkos hdf5)

# ── Test: HDF5 roundtrip ─────────────────────────────────────────────────
add_executable(test_hdf5_roundtrip
    tests/test_hdf5_roundtrip.cpp
    src/hdf5_io.cpp
)
target_include_directories(test_hdf5_roundtrip PRIVATE src)
target_link_libraries(test_hdf5_roundtrip hdf5)

# ── Test: MT utils ───────────────────────────────────────────────────────
add_executable(test_mt_utils
    tests/test_mt_to_csv.cpp
    src/mt_utils.cpp
)
target_include_directories(test_mt_utils PRIVATE src)

# XCorr kernel test (Task 12)
add_executable(test_xcorr tests/test_xcorr.cpp)
target_include_directories(test_xcorr PRIVATE src)
target_link_libraries(test_xcorr Kokkos::kokkos)

# Data Cache test (Task 11)
add_executable(test_data_cache tests/test_data_cache.cpp)
target_include_directories(test_data_cache PRIVATE src)
target_link_libraries(test_data_cache Kokkos::kokkos hdf5)

# Polarity + PSR kernel test (Task 13)
add_executable(test_misfit_kernels tests/test_misfit_kernels.cpp)
target_include_directories(test_misfit_kernels PRIVATE src)
target_link_libraries(test_misfit_kernels Kokkos::kokkos)