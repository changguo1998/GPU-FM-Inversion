cmake_minimum_required(VERSION 3.15)
project(fm_forward LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# System HDF5 (serial) headers at /usr/include/hdf5/serial, libs at /usr/lib/x86_64-linux-gnu/hdf5/serial
set(HDF5_ROOT /usr/lib/x86_64-linux-gnu/hdf5/serial)
include_directories(/usr/include/hdf5/serial)
link_directories(/usr/lib/x86_64-linux-gnu/hdf5/serial/lib)

find_package(OpenMP REQUIRED)
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
target_link_libraries(forward OpenMP::OpenMP_CXX hdf5)
