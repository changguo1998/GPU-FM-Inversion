cmake_minimum_required(VERSION 3.15)
project(fm_forward LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Kokkos REQUIRED)
find_package(HDF5 REQUIRED)

add_executable(forward src/main.cpp)
target_link_libraries(forward Kokkos::kokkos hdf5)