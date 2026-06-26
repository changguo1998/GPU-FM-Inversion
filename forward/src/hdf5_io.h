#ifndef HDF5_IO_H
#define HDF5_IO_H

#include <hdf5.h>
#include <stdexcept>
#include <string>
#include <vector>

struct Hdf5Handle {
    hid_t file_id = -1;

    // Open/close
    void open(const char *path, unsigned flags);
    void close();

    // Scalar readers
    int read_int_scalar(const char *path);
    double read_double_scalar(const char *path);

    // 1D readers
    std::vector<int> read_int_1d(const char *path);
    std::vector<double> read_double_1d(const char *path);

    // 2D reader — returns data as flat vector, outputs rows/cols
    std::vector<double> read_double_2d(const char *path, int &rows, int &cols);

    // Group ops
    bool group_exists(const char *path);
    void create_group(const char *path);

    // Writer
    void write_double_2d(const char *path, const double *data, hsize_t dim1, hsize_t dim2);
};

#endif // HDF5_IO_H