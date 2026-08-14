#ifndef HDF5_IO_H
#define HDF5_IO_H

#include <hdf5.h>
#include <stdexcept>
#include <string>
#include <vector>

struct Hdf5Handle {
    hid_t file_id = -1;
    /// Silent link-exists check (no HDF5-DIAG noise for missing paths).
    static bool link_exists(hid_t file, const char *path);

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

    // 3D reader - returns data as flat vector, outputs all three dims
    std::vector<double> read_double_3d(const char *path, int &dim1, int &dim2, int &dim3);

    // String readers (variable-length strings)
    std::string read_string_scalar(const char *path);
    std::vector<std::string> read_string_1d(const char *path);

    // Group ops
    bool group_exists(const char *path);
    void create_group(const char *path);
    void delete_group(const char *path);

    // Writer
    void write_double_2d(const char *path, const double *data, hsize_t dim1, hsize_t dim2);
    void write_int32_2d(const char *path, const int32_t *data, hsize_t dim1, hsize_t dim2);
    void write_int8_2d(const char *path, const int8_t *data, hsize_t dim1, hsize_t dim2);
};

#endif // HDF5_IO_H