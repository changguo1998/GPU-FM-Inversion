#include "hdf5_io.h"
#include <stdexcept>
#include <string>
#include <cstring>

// Helper: check HDF5 return code, throw if negative
static void check_herr(const char* context, herr_t status) {
    if (status < 0) {
        std::string msg = std::string("HDF5 error in ") + context;
        throw std::runtime_error(msg);
    }
}

static void check_null(const char* context, hid_t id) {
    if (id < 0) {
        std::string msg = std::string("HDF5 error (invalid id) in ") + context;
        throw std::runtime_error(msg);
    }
}

void Hdf5Handle::open(const char* path, unsigned flags) {
    file_id = H5Fopen(path, flags, H5P_DEFAULT);
    check_null("H5Fopen", file_id);
}

void Hdf5Handle::close() {
    if (file_id >= 0) {
        check_herr("H5Fclose", H5Fclose(file_id));
        file_id = -1;
    }
}

// --- Scalar readers ---

int Hdf5Handle::read_int_scalar(const char* path) {
    hid_t dset = H5Dopen(file_id, path, H5P_DEFAULT);
    check_null("H5Dopen (int_scalar)", dset);

    int value = 0;
    check_herr("H5Dread (int_scalar)", H5Dread(dset, H5T_NATIVE_INT,
          H5S_ALL, H5S_ALL, H5P_DEFAULT, &value));
    check_herr("H5Dclose (int_scalar)", H5Dclose(dset));
    return value;
}

double Hdf5Handle::read_double_scalar(const char* path) {
    hid_t dset = H5Dopen(file_id, path, H5P_DEFAULT);
    check_null("H5Dopen (double_scalar)", dset);

    double value = 0.0;
    check_herr("H5Dread (double_scalar)", H5Dread(dset, H5T_NATIVE_DOUBLE,
          H5S_ALL, H5S_ALL, H5P_DEFAULT, &value));
    check_herr("H5Dclose (double_scalar)", H5Dclose(dset));
    return value;
}

// --- 1D readers ---

std::vector<int> Hdf5Handle::read_int_1d(const char* path) {
    hid_t dset = H5Dopen(file_id, path, H5P_DEFAULT);
    check_null("H5Dopen (int_1d)", dset);

    hid_t space = H5Dget_space(dset);
    check_null("H5Dget_space (int_1d)", space);

    int ndims = H5Sget_simple_extent_ndims(space);
    if (ndims != 1) {
        H5Sclose(space);
        H5Dclose(dset);
        throw std::runtime_error("read_int_1d: dataset is not 1-dimensional");
    }

    hsize_t dims[1] = {0};
    check_herr("H5Sget_simple_extent_dims (int_1d)",
               H5Sget_simple_extent_dims(space, dims, nullptr));

    std::vector<int> result(dims[0]);
    check_herr("H5Dread (int_1d)", H5Dread(dset, H5T_NATIVE_INT,
          H5S_ALL, H5S_ALL, H5P_DEFAULT, result.data()));

    check_herr("H5Sclose (int_1d)", H5Sclose(space));
    check_herr("H5Dclose (int_1d)", H5Dclose(dset));
    return result;
}

std::vector<double> Hdf5Handle::read_double_1d(const char* path) {
    hid_t dset = H5Dopen(file_id, path, H5P_DEFAULT);
    check_null("H5Dopen (double_1d)", dset);

    hid_t space = H5Dget_space(dset);
    check_null("H5Dget_space (double_1d)", space);

    int ndims = H5Sget_simple_extent_ndims(space);
    if (ndims != 1) {
        H5Sclose(space);
        H5Dclose(dset);
        throw std::runtime_error("read_double_1d: dataset is not 1-dimensional");
    }

    hsize_t dims[1] = {0};
    check_herr("H5Sget_simple_extent_dims (double_1d)",
               H5Sget_simple_extent_dims(space, dims, nullptr));

    std::vector<double> result(dims[0]);
    check_herr("H5Dread (double_1d)", H5Dread(dset, H5T_NATIVE_DOUBLE,
          H5S_ALL, H5S_ALL, H5P_DEFAULT, result.data()));

    check_herr("H5Sclose (double_1d)", H5Sclose(space));
    check_herr("H5Dclose (double_1d)", H5Dclose(dset));
    return result;
}

// --- 2D reader ---

std::vector<double> Hdf5Handle::read_double_2d(const char* path, int& rows, int& cols) {
    hid_t dset = H5Dopen(file_id, path, H5P_DEFAULT);
    check_null("H5Dopen (double_2d)", dset);

    hid_t space = H5Dget_space(dset);
    check_null("H5Dget_space (double_2d)", space);

    int ndims = H5Sget_simple_extent_ndims(space);
    if (ndims != 2) {
        H5Sclose(space);
        H5Dclose(dset);
        throw std::runtime_error("read_double_2d: dataset is not 2-dimensional");
    }

    hsize_t dims[2] = {0, 0};
    check_herr("H5Sget_simple_extent_dims (double_2d)",
               H5Sget_simple_extent_dims(space, dims, nullptr));

    rows = static_cast<int>(dims[0]);
    cols = static_cast<int>(dims[1]);

    std::vector<double> result(rows * cols);
    check_herr("H5Dread (double_2d)", H5Dread(dset, H5T_NATIVE_DOUBLE,
          H5S_ALL, H5S_ALL, H5P_DEFAULT, result.data()));

    check_herr("H5Sclose (double_2d)", H5Sclose(space));
    check_herr("H5Dclose (double_2d)", H5Dclose(dset));
    return result;
}

// --- Group ops ---

bool Hdf5Handle::group_exists(const char* path) {
    hid_t lapl = H5Pcreate(H5P_LINK_ACCESS);
    if (lapl < 0) return false;
    htri_t status = H5Lexists(file_id, path, lapl);
    H5Pclose(lapl);
    return status > 0;
}

void Hdf5Handle::create_group(const char* path) {
    hid_t grp = H5Gcreate(file_id, path, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    check_null("H5Gcreate", grp);
    check_herr("H5Gclose", H5Gclose(grp));
}

// --- Writer ---

void Hdf5Handle::write_double_2d(const char* path, const double* data,
                                  hsize_t dim1, hsize_t dim2) {
    hsize_t dims[2] = {dim1, dim2};
    hid_t space = H5Screate_simple(2, dims, nullptr);
    check_null("H5Screate_simple (write_double_2d)", space);

    hid_t dset = H5Dcreate(file_id, path, H5T_NATIVE_DOUBLE,
                           space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    check_null("H5Dcreate (write_double_2d)", dset);

    check_herr("H5Dwrite (write_double_2d)", H5Dwrite(dset, H5T_NATIVE_DOUBLE,
          H5S_ALL, H5S_ALL, H5P_DEFAULT, data));

    check_herr("H5Sclose (write_double_2d)", H5Sclose(space));
    check_herr("H5Dclose (write_double_2d)", H5Dclose(dset));
}