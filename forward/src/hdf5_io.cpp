#include "hdf5_io.h"
#include "validation.h"
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

// Helper: check HDF5 return code, throw if negative
static void check_herr(const char *context, herr_t status) {
    if (status < 0) {
        std::string msg = std::string("HDF5 error in ") + context;
        throw std::runtime_error(msg);
    }
}

static void check_null(const char *context, hid_t id) {
    if (id < 0) {
        std::string msg = std::string("HDF5 error (invalid id) in ") + context;
        throw std::runtime_error(msg);
    }
}

static size_t checked_hsize(hsize_t value, const char *context) {
    if (value > static_cast<hsize_t>(std::numeric_limits<size_t>::max()))
        throw std::runtime_error(std::string(context) + ": dimension exceeds size_t");
    return static_cast<size_t>(value);
}

static int checked_int_dimension(hsize_t value, const char *context) {
    if (value > static_cast<hsize_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error(std::string(context) + ": dimension exceeds INT_MAX");
    return static_cast<int>(value);
}

void Hdf5Handle::open(const char *path, unsigned flags) {
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

int Hdf5Handle::read_int_scalar(const char *path) {
    hid_t dset = H5Dopen(file_id, path, H5P_DEFAULT);
    check_null("H5Dopen (int_scalar)", dset);

    int value = 0;
    check_herr("H5Dread (int_scalar)",
               H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value));
    check_herr("H5Dclose (int_scalar)", H5Dclose(dset));
    return value;
}

double Hdf5Handle::read_double_scalar(const char *path) {
    hid_t dset = H5Dopen(file_id, path, H5P_DEFAULT);
    check_null("H5Dopen (double_scalar)", dset);

    double value = 0.0;
    check_herr("H5Dread (double_scalar)",
               H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value));
    check_herr("H5Dclose (double_scalar)", H5Dclose(dset));
    return value;
}

// --- 1D readers ---

std::vector<int> Hdf5Handle::read_int_1d(const char *path) {
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

    std::vector<int> result(checked_hsize(dims[0], "read_int_1d"));
    check_herr("H5Dread (int_1d)",
               H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, result.data()));

    check_herr("H5Sclose (int_1d)", H5Sclose(space));
    check_herr("H5Dclose (int_1d)", H5Dclose(dset));
    return result;
}

std::vector<double> Hdf5Handle::read_double_1d(const char *path) {
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

    std::vector<double> result(checked_hsize(dims[0], "read_double_1d"));
    check_herr("H5Dread (double_1d)",
               H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, result.data()));

    check_herr("H5Sclose (double_1d)", H5Sclose(space));
    check_herr("H5Dclose (double_1d)", H5Dclose(dset));
    return result;
}

// --- 2D reader ---

std::vector<double> Hdf5Handle::read_double_2d(const char *path, int &rows, int &cols) {
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

    rows = checked_int_dimension(dims[0], "read_double_2d rows");
    cols = checked_int_dimension(dims[1], "read_double_2d cols");

    std::vector<double> result(
        fm::checked_mul(static_cast<size_t>(rows), static_cast<size_t>(cols), "read_double_2d"));
    check_herr("H5Dread (double_2d)",
               H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, result.data()));

    check_herr("H5Sclose (double_2d)", H5Sclose(space));
    check_herr("H5Dclose (double_2d)", H5Dclose(dset));
    return result;
}

// --- 3D reader ---

std::vector<double> Hdf5Handle::read_double_3d(const char *path, int &dim1, int &dim2, int &dim3) {
    hid_t dset = H5Dopen(file_id, path, H5P_DEFAULT);
    check_null("H5Dopen (double_3d)", dset);

    hid_t space = H5Dget_space(dset);
    check_null("H5Dget_space (double_3d)", space);

    int ndims = H5Sget_simple_extent_ndims(space);
    if (ndims != 3) {
        H5Sclose(space);
        H5Dclose(dset);
        throw std::runtime_error("read_double_3d: dataset is not 3-dimensional");
    }

    hsize_t dims[3] = {0, 0, 0};
    check_herr("H5Sget_simple_extent_dims (double_3d)",
               H5Sget_simple_extent_dims(space, dims, nullptr));

    dim1 = checked_int_dimension(dims[0], "read_double_3d dim1");
    dim2 = checked_int_dimension(dims[1], "read_double_3d dim2");
    dim3 = checked_int_dimension(dims[2], "read_double_3d dim3");

    const size_t first_two =
        fm::checked_mul(static_cast<size_t>(dim1), static_cast<size_t>(dim2), "read_double_3d");
    std::vector<double> result(
        fm::checked_mul(first_two, static_cast<size_t>(dim3), "read_double_3d"));
    check_herr("H5Dread (double_3d)",
               H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, result.data()));

    check_herr("H5Sclose (double_3d)", H5Sclose(space));
    check_herr("H5Dclose (double_3d)", H5Dclose(dset));
    return result;
}

// --- Group ops ---

bool Hdf5Handle::link_exists(hid_t file, const char *path) {
    const std::string full_path(path);
    if (full_path.empty() || full_path == "/")
        return true;

    std::string current = full_path.front() == '/' ? "/" : "";
    size_t begin = full_path.front() == '/' ? 1 : 0;
    while (begin < full_path.size()) {
        const size_t end = full_path.find('/', begin);
        const std::string component = full_path.substr(begin, end - begin);
        if (!component.empty()) {
            if (current.size() > 1 && current.back() != '/')
                current += "/";
            current += component;
            htri_t status;
            H5E_BEGIN_TRY {
                status = H5Lexists(file, current.c_str(), H5P_DEFAULT);
            }
            H5E_END_TRY;
            if (status < 0)
                throw std::runtime_error("HDF5 error in H5Lexists: " + current);
            if (status == 0)
                return false;
        }
        if (end == std::string::npos)
            break;
        begin = end + 1;
    }
    return true;
}
bool Hdf5Handle::group_exists(const char *path) {
    return link_exists(file_id, path);
}

void Hdf5Handle::create_group(const char *path) {
    hid_t grp = H5Gcreate(file_id, path, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    check_null("H5Gcreate", grp);
    check_herr("H5Gclose", H5Gclose(grp));
}

void Hdf5Handle::delete_group(const char *path) {
    if (!group_exists(path))
        return;
    check_herr("H5Ldelete", H5Ldelete(file_id, path, H5P_DEFAULT));
}

void Hdf5Handle::move_link(const char *source, const char *destination) {
    if (!group_exists(source))
        throw std::runtime_error(std::string("HDF5 source link does not exist: ") + source);
    if (group_exists(destination))
        throw std::runtime_error(std::string("HDF5 destination link already exists: ") +
                                 destination);
    check_herr("H5Lmove", H5Lmove(file_id, source, file_id, destination, H5P_DEFAULT, H5P_DEFAULT));
}

void Hdf5Handle::flush() {
    check_herr("H5Fflush", H5Fflush(file_id, H5F_SCOPE_GLOBAL));
}

void Hdf5Handle::validate_dataset_2d(const char *path, hid_t expected_type, hsize_t dim1,
                                     hsize_t dim2) {
    if (!group_exists(path))
        throw std::runtime_error(std::string(path) + ": dataset does not exist");
    hid_t dset = H5Dopen(file_id, path, H5P_DEFAULT);
    check_null("H5Dopen (validate_dataset_2d)", dset);
    hid_t space = H5Dget_space(dset);
    check_null("H5Dget_space (validate_dataset_2d)", space);
    if (H5Sget_simple_extent_ndims(space) != 2) {
        H5Sclose(space);
        H5Dclose(dset);
        throw std::runtime_error(std::string(path) + ": expected a 2D dataset");
    }
    hsize_t dims[2] = {0, 0};
    check_herr("H5Sget_simple_extent_dims (validate_dataset_2d)",
               H5Sget_simple_extent_dims(space, dims, nullptr));
    hid_t datatype = H5Dget_type(dset);
    check_null("H5Dget_type (validate_dataset_2d)", datatype);
    const htri_t type_equal = H5Tequal(datatype, expected_type);
    check_herr("H5Tclose (validate_dataset_2d)", H5Tclose(datatype));
    check_herr("H5Sclose (validate_dataset_2d)", H5Sclose(space));
    check_herr("H5Dclose (validate_dataset_2d)", H5Dclose(dset));
    if (type_equal <= 0)
        throw std::runtime_error(std::string(path) + ": unexpected datatype");
    if (dims[0] != dim1 || dims[1] != dim2)
        throw std::runtime_error(std::string(path) + ": unexpected shape");
}

// --- Writer ---

void Hdf5Handle::write_double_2d(const char *path, const double *data, hsize_t dim1, hsize_t dim2) {
    hsize_t dims[2] = {dim1, dim2};
    hid_t space = H5Screate_simple(2, dims, nullptr);
    check_null("H5Screate_simple (write_double_2d)", space);

    hid_t dset =
        H5Dcreate(file_id, path, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    check_null("H5Dcreate (write_double_2d)", dset);

    check_herr("H5Dwrite (write_double_2d)",
               H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data));

    check_herr("H5Sclose (write_double_2d)", H5Sclose(space));
    check_herr("H5Dclose (write_double_2d)", H5Dclose(dset));
}

// --- String readers ---

std::string Hdf5Handle::read_string_scalar(const char *path) {
    hid_t dset = H5Dopen(file_id, path, H5P_DEFAULT);
    check_null("H5Dopen (string_scalar)", dset);

    hid_t filetype = H5Dget_type(dset);
    std::string result;
    if (H5Tis_variable_str(filetype)) {
        hid_t memtype = H5Tcopy(filetype);
        char *buf = nullptr;
        check_herr("H5Dread (string_scalar vlen)",
                   H5Dread(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &buf));
        result = buf ? std::string(buf) : std::string();
        if (buf)
            std::free(buf);
        H5Tclose(memtype);
    } else {
        size_t sz = H5Tget_size(filetype);
        std::vector<char> buf(sz + 1, '\0');
        check_herr("H5Dread (string_scalar fixed)",
                   H5Dread(dset, filetype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data()));
        result = std::string(buf.data());
    }
    H5Tclose(filetype);
    H5Dclose(dset);
    return result;
}

std::vector<std::string> Hdf5Handle::read_string_1d(const char *path) {
    hid_t dset = H5Dopen(file_id, path, H5P_DEFAULT);
    check_null("H5Dopen (string_1d)", dset);

    hid_t filetype = H5Dget_type(dset);
    hid_t space = H5Dget_space(dset);
    check_null("H5Dget_space (string_1d)", space);
    hsize_t dims[1] = {0};
    H5Sget_simple_extent_dims(space, dims, nullptr);

    std::vector<std::string> result;
    if (H5Tis_variable_str(filetype)) {
        std::vector<char *> buf(dims[0]);
        hid_t memtype = H5Tcopy(filetype);
        check_herr("H5Dread (string_1d vlen)",
                   H5Dread(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data()));
        for (hsize_t i = 0; i < dims[0]; ++i) {
            result.push_back(buf[i] ? std::string(buf[i]) : std::string());
            if (buf[i])
                std::free(buf[i]);
        }
        H5Tclose(memtype);
    } else {
        size_t sz = H5Tget_size(filetype);
        std::vector<char> buf(
            fm::checked_mul(checked_hsize(dims[0], "read_string_1d"), sz, "read_string_1d buffer"),
            '\0');
        check_herr("H5Dread (string_1d fixed)",
                   H5Dread(dset, filetype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data()));
        for (hsize_t i = 0; i < dims[0]; ++i)
            result.push_back(std::string(buf.data() + i * sz));
    }
    H5Tclose(filetype);
    H5Sclose(space);
    H5Dclose(dset);
    return result;
}

// --- Int writers ---

void Hdf5Handle::write_int32_2d(const char *path, const int32_t *data, hsize_t dim1, hsize_t dim2) {
    hsize_t dims[2] = {dim1, dim2};
    hid_t space = H5Screate_simple(2, dims, nullptr);
    check_null("H5Screate_simple (write_int32_2d)", space);
    hid_t dset =
        H5Dcreate(file_id, path, H5T_NATIVE_INT32, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    check_null("H5Dcreate (write_int32_2d)", dset);
    check_herr("H5Dwrite (write_int32_2d)",
               H5Dwrite(dset, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, data));
    H5Sclose(space);
    H5Dclose(dset);
}

void Hdf5Handle::write_int8_2d(const char *path, const int8_t *data, hsize_t dim1, hsize_t dim2) {
    hsize_t dims[2] = {dim1, dim2};
    hid_t space = H5Screate_simple(2, dims, nullptr);
    check_null("H5Screate_simple (write_int8_2d)", space);
    hid_t dset =
        H5Dcreate(file_id, path, H5T_NATIVE_INT8, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    check_null("H5Dcreate (write_int8_2d)", dset);
    check_herr("H5Dwrite (write_int8_2d)",
               H5Dwrite(dset, H5T_NATIVE_INT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, data));
    H5Sclose(space);
    H5Dclose(dset);
}
