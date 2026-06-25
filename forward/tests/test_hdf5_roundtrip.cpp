#include "hdf5_io.h"
#include <cmath>
#include <cstdlib>
#include <iostream>

static int failures = 0;

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "FAIL: " << msg << std::endl;                               \
      failures++;                                                              \
    } else {                                                                   \
      std::cout << "  OK: " << msg << std::endl;                               \
    }                                                                          \
  } while (0)

static void write_scalar_int(hid_t file_id, const char *path, int value) {
  hid_t space = H5Screate(H5S_SCALAR);
  hid_t dset = H5Dcreate(file_id, path, H5T_NATIVE_INT, space, H5P_DEFAULT,
                         H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
  H5Dclose(dset);
  H5Sclose(space);
}

static void write_scalar_double(hid_t file_id, const char *path, double value) {
  hid_t space = H5Screate(H5S_SCALAR);
  hid_t dset = H5Dcreate(file_id, path, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT,
                         H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
  H5Dclose(dset);
  H5Sclose(space);
}

static void write_int_1d(hid_t file_id, const char *path, const int *data,
                         hsize_t n) {
  hid_t space = H5Screate_simple(1, &n, nullptr);
  hid_t dset = H5Dcreate(file_id, path, H5T_NATIVE_INT, space, H5P_DEFAULT,
                         H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
  H5Dclose(dset);
  H5Sclose(space);
}

static void write_double_1d(hid_t file_id, const char *path, const double *data,
                            hsize_t n) {
  hid_t space = H5Screate_simple(1, &n, nullptr);
  hid_t dset = H5Dcreate(file_id, path, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT,
                         H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
  H5Dclose(dset);
  H5Sclose(space);
}

int main() {
  const char *test_file = "/tmp/test_hdf5_roundtrip.h5";

  // Setup: create file and write known data using raw HDF5 API
  {
    hid_t file_id =
        H5Fcreate(test_file, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    write_scalar_int(file_id, "/int_val", 42);
    write_scalar_double(file_id, "/double_val", 3.14159);

    int ints[] = {10, 20, 30, 40, 50};
    write_int_1d(file_id, "/int_array", ints, 5);

    double doubles[] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    write_double_1d(file_id, "/double_array", doubles, 6);

    // 2D array via our write_double_2d interface
    Hdf5Handle h;
    h.file_id = file_id;
    double data_2d[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    h.write_double_2d("/double_2d", data_2d, 2, 3);
    H5Fclose(file_id);
  }
  std::cout << "Setup: file written." << std::endl;

  // Round-trip: open and read back
  {
    Hdf5Handle h;
    h.open(test_file, H5F_ACC_RDWR);

    // Scalar reads
    int iv = h.read_int_scalar("/int_val");
    CHECK(iv == 42, "read_int_scalar: expected 42, got " << iv);

    double dv = h.read_double_scalar("/double_val");
    CHECK(std::abs(dv - 3.14159) < 1e-10,
          "read_double_scalar: expected 3.14159, got " << dv);

    // 1D int array
    auto iv1d = h.read_int_1d("/int_array");
    CHECK(iv1d.size() == 5,
          "read_int_1d size: expected 5, got " << iv1d.size());
    if (iv1d.size() == 5) {
      CHECK(iv1d[0] == 10 && iv1d[1] == 20 && iv1d[2] == 30 && iv1d[3] == 40 &&
                iv1d[4] == 50,
            "read_int_1d values: [10,20,30,40,50]");
    }

    // 1D double array
    auto dv1d = h.read_double_1d("/double_array");
    CHECK(dv1d.size() == 6,
          "read_double_1d size: expected 6, got " << dv1d.size());
    if (dv1d.size() == 6) {
      CHECK(std::abs(dv1d[0] - 1.1) < 1e-10 &&
                std::abs(dv1d[2] - 3.3) < 1e-10 &&
                std::abs(dv1d[5] - 6.6) < 1e-10,
            "read_double_1d values ok");
    }

    // 2D double
    int rows = 0, cols = 0;
    auto dv2d = h.read_double_2d("/double_2d", rows, cols);
    CHECK(rows == 2, "read_double_2d rows: expected 2, got " << rows);
    CHECK(cols == 3, "read_double_2d cols: expected 3, got " << cols);
    if (rows == 2 && cols == 3) {
      CHECK(std::abs(dv2d[0] - 1.0) < 1e-10 &&
                std::abs(dv2d[3] - 4.0) < 1e-10 &&
                std::abs(dv2d[5] - 6.0) < 1e-10,
            "read_double_2d values ok");
    }

    // Group existence
    CHECK(h.group_exists("/"), "group_exists('/') true");
    CHECK(!h.group_exists("/nonexistent"),
          "group_exists('/nonexistent') false");

    // Create group
    h.create_group("/test_group");
    CHECK(h.group_exists("/test_group"), "created group exists");

    h.close();
  }

  // Verify close works
  {
    Hdf5Handle h;
    h.open(test_file, H5F_ACC_RDONLY);
    h.close();
    h.close(); // double close should be safe
    CHECK(true, "double close is safe");
  }

  std::cout << std::endl;
  if (failures == 0) {
    std::cout << "All HDF5 round-trip tests passed." << std::endl;
    return 0;
  } else {
    std::cerr << failures << " test(s) FAILED." << std::endl;
    return 1;
  }
}