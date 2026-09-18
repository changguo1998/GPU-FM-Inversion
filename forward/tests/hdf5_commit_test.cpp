#include "hdf5_io.h"
#include "intermediates_transaction.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

void create_empty_file(const std::string &path) {
    const hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0)
        throw std::runtime_error("failed to create test HDF5 file");
    if (H5Fclose(file) < 0)
        throw std::runtime_error("failed to close test HDF5 file");
}

void write_marker(Hdf5Handle &file, const std::string &root, double value) {
    file.create_group(root.c_str());
    file.write_double_2d((root + "/value").c_str(), &value, 1, 1);
}

void validate_marker(Hdf5Handle &file, const std::string &root) {
    file.validate_dataset_2d((root + "/value").c_str(), H5T_NATIVE_DOUBLE, 1, 1);
}

double read_marker(Hdf5Handle &file) {
    int rows = 0;
    int cols = 0;
    const std::vector<double> values = file.read_double_2d("/intermediates/value", rows, cols);
    if (rows != 1 || cols != 1 || values.size() != 1)
        throw std::runtime_error("invalid marker shape");
    return values[0];
}

void require(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}

} // namespace

int main() {
    std::string path_template =
        (std::filesystem::temp_directory_path() / "fm-forward-hdf5-commit-XXXXXX").string();
    std::vector<char> path_buffer(path_template.begin(), path_template.end());
    path_buffer.push_back('\0');
    const int temporary_fd = mkstemp(path_buffer.data());
    if (temporary_fd < 0)
        throw std::runtime_error("failed to reserve test HDF5 path");
    close(temporary_fd);
    const std::string path(path_buffer.data());

    const std::vector<std::pair<fm::CommitFaultPoint, double>> fault_cases {
        {fm::CommitFaultPoint::AfterTemporaryWrite, 1.0},
        {fm::CommitFaultPoint::AfterTemporaryFlush, 1.0},
        {fm::CommitFaultPoint::AfterBackupMove, 1.0},
        {fm::CommitFaultPoint::AfterPromotion, 2.0},
        {fm::CommitFaultPoint::AfterFinalFlush, 2.0},
    };

    for (const auto &[fault, expected] : fault_cases) {
        create_empty_file(path);
        Hdf5Handle file;
        file.open(path.c_str(), H5F_ACC_RDWR);
        write_marker(file, "/intermediates", 1.0);
        file.flush();
        try {
            fm::commit_intermediates(
                file, [&](const std::string &root) { write_marker(file, root, 2.0); },
                [&](const std::string &root) { validate_marker(file, root); }, fault);
            throw std::runtime_error("fault injection did not fail");
        } catch (const fm::InjectedCommitFailure &) {
        }
        file.close();

        file.open(path.c_str(), H5F_ACC_RDWR);
        fm::recover_intermediates(file,
                                  [&](const std::string &root) { validate_marker(file, root); });
        require(std::abs(read_marker(file) - expected) < 1e-12, "wrong recovered marker");
        require(!file.group_exists("/intermediates.__tmp__"), "temporary group remains");
        require(!file.group_exists("/intermediates.__backup__"), "backup group remains");
        file.close();
    }

    create_empty_file(path);
    Hdf5Handle file;
    file.open(path.c_str(), H5F_ACC_RDWR);
    write_marker(file, "/intermediates.__backup__", 1.0);
    file.create_group("/intermediates");
    fm::recover_intermediates(file, [&](const std::string &root) { validate_marker(file, root); });
    require(std::abs(read_marker(file) - 1.0) < 1e-12, "malformed final did not restore backup");
    file.close();

    create_empty_file(path);
    file.open(path.c_str(), H5F_ACC_RDWR);
    write_marker(file, "/intermediates.__backup__", 1.0);
    write_marker(file, "/intermediates.__tmp__", 3.0);
    write_marker(file, "/intermediates", 2.0);
    fm::recover_intermediates(file, [&](const std::string &root) { validate_marker(file, root); });
    require(std::abs(read_marker(file) - 2.0) < 1e-12, "valid final was not preserved");
    require(!file.group_exists("/intermediates.__tmp__"), "temporary group remains");
    require(!file.group_exists("/intermediates.__backup__"), "backup group remains");
    file.close();

    create_empty_file(path);
    file.open(path.c_str(), H5F_ACC_RDWR);
    write_marker(file, "/intermediates.__tmp__", 3.0);
    fm::recover_intermediates(file, [&](const std::string &root) { validate_marker(file, root); });
    require(!file.group_exists("/intermediates"), "temporary-only state created final");
    require(!file.group_exists("/intermediates.__tmp__"), "temporary-only state remains");
    file.close();

    std::filesystem::remove(path);
    std::cout << "hdf5_commit_test: PASS" << std::endl;
    return 0;
}
