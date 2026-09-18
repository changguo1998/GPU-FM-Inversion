#include "intermediates_transaction.h"

namespace fm {
namespace {

constexpr const char *FINAL_PATH = "/intermediates";
constexpr const char *TEMP_PATH = "/intermediates.__tmp__";
constexpr const char *BACKUP_PATH = "/intermediates.__backup__";

void inject_if_requested(CommitFaultPoint actual, CommitFaultPoint requested) {
    if (actual == requested)
        throw InjectedCommitFailure("injected intermediates commit failure");
}

bool validates(const IntermediatesAction &validator, const std::string &root) {
    try {
        validator(root);
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

} // namespace

void recover_intermediates(Hdf5Handle &file, const IntermediatesAction &validator) {
    const bool has_final = file.group_exists(FINAL_PATH);
    const bool has_temp = file.group_exists(TEMP_PATH);
    const bool has_backup = file.group_exists(BACKUP_PATH);

    if (has_final) {
        if (has_backup && !validates(validator, FINAL_PATH)) {
            file.delete_group(FINAL_PATH);
            file.move_link(BACKUP_PATH, FINAL_PATH);
            validator(FINAL_PATH);
        } else {
            validator(FINAL_PATH);
            if (has_backup)
                file.delete_group(BACKUP_PATH);
        }
        if (has_temp)
            file.delete_group(TEMP_PATH);
        file.flush();
        return;
    }

    if (has_backup) {
        file.move_link(BACKUP_PATH, FINAL_PATH);
        validator(FINAL_PATH);
        if (has_temp)
            file.delete_group(TEMP_PATH);
        file.flush();
        return;
    }

    if (has_temp) {
        file.delete_group(TEMP_PATH);
        file.flush();
    }
}

void commit_intermediates(Hdf5Handle &file, const IntermediatesAction &writer,
                          const IntermediatesAction &validator, CommitFaultPoint fault) {
    if (file.group_exists(TEMP_PATH) || file.group_exists(BACKUP_PATH))
        throw std::runtime_error("intermediates transaction was not recovered before commit");

    writer(TEMP_PATH);
    inject_if_requested(CommitFaultPoint::AfterTemporaryWrite, fault);
    validator(TEMP_PATH);
    file.flush();
    inject_if_requested(CommitFaultPoint::AfterTemporaryFlush, fault);

    if (file.group_exists(FINAL_PATH))
        file.move_link(FINAL_PATH, BACKUP_PATH);
    inject_if_requested(CommitFaultPoint::AfterBackupMove, fault);

    file.move_link(TEMP_PATH, FINAL_PATH);
    inject_if_requested(CommitFaultPoint::AfterPromotion, fault);
    file.flush();
    inject_if_requested(CommitFaultPoint::AfterFinalFlush, fault);

    if (file.group_exists(BACKUP_PATH))
        file.delete_group(BACKUP_PATH);
    file.flush();
}

} // namespace fm
