#ifndef FM_INTERMEDIATES_TRANSACTION_H
#define FM_INTERMEDIATES_TRANSACTION_H

#include "hdf5_io.h"

#include <functional>
#include <stdexcept>
#include <string>

namespace fm {

enum class CommitFaultPoint {
    None,
    AfterTemporaryWrite,
    AfterTemporaryFlush,
    AfterBackupMove,
    AfterPromotion,
    AfterFinalFlush,
};

class InjectedCommitFailure : public std::runtime_error {
  public:
    explicit InjectedCommitFailure(const std::string &message) : std::runtime_error(message) {
    }
};

using IntermediatesAction = std::function<void(const std::string &root)>;

/// Recover interrupted /intermediates link swaps to one validated final group.
void recover_intermediates(Hdf5Handle &file, const IntermediatesAction &validator);

/// Write, validate, and atomically promote a temporary /intermediates group.
void commit_intermediates(Hdf5Handle &file, const IntermediatesAction &writer,
                          const IntermediatesAction &validator,
                          CommitFaultPoint fault = CommitFaultPoint::None);

} // namespace fm

#endif // FM_INTERMEDIATES_TRANSACTION_H
