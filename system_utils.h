#ifndef SYSTEM_UTILS_H
#define SYSTEM_UTILS_H

#include <cstdlib>
#include <cstring>
#include <stdexcept>

inline void discardSystemResult(const char *command)
{
    // CUDA errors used to execute an unavailable Windows pause command and
    // continue with corrupt state. A failed CUDA operation must fail the run.
    if (std::strcmp(command, "pause") == 0)
        throw std::runtime_error("CUDA operation failed (see preceding error log)");
    const int result = std::system(command);
    (void)result;
}

#endif
