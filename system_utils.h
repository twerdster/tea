#ifndef SYSTEM_UTILS_H
#define SYSTEM_UTILS_H

#include <cstdlib>

inline void discardSystemResult(const char *command)
{
    const int result = std::system(command);
    (void)result;
}

#endif
