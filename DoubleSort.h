#ifndef TEA_DOUBLE_SORT_H
#define TEA_DOUBLE_SORT_H
#include "defines.h"
#include <algorithm>

// Partition paired sample/index arrays without changing split membership.
// Empty, odd-sized and singleton children need no artificial alignment.
inline int partition(Sample *samples, uint *indices, int start, int end)
{
    int left = start;
    int right = end;
    while (left <= right) {
        while (left <= right && samples[left].bestTraj == 0) ++left;
        while (left <= right && samples[right].bestTraj != 0) --right;
        if (left < right) {
            std::swap(samples[left], samples[right]);
            std::swap(indices[left], indices[right]);
            ++left;
            --right;
        }
    }
    return left;
}
#endif
