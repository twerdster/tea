# References

## Tea In Published Work

Aaron Wetzler, Ron Slossberg, and Ron Kimmel. *Rule Of Thumb: Deep derotation for improved fingertip detection*. BMVC 2015.

- arXiv: <https://arxiv.org/abs/1507.05726>

Tea was used in that work for large-scale decision-tree training over precomputed features.

## Algorithmic Background

Tea is a GPU-oriented trainer for randomized decision trees over large feature banks. The implementation is not presented in this repository as a formal standalone paper, but it belongs to the practical family of:

- randomized decision-tree and random-forest training
- histogram-based split evaluation
- multi-GPU batched feature evaluation

## Repository Scope

This documentation site is meant to describe:

- what Tea does
- how to build and verify it today
- how to prepare datasets
- how to run maintained tests and benchmarks

It is not intended to claim that every historical research configuration is fully regression-tested in the modernized repository state.
