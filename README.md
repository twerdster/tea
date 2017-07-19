# Tea

Tea is a randomized decision tree learning framework designed for fast learning on massive tera-byte sized datasets.
It can handle up to 2B examples with thousands of features and up to 256 separate classes.

It is the first such framework which can take advantage of multiple GPUs on a single computing node to perform highly parallel decision tree
training. 

## Making Tea

This has currently only been tested on Ubuntu 14.04. You will need Boost version >= 1.54 and Cuda version >= 7.5.
Run `make Tea` to make Tea. You can then run `./Tea` to see current options.

## Reading Tea leaves

The default output of Tea is an unpruned randomized decision tree with its leaves containing the apriori class distribution of a given
input example. Performing inference using Tea is incredibly fast and we provide a number of different implementations for different
platforms using CPU, GPU and FPGA.   

## Bagging with Tea

Tea can perform bagging efficiently during training.   

## Whats in a name?

Tea takes its name from the ancient beverage: tea. The beverage has a long and interesting history
and is ultimately created by a collection of leaves which can, when properly brewed, infuse a liquid with an
almost infinite variety of different tastes. 

## Core elements

The underlying training algorithm is loosely based on ID3 and includes a number of novel modifications
to enable mapping of the training process to multiple processing units and back again while minimizing and hiding 
data transmission overhead.
It has been developed by Aaron Wetzler as part of his PhD thesis in the GIP lab at the Technion Israel Institute of Technology.


## TODO:
a lot ...
* CMake
* Python and Matlab interfaces
* Pruning and other utilities
* Refactoring
* Examples
* Better logging
* Regression
* Boosting
* Missing values
* Forests
* Preloaded GPU data pool
* Best Delta node updates




