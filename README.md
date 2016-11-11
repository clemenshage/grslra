## GRSLRA. What does this tongue-twisting abbreviation stand for?
GRSLRA stands for Grassmannian Robust (Structured) Low-Rank Approximation. The grslra package implements various algorithms for Low-Rank Approximation on the Grassmann manifold using a smooth loss function based on the non-convex Lp norm. The algorithms are related to publications on [Robust PCA], [Robust Subspace Tracking] and [Robust Structured Low-Rank Approximation]. Applications are in Big Data analysis, video segmentation as well as time series analysis and forecasting.
The implementation is done in Python 2.7 using NumPy and SciPy, visualization is performed using Matplotlib. The package contains C-extensions for the cost function, whose compilation requires CBLAS and OpenMP. Some functions in the video processing module require OpenCV and ffmpeg, but they are not required for installing the package.
The code has been developed and tested under Ubuntu 16.04 within a virtual environment.

## Main modules
The folder 'grslra' contains the main modules of the package, containg classes or methods that describe the problem, its geometry, optimization methods and the respective algorithms

problems.py: required variables, cost functions and gradients for the problems to be solved
optimization.py: first-order optimization methods
spaces.py: spaces over which the optimization is performed

grpca.py: algorithm for Robust PCA
grst.py: algorithm for Robust Subspace Tracking
grslra_batch.py: batch algorithm for Robust Structured Low-Rank Approximation
grslra_online.py: algorithm for online Time Series Analysis

and some more tools and method for pre- and post processing and visualization

## Where should I start?
The folder 'experiments' contains various scripts to the following experiments:

4_grpca: Robust PCA experiments
- Comparison of phase transitions in rank and sparsity to other state-of-the-art methods. See the README.txt file in the 'matlab' folder for information on where to obtain those algorithms.
- Real-world video segmentation on the 'escalator' sequence from [Li et al, 2004]
5_grst: Robust Subspace Tracking experiments
- Simulations on generated data
- see Florian Seidel's implementation of [pROST] for a practical application on video segmentation
6_grslra: Robust Structured Low-Rank Approximation
- Simulations on System Identification of LTI and LTV systems
- Real-world time series forecasting on [airline passenger data]

## License
The software package is licensed under the MIT license. See the license file for details.

[Robust PCA]:https://arxiv.org/abs/1210.0805
[Robust Subspace Tracking]:https://arxiv.org/abs/1302.2073
[Robust Structured Low-Rank Approximation]:https://arxiv.org/abs/1506.03958
[Li et al, 2004]:http://ieeexplore.ieee.org/document/1344037/
[pROST]:https://github.com/FlorianSeidel/GOL
[airline passenger data]:http://www.rita.dot.gov/bts
