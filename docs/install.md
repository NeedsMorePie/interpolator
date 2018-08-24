# Installation

## Prerequisites

*   Anaconda3
*   CMake (version >= 3.5)
*   CUDA toolkit (version recommended by [Tensorflow installation](https://www.tensorflow.org/install/install_windows#requirements_to_run_tensorflow_with_gpu_support))
*   Windows only
    *   Visual studio 14 (2015)

## Python environment

Create a conda env with the following packages:
*   tensorflow-gpu
*   opencv-python
*   matplotlib
*   joblib
*   pillow

```
conda create --name interpolator python=3.5
conda activate interpolator
pip install tensorflow-gpu opencv-python matplotlib joblib pillow
```

## Building ops

### Linux

In a terminal with the conda environment activated:

```
mkdir build
cd build
cmake ..
cmake --build . --config Release --target install
```

Note that the built custom ops (e.g libcorrelation_op.so) must be in the `build` folder.

### Windows

In powershell with the conda environment activated:

```
mkdir build
cd build
cmake -G "Visual Studio 14 2015 Win64" ..
cmake --build . --config Release --target install
```

Note that the built custom ops (e.g correlation_op.dll) must be in the `build` folder.

### Common gotchas

After running cmake, you must look for ```CUDA was found``` in the console output:

If cmake can't find CUDA, you may need to manually pass it the toolkit's root path:

```
cmake -D CUDA_TOOLKIT_ROOT_DIR=/path/to/cuda ..
```
