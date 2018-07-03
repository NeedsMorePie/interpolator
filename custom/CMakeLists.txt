cmake_minimum_required(VERSION 3.5)

# Get Tf include dirs, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE TF_INCLUDE_DIRS)
include_directories(${TF_INCLUDE_DIRS})

# Get Tf library dirs, see https://github.com/tensorflow/tensorflow/issues/13607
execute_process(COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_lib())" OUTPUT_VARIABLE TF_LIB_DIR)

# There's also a trailing newline that needs to be stripped:
# https://stackoverflow.com/questions/39496043/how-to-strip-trailing-whitespace-in-cmake-variable
string(REGEX REPLACE "\n$" "" TF_LIB_DIR "${TF_LIB_DIR}")

find_library(TF_FRAMEWORK tensorflow_framework HINTS "${TF_LIB_DIR}")
message(STATUS "${TF_FRAMEWORK}")

find_package(CUDA)
set(CMAKE_CXX_FLAGS "--std=c++11 -fPIC -O2")
set(CUDA_PROPAGATE_HOST_FLAGS ON)

# Create library.
cuda_add_library(
    cuda_op SHARED
    src/cuda_op_kernel.cu
    src/cuda_op_kernel.cc)

# Link libraries.
target_link_libraries(cuda_op "${TF_FRAMEWORK}")

# copy test file to build folder
configure_file(src/test.py test.py COPYONLY)