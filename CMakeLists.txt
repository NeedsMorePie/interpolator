# Initially copied from https://stackoverflow.com/questions/44403127/adding-a-gpu-op-in-tensorflow--
# A few changes were required to make this work with TensorFlow 1.8.0.
cmake_minimum_required(VERSION 3.5)

# Get Tf include dirs, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE TF_INCLUDE_DIRS)

# Get Tf library dirs, see https://github.com/tensorflow/tensorflow/issues/13607
execute_process(COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_lib())" OUTPUT_VARIABLE TF_LIB_DIR)

# There's also a trailing newline that needs to be stripped:
# https://stackoverflow.com/questions/39496043/how-to-strip-trailing-whitespace-in-cmake-variable
string(REGEX REPLACE "\n$" "" TF_LIB_DIR ${TF_LIB_DIR})

find_package(CUDA)
find_library(TF_FRAMEWORK tensorflow_framework HINTS ${TF_LIB_DIR})

set(CMAKE_CXX_FLAGS "-fPIC -std=c++11 ${CMAKE_CXX_FLAGS} -march=native -D_GLIBCXX_USE_CXX11_ABI=0")
if(CUDA_FOUND)
    message(STATUS "CUDA was found")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGOOGLE_CUDA=1")
    set(CUDA_PROPAGATE_HOST_FLAGS ON)

    # Create symlink to cuda home. This is needed due to tensorflow headers looking for "cuda/include/cuda.h".
    set(CUDA_SYMLINK "cuda")
    message(STATUS "Creating symlink ${CUDA_SYMLINK} to $ENV{CUDA_BIN_PATH}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink $ENV{CUDA_BIN_PATH} ${CUDA_SYMLINK})
else()
    set(CUDA_LIBRARIES "")
    message(STATUS "Compilation of ops will be for the CPU only")
endif()

# Set header include directories.
include_directories("build")
include_directories(${TF_INCLUDE_DIRS})

if (CUDA_FOUND)
    cuda_add_library(correlation_op SHARED 
        "common/ops/cost_volume/correlation_op.cc"
        "common/ops/cost_volume/correlation_op.cc.cu"
    )
else()
    add_library(correlation_op SHARED 
        "common/ops/cost_volume/correlation_op.cc"
        "common/ops/cost_volume/correlation_op.cc.cu"
    )
endif()

# Link libraries.
target_link_libraries(correlation_op ${CUDA_LIBRARIES} ${TF_FRAMEWORK})
set_target_properties(correlation_op PROPERTIES SUFFIX ".so")
