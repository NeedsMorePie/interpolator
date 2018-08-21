# Initially copied from https://stackoverflow.com/questions/44403127/adding-a-gpu-op-in-tensorflow--
# A few changes were required to make this work with TensorFlow 1.8.0.
# Windows version documented here: https://github.com/tensorflow/models/issues/1103
cmake_minimum_required(VERSION 3.5)

# Get Tf include dirs, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE TF_INCLUDE_DIRS)
message(STATUS "TF_INCLUDE_DIRS is ${TF_INCLUDE_DIRS}")

# Get Tf library dirs, see https://github.com/tensorflow/tensorflow/issues/13607
execute_process(COMMAND python -c "import tensorflow as tf; print(tf.sysconfig.get_lib())" OUTPUT_VARIABLE TF_LIB_DIR)
message(STATUS "TF_LIB_DIR is ${TF_LIB_DIR}")

# There's also a trailing newline that needs to be stripped:
# https://stackoverflow.com/questions/39496043/how-to-strip-trailing-whitespace-in-cmake-variable
string(REGEX REPLACE "\n$" "" TF_LIB_DIR ${TF_LIB_DIR})

find_package(CUDA)

set(CMAKE_CXX_FLAGS "-fPIC -std=c++11 ${CMAKE_CXX_FLAGS} -march=native -D_GLIBCXX_USE_CXX11_ABI=0")
if(CUDA_FOUND)
    message(STATUS "CUDA was found")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGOOGLE_CUDA=1")
    set(CUDA_PROPAGATE_HOST_FLAGS ON)

    # tf assumes in various places header files to be in cuda/include.
    if(WIN32)
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/cmake/CMakeLists.txt
        FILE(COPY
            ${CUDA_TOOLKIT_ROOT_DIR}/include
            DESTINATION cuda
        )
    else()
        # Unix-based operating systems can create symlinks.
        set(CUDA_SYMLINK "cuda")
        message(STATUS "Creating symlink ${CUDA_SYMLINK} to ${CUDA_TOOLKIT_ROOT_DIR}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SYMLINK})
    endif()
else()
    set(CUDA_LIBRARIES "")
    message(STATUS "Compilation of ops will be for the CPU only")
endif()

# Set header include directories.
include_directories("build")
include_directories(${TF_INCLUDE_DIRS})

if (CUDA_FOUND)
    cuda_add_library(correlation_op SHARED 
        "pwcnet/cost_volume/correlation_op.cc"
        "pwcnet/cost_volume/correlation_op.cc.cu"
    )
    cuda_add_library(backward_warp_op SHARED
        "pwcnet/warp/backward_warp_op.cc"
        "pwcnet/warp/backward_warp_op.cc.cu"
    )
else()
    add_library(correlation_op SHARED 
        "pwcnet/cost_volume/correlation_op.cc"
        "pwcnet/cost_volume/correlation_op.cc.cu"
    )
    add_library(backward_warp_op SHARED
        "pwcnet/warp/backward_warp_op.cc"
        "pwcnet/warp/backward_warp_op.cc.cu"
    )
endif()

# Windows definitions.
if(WIN32)
    add_definitions(-DEIGEN_AVOID_STL_ARRAY)
    add_definitions(-DNOMINMAX -D_WIN32_WINNT=0x0A00 -DLANG_CXX11 -DCOMPILER_MSVC)
    add_definitions(-DWIN32 -DOS_WIN -D_MBCS -DWIN64 -DWIN32_LEAN_AND_MEAN -DNOGDI -DPLATFORM_WINDOWS)
    add_definitions(-DTENSORFLOW_USE_EIGEN_THREADPOOL -DEIGEN_HAS_C99_MATH -D_ITERATOR_DEBUG_LEVEL=0)
    add_definitions(/bigobj /nologo /EHsc /GF /FC /MP /Gm-)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP -DTF_EXTERN=\"extern __declspec(dllimport)\"")
endif()

# Link libraries.
if(WIN32)
    FIND_LIBRARY(TF_INTERNAL NAMES pywrap_tensorflow_internal HINTS ${TF_LIB_DIR}/python)
    message(STATUS "TF_INTERNAL is ${TF_INTERNAL}")

    target_link_libraries(correlation_op ${CUDA_LIBRARIES} ${TF_INTERNAL})
    set_target_properties(correlation_op PROPERTIES SUFFIX ".dll")

    target_link_libraries(backward_warp_op ${CUDA_LIBRARIES} ${TF_INTERNAL})
    set_target_properties(backward_warp_op PROPERTIES SUFFIX ".dll")
else()
    find_library(TF_FRAMEWORK tensorflow_framework HINTS ${TF_LIB_DIR})
    message(STATUS "TF_FRAMEWORK is ${TF_FRAMEWORK}")

    target_link_libraries(correlation_op ${CUDA_LIBRARIES} ${TF_FRAMEWORK})
    set_target_properties(correlation_op PROPERTIES SUFFIX ".so")

    target_link_libraries(backward_warp_op ${CUDA_LIBRARIES} ${TF_FRAMEWORK})
    set_target_properties(backward_warp_op PROPERTIES SUFFIX ".so")
endif()
