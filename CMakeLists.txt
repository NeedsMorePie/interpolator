# Initially copied from https://stackoverflow.com/questions/44403127/adding-a-gpu-op-in-tensorflow--
# A few changes were required to make this work with TensorFlow 1.8.0.
# Windows version documented here: https://github.com/tensorflow/models/issues/1103
cmake_minimum_required(VERSION 3.5)
include(CMakeParseArguments)
project(interpolator)

# Get Tf include dirs, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE TF_INCLUDE_DIRS)
string(REGEX REPLACE "\n$" "" TF_INCLUDE_DIRS ${TF_INCLUDE_DIRS})
if("${TF_INCLUDE_DIRS}" STREQUAL "")
    message(FATAL_ERROR "TF_INCLUDE_DIRS could not be found. Make sure you are running this in an environment with Tensorflow installed.")
endif()
message(STATUS "TF_INCLUDE_DIRS is ${TF_INCLUDE_DIRS}")

# Get Tf library dirs, see https://github.com/tensorflow/tensorflow/issues/13607
execute_process(COMMAND python -c "import tensorflow as tf; print(tf.sysconfig.get_lib())" OUTPUT_VARIABLE TF_LIB_DIR)
string(REGEX REPLACE "\n$" "" TF_LIB_DIR ${TF_LIB_DIR})
if("${TF_LIB_DIR}" STREQUAL "")
    message(FATAL_ERROR "TF_LIB_DIR could not be found. Make sure you are running this in an environment with Tensorflow installed.")
endif()
message(STATUS "TF_LIB_DIR is ${TF_LIB_DIR}")

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

# Windows definitions.
if(WIN32)
    add_definitions(-DEIGEN_AVOID_STL_ARRAY)
    add_definitions(-DNOMINMAX -D_WIN32_WINNT=0x0A00 -DLANG_CXX11 -DCOMPILER_MSVC)
    add_definitions(-DWIN32 -DOS_WIN -D_MBCS -DWIN64 -DWIN32_LEAN_AND_MEAN -DNOGDI -DPLATFORM_WINDOWS)
    add_definitions(-DTENSORFLOW_USE_EIGEN_THREADPOOL -DEIGEN_HAS_C99_MATH -D_ITERATOR_DEBUG_LEVEL=0)
    add_definitions(/bigobj /nologo /EHsc /GF /FC /MP /Gm-)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP -DTF_EXTERN=\"extern __declspec(dllimport)\"")
endif()

# Find the Tensorflow library to link against.
if(WIN32)
    FIND_LIBRARY(TF_INTERNAL NAMES pywrap_tensorflow_internal HINTS ${TF_LIB_DIR}/python)
    message(STATUS "TF_INTERNAL is ${TF_INTERNAL}")
else()
    find_library(TF_FRAMEWORK tensorflow_framework HINTS ${TF_LIB_DIR})
    message(STATUS "TF_FRAMEWORK is ${TF_FRAMEWORK}")
endif()

macro(add_op_library)
    set(_OPTIONS_ARGS)
    set(_ONE_VALUE_ARGS NAME)
    set(_MULTI_VALUE_ARGS SOURCES)
    cmake_parse_arguments(_ADD_OP_LIBRARY "${_OPTIONS_ARGS}" "${_ONE_VALUE_ARGS}" "${_MULTI_VALUE_ARGS}" ${ARGN} )

    if(NOT _ADD_OP_LIBRARY_NAME)
        message(FATAL_ERROR "add_op_library: 'NAME' argument required.")
    endif()
    if(NOT _ADD_OP_LIBRARY_SOURCES)
        message(FATAL_ERROR "add_op_library: 'SOURCES' argument required.")
    endif()

    message(STATUS "Adding library ${_ADD_OP_LIBRARY_NAME} with files ${_ADD_OP_LIBRARY_SOURCES}")
    if (CUDA_FOUND)
        cuda_add_library(${_ADD_OP_LIBRARY_NAME} SHARED
            ${_ADD_OP_LIBRARY_SOURCES}
        )
    else()
        add_library(${_ADD_OP_LIBRARY_NAME} SHARED
            ${_ADD_OP_LIBRARY_SOURCES}
        )
    endif()

    # Link libraries.
    if(WIN32)
        target_link_libraries(${_ADD_OP_LIBRARY_NAME} ${CUDA_LIBRARIES} ${TF_INTERNAL})
        set_target_properties(${_ADD_OP_LIBRARY_NAME} PROPERTIES SUFFIX ".dll")
    else()
        target_link_libraries(${_ADD_OP_LIBRARY_NAME} ${CUDA_LIBRARIES} ${TF_FRAMEWORK})
        set_target_properties(${_ADD_OP_LIBRARY_NAME} PROPERTIES SUFFIX ".so")
    endif()

    install(TARGETS ${_ADD_OP_LIBRARY_NAME} DESTINATION ${CMAKE_BINARY_DIR})
endmacro()

# Add subdirectories.
file(GLOB_RECURSE SUB_DIR_LISTS ${CMAKE_SOURCE_DIR}/*/CMakeLists.txt)
foreach(SUB_DIR_LIST ${SUB_DIR_LISTS})
    message(STATUS "Found package ${SUB_DIR_LIST}")
    string(REGEX REPLACE "CMakeLists.txt" "" SUB_DIR_LIST ${SUB_DIR_LIST})
    add_subdirectory(${SUB_DIR_LIST})
endforeach()
