cmake_minimum_required(VERSION 3.5)

add_op_library(NAME correlation_op SOURCES
    "correlation_op.cc"
    "correlation_op.cc.cu"
)
