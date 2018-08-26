cmake_minimum_required(VERSION 3.5)

add_op_library(NAME backward_warp_op SOURCES
    "backward_warp_op.cc"
    "backward_warp_op.cc.cu"
)
