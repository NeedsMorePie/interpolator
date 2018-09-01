cmake_minimum_required(VERSION 3.5)

add_op_library(NAME forward_warp_op SOURCES
    "forward_warp_op.cc"
    "forward_warp_op.cc.cu"
)
