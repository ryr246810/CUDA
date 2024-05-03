#Cmake file for library ReadTool.so
#Author:   yogurt
#Created:  2022-06-04

cmake_minimum_required(VERSION 3.9)

project(ReadTool)

message(STATUS "Start to make ReadTool.")

set(
    lib_dir
    .
    ../TxBase
    ../BaseDefine
)
include_directories(${lib_dir})

aux_source_directory(. source_list)

link_directories(${CEMPIC_LIB})

add_library(ReadTool SHARED ${source_list})

target_link_libraries(ReadTool txbase)