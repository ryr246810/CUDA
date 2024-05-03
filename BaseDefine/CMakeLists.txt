#Cmake file for library basedefine.so
#Author:   yogurt
#Created:  2021-10-10 
#Modified: 2022-11-28

cmake_minimum_required(VERSION 3.9)

project(BaseDefine)

message(STATUS "Start to make basedefine.")
set(
    lib_dir
    .
    ../Cuda_Files
)
include_directories(${lib_dir})

aux_source_directory(. source_list)

add_library(basedefine SHARED ${source_list})

target_link_libraries(basedefine Cuda_Files)