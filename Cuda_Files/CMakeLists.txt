#Cmake file for library Cuda_Files.so
#Author:   yogurt
#Created:  2021-12-01
#Modified: 2022-06-22  

cmake_minimum_required(VERSION 3.9)

project(Cuda_Files)

message(STATUS "Start to make Cuda_Files.")

include_directories(.) 

aux_source_directory(. source_list)     # 搜索当前目录下的所有文件

add_library(Cuda_Files SHARED ${source_list})   # 生成静态库

# target_compile_options(Cuda_Files PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_80>)