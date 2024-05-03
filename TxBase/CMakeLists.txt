#Cmake file for library txbase.so
#Author:   yogurt
#Created:  2021-10-10
#Modified: 2022-11-28

cmake_minimum_required(VERSION 3.9)

project(txbase)

message(STATUS "STart to make txbase.")

include_directories(.)
include_directories(../Cuda_Files)

aux_source_directory(. source_list)

add_library(txbase SHARED ${source_list})