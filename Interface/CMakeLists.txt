#Cmake file for library MaterialDefineInterface.so
#Author:   yogurt
#Created:  2021-10-10
#Modified: NULL

cmake_minimum_required(VERSION 3.9)

project(Interface)

message(STATUS "Start to make MaterialDefineInterface.")

set(
    lib_dir
    .
    ../OCAF_LIB/include
    ${CEMPIC_OCCT_INC}
    ../BaseDefine
)
include_directories(${lib_dir})

aux_source_directory(. source_list)

link_directories(${CEMPIC_LIB})
link_directories(${CEMPIC_OCCT_LIB})

add_library(MaterialDefineInterface SHARED ${source_list})

target_link_libraries(MaterialDefineInterface OCAFTool)
