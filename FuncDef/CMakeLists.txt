#Cmake file for library FuncDef.so
#Author:   yogurt
#Created:  2021-10-10 
#Modified: 2022-11-28

cmake_minimum_required(VERSION 3.9)

project(FuncDef)

message(STATUS "Start to make FuncDef.")

set(
    lib_dir
    ./include
    ../BaseDefine
    ../TxBase
)
include_directories(${lib_dir})

aux_source_directory(./ExpressionFunc   source_list)
aux_source_directory(./newDef           source_list)
aux_source_directory(./NoArgFunc        source_list)
aux_source_directory(./STFunc           source_list)
aux_source_directory(./TFunc            source_list)

link_directories(${CEMPIC_LIB})

add_library(FuncDef SHARED ${source_list})

target_link_libraries(FuncDef txbase basedefine)