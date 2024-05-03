#Cmake file for library FuncDef_expreval.so
#Author:      yogurt
#Created:     2022-10-10
#Modified:    NULL

project(FuncDef_expreval)
cmake_minimum_required(VERSION 3.9)

message(STATUS "Start to Make FuncDef_expreval.")

set(
    lib_dir
    ./include
    ./expreval35
    ../BaseDefine
    ../TxBase
)
include_directories(${lib_dir})

aux_source_directory( ./expreval35 SRC)
aux_source_directory( ./src SRC)

link_directories( ${CEMPIC_LIB})

add_library(FuncDef_expreval STATIC ${SRC})

target_link_libraries(FuncDef_expreval txbase basedefine)
