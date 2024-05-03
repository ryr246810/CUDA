#Cmake file for library FieldsBase.so
#Author:   yogurt
#Created:  2021-10-10  
#Modified: 2022-11-28  

cmake_minimum_required(VERSION 3.9)

project(FieldsBase)

message(STATUS "Start to make FieldsBase.")

set(
    lib_dir
    ./include
    ../TxBase
    ../BaseDefine
    ../DynBase
    ../GridData/include
)

include_directories(${lib_dir})

aux_source_directory(./Dgn source_list)
aux_source_directory(./src source_list)

link_directories(${CEMPIC_LIB})
link_directories(${CEMPIC_HDF5_LIB})

add_library(FieldsBase SHARED ${source_list})

target_link_libraries(FieldsBase txbase GridData hdf5)