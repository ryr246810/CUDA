#Cmake file for library GridData.so
#Author:   yogurt
#Created:  2021-10-10
#Modified: 2022-11-28  

cmake_minimum_required(VERSION 3.9)

project(GridData)

message(STATUS "Start to make GridData.")

set(
    lib_dir
    ./include
    ../FuncDef/include
    ../BaseDefine
    ../TxBase
    ${CEMPIC_HDF5_INC}
    ../Cuda_Files
)

include_directories(${lib_dir})

aux_source_directory(./DataElement  source_list)
aux_source_directory(./GridBnd      source_list)
aux_source_directory(./GridData     source_list)
aux_source_directory(./GridElement  source_list)
aux_source_directory(./GridGeom     source_list)
aux_source_directory(./PMLData      source_list)
aux_source_directory(./TElement     source_list)

link_directories(${CEMPIC_LIB})
link_directories(${CEMPIC_HDF5_LIB})

add_library(GridData SHARED ${source_list})

target_link_libraries(GridData FuncDef basedefine txbase hdf5)