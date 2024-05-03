#Cmake file for library NodeFields.so
#Author:   yogurt
#Created:  2021-10-10  
#Modified: 2022-11-28

cmake_minimum_required(VERSION 3.9)

project(NodeFields)

message(STATUS "Start to make NodeFields.")

set(
    lib_dir
    ./include
    ../TxBase
    ../FuncDef/include
    ../FuncDef_expreval/include
    ../FuncDef_expreval/expreval35
    ${CEMPIC_HDF5_INC}
    ../BaseDefine
    ../DynBase
    ../FieldsBase/include
    ../ComboConformalFields/include
    ../ComboConformalFields/include_SI_SC_Matrix
    ../GridData/include
    ${tecIO_dir}
)

include_directories(${lib_dir})

aux_source_directory(./src source_list)

link_directories(${CEMPIC_LIB})
link_directories(${CEMPIC_HDF5_LIB})
link_directories(${tecIO_dir})

add_library(NodeFields SHARED ${source_list})

target_link_libraries(NodeFields ComboFields FieldsBase GridData FuncDef FuncDef_expreval txbase hdf5 tecio)