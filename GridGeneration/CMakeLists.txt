#Cmake file for library GridGeneration.so
#Author:   yogurt
#Created:  2021-10-10
#Modified: 2022-11-28  

cmake_minimum_required(VERSION 3.9)

project(GridGeneration)

message(STATUS "Start to make GridGeneration")

set(
    lib_dir
    ./include
    ${CEMPIC_OCCT_INC}
    ../OCAF_LIB/include
    ../GridData/include
    ../BaseDefine
    ../TxBase
    ../Interface
    # ${CEMPIC_FREETYPE_INC}
)

include_directories(${lib_dir})

aux_source_directory(./Ctrl         source_list)
aux_source_directory(./GridTool     source_list)
aux_source_directory(./Ray_Tracing  source_list)
aux_source_directory(./ReadStd      source_list)
aux_source_directory(./Tool         source_list)
aux_source_directory(./Writer       source_list)

link_directories(${CEMPIC_LIB})
link_directories(${CEMPIC_OCCT_LIB})
link_directories(${CEMPIC_HDF5_LIB})

add_library(GridGeneration SHARED ${source_list})

target_link_libraries(GridGeneration txbase GridData MaterialDefineInterface OCAFTool hdf5
${OCCLIBS_Draw} ${OCCLIBS_DataExchange} ${OCCLIBS_ApplicationFramework} ${OCCLIBS_Visualization}  
${OCCLIBS_ModelingAlgorithms} ${OCCLIBS_ModelingData} ${OCCLIBS_Foundation})