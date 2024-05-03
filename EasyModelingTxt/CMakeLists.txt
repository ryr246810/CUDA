#Cmake file for library GeomTxtBuilder.so
#Author:   yogurt
#Created:  2021-10-10  
#Modified: NULL    

cmake_minimum_required(VERSION 3.9)

project(GeomTxtBuilder)

message(STATUS "Start to make GeomTxtBuilder")

set(
    lib_dir
    .
    ../OCAF_LIB/include
    ${CEMPIC_OCCT_INC}
    ../BaseDefine
    ../TxBase
)

include_directories(${lib_dir})

aux_source_directory(. source_list)

link_directories(${CEMPIC_LIB})
link_directories(${CEMPIC_OCCT_LIB})

add_library(GeomTxtBuilder SHARED ${source_list})

target_link_libraries(GeomTxtBuilder OCAFTool txbase ${OCCLIBS_Draw} ${OCCLIBS_DataExchange} ${OCCLIBS_ApplicationFramework} ${OCCLIBS_Visualization}  
${OCCLIBS_ModelingAlgorithms} ${OCCLIBS_ModelingData} ${OCCLIBS_Foundation})