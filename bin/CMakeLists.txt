#Cmake file for file EMPIC_CYL.exe
#Author:   yogurt
#Created:  2021-10-10    
#Modified: 2022-11-28

cmake_minimum_required(VERSION 3.9)

project(bin)

message(STATUS "Start to make bin")

set(
    lib_dir
    ${CEMPIC_OCCT_INC}
    ${CEMPIC_HDF5_INC}
    # ${CEMPIC_FREETYPE_INC}
    ${tecIO_dir}
    ../TxBase
    ../FuncDef/include
    ../FuncDef_expreval/include
    ../FuncDef_expreval/expreval35
    ../BaseDefine
    ../ReadFileTool
    ../DynBase
    ../GridData/include
    ../OCAF_LIB/include
    ../Interface
    ../GridGeneration/include
    ../EasyModelingTxt
    ../FieldsBase/include
    ../ComboConformalFields/include
    ../ComboConformalFields/include_SI_SC_Matrix
    ../ComboNodeFields/include
    ../Ptcl_Cyl3D
    ../Cuda_Files
)

include_directories(${lib_dir})

aux_source_directory(. source_list)

link_directories(${CEMPIC_LIB})
link_directories(${CEMPIC_OCCT_LIB})
link_directories(${CEMPIC_HDF5_LIB})
# link_directories(${CEMPIC_FREETYPE_LIB})
link_directories(${tecIO_dir})

add_executable(EMPIC_CYL ${source_list})

target_link_libraries(EMPIC_CYL "-Wl,--start-group" ComboFields NodeFields FieldsBase basedefine MaterialDefineInterface txbase FuncDef FuncDef_expreval ReadTool GridData GridGeneration OCAFTool GeomTxtBuilder Ptcl_Cyl3D  Cuda_Files ${OCCLIBS_Draw} ${OCCLIBS_DataExchange} ${OCCLIBS_ApplicationFramework} ${OCCLIBS_Visualization} ${OCCLIBS_ModelingAlgorithms} ${OCCLIBS_ModelingData} ${OCCLIBS_Foundation} "-Wl,--end-group" 
hdf5 tecio)
# target_compile_options(EMPIC_CYL PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)