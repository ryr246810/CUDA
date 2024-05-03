#Cmake file for library Ptcl_Cyl3D.so
#Author:   yogurt
#Created:  2021-10-10
#Modified: 2022-11-28   

cmake_minimum_required(VERSION 3.9)

project(Ptcl_Cyl3D)

message(STATUS "Start to make Ptcl_Cyl3D")

set(
    lib_dir
    .
    ../Cuda_Files
    ../TxBase
    ../FuncDef/include
    ../BaseDefine
    ../DynBase
    ../FieldsBase/include
    ../GridData/include
    ../ComboNodeFields/include
    ../ComboConformalFields/include
    ../ComboConformalFields/include_SI_SC_Matrix
    ${tecIO_dir}
)

include_directories(${lib_dir})

aux_source_directory(. source_list)
aux_source_directory(./cu_Source source_list)

link_directories(${CEMPIC_LIB})
link_directories(${CEMPIC_OCCT_LIB})
link_directories(${CEMPIC_HDF5_LIB})
link_directories(${tecIO_dir})

add_library(Ptcl_Cyl3D SHARED ${source_list})

target_link_libraries(Ptcl_Cyl3D "-Wl,--start-group" ${OCCLIBS_Draw} ${OCCLIBS_DataExchange} ${OCCLIBS_ApplicationFramework} ${OCCLIBS_Visualization}  
${OCCLIBS_ModelingAlgorithms} ${OCCLIBS_ModelingData} ${OCCLIBS_Foundation} "-Wl,--end-group" FieldsBase GridData FuncDef txbase NodeFields hdf5 tecio Cuda_Files)

# target_compile_options(Ptcl_Cyl3D PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
target_compile_options(Ptcl_Cyl3D PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>)
target_compile_options(Ptcl_Cyl3D PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--prec-sqrt=true>)
target_compile_options(Ptcl_Cyl3D PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--prec-div=true>)
target_compile_options(Ptcl_Cyl3D PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false>)
# target_compile_options(Ptcl_Cyl3D PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_80>)