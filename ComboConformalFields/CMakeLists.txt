#Cmake file for library ComboFields.so
#Author:   yogurt
#Created:  2021-10-10
#Modified: 2022-11-28

cmake_minimum_required(VERSION 3.9)

project(ComboFields)

message(STATUS "Start to make ComboFields.")

set(
    lib_dir
    ./include
    ./include_SI_SC_Matrix
    ../TxBase
    ../BaseDefine
    ../FuncDef/include
    ../FuncDef_expreval/include
    ../FuncDef_expreval/expreval35
    ../DynBase
    ../FieldsBase/include
    ../GridData/include
    ../Cuda_Files
)

include_directories(${lib_dir})

aux_source_directory( ./src SRC)
aux_source_directory( ./src/EdgeSrc SRC)
aux_source_directory( ./src/FldModeLoad SRC)
aux_source_directory( ./src/SI_SC_Flds SRC)
aux_source_directory( ./src/SI_SC_Func SRC)
aux_source_directory( ./src/SI_SC_Matrix SRC)
aux_source_directory( ./src/SweptFaceSrc SRC)
aux_source_directory( ./src/VertexSrc SRC)
aux_source_directory( ./src/VoltagePort SRC)

link_directories(${CEMPIC_LIB})
link_directories(${CEMPIC_HDF5_LIB})

add_library(ComboFields SHARED ${SRC})

target_link_libraries(ComboFields FieldsBase GridData FuncDef FuncDef_expreval basedefine txbase hdf5 Cuda_Files)
target_compile_options(ComboFields PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>)
target_compile_options(ComboFields PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--prec-sqrt=true>)
target_compile_options(ComboFields PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--prec-div=true>)
target_compile_options(ComboFields PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false>)


# target_compile_options(ComboFields PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)
# set_target_properties(ComboFields PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# target_compile_options(ComboFields PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
# target_compile_options(ComboFields PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--prec-sqrt=true>)
# target_compile_options(ComboFields PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--prec-div=true>)
# target_compile_options(ComboFields PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false>)
# target_compile_options(ComboFields PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_80>)