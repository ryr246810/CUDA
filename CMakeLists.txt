#Cmake file for Executable build
#Author:   yogurt
#Created:  2022-06-04

cmake_minimum_required(VERSION 3.9)

project(CEMPIC_3D LANGUAGES CXX CUDA)

if (WIN32)
    message(STATUS "Now is Windows System.")
elseif (APPLE)
    message(STATUS "Now is Apple System.")
elseif (UNIX)
    message(STATUS "Now is Linux System.")
endif ()

# start to set the path of user's surroundings 
# If you want to make and run this project, please change the path of user's surroundings.
# A total of six dependency packages need to be installed in the user's surroundings.
# Dependency packages: opencascade-7.2.0, hdf5-1.10.5, mpich2-1.4.1, freetype-2.9.1, tecIO, blitz.

set(CASROOT ${PROJECT_SOURCE_DIR}/Pack/OCCT)
set(hdf5_prefix ${PROJECT_SOURCE_DIR}/Pack/HDF5)
set(tecIO_dir ${PROJECT_SOURCE_DIR}/Pack/TECIO)
set(CEMPIC_CUDA_RUNTIME_DIR "/usr/local/cuda/include")
set(CEMPIC_CUDA_SAMPLES_DIR "/usr/local/cuda/samples/common/inc")

# end to set the path of user's surroundings 

# start to set the relative directory of dependency packages

set(CEMPIC_OCCT_DIR ${CASROOT})
set(CEMPIC_OCCT_INC ${CEMPIC_OCCT_DIR}/include)
set(CEMPIC_OCCT_LIB ${CEMPIC_OCCT_DIR}/lib)

set (CEMPIC_HDF5_DIR ${hdf5_prefix})
set (CEMPIC_HDF5_INC ${CEMPIC_HDF5_DIR}/include)
set (CEMPIC_HDF5_LIB ${CEMPIC_HDF5_DIR}/lib)

# end to set the relative directory of dependency packages

# start to check CXX compiler

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

if(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS_cc " -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no c++0x support. Please use a different C++ compiler.")
endif()

# end to check CXX compiler

# start to set the option of compiler

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_cc} -m64")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_HDF5")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MPI")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMPICH_SKIP_MPICXX")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_OCC64")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCSFDB")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOCC_CONVERT_SIGNALS")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLIN")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLININTEL")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_CONFIG_H")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe") 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_REENTRANT")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -DNDEBUG")
message(STATUS "Release Model.")

# end to set the option of compiler

# start to set another name of used OCCT libraries 

set(
OCCLIBS_Foundation
TKMath TKernel
)

set(
OCCLIBS_ModelingData
TKBRep TKGeomBase TKG2d TKG3d
)

set(
OCCLIBS_ModelingAlgorithms
TKBO TKHLR TKOffset TKTopAlgo TKShHealing TKBool TKFeat TKFillet TKGeomAlgo TKMesh TKPrim
)

set(
OCCLIBS_Visualization
TKV3d TKService TKOpenGl
)

set(
OCCLIBS_ApplicationFramework
TKBin TKVCAF TKLCAF TKCAF TKCDF TKXml
)

set(
OCCLIBS_DataExchange
TKIGES TKSTEP TKXSBase TKSTEPBase TKSTL TKVRML TKXSBase TKSTEP209 TKSTEPAttr
)

# end to set another name of used OCCT libraries 

# start to set the basic path of project

set(CEMPIC_LIB ${PROJECT_SOURCE_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${CEMPIC_LIB})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# end to set the basic path of project

include_directories(${CEMPIC_CUDA_RUNTIME_DIR})
include_directories(${CEMPIC_CUDA_SAMPLES_DIR})

message(STATUS "CMake start!")

add_subdirectory(Cuda_Files)        # 添加CMakeList子目录
add_subdirectory(OCAF_LIB)
add_subdirectory(Interface)
add_subdirectory(TxBase)
add_subdirectory(BaseDefine)
add_subdirectory(ReadFileTool)
add_subdirectory(FuncDef)
add_subdirectory(FuncDef_expreval)
add_subdirectory(GridData)
add_subdirectory(FieldsBase)
add_subdirectory(ComboConformalFields)
add_subdirectory(ComboNodeFields)
add_subdirectory(EasyModelingTxt)
add_subdirectory(GridGeneration)
add_subdirectory(Ptcl_Cyl3D)
add_subdirectory(bin)

message(STATUS "CMake end!")