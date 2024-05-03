PROJECT(LIBEXPREVAL)

#set(CMAKE_CXX_STANDARD 17)

SET(CMAKE_CXX_COMPILER ${CXX_COMPILER})
SET(CMAKE_C_COMPILER ${C_COMPILER})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_CFLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_CFLAGS}")

INCLUDE_DIRECTORIES("${LIBEXPREVAL_SOURCE_DIR}")
INCLUDE_DIRECTORIES("${boost_prefix}/include")

set(EXPREVAL_INCLUDE_FILES
  alglibinternal.h
  ap.h
  defs.h
  except.h
  expreval.h
  expr.h
  funclist.h
  node.h
  parser.h
  specialfunctions.h
  vallist.h)

set(EXPREVAL_SRC
  except.cpp
  expr.cpp
  func.cpp
  funclist.cpp
  node.cpp
  parser.cpp
  vallist.cpp
  alglibinternal.cpp
  ap.cpp
  specialfunctions.cpp)

#CUDA_ADD_LIBRARY(expreval SHARED ${EXPREVAL_SRC})
ADD_LIBRARY(expreval SHARED ${EXPREVAL_SRC})
#SET(LIBRARY_OUTPUT_PATH "${CEMPIC_SOURCE_DIR}/lib")
SET(LIBRARY_OUTPUT_PATH "${CEMPIC_LIB}")
