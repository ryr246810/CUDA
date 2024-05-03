//--------------------------------------------------------------------
//
// File:    TxTensor.cpp:
//
// Purpose: Implementation and instantiation of ref counted tensor 
//
// Version: $Id: TxTensor.cpp 91 2007-03-04 19:59:18Z cary $
//
// Copyright (c) 1996-1999 by Tech-X Corporation.  All rights reserved.
//
//--------------------------------------------------------------------

#define TX_TENSOR_CPP

// txbase includes
#include "TxTensor.h"

// ansi instantiation

template class TxTensor<float, 1>;
template class TxTensor<float, 2>;
template class TxTensor<float, 3>;
template class TxTensor<float, 4>;

template class TxTensor<double, 1>;
template class TxTensor<double, 2>;
template class TxTensor<double, 3>;
template class TxTensor<double, 4>;

template class TxTensor<int, 1>;
template class TxTensor<int, 2>;
template class TxTensor<int, 3>;
template class TxTensor<int, 4>;

template class TxTensor<std::complex<double>, 1>;
template class TxTensor<std::complex<double>, 2>;
template class TxTensor<std::complex<double>, 3>;
template class TxTensor<std::complex<double>, 4>;

template class TxTensor<void*, 1>;
template class TxTensor<void*, 2>;
template class TxTensor<void*, 3>;
template class TxTensor<void*, 4>;


#ifndef __HP_aCC
// Not compiling on hpux as aCC runs out of memory
template class TxTensor<size_t, 1>;
template class TxTensor<size_t, 2>;
template class TxTensor<size_t, 3>;
template class TxTensor<size_t, 4>;
#endif

#ifndef __hpux
// Not present on hpux
template class TxTensor<long double, 1>;
template class TxTensor<long double, 2>;
template class TxTensor<long double, 3>;
template class TxTensor<long double, 4>;
#endif

