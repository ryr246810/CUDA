// -----------------------------------------------------------------------
// File:      TxVector.cpp
// Purpose:   Templated class for representing small groups of data,like a particle position or velocity,
//            that can then be manipulated as a standard 1-D mathematical vector.
// -----------------------------------------------------------------------

#define VP_VECTOR_CPP

#include "TxVector.h"

template class TxVector<double>;

template class TxVector<int>;

template class TxVector<size_t>;

template class TxVector<bool>;
