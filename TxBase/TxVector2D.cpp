// -----------------------------------------------------------------------
// File:      TxVector2D.cpp
// Purpose:   Templated class for representing small groups of data,like a particle position or velocity,
//            that can then be manipulated as a standard 1-D mathematical vector.
// -----------------------------------------------------------------------

#define VP_VECTOR_CPP

#include "TxVector2D.h"

template class TxVector2D<double>;

template class TxVector2D<int>;

template class TxVector2D<size_t>;

template class TxVector2D<bool>;
