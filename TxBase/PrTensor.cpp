//--------------------------------------------------------------------
//
// File:    PrTensor.C
//
// Purpose: Implementation of tensor letter class
//
// Version: $Id: PrTensor.cpp 121 2007-11-21 21:07:25Z swsides $
//
// Copyright 1996, 1997, 1998 by Tech-X Corporation
//
//--------------------------------------------------------------------

#define PR_TENSOR_CPP

// std includes
#include <complex>

// txbase includes
#include "PrTensor.h"
#include "TxDebugExcept.h"

template <class Type, size_t dimen>
PrTensor<Type, dimen>::PrTensor() : PrRefCount() {
  size_t i;
  for (i=0; i<dimen; i++) {
    lengths[i] = 0;
    size[i+1] = 1;
  }
  size[0] = 0;
  data = 0;
}

template <class Type, size_t dimen>
PrTensor<Type, dimen>::PrTensor(size_t* len) {
  setupLengths(len);
  getDataMemory();
}

template <class Type, size_t dimen>
PrTensor<Type, dimen>::PrTensor(const PrTensor<Type, dimen>& prt) : 
	PrRefCount(prt) {
  setupLengths(prt.lengths);
  getDataMemory();
  for (size_t i=0; i<size[0]; i++) data[i] = prt.data[i];
}

//  Constructors giving dimensions.  Ignore irrelevant arguments
template <class Type, size_t dimen>
PrTensor<Type, dimen>::PrTensor(size_t i) {
  lengths[0] = i;
  size_t il; 
  for (il=1; il<dimen; il++) lengths[il] = 1;
  getDataMemory();
}

template <class Type, size_t dimen>
PrTensor<Type, dimen>::PrTensor(size_t i, size_t j) {
  switch (dimen) {
    case 2:
      lengths[1] = j;
    case 1:
      lengths[0] = i;
    case 0:
      break;
  }
  size_t il;
  for (il=2; il<dimen; il++) lengths[il] = 1;
  getDataMemory();
}

template <class Type, size_t dimen>
PrTensor<Type, dimen>::PrTensor(size_t i, size_t j, size_t k) {
  switch (dimen) {
    case 3:
      lengths[2] = k;
    case 2:
      lengths[1] = j;
    case 1:
      lengths[0] = i;
    case 0:
      break;
  }
  size_t il;
  for (il=3; il<dimen; il++) lengths[il] = 1;
  getDataMemory();
}

template <class Type, size_t dimen>
PrTensor<Type, dimen>::PrTensor(size_t i, size_t j, size_t k, size_t l) {
  switch (dimen) {
    case 4:
      lengths[3] = l;
    case 3:
      lengths[2] = k;
    case 2:
      lengths[1] = j;
    case 1:
      lengths[0] = i;
    case 0:
      break;
  }
  size_t il;
  for (il=4; il<dimen; il++) lengths[il] = 1;
  getDataMemory();
}

template <class Type, size_t dimen>
PrTensor<Type, dimen>::PrTensor(size_t i, size_t j, size_t k, size_t l, 
	size_t m) {
  switch (dimen) {
    case 5:
      lengths[4] = m;
    case 4:
      lengths[3] = l;
    case 3:
      lengths[2] = k;
    case 2:
      lengths[1] = j;
    case 1:
      lengths[0] = i;
    case 0:
      break;
  }
  size_t il;
  for (il=5; il<dimen; il++) lengths[il] = 1;
  getDataMemory();
}

template <class Type, size_t dimen>
PrTensor<Type, dimen>::PrTensor(size_t i, size_t j, size_t k, size_t l, 
	size_t m, size_t n) {
  switch (dimen) {
    case 6:
      lengths[5] = n;
    case 5:
      lengths[4] = m;
    case 4:
      lengths[3] = l;
    case 3:
      lengths[2] = k;
    case 2:
      lengths[1] = j;
    case 1:
      lengths[0] = i;
    case 0:
      break;
  }
  size_t il;
  for (il=6; il<dimen; il++) lengths[il] = 1;
  getDataMemory();
}

template <class Type, size_t dimen>
PrTensor<Type, dimen>::PrTensor(size_t i, size_t j, size_t k, size_t l, 
	size_t m, size_t n, size_t o) {
  switch (dimen) {
    case 7:
      lengths[6] = o;
    case 6:
      lengths[5] = n;
    case 5:
      lengths[4] = m;
    case 4:
      lengths[3] = l;
    case 3:
      lengths[2] = k;
    case 2:
      lengths[1] = j;
    case 1:
      lengths[0] = i;
    case 0:
      break;
  }
  size_t il;
  for (il=7; il<dimen; il++) lengths[il] = 1;
  getDataMemory();
}

//  Auxiliary functions
template <class Type, size_t dimen>
void PrTensor<Type, dimen>::setupLengths(const size_t* len) {
  size_t i;
  for (i=0; i<dimen; i++) {
#ifdef SAFEPRTENSOR
    assert(len[i]);
#endif
    lengths[i] = len[i];
  }
}

template <class Type, size_t dimen>
void PrTensor<Type, dimen>::getDataMemory() {
  for (size_t j=0; j<dimen; ++j) size[j] = 1;
  for (size_t i=0; i<dimen; ++i) {
    for (size_t j=0; j<dimen-i; ++j) {
      size[j] *= lengths[i+j];
    }
  }
  size[dimen] = 1;
  tensorRank = dimen;
  data = new Type[size[0]];
}

template <class Type, size_t dimen>
PrTensor<Type, dimen>::~PrTensor() {
  delete [] data;
}


template <class Type, size_t dimen>
void PrTensor<Type, dimen>::setLength(size_t lengthIndex, size_t lengthValue) {
  lengths[lengthIndex] = lengthValue;
  delete [] data;
  getDataMemory();
}

template <class Type, size_t dimen>
void PrTensor<Type, dimen>::setLengths(size_t lens[dimen]) {
  for (size_t i=0; i<dimen; ++i) lengths[i] = lens[i];
  delete [] data;
  getDataMemory();
}


template <class Type, size_t dimen>
const PrTensor<Type, dimen>& 
PrTensor<Type, dimen>::operator= (const PrTensor<Type, dimen>& prt) {
  if ( this == &prt ) return *this;
  setupLengths(prt.lengths);
  delete [] data;
  getDataMemory();
  for (size_t i=0; i<size[0]; i++) data[i] = prt.data[i];
  return *this;
}

template <class Type, size_t dimen>
const PrTensor<Type, dimen>& PrTensor<Type, dimen>::operator=(Type t) {
  for (size_t i=0; i<size[0]; i++) data[i] = t;
  return *this;
}

template <class Type, size_t dimen>
Type PrTensor<Type, dimen>::operator()(size_t* indices) const {
  size_t indx = indices[0];
  for (size_t i=1; i<dimen; i++) {
    indx *= lengths[i];
    indx += indices[i];
  }
  return data[indx];
}

/*
template <class Type, size_t dimen>
Type 
PrTensor<Type, dimen>::
operator()(size_t i) const {
#ifdef SAFEPRTENSOR
  assert(dimen >= 1);
#endif
  return data[i];
}
*/

/* Now in header
template <class Type, size_t dimen>
Type
PrTensor<Type, dimen>::
operator()(size_t i, size_t j) const {
#ifdef SAFEPRTENSOR
  assert(dimen >= 2);
#endif

  size_t indx=0;
  switch (dimen) {
    case 2:
      indx = i*lengths[dimen-1];
    case 1:
      indx += j;
      break;
    default:
      throw TxDebugExcept("More indices than dimensions");
  }

// Desired size_t indx = i*lengths[dimen-1] + j;
  return data[indx];
}
*/

template <class Type, size_t dimen>
Type PrTensor<Type, dimen>::operator()(size_t i, size_t j, size_t k) const {
#ifdef SAFEPRTENSOR
  assert(dimen >= 3);
#endif

  size_t indx=0;
  int dim(dimen);  
  switch (dimen) {
    case 3:
      indx = i*lengths[dim-2];
    case 2:
      indx = (indx + j)*lengths[dim-1];
    case 1:
      indx += k;
      break;
    default:
      throw TxDebugExcept("More indices than dimensions");
  }

// Desired: size_t indx = (i*lengths[dimen-2] + j)*lengths[dimen-1] + k;

  return data[indx];
}

template <class Type, size_t dimen>
Type
PrTensor<Type, dimen>::
operator()(size_t i, size_t j, size_t k, size_t l) const {
#ifdef SAFEPRTENSOR
  assert(dimen >= 4);
#endif

  size_t indx=0;
  int dim(dimen);
  switch (dimen) {
    case 4:
      indx = i*lengths[dim-3];
    case 3:
      indx = (indx + j)*lengths[dim-2];
    case 2:
      indx = (indx + k)*lengths[dim-1];
    case 1:
      indx += l;
      break;
    default:
      throw TxDebugExcept("More indices than dimensions");
  }

// Desired: size_t indx = ( (i*lengths[dimen-3] + j)*lengths[dimen-2] + k)
//  *lengths[dimen-1] + l;

  return data[indx];
}

template <class Type, size_t dimen>
Type
PrTensor<Type, dimen>::
operator()(size_t i, size_t j, size_t k, size_t l, size_t m) const {
#ifdef SAFEPRTENSOR
  assert(dimen >= 5);
#endif

  size_t indx=0;
  int dim(dimen);  
  switch (dimen) {
    case 5:
      indx = i*lengths[dim-4];
    case 4:
      indx = (indx + j)*lengths[dim-3];
    case 3:
      indx = (indx + k)*lengths[dim-2];
    case 2:
      indx = (indx + l)*lengths[dim-1];
    case 1:
      indx += m;
      break;
    default:
      throw TxDebugExcept("More indices than dimensions");
  }

// Desired: size_t indx = ( ( (i*lengths[dimen-4] + j)*lengths[dimen-3] + k)
//  *lengths[dimen-2] + l)*lengths[dimen-1] + m;

  return data[indx];
}

template <class Type, size_t dimen>
Type
PrTensor<Type, dimen>::
operator()(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n) const {
#ifdef SAFEPRTENSOR
  assert(dimen >= 6);
#endif

  size_t indx=0;
  int dim(dimen); 
  switch (dimen) {
    case 6:
      indx = i*lengths[dim-5];
    case 5:
      indx = (indx + j)*lengths[dim-4];
    case 4:
      indx = (indx + k)*lengths[dim-3];
    case 3:
      indx = (indx + l)*lengths[dim-2];
    case 2:
      indx = (indx + m)*lengths[dim-1];
    case 1:
      indx += n;
      break;
    default:
      throw TxDebugExcept("More indices than dimensions");
  }

// Desired: size_t indx = ( ( ( (i*lengths[dimen-5] + j)*lengths[dimen-4] + k)
//  *lengths[dimen-3] + l)*lengths[dimen-2] + m)*lengths[dimen-1] + n;

  return data[indx];
}

template <class Type, size_t dimen>
Type
PrTensor<Type, dimen>::
operator()(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n, size_t o) const {
#ifdef SAFEPRTENSOR
  assert(dimen >= 7);
#endif

  size_t indx=0;
  int dim(dimen);  
  switch (dimen) {
    case 7:
      indx = i*lengths[dim-6];
    case 6:
      indx = (indx + j)*lengths[dim-5];
    case 5:
      indx = (indx + k)*lengths[dim-4];
    case 4:
      indx = (indx + l)*lengths[dim-3];
    case 3:
      indx = (indx + m)*lengths[dim-2];
    case 2:
      indx = (indx + n)*lengths[dim-1];
    case 1:
      indx += o;
      break;
    default:
      throw TxDebugExcept("More indices than dimensions");
  }

// Desired: size_t indx = ( ( ( ( (i*lengths[dimen-6] + j)*lengths[dimen-5] + k)
//  *lengths[dimen-4] + l)*lengths[dimen-3] + m)*lengths[dimen-2] + n)
//  *lengths[dimen-1] + o;

  return data[indx];
}

template <class Type, size_t dimen>
Type&
PrTensor<Type, dimen>::
operator()(size_t* indices) {
  size_t indx = indices[0];
  size_t i;
  for (i=1; i<dimen; i++) {
    indx *= lengths[i];
    indx += indices[i];
  }
  return data[indx];
}

/* Now in header file
template <class Type, size_t dimen>
Type&
PrTensor<Type, dimen>::
operator()(size_t i) {
#ifdef SAFEPRTENSOR
  assert(dimen >= 1);
#endif
  return data[i];
}
*/

/* Now in header file
template <class Type, size_t dimen>
Type&
PrTensor<Type, dimen>::
operator()(size_t i, size_t j) {
#ifdef SAFEPRTENSOR
  assert(dimen >= 2);
#endif

  size_t indx=0;
  switch (dimen) {
    case 2:
      indx = i*lengths[dimen-1];
    case 1:
      indx += j;
      break;
    default:
      throw TxDebugExcept("More indices than dimensions");
  }

// Desired: size_t indx = i*lengths[dimen-1] + j;

  return data[indx];
}
*/

template <class Type, size_t dimen>
Type&
PrTensor<Type, dimen>::
operator()(size_t i, size_t j, size_t k) {
#ifdef SAFEPRTENSOR
  assert(dimen >= 3);
#endif

  size_t indx=0;
  int dim(dimen);  
  switch (dimen) {
    case 3:
      indx = i*lengths[dim-2];
    case 2:
      indx = (indx + j)*lengths[dim-1];
    case 1:
      indx += k;
      break;
    default:
      throw TxDebugExcept("More indices than dimensions");
  }

// Desired: size_t indx = (i*lengths[dimen-2] + j)*lengths[dimen-1] + k;

  return data[indx];
}

template <class Type, size_t dimen>
Type&
PrTensor<Type, dimen>::
operator()(size_t i, size_t j, size_t k, size_t l) {
#ifdef SAFEPRTENSOR
  assert(dimen >= 4);
#endif

  size_t indx=0;
  int dim(dimen);  
  switch (dimen) {
    case 4:
      indx = i*lengths[dim-3];
    case 3:
      indx = (indx + j)*lengths[dim-2];
    case 2:
      indx = (indx + k)*lengths[dim-1];
    case 1:
      indx += l;
      break;
    default:
      throw TxDebugExcept("More indices than dimensions");
  }

// Desired: size_t indx = ( (i*lengths[dimen-3] + j)*lengths[dimen-2] + k)
//  *lengths[dimen-1] + l;

  return data[indx];
}

template <class Type, size_t dimen>
Type&
PrTensor<Type, dimen>::
operator()(size_t i, size_t j, size_t k, size_t l, size_t m) {
#ifdef SAFEPRTENSOR
  assert(dimen >= 5);
#endif

  size_t indx=0;
  int dim(dimen); 
  switch (dimen) {
    case 5:
      indx = i*lengths[dim-4];
    case 4:
      indx = (indx + j)*lengths[dim-3];
    case 3:
      indx = (indx + k)*lengths[dim-2];
    case 2:
      indx = (indx + l)*lengths[dim-1];
    case 1:
      indx += m;
      break;
    default:
      throw TxDebugExcept("More indices than dimensions");
  }

// Desired: size_t indx = ( ( (i*lengths[dimen-4] + j)*lengths[dimen-3] + k)
//  *lengths[dimen-2] + l)*lengths[dimen-1] + m;

  return data[indx];
}

template <class Type, size_t dimen>
Type&
PrTensor<Type, dimen>::
operator()(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n) {
#ifdef SAFEPRTENSOR
  assert(dimen >= 6);
#endif

  size_t indx=0;
  int dim(dimen);  
  switch (dimen) {
    case 6:
      indx = i*lengths[dim-5];
    case 5:
      indx = (indx + j)*lengths[dim-4];
    case 4:
      indx = (indx + k)*lengths[dim-3];
    case 3:
      indx = (indx + l)*lengths[dim-2];
    case 2:
      indx = (indx + m)*lengths[dim-1];
    case 1:
      indx += n;
      break;
    default:
      throw TxDebugExcept("More indices than dimensions");
  }

// Desired: size_t indx = ( ( ( (i*lengths[dimen-5] + j)*lengths[dimen-4] + k)
//  *lengths[dimen-3] + l)*lengths[dimen-2] + m)*lengths[dimen-1] + n;

  return data[indx];
}

template <class Type, size_t dimen>
Type&
PrTensor<Type, dimen>::
operator()(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n, size_t o) {
#ifdef SAFEPRTENSOR
  assert(dimen >= 7);
#endif

  size_t indx=0;
  int dim(dimen); 
  switch (dimen) {
    case 7:
      indx = i*lengths[dim-6];
    case 6:
      indx = (indx + j)*lengths[dim-5];
    case 5:
      indx = (indx + k)*lengths[dim-4];
    case 4:
      indx = (indx + l)*lengths[dim-3];
    case 3:
      indx = (indx + m)*lengths[dim-2];
    case 2:
      indx = (indx + n)*lengths[dim-1];
    case 1:
      indx += o;
      break;
    default:
      throw TxDebugExcept("More indices than dimensions");
  }

// Desired: size_t indx = ( ( ( ( (i*lengths[dimen-6] + j)*lengths[dimen-5] + k)
//  *lengths[dimen-4] + l)*lengths[dimen-3] + m)*lengths[dimen-2] + n)
//  *lengths[dimen-1] + o;


  return data[indx];
}

//  Booleans - only equality defined
template <class Type, size_t dimen>
bool PrTensor<Type, dimen>::operator==(const PrTensor<Type, dimen>& prt) const {
//  All lengths must be equal
  size_t i;
  for (i=0; i<dimen; i++) if (getLength(i) != prt.getLength(i)) return 0;
//  All elements must be equal
  for (i=0; i<size[0]; i++) if ( data[i] != prt.data[i] ) return 0;
  return 1;
}

//  I/O
template <class Type, size_t dimen>
void PrTensor<Type, dimen>::write(std::ostream& ostr) const{
  ostr << "Size of TxTensor is " << size[0] << 
    "   (only nonzero values written):" << std::endl;
  size_t i;
  for (i=0; i<size[0]; i++) {
    Type zero = 0;                     // only write out non-zero elements
    if ( data[i] == zero ) continue;   // DLB 11/18/97

    size_t rem = i;
    size_t remsize = size[0];
    size_t j;
    for (j=0; j<dimen; j++) {
      remsize /= lengths[j];
      size_t indx = rem/remsize;
      rem -= indx*remsize;
      ostr.width(7);
      ostr << indx;
    }
    ostr.width(20);
    ostr << data[i] << std::endl;
  }
}

// I/O for testing w/ script
template <class Type, size_t dimen>
void PrTensor<Type, dimen>::writeTestOutput(std::ostream& ostr) const{
  ostr << "String: Size of TxTensor is " << size[0] << 
    "   (only nonzero values written):" << std::endl;
  size_t i;
  for (i=0; i<size[0]; i++) {
    Type zero = 0;                     // only write out non-zero elements
    if ( data[i] == zero ) continue;   // DLB 11/18/97

    size_t rem = i;
    size_t remsize = size[0];
    size_t j;
    ostr << "Result: ";
    for (j=0; j<dimen; j++) {
      remsize /= lengths[j];
      size_t indx = rem/remsize;
      rem -= indx*remsize;
      ostr.width(12);
      //ostr << "String: " << indx;
      ostr << indx;
    }
    ostr.width(12);
    ostr << data[i] << std::endl;
  }
}

// ansi instantiation

template class PrTensor<float, 1>;
template class PrTensor<float, 2>;
template class PrTensor<float, 3>;
template class PrTensor<float, 4>;

template class PrTensor<double, 1>;
template class PrTensor<double, 2>;
template class PrTensor<double, 3>;
template class PrTensor<double, 4>;

template class PrTensor<int, 1>;
template class PrTensor<int, 2>;
template class PrTensor<int, 3>;
template class PrTensor<int, 4>;

template class PrTensor<std::complex<double>, 1>;
template class PrTensor<std::complex<double>, 2>;
template class PrTensor<std::complex<double>, 3>;
template class PrTensor<std::complex<double>, 4>;

template class PrTensor<void*, 1>;
template class PrTensor<void*, 2>;
template class PrTensor<void*, 3>;
template class PrTensor<void*, 4>;

#ifndef __HP_aCC
// Not compiling on hpux as aCC runs out of memory
template class PrTensor<size_t, 1>;
template class PrTensor<size_t, 2>;
template class PrTensor<size_t, 3>;
template class PrTensor<size_t, 4>;
#endif

#ifndef __hpux
// Not present on hpux
template class PrTensor<long double, 1>;
template class PrTensor<long double, 2>;
template class PrTensor<long double, 3>;
template class PrTensor<long double, 4>;
#endif

