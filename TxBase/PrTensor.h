//-----------------------------------------------------------------------------
//
// File:	PrTensor.h
//
// Purpose:	Letter class for tensor
//
// Version:	$Id: PrTensor.h 122 2007-11-22 03:15:01Z swsides $
//
// Copyright 1996, 1997, 1998 by Tech-X Corporation
//
//-----------------------------------------------------------------------------

#ifndef PR_TENSOR_H
#define PR_TENSOR_H

// Uncomment to prevent checking number of indices
// #define SAFEPRTENSOR

// system
#include <assert.h>

// std
#include <complex>


#include <PrRefCount.h>
#include <TxDebugExcept.h>
#include "TxBase.h"

template <class Type, size_t dimen> class TxTensor;

/**
 *    --  Templated array class with reference counting.
 *
 *  The components are stored in a linear array.
 *  The class has booleans for comparison of lexigraphical ordering.
 *
 *
 *  Copyright 1996, 1997, 1998 by Tech-X Corporation
 *
 *  @author  John R. Cary
 *
 *  @version $Id: PrTensor.h 122 2007-11-22 03:15:01Z swsides $
 *
 *  @param Type   The type of the objects in the array (int, double, complex, etc).
 *  @param dimen  The dimension of the tensor.
 */
template <class Type, size_t dimen>
class TXBASE_API PrTensor : public PrRefCount 
{
  /**
   * Friend envelope class
   */
  friend class TxTensor<Type, dimen>;
  
 protected:
  /**
   * Construct an empty tensor
   */
  PrTensor();
  
  /**
   * Construct a tensor from a length vector
   *
   * @param len the vector holding the lengths of the dimensions
   */
  PrTensor(size_t* len);
  
  /**
   * Construct a tensor from another tensor
   *g
   * @param prt the tensor to copy from
   */
  PrTensor(const PrTensor<Type, dimen>& prt);
  
  /**
   * Construct from a single length
   *
   * @param i the length of the first dimension, others assumed to be 1.
   */
  PrTensor(size_t i);
  
  /**
   * Construct from two lengths
   *
   * @param i the length of the first dimension.
   * @param j the length of the second dimension, others assumed to be 1,g
   *	excess lengths ignored.
   */
  PrTensor(size_t i, size_t j);
  
  /**
   * Construct from three lengths
   *
   * @param i the length of the first dimension.
   * @param j the length of the second dimension.
   * @param k the length of the third dimension, others assumed to be 1,
   *      excess lengths ignored.
   */
  PrTensor(size_t i, size_t j, size_t k);
  
  /**
   * Construct from four lengths
   *
   * @param i the length of the first dimension.
   * @param j the length of the second dimension.
   * @param k the length of the third dimension.
   * @param l the length of the fourth dimension, others assumed to be 1,
   *      excess lengths ignored.
   */
  PrTensor(size_t i, size_t j, size_t k, size_t l);
  
  /**
   * Construct from five lengths
   *
   * @param i the length of the first dimension.
   * @param j the length of the second dimension.
   * @param k the length of the third dimension.
   * @param l the length of the fourth dimension.
   * @param m the length of the fifth dimension, others assumed to be 1,
   *      excess lengths ignored.
   */
  PrTensor(size_t i, size_t j, size_t k, size_t l, size_t m);
  
  /**
   * Construct from six lengths
   *
   * @param i the length of the first dimension.
   * @param j the length of the second dimension.
   * @param k the length of the third dimension.
   * @param l the length of the fourth dimension.
   * @param m the length of the fifth dimension.
   * @param n the length of the sixth dimension, others assumed to be 1,
   *      excess lengths ignored.
   */
  PrTensor(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n);
  
  /**
   * Construct from seven lengths
   *
   * @param i the length of the first dimension.
   * @param j the length of the second dimension.
   * @param k the length of the third dimension.
   * @param l the length of the fourth dimension.
   * @param m the length of the fifth dimension.
   * @param n the length of the sixth dimension.
   * @param o the length of the seventh dimension, others assumed to be 1,
   *      excess lengths ignored.
   */
  PrTensor(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n, size_t o);
  
  /**
   * Destructor
   */
  virtual ~PrTensor();
  
  /// Assignment
  const PrTensor<Type, dimen>& operator=(Type);
  
  /// Assignment
  const PrTensor<Type, dimen>& operator=
    (const PrTensor<Type, dimen>& prt);
  
  /**
   * Return pointer to the data member
   *
   * @return a Type* data pointer
   */
  const Type* getTensorDataPtr() const{
    return data;
  }
  
  /**
   * Get rank of tensor
   *
   * @return rank of tensor, ie dimen in PrTensor
   */
  size_t getTensorRank() const { return tensorRank;}
  
  
  /// get the length of the i'th dimension
  size_t getLength(size_t i) const {return lengths[i];}
  
  /**
   * set the length of the i'th dimension
   *
   * @param dir the direction to set the length for
   * @param len the new length
   */
  void setLength(size_t dir, size_t len);
  
  /**
   * set the lengths of all of the dimensions
   *
   * @param len array continaing the new lengths
   */
  void setLengths(size_t len[dimen]);
  
  /**
   * get the number of elements for the full range of an index
   *
   * @param dir the index of the dimension
   *
   * @return the increment for the full range of the index
   *	returns unity for dir = dimen
   */
  size_t getSize(size_t dir) const { return size[dir];}
  
  /** get the total number of elements
   *
   * @return the total number of elements in the tensor
   */
  size_t getSize() const { return size[0];}
  
  /**
   * Fill an index vector corresponding to a position
   * in the array, so that one can loop over all elements in
   * the array and know the corresponding indices.
   *
   * @param i the index of the element in the array.
   * @param indxVec a pointer to the indices and will be filled by the method.
   *	User responsible for assuring this has sufficient memory
   */
  void fillIndexVec(size_t i, size_t* indxVec) const {
    size_t rem = i;
    for (int j=0; j<(int)dimen-1; ++j) {
      indxVec[j] = rem/size[j+1];
      rem = rem - indxVec[j]*size[j+1];
    }
    indxVec[dimen-1] = rem;
  }
  
  /**
   * Get the element corresponding to an index vector
   *
   * @param indices the index vector.
   *
   * @return the element value.
   */
  Type operator()(size_t* indices) const;
  
  /**
   * Get the i_th element of the array.
   *
   * @param i the last index, as if all indices below the last are unity.
   *
   * @return the element value.
   */
  Type operator()(size_t i) const {
    return data[i];
  }
  
  /**
   * Get the (i, j)_th element of the array.
   *
   * @param i the second to last index, as if all indices below the second
   *	to last are unity.
   * @param j the last index.
   *
   * @return the element value.
   */
  Type operator()(size_t i, size_t j) const {
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
  
  /**
   * Get the (i, j, k)_th element of the array.
   *
   * @param i the third to last index, as if all indices below the third
   *	to last are unity.
   * @param j the second to last index.
   * @param k the last index.
   *
   * @return the element value.
   */
  Type operator()(size_t i, size_t j, size_t k) const;
  
  /**
   * Get the (i, j, k, l)_th element of the array.
   *
   * @param i the fourth to last index, as if all indices below the fourth
   *	to last are unity.
   * @param j the third to last index.
   * @param k the second to last index.
   * @param l the last index.
   *
   * @return the element value.
   */
  Type operator()(size_t i, size_t j, size_t k, size_t l) const;
  
  /**
   * Get the (i, j, k, l, m)_th element of the array.
   *
   * @param i the fifth to last index, as if all indices below the fifth
   *	to last are unity.
   * @param j the fourth to last index.
   * @param k the third to last index.
   * @param l the second to last index.
   * @param m the last index.
   *
   * @return the element value.
   */
  Type operator()(size_t i, size_t j, size_t k, size_t l, size_t m) const;
  
  /**
   * Get the (i, j, k, l, m, n)_th element of the array.
   *
   * @param i the sixth to last index, as if all indices below the sixth
   *	to last are unity.
   * @param j the fifth to last index.
   * @param k the fourth to last index.
   * @param l the third to last index.
   * @param m the second to last index.
   * @param n the last index.
   *
   * @return the element value.
   */
  Type operator()(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n) const;
  
  /**
   * Get the (i, j, k, l, m, n)_th element of the array.
   *
   * @param i the seventh to last index, as if all indices below the seventhg
   *	to last are unity.
   * @param j the sixth to last index
   * @param k the fifth to last index.
   * @param l the fourth to last index.
   * @param m the third to last index.
   * @param n the second to last index.
   * @param o the last index.
   *
   * @return the element value.
   */
  Type operator()(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n, size_t o) const;
  
  /// getting the elements of the array
  Type& operator()(size_t* indices);
  
  /**
   * Get a reference to a value in the tensor
   *
   * @param i the index
   *
   * @return the reference to that member of the tensor
   */
  Type& operator()(size_t i) {
    return data[i];
  }
  
  /**
   * Get a reference to a value in the tensor
   *
   * @param i the second to last index, as if all indices below the second
   *	to last are unity.
   * @param j the last index.
   *
   * @return the reference to that member of the tensor
   */
  Type& operator()(size_t i, size_t j) {
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
  
  /**
   * Get a reference to the (i, j, k)_th value in the tensor.
   *
   * @param i the third to last index, as if all indices below the third
   *	to last are unity.
   * @param j the second to last index.
   * @param k the last index.
   *
   * @return the reference to that member of the tensor
   */
  Type& operator()(size_t i, size_t j, size_t k);
  
  /**
   * Get a reference to the (i, j, k, l)_th value in the tensor.
   *
   * @param i the fourth to last index, as if all indices below the fourth
   *	to last are unity.
   * @param j the third to last index, as if all indices below the third
   * @param k the second to last index.
   * @param l the last index.
   *
   * @return the reference to that member of the tensor
   */
  Type& operator()(size_t i, size_t j, size_t k, size_t l);
  
  /**
   * Get a reference to the (i, j, k, l, m)_th value in the tensor.
   *
   * @param i the fifth to last index, as if all indices below the fifthg
   *	to last are unity.
   * @param j the fourth to last index, as if all indices below the fourth
   * @param k the third to last index, as if all indices below the third
   * @param l the second to last index.
   * @param m the last index.
   *
   * @return the reference to that member of the tensor
   */
  Type& operator()(size_t i, size_t j, size_t k, size_t l, size_t m);
  
  /**
   * Get a reference to the (i, j, k, l, m)_th value in the tensor.
   *
   * @param i the sixth to last index, as if all indices below the sixth
   *	to last are unity.
   * @param j the fifth to last index, as if all indices below the fifthg
   * @param k the fourth to last index, as if all indices below the fourth
   * @param l the third to last index, as if all indices below the third
   * @param m the second to last index.
   * @param n the last index.
   *
   * @return the reference to that member of the tensor
   */
  Type& operator()(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n);
  
  /**
   * Get a reference to the (i, j, k, l, m, n)_th value in the tensor.
   *
   * @param i the seventh to last index, as if all indices below the seventh
   *	to last are unity.
   * @param j the sixth to last index, as if all indices below the sixth
   * @param k the fifth to last index, as if all indices below the fifthg
   * @param l the fourth to last index, as if all indices below the fourth
   * @param m the third to last index, as if all indices below the third
   * @param n the second to last index.
   * @param o the last index.
   *
   * @return the reference to that member of the tensor
   */
  Type& operator()(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n, size_t o);
  
  /// Booleans (only equality defined as ordering depends on context).
  bool operator==(const PrTensor<Type, dimen>&) const ;
  
  /// I/O
  virtual void write(std::ostream& ostr) const;
  
  /// I/O for testing w/ script
  virtual void writeTestOutput(std::ostream& ostr) const;
 protected:
  
  /// setup
  void setupLengths(const size_t* len);
  
  /// setup
  void getDataMemory();
  
  /// setup
  size_t tensorRank;
  
  /// this is the actual data
  Type* data;
  
  /// array of unsigned integers (size_t's) specifying length of each dimension
  size_t lengths[dimen];
  
  /// total size of the array
  size_t size[dimen + 1];
  
};

#if defined(__DECCXX) && !defined(PR_TENSOR_CPP)

// #pragma do_not_instantiate PrTensor<float, 1>
// #pragma do_not_instantiate PrTensor<float, 2>
// #pragma do_not_instantiate PrTensor<float, 3>
// #pragma do_not_instantiate PrTensor<float, 4>

// #pragma do_not_instantiate PrTensor<double, 1>
// #pragma do_not_instantiate PrTensor<double, 2>
// #pragma do_not_instantiate PrTensor<double, 3>
// #pragma do_not_instantiate PrTensor<double, 4>

// #ifndef __hpux
// #pragma do_not_instantiate PrTensor<long double, 1>
// #pragma do_not_instantiate PrTensor<long double, 2>
// #pragma do_not_instantiate PrTensor<long double, 3>
// #pragma do_not_instantiate PrTensor<long double, 4>
// #endif

// #pragma do_not_instantiate PrTensor<int, 1>
// #pragma do_not_instantiate PrTensor<int, 2>
// #pragma do_not_instantiate PrTensor<int, 3>
// #pragma do_not_instantiate PrTensor<int, 4>

// #pragma do_not_instantiate PrTensor<size_t, 1>
// #pragma do_not_instantiate PrTensor<size_t, 2>
// #pragma do_not_instantiate PrTensor<size_t, 3>
// #pragma do_not_instantiate PrTensor<size_t, 4>

// #pragma do_not_instantiate PrTensor<std::complex<double>, 1>
// #pragma do_not_instantiate PrTensor<std::complex<double>, 2>
// #pragma do_not_instantiate PrTensor<std::complex<double>, 3>
// #pragma do_not_instantiate PrTensor<std::complex<double>, 4>

// #pragma do_not_instantiate PrTensor<void*, 1>
// #pragma do_not_instantiate PrTensor<void*, 2>
// #pragma do_not_instantiate PrTensor<void*, 3>
// #pragma do_not_instantiate PrTensor<void*, 4>

#endif

#endif   // PR_TENSOR_H
