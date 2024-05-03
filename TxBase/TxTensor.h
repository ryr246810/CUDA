//-----------------------------------------------------------------------------
//
// File:	TxTensor.h
//
// Purpose:	Interface of ref counted tensor
//
// Version:	$Id: TxTensor.h 122 2007-11-22 03:15:01Z swsides $
//
// Copyright (c) 1996-2003 by Tech-X Corporation.  All rights reserved.
//
//-----------------------------------------------------------------------------

#ifndef TX_TENSOR_H
#define TX_TENSOR_H

// system includes

// std includes
#include <complex>

// txbase includes
#include "TxRefCount.h"
#include "PrTensor.h"
#include "TxBase.h"
/** Templated array class with reference counting.
 *  The components are stored in a linear array.
 *  The class has booleans for comparison of lexigraphical ordering.
 *
 *  Review of reference counting technique:
 *    The envelope class determines whether the copy constructor is used.
 *    If the class is changing the internal data, then make a new copy.
 *
 *  TxTensor is envelope class -- presents interface to the world.
 *  PrTensor is  letter  class -- does work and keeps track of references.
 *
 *  @author  John R. Cary
 *  @version $Id: TxTensor.h 122 2007-11-22 03:15:01Z swsides $
 *  @param Type   The type of the objects in the array:
 *                                         int, double, complex, ...
 *  @param dimen  The dimension of the tensor.
 */

template <class Type, size_t dimen>
class TXBASE_API TxTensor : public TxRefCount 
{

  public:

/**
 * Default constructor: create an empty tensor
 */
    TxTensor() 
      : TxRefCount(new PrTensor<Type, dimen>()) {}

/**
 * Copy constructor: increment reference count
 */
    TxTensor(const TxTensor<Type, dimen>& tt) 
      : TxRefCount( tt ) {}

/**
 * Create a tensor with dimensions given by the vector len and fill
 * it with zeros.
 *
 * @param len the vector giving shape (dimensions, or range of
 * 	indices)
 */
    TxTensor(size_t* len) 
      : TxRefCount(new PrTensor<Type, dimen>(len)) {}

/**
 * Construct a tensor of given shape
 *
 * @param i the range of the last index
 *
 * Unspecified indices have unit range.
 */
    TxTensor(size_t i)
        : TxRefCount(new PrTensor<Type, dimen>(i)) {}

/**
 * Construct a tensor of given shape
 *
 * @param i the range of the second to last index
 * @param j the range of the last index
 *
 * Unspecified indices have unit range.  Excess arguments are ignored.
 */
    TxTensor(size_t i, size_t j)
        : TxRefCount(new PrTensor<Type, dimen>(i, j)) {}

/**
 * Construct a tensor of given shape
 *
 * @param i the range of the third to last index
 * @param j the range of the second to last index
 * @param k the range of the last index
 *
 * Unspecified indices have unit range.  Excess arguments are ignored.
 */
    TxTensor(size_t i, size_t j, size_t k)
        : TxRefCount(new PrTensor<Type, dimen>(i, j, k)) {}

/**
 * Construct a tensor of given shape
 *
 * @param i the range of the fourth to last index
 * @param j the range of the third to last index
 * @param k the range of the second to last index
 * @param l the range of the last index
 *
 * Unspecified indices have unit range.  Excess arguments are ignored.
 */
    TxTensor(size_t i, size_t j, size_t k, size_t l)
        : TxRefCount(new PrTensor<Type, dimen>(i, j, k, l)) {}

/**
 * Construct a tensor of given shape
 *
 * @param i the range of the fifth to last index
 * @param j the range of the fourth to last index
 * @param k the range of the third to last index
 * @param l the range of the second to last index
 * @param m the range of the last index
 *
 * Unspecified indices have unit range.  Excess arguments are ignored.
 */
    TxTensor(size_t i, size_t j, size_t k,
             size_t l, size_t m)
        : TxRefCount(new PrTensor<Type, dimen>(i, j, k, l, m)) {}

/**
 * Construct a tensor of given shape
 *
 * @param i the range of the sixth to last index
 * @param j the range of the fifth to last index
 * @param k the range of the fourth to last index
 * @param l the range of the third to last index
 * @param m the range of the second to last index
 * @param n the range of the last index
 *
 * Unspecified indices have unit range.  Excess arguments are ignored.
 */
    TxTensor(size_t i, size_t j, size_t k,
             size_t l, size_t m, size_t n)
        : TxRefCount(new PrTensor<Type, dimen>(i, j, k, l, m, n)) {}

/**
 * Construct a tensor of given shape
 *
 * @param i the range of the seventh to last index
 * @param j the range of the sixth to last index
 * @param k the range of the fifth to last index
 * @param l the range of the fourth to last index
 * @param m the range of the third to last index
 * @param n the range of the second to last index
 * @param o the range of the last index
 *
 * Unspecified indices have unit range.  Excess arguments are ignored.
 */
    TxTensor(size_t i, size_t j, size_t k,
             size_t l, size_t m, size_t n, size_t o)
        : TxRefCount(new PrTensor<Type, dimen>(i, j, k, l, m, n, o)) {}

/**
 * Destructor: defer to base class to decrement count
 */
    virtual ~TxTensor() {
    }

/**
 * Assignment of all variables to a single value
 *
 * @param t the value to set to
 *
 * @return a reference to the new value
 */
    const TxTensor<Type, dimen>& operator=(Type t) {
      makeUnique();
      *getTensor() = t;
      return *this;
    }

/**
 * Assignment to another tensor
 *
 * @param tt the tensor to set to
 *
 * @return a reference to the new value
 */
    const TxTensor<Type, dimen>& operator=(const TxTensor<Type, dimen>& tt) {
      return *(  (TxTensor<Type, dimen>*) (&( TxRefCount::operator=(tt) ))  );
    }


/**
 * Return pointer to the data member
 *
 * @return a Type* data pointer
 */
    const Type* getTensorDataPtr() const { return getTensor()->getTensorDataPtr();}

/**
 * Get rank of tensor
 *
 * @return rank of tensor, ie dimen in PrTensor
 */
    size_t getTensorRank() const { return getTensor()->getTensorRank();}

/**
 * Get a length (index range)
 *
 * @param i the dimension to get the length of
 *
 * @return the length = index range of a given index
 */
    size_t getLength(size_t i) const { return getTensor()->getLength(i);}

/**
 * Set a length (index range)
 *
 * @param li the direction to set the length for
 * @param lv the new length
 */
    void setLength(size_t li, size_t lv) { getTensor()->setLength(li, lv);}

/**
 * set the lengths of all of the dimensions
 *
 * @param len array continaing the new lengths
 */
    void setLengths(size_t len[dimen]) { getTensor()->setLengths(len);}

/**
 * get the number of elements for the full range of an index
 *
 * @param dir the index of the dimension
 *
 * @return the increment for the full range of the index,
 *	returns unity for dir = dimen
 */
    size_t getSize(size_t dir) const { return getTensor()->getSize(dir);}


/**
 * Get the size (number of elements contained)
 *
 * @return the size = number of elements contained.
 */
    size_t getSize() const { return getTensor()->getSize();}

/**
 * Fill an index vector corresponding to a position
 * in the array, so that one can loop over all elements in
 * the array and know the corresponding indices.
 *
 * @param i the index of the element in the array.
 * @param indxVec a pointer to the indices and will be filled by the method.
 *      User responsible for assuring this has sufficient memory
 */
    void fillIndexVec(size_t i, size_t* indxVec) const {
      getTensor()->fillIndexVec(i, indxVec);
    }

//
// getting the data elements of the array
//

/**
 * Get an element as an lval (not settable)
 *
 * @param indices the indices of the element to get
 *
 * @return the value of the element as an lval
 */
    Type operator()(size_t* indices) const {
      return getTensor()->operator()(indices);
    }

/**
 * Get an element as an lval (not settable)
 *
 * @param i the range of the last index
 *
 * Unspecified indices are interpreted as unity
 *
 * @return the value of the element as an lval
 */
    Type operator()(size_t i) const {
      // return getTensor()->operator()(i);
      return getTensor()->data[i];
    }

/**
 * Get an element as an lval (not settable)
 *
 * @param i the range of the second to last index
 * @param j the range of the last index
 *
 * Unspecified indices are interpreted as unity
 *
 * @return the value of the element as an lval
 */
    Type operator()(size_t i, size_t j) const {
      return getTensor()->operator()(i, j);
    }

/**
 * Get an element as an lval (not settable)
 *
 * @param i the range of the third to last index
 * @param j the range of the second to last index
 * @param k the range of the last index
 *
 * Unspecified indices are interpreted as unity
 *
 * @return the value of the element as an lval
 */
    Type operator()(size_t i, size_t j, size_t k) const {
      return getTensor()->operator()(i, j, k);
    }

/**
 * Get an element as an lval (not settable)
 *
 * @param i the range of the fourth to last index
 * @param j the range of the third to last index
 * @param k the range of the second to last index
 * @param l the range of the last index
 *
 * Unspecified indices are interpreted as unity
 *
 * @return the value of the element as an lval
 */
    Type operator()(size_t i, size_t j, size_t k, size_t l) const {
      return getTensor()->operator()(i, j, k, l);
    }

/**
 * Get an element as an lval (not settable)
 *
 * @param i the range of the fifth to last index
 * @param j the range of the fourth to last index
 * @param k the range of the third to last index
 * @param l the range of the second to last index
 * @param m the range of the last index
 *
 * Unspecified indices are interpreted as unity
 *
 * @return the value of the element as an lval
 */
    Type operator()(size_t i, size_t j, size_t k, size_t l, size_t m) const {
      return getTensor()->operator()(i, j, k, l, m);
    }

/**
 * Get an element as an lval (not settable)
 *
 * @param i the range of the sixth to last index
 * @param j the range of the fifth to last index
 * @param k the range of the fourth to last index
 * @param l the range of the third to last index
 * @param m the range of the second to last index
 * @param n the range of the last index
 *
 * Unspecified indices are interpreted as unity
 *
 * @return the value of the element as an lval
 */
    Type operator()(size_t i, size_t j, size_t k,
                     size_t l, size_t m, size_t n) const {
      return getTensor()->operator()(i, j, k, l, m, n);
    }

/**
 * Get an element as an lval (not settable)
 *
 * @param i the range of the seventh to last index
 * @param j the range of the sixth to last index
 * @param k the range of the fifth to last index
 * @param l the range of the fourth to last index
 * @param m the range of the third to last index
 * @param n the range of the second to last index
 * @param o the range of the last index
 *
 * Unspecified indices are interpreted as unity
 *
 * @return the value of the element as an lval
 */
    Type operator()(size_t i, size_t j, size_t k,
                     size_t l, size_t m,
                     size_t n, size_t o) const {
      return getTensor()->operator()(i, j, k, l, m, n, o);
    }

//
// setting the data elements of the array
//

/**
 * Get an element as an lval (left value)
 *
 * @param indices the range of the last index
 *                Unspecified indices are interpreted as unity
 *
 * @return the value of the element as an rval
 */
    Type& operator()(size_t* indices) {
      makeUnique();
      return getTensor()->operator()(indices);
    }

/**
 * Get an element as an lval (left value)
 *
 * @param i the range of the last index
 *
 * Unspecified indices are interpreted as unity
 *
 * @return the value of the element as an rval
 */
    Type& operator()(size_t i) {
      makeUnique();
      return getTensor()->operator()(i);
    }

/**
 * Get an element as an lval (left value)
 *
 * @param i the range of the second to last index
 * @param j the range of the last index
 *
 * Unspecified indices are interpreted as unity
 *
 * @return the value of the element as an rval
 */
    Type& operator()(size_t i,  size_t j) {
      makeUnique();
      return getTensor()->operator()(i, j);
    }

/**
 * Get an element as an lval (left value)
 *
 * @param i the range of the third to last index
 * @param j the range of the second to last index
 * @param k the range of the last index
 *
 * Unspecified indices are interpreted as unity
 *
 * @return the value of the element as an rval
 */
    Type& operator()(size_t i, size_t j, size_t k) {
      makeUnique();
      return getTensor()->operator()(i, j, k);
    }

/**
 * Get an element as an lval (left value)
 *
 * @param i the range of the fourth to last index
 * @param j the range of the third to last index
 * @param k the range of the second to last index
 * @param l the range of the last index
 *
 * Unspecified indices are interpreted as unity
 *
 * @return the value of the element as an lval
 */
    Type& operator()(size_t i, size_t j, size_t k, size_t l) {
      makeUnique();
      return getTensor()->operator()(i, j, k, l);
    }

/**
 * Get an element as an lval (left value)
 *
 * @param i the range of the fifth to last index
 * @param j the range of the fourth to last index
 * @param k the range of the third to last index
 * @param l the range of the second to last index
 * @param m the range of the last index
 *
 * Unspecified indices are interpreted as unity
 *
 * @return the value of the element as an lval
 */
    Type& operator()(size_t i, size_t j, size_t k,
                      size_t l, size_t m) {
      makeUnique();
      return getTensor()->operator()(i, j, k, l, m);
    }

/**
 * Get an element as an lval (left value)
 *
 * @param i the range of the sixth to last index
 * @param j the range of the fifth to last index
 * @param k the range of the fourth to last index
 * @param l the range of the third to last index
 * @param m the range of the second to last index
 * @param n the range of the last index
 *
 * Unspecified indices are interpreted as unity
 *
 * @return the value of the element as an lval
 */
    Type& operator()(size_t i, size_t j, size_t k,
                      size_t l, size_t m, size_t n) {
      makeUnique();
      return getTensor()->operator()(i, j, k, l, m, n);
    }

/**
 * Get an element as an lval (left value)
 *
 * @param i the range of the seventh to last index
 * @param j the range of the sixth to last index
 * @param k the range of the fifth to last index
 * @param l the range of the fourth to last index
 * @param m the range of the third to last index
 * @param n the range of the second to last index
 * @param o the range of the last index
 *
 * Unspecified indices are interpreted as unity
 *
 * @return the value of the element as an lval
 */
    Type& operator()(size_t i, size_t j, size_t k,
                      size_t l, size_t m, size_t n,
                      size_t o) {
      makeUnique();
      return getTensor()->operator()(i, j, k, l, m, n, o);
    }

/**
 * Boolean for equality - all members equal
 *
 * @param txt the tensor to compare to
 *
 * @return true if all terms equal
 */
    bool operator==(const TxTensor<Type, dimen>& txt) const {
      return ( getTensor()->operator==(*(txt.getTensor())) );
    }

/**
 * Boolean for inequality - one member unequal
 *
 * @param txt the tensor to compare to
 *
 * @return false if all terms equal
 */
    bool operator!=(const TxTensor<Type, dimen>& txt) const {
      return !( getTensor()->operator==(*(txt.getTensor())) );
    }

/**
 * output
 *
 * @param ostr the stream to write to
 */
    void write(std::ostream& ostr) const {
      getTensor()->write(ostr);
    }

/**
 * output for test script
 *
 * @param ostr the stream to write to
 */
    void writeTestOutput(std::ostream& ostr) const {
      getTensor()->writeTestOutput(ostr);
    }

/*
 * friend output operator: This is not needed, as
 * this operator only calls write, which is public.
#ifdef HAVE_NONTYPE_TEMPLATE_OPERATORS
  #ifdef TEMPLATE_FRIENDS_NEED_BRACKETS
    friend std::ostream& operator<< <> (std::ostream& ostr,
	const TxTensor<Type, dimen>& tt);
  #else
    friend std::ostream& operator<< (std::ostream& ostr,
	const TxTensor<Type, dimen>& tt);
  #endif
#endif
 */

  protected:

/**
 * Construct an envelope object (TxTensor) from a
 * letter object (PrTensor).
 */
    TxTensor(PrTensor<Type, dimen>* p) : TxRefCount(p) {
    }

/**
 * get a pointer to the private array data
 *
 * @return a pointer to the private data array
 */
    PrTensor<Type, dimen>* getTensor() const {
      return (PrTensor<Type, dimen>*) prc;
    }

  private:

/**
 * Ensure that this is has a unique private data
 */
    virtual void makeUnique() {
      if ( getRefCount() > 1 ) {
         // std::cerr << "prc = " << prc << ", count = " <<
		// getRefCount() << std::endl;
         getTensor()->decrementCount();
         prc = new PrTensor<Type, dimen>(*getTensor());
         // std::cerr << "Now prc = " << prc << ", and count = " <<
		// getRefCount() << std::endl;
      }
    }

};


/**
 * Output operator for tensors
 *
 * @param ostr the stream to write to
 * @param tt the tensor to output
 *
 * @return a reference to the output stream
 */
template <class Type, size_t dimen> std::ostream&
operator<<(std::ostream& ostr, const TxTensor<Type, dimen>& tt) {
  tt.write(ostr);
  return ostr;
}

#if defined(__DECCXX) && !defined(TX_TENSOR_CPP)

// #pragma do_not_instantiate TxTensor<float, 1>
// #pragma do_not_instantiate TxTensor<float, 2>
// #pragma do_not_instantiate TxTensor<float, 3>
// #pragma do_not_instantiate TxTensor<float, 4>

// #pragma do_not_instantiate TxTensor<double, 1>
// #pragma do_not_instantiate TxTensor<double, 2>
// #pragma do_not_instantiate TxTensor<double, 3>
// #pragma do_not_instantiate TxTensor<double, 4>

// #ifndef __hpux
// #pragma do_not_instantiate TxTensor<long double, 1>
// #pragma do_not_instantiate TxTensor<long double, 2>
// #pragma do_not_instantiate TxTensor<long double, 3>
// #pragma do_not_instantiate TxTensor<long double, 4>
// #endif

// #pragma do_not_instantiate TxTensor<int, 1>
// #pragma do_not_instantiate TxTensor<int, 2>
// #pragma do_not_instantiate TxTensor<int, 3>
// #pragma do_not_instantiate TxTensor<int, 4>

// #pragma do_not_instantiate TxTensor<size_t, 1>
// #pragma do_not_instantiate TxTensor<size_t, 2>
// #pragma do_not_instantiate TxTensor<size_t, 3>
// #pragma do_not_instantiate TxTensor<size_t, 4>

// #pragma do_not_instantiate TxTensor<std::complex<double>, 1>
// #pragma do_not_instantiate TxTensor<std::complex<double>, 2>
// #pragma do_not_instantiate TxTensor<std::complex<double>, 3>
// #pragma do_not_instantiate TxTensor<std::complex<double>, 4>

// #pragma do_not_instantiate TxTensor<void*, 1>
// #pragma do_not_instantiate TxTensor<void*, 2>
// #pragma do_not_instantiate TxTensor<void*, 3>
// #pragma do_not_instantiate TxTensor<void*, 4>

#endif


#endif   // TX_TENSOR_H
