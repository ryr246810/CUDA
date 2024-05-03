//------------------------------------------------------------------------
// File:	TxTensorCopy.h
// Purpose:	Class for copying TxTensors 
//------------------------------------------------------------------------

#ifndef _TxTensorCopy_HeaderFile
#define _TxTensorCopy_HeaderFile

template <class Type, size_t dimen> class TxTensor;

//allows for three spatial dimentions and one component dimention
size_t lens[4];

template <class Type, size_t dimen, size_t dir>
class TxTensorCopy 
{
 public:
  static inline void copy(TxTensor<Type, dimen>& from, 
			  TxTensor<Type, dimen>& to, 
			  const size_t* startLens) {
    for (size_t i=0;i<from.getLength(dimen-dir);++i) {
      lens[dimen-dir]=i;
      TxTensorCopy<Type, dimen, dir-1>::copy(from, to, startLens);
    }
  }
};

template <class Type, size_t dimen>
class TxTensorCopy<Type, dimen, 0> 
{
 public:
  static inline void vs(const size_t* x, const size_t* y, size_t* r) {
    for (size_t i=0;i<dimen;++i) r[i]=x[i]+y[i];
  }
  static inline void copy(TxTensor<Type, dimen>& from, 
			  TxTensor<Type, dimen>& to,
			  const size_t* startLens) {
    size_t r[dimen];
    vs(lens,startLens,r);
    to(r)=from(lens);
  }
};

#endif
