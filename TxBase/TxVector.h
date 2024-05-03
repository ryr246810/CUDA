// -----------------------------------------------------------------------
// File:      TxVector.h
// -----------------------------------------------------------------------
#ifndef VP_VECTOR_H
#define VP_VECTOR_H

// std includes
#include <vector>
#include <cmath>
// txbase includes
#include <TxStreams.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define HOST_DEVICE0 __host__ __device__
// #define HOST_DEVICE 

template <class TYPE> class TxVector
{
public:
  HOST_DEVICE0 TxVector() { for(size_t i=0; i<3; ++i) data[i] = 0; }
  HOST_DEVICE0 TxVector(const TxVector& tv) { for(size_t i=0; i<3; ++i) data[i] = tv.data[i];  }
  HOST_DEVICE0 TxVector(const TYPE v[3]) { for(size_t i=0; i<3; ++i) data[i] = v[i]; }
  HOST_DEVICE0 TxVector(const TYPE X,const TYPE Y,const TYPE Z){data[0] = X; data[1]=Y; data[2]=Z; }

  HOST_DEVICE0 TxVector& operator=(const TxVector& tv)
    {
      for(size_t i=0; i<3; ++i) data[i] = tv.data[i];
      return *this;
    }
  /*
  TxVector& operator=(TYPE x)
    {
      for(size_t i=0; i<3; ++i) data[i] = x;
      return *this;
    }
  //*/

  HOST_DEVICE0 TYPE& operator[](size_t i) { return data[i];  }
  HOST_DEVICE0 TYPE operator[](size_t i) const { return data[i]; }

  HOST_DEVICE0 bool operator==(const TxVector& tv) const
  {
    for (size_t i=0;i<3;++i)
      {
	if (operator[](i)!=tv[i]) return false;
      }
    return true;
  }
  HOST_DEVICE0 bool operator!=(const TxVector& tv) const
  {
    return !(operator==(tv));
  }

  HOST_DEVICE0 TxVector operator+(const TxVector& tv) const
  {
    TxVector res;
    for(size_t i=0; i<3; ++i) res.data[i] = data[i] + tv.data[i];
    return res;
  }

  HOST_DEVICE0 TxVector operator-(const TxVector& tv) const
  {
    TxVector res;
    for(size_t i=0; i<3; ++i) res.data[i] = data[i] - tv.data[i];
    return res;
  }


  HOST_DEVICE0 TYPE Dot(const TxVector& tv) const{
    TYPE res = 0;
    for(size_t i=0; i<3; ++i) res += data[i]* tv.data[i];
    return res;
  }


  HOST_DEVICE0 TYPE operator *(const TxVector& tv) const{
    return Dot(tv);
  }


  HOST_DEVICE0 void Cross(const TxVector& Right){
    TYPE Xresult = data[1] * Right.data[2] - data[2] * Right.data[1];
    TYPE Yresult = data[2] * Right.data[0] - data[0] * Right.data[2];
    TYPE Zresult = data[0] * Right.data[1] - data[1] * Right.data[0];
    data[0] = Xresult;
    data[1] = Yresult;
    data[2] = Zresult;
  }
  
  HOST_DEVICE0 TxVector Cross2(const TxVector& Right){
    TYPE Xresult = data[1] * Right.data[2] - data[2] * Right.data[1];
    TYPE Yresult = data[2] * Right.data[0] - data[0] * Right.data[2];
    TYPE Zresult = data[0] * Right.data[1] - data[1] * Right.data[0];   
	return TxVector(Xresult,Yresult,Zresult);
  }

  HOST_DEVICE0 TxVector Crossed(const TxVector& Right) const{

    TYPE Xresult = data[1] * Right.data[2] - data[2] * Right.data[1];
    TYPE Yresult = data[2] * Right.data[0] - data[0] * Right.data[2];
    TYPE Zresult = data[0] * Right.data[1] - data[1] * Right.data[0];

    return TxVector (Xresult,Yresult,Zresult);
  }



  HOST_DEVICE0 bool operator<(const TxVector& tv) const {
    for (size_t i=0;i<3;++i) {
      if (operator[](i)<tv[i]) 
	return true;
      if (operator[](i)>tv[i]) 
	return false;
    }
    return false;
  }
  
  
  HOST_DEVICE0 bool operator>(const TxVector& tv) const {
    for (size_t i=0;i<3;++i) {
      if (operator[](i)<tv[i]) 
	return false;
      if (operator[](i)>tv[i]) 
	return true;
    }
    return false;
  }
  
  HOST_DEVICE0 bool operator>=(const TxVector& tv) const {
    for (size_t i=0;i<3;++i) {
      if (operator[](i)<tv[i]) 
	return false;
      if (operator[](i)>tv[i]) 
	return true;
    }
    return true;
  }
  


  HOST_DEVICE0 void operator ^=(const TxVector& Right) {
    Cross(Right);
  }

  HOST_DEVICE0 TxVector operator ^(const TxVector& Right) const {
    return Crossed(Right);
  }
  
  HOST_DEVICE0 TxVector& operator+=(const TxVector& tv) {
    for(size_t i=0; i<3; ++i) data[i] += tv.data[i];
    return *this;
  }

  HOST_DEVICE0 TxVector& operator*=(TYPE fac){
    for(size_t i=0; i<3; ++i) data[i] *= fac;
    return *this;
  }
  HOST_DEVICE0 TxVector& operator/=(TYPE fac){
    for(size_t i=0; i<3; ++i) data[i] /= fac;
    return *this;
  }

  HOST_DEVICE0 TxVector operator*(TYPE fac) const{
    TxVector res;
    for(size_t i=0; i<3; ++i) res.data[i] = fac*data[i];
    return res;
  }

  HOST_DEVICE0 TxVector operator/(TYPE fac) const{
    TxVector res;
    for(size_t i=0; i<3; ++i) res.data[i] = data[i]/fac;
    return res;
  }

  HOST_DEVICE0 TYPE length() const{
    TYPE value = 0;
    for(size_t i=0; i<3; ++i) value += data[i]*data[i];
    return TYPE(sqrt(value));
  }

  HOST_DEVICE0 TYPE squrelength() const{
    TYPE value = 0;
    for(size_t i=0; i<3; ++i) value += data[i]*data[i];
    return value;
  }

  void write(ostream& o) const{
    o << operator[](0) ;
    for (size_t i=1;i<3;++i) o<<", " << operator[](i);
    o <<endl;
  }

 private:
  TYPE data[3];
};
#endif
