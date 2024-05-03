// -----------------------------------------------------------------------
// File:      TxVector2D.h
// -----------------------------------------------------------------------
#ifndef VP_VECTOR2D_H
#define VP_VECTOR2D_H

// std includes
#include <vector>
#include <cmath>
// txbase includes
#include <TxStreams.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define HOST_DEVICE2 __host__ __device__

template <class TYPE> class TxVector2D
{
public:
  HOST_DEVICE2 TxVector2D() { for(size_t i=0; i<2; ++i) data[i] = 0; }
  HOST_DEVICE2 TxVector2D(const TxVector2D& tv) { for(size_t i=0; i<2; ++i) data[i] = tv.data[i];  }
  HOST_DEVICE2 TxVector2D(const TYPE v[2]) { for(size_t i=0; i<2; ++i) data[i] = v[i]; }
  HOST_DEVICE2 TxVector2D(const TYPE X0,const TYPE X1){data[0] = X0; data[1]=X1; }

  HOST_DEVICE2 TxVector2D& operator=(const TxVector2D& tv)
  {
    for(size_t i=0; i<2; ++i) data[i] = tv.data[i];
    return *this;
  }

  HOST_DEVICE2 TYPE& operator[](size_t i) { return data[i];  }
  HOST_DEVICE2 TYPE operator[](size_t i) const { return data[i]; }

  HOST_DEVICE2 bool operator==(const TxVector2D& tv) const
  {
    for (size_t i=0;i<2;++i)
      {
	if (operator[](i)!=tv[i]) return false;
      }
    return true;
  }
  HOST_DEVICE2 bool operator!=(const TxVector2D& tv) const
  {
    return !(operator==(tv));
  }

  HOST_DEVICE2 TxVector2D operator+(const TxVector2D& tv) const
  {
    TxVector2D res;
    for(size_t i=0; i<2; ++i) res.data[i] = data[i] + tv.data[i];
    return res;
  }

  HOST_DEVICE2 TxVector2D operator-(const TxVector2D& tv) const
  {
    TxVector2D res;
    for(size_t i=0; i<2; ++i) res.data[i] = data[i] - tv.data[i];
    return res;
  }


  HOST_DEVICE2 TYPE Dot(const TxVector2D& tv) const{
    TYPE res = 0;
    for(size_t i=0; i<2; ++i) res += data[i]* tv.data[i];
    return res;
  }


  HOST_DEVICE2 TYPE operator *(const TxVector2D& tv) const{
    return Dot(tv);
  }

  HOST_DEVICE2 bool operator<(const TxVector2D& tv) const {
    for (size_t i=0;i<2;++i) {
      if (operator[](i)<tv[i]) 
	return true;
      if (operator[](i)>tv[i]) 
	return false;
    }
    return false;
  }
  
  
  HOST_DEVICE2 bool operator>(const TxVector2D& tv) const {
    for (size_t i=0;i<3;++i) {
      if (operator[](i)<tv[i]) 
	return false;
      if (operator[](i)>tv[i]) 
	return true;
    }
    return false;
  }
  
  HOST_DEVICE2 bool operator>=(const TxVector2D& tv) const {
    for (size_t i=0;i<2;++i) {
      if (operator[](i)<tv[i]) 
	return false;
      if (operator[](i)>tv[i]) 
	return true;
    }
    return true;
  }
  
  HOST_DEVICE2 TxVector2D& operator+=(const TxVector2D& tv) {
    for(size_t i=0; i<2; ++i) data[i] += tv.data[i];
    return *this;
  }

  HOST_DEVICE2 TxVector2D& operator*=(TYPE fac){
    for(size_t i=0; i<2; ++i) data[i] *= fac;
    return *this;
  }

  HOST_DEVICE2 TxVector2D& operator/=(TYPE fac){
    for(size_t i=0; i<2; ++i) data[i] /= fac;
    return *this;
  }

  HOST_DEVICE2 TxVector2D operator*(TYPE fac) const{
    TxVector2D res;
    for(size_t i=0; i<2; ++i) res.data[i] = fac*data[i];
    return res;
  }

  HOST_DEVICE2 TxVector2D operator/(TYPE fac) const{
    TxVector2D res;
    for(size_t i=0; i<2; ++i) res.data[i] = data[i]/fac;
    return res;
  }

  HOST_DEVICE2 TYPE length() const{
    TYPE value = 0;
    for(size_t i=0; i<2; ++i) value += data[i]*data[i];
    return TYPE(sqrt(value));
  }

  HOST_DEVICE2 TYPE squrelength() const{
    TYPE value = 0;
    for(size_t i=0; i<2; ++i) value += data[i]*data[i];
    return value;
  }

  void write(ostream& o) const{
    o << operator[](0) <<", " << operator[](1) <<endl;
  }

 private:
  TYPE data[2];
};
#endif
