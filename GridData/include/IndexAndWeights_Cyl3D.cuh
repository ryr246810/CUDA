#ifndef _IndexAndWeights_Cyl3D_HeaderFile
#define _IndexAndWeights_Cyl3D_HeaderFile

#include "Standard_TypeDefine.hxx"
#include "../../Cuda_Files/CUDAHeader.cuh"

struct IndexAndWeights_Cyl3D
{
public:
  HOST_DEVICE IndexAndWeights_Cyl3D()
  {
    for(Standard_Integer i = 0; i < 3; i++){
      indx[i] = 0;
      wu[i] = 0.0;
      wl[i] = 0.0;
    }
  }
  HOST_DEVICE IndexAndWeights_Cyl3D(Standard_Size _indx[3],  Standard_Real _wu[3], Standard_Real _wl[3])
  {
    for(Standard_Integer i = 0; i < 3; i++){
      indx[i] = _indx[i];
      wu[i] = _wu[i];
      wl[i] = _wl[i];
    }
  }

public:
  HOST_DEVICE inline bool operator!=(const IndexAndWeights_Cyl3D& t) { return !operator==(t); }

  HOST_DEVICE inline IndexAndWeights_Cyl3D& operator=(const IndexAndWeights_Cyl3D& tv)
  {
    for(size_t i=0; i<3; ++i){
      indx[i] = tv.indx[i];
      wu[i] = tv.wu[i];
      wl[i] = tv.wl[i];
    }
    return *this;
  }
  
  
  HOST_DEVICE inline bool operator==(const IndexAndWeights_Cyl3D& t)
  {
    bool result=true;
    for (Standard_Size i=0;i<3;++i){
      if (wu[i]!=t.wu[i]){
        result=false;
        break;
      }
      if (wl[i]!=t.wl[i]){
        result=false;
        break;
      }
      if (indx[i]!=t.indx[i]){
        result=false;
        break;
      }
    }
    return result;
  }

  
public:
  Standard_Size indx[3];
  Standard_Real wu[3];
  Standard_Real wl[3];


};

/*
class IndexAndWeights_Cyl3D
{
public:
  Standard_Size indx[3];
  Standard_Real wu[3];
  Standard_Real wl[3];
  
  bool operator!=(const IndexAndWeights_Cyl3D& t) { return !operator==(t); }
  
  bool operator==(const IndexAndWeights_Cyl3D& t)
  {
    bool result=true;
    for (Standard_Size i=0;i<3;++i){
      if (wu[i]!=t.wu[i]){
	result=false;
	break;
      }
      if (wl[i]!=t.wl[i]){
	result=false;
	break;
      }
      if (indx[i]!=t.indx[i]){
	result=false;
	break;
      }
    }
    return result;
  }
};
*/


#endif
