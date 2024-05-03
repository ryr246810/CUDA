#ifndef _IndexAndWeights_HeaderFile
#define _IndexAndWeights_HeaderFile

#include <Standard_TypeDefine.hxx>

class IndexAndWeights
{
public:
  Standard_Size indx[2];
  Standard_Real wu[2];
  Standard_Real wl[2];
  
  bool operator!=(const IndexAndWeights& t) { return !operator==(t); }
  
  bool operator==(const IndexAndWeights& t)
  {
    bool result=true;
    for (Standard_Size i=0;i<2;++i){
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


// class IndexAndWeights
// {
// public:
  // int indx[2];
  // double wu[2];
  // double wl[2];
  
  // bool operator!=(const IndexAndWeights& t) { return !operator==(t); }
  
  // bool operator==(const IndexAndWeights& t)
  // {
    // bool result=true;
    // for (int i=0;i<2;++i){
      // if (wu[i]!=t.wu[i]){
	// result=false;
	// break;
      // }
      // if (wl[i]!=t.wl[i]){
	// result=false;
	// break;
      // }
      // if (indx[i]!=t.indx[i]){
	// result=false;
	// break;
      // }
    // }
    // return result;
  // }
// };


#endif
