#include <Standard_TypeDefine.hxx>

#include <cfloat>


Standard_Real RealEpsilon()
{
  return DBL_EPSILON;
}

Standard_Real RealFirst()
{ 
  return -DBL_MAX;
}
  
Standard_Real RealLast()
{ 
  return  DBL_MAX; 
}
