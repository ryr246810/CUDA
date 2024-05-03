// --------------------------------------------------------------------
//
// File:        TConstantFunc.cxx
//
// Purpose:     Implementation of a constant space-time functor
//
// --------------------------------------------------------------------

#include <TConstantFunc.hxx>


TConstantFunc::TConstantFunc()
{
  amplitude = 0.;
}


void TConstantFunc::setAttrib(const TxHierAttribSet& tas)
{
  if(tas.hasParam("amplitude")) amplitude = tas.getParam("amplitude");
}


Standard_Real TConstantFunc::operator()(Standard_Real t) const 
{
  return amplitude;
}

