// --------------------------------------------------------------------
//
// File:        STConstantFunc.cxx
//
// Purpose:     Implementation of a constant space-time functor
//
// --------------------------------------------------------------------

#include <STConstantFunc.hxx>


STConstantFunc::STConstantFunc()
{
  amplitude = 0.;
}


void STConstantFunc::setAttrib(const TxHierAttribSet& tas)
{
  if(tas.hasParam("amplitude")) amplitude = tas.getParam("amplitude");
}


Standard_Real STConstantFunc::operator()(Standard_Real* x, Standard_Real t) const 
{
  return amplitude;
}

