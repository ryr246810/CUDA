//--------------------------------------------------------------------
//
// File:	TGaussianEnvelopePulse.cxx
//
// Purpose:	Implementation of a functor for a rising with the half-cosine in time.
//
//--------------------------------------------------------------------

#include <TGaussianEnvelopePulse.hxx>
#include <PhysConsts.hxx>
#include <TxStreams.h>



TGaussianEnvelopePulse::
TGaussianEnvelopePulse()
{
  m_Frequency = 0.;
  m_CenterTime = 0.;
  m_Tau = 0.;
  m_Amp = 0.;
}



void TGaussianEnvelopePulse::
setAttrib(const TxHierAttribSet& tas)
{
  // Set parameters of this object
  if(tas.hasParam("centerTime")) m_CenterTime = tas.getParam("centerTime");
  if(tas.hasParam("tau"))        m_Tau = tas.getParam("tau");
  if(tas.hasParam("amplitude"))  m_Amp = tas.getParam("amplitude");
  if(tas.hasParam("frequency"))  m_Frequency = tas.getParam("frequency");
}



Standard_Real 
TGaussianEnvelopePulse::
operator()(Standard_Real t) const 
{
  // Get pulse amplitude
  Standard_Real pulse = m_Amp*cos(mksConsts.twopi*m_Frequency*t)*exp( - mksConsts.fourpi*(t-m_CenterTime)*(t-m_CenterTime)/m_Tau/m_Tau);

  return pulse;
}

