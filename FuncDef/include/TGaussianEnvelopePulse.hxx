// --------------------------------------------------------------------
//
// File:	TGaussianEnvelopePulse.hxx
//
// Purpose:     Interface of a functor for a rising, plane-wave
//              pulse - one that is stationary spatially but rises
//              with the half-cosine in time.
// --------------------------------------------------------------------


#ifndef _TGaussianEnvelopePulse_HeaderFile
#define _TGaussianEnvelopePulse_HeaderFile


#include <TFunc.hxx>


class TGaussianEnvelopePulse 
  : public TFunc
{
public:  
  TGaussianEnvelopePulse();
  
  virtual ~TGaussianEnvelopePulse(){};
  
  virtual void setAttrib(const TxHierAttribSet& tas);
  
  virtual Standard_Real operator()(Standard_Real t) const ;


protected:
  Standard_Real m_Frequency;
  Standard_Real m_Amp;


  /** the time to end raising the pulse amplitude */
  Standard_Real m_CenterTime;
  
  /** the frequency used in the rising time */
  Standard_Real m_Tau;

};

#endif
