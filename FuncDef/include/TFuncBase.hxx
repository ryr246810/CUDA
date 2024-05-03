// --------------------------------------------------------------------
//
// File:	TFuncBase.hxx
//
// Purpose:     Interface of a functor for a rising, plane-wave
//              pulse - one that is stationary spatially but rises
//              with the half-cosine in time.
// --------------------------------------------------------------------


#ifndef _TFuncBase_HeaderFile
#define _TFuncBase_HeaderFile


#include <FuncDefBase.hxx>
#include <vector>

class TFuncBase : public FuncDefBase
{
public:  
  TFuncBase();
  
  virtual ~TFuncBase(){}

  virtual void setAttrib_Param(const TxHierAttribSet& tas);
  virtual void setAttrib_DFL(const TxHierAttribSet& tas);

  virtual Standard_Real operator()(Standard_Real t) const ;

protected:
  std::vector<Standard_Real> m_TPntVec;
  std::vector<std::string>  m_tFuncsNames;
};

#endif
