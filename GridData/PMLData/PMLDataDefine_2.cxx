#include <math.h>
#include <PMLDataDefine.hxx>

#include <iostream>



Standard_Real 
PMLDataDefine::
ComputePMLParamtPowFunc_2(const Standard_Real _gridStep, 
			  const Standard_Real _distance, 
			  const Standard_Integer _PMLNum)
{
  Standard_Real POrder = _distance/_gridStep;
  Standard_Real result = pow(m_GeomProg, POrder)/(pow(m_GeomProg, _PMLNum)-1.0);
  return result;
}



Standard_Real 
PMLDataDefine::
ComputePMLSigmaOpt_2(const Standard_Real _gridStep, 
		     const Standard_Integer _PMLNum)
{
  Standard_Real PI = 2*asin(1);
  Standard_Real FreeSpace_Impedance = 120.*PI;
  Standard_Real Reflection_Error = exp(-16); // exp(-16)~~1e-6;

  Standard_Real Sigma_Opt = -log(m_GeomProg)*log(Reflection_Error)/(2.0*_gridStep*FreeSpace_Impedance);


  return Sigma_Opt;
}



Standard_Real 
PMLDataDefine::
ComputePMLSigma_2(const Standard_Real _gridStep, 
		  const Standard_Real _distance, 
		  const Standard_Integer _PMLNum)
{
  Standard_Real Sigma_Opt = ComputePMLSigmaOpt_2(_gridStep, _PMLNum);
  Standard_Real Sigma_Max = m_SigmaRatio * Sigma_Opt;
  
  Standard_Real result = Sigma_Max * ComputePMLParamtPowFunc_2(_gridStep, _distance, _PMLNum);

  return result;
}



Standard_Real 
PMLDataDefine::
ComputePMLKappa_2(const Standard_Real _gridStep, 
		  const Standard_Real _distance, 
		  const Standard_Integer _PMLNum)
{
  Standard_Real result = 1.0 + (m_KappaMax-1)*ComputePMLParamtPowFunc_2(_gridStep, _distance, _PMLNum);
  return result;
}
