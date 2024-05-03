#include <math.h>
#include <PMLDataDefine.hxx>

#include <iostream>



Standard_Real 
PMLDataDefine::
ComputePMLParamtPowFunc(const Standard_Real _gridStep, 
			const Standard_Real _distance, 
			const Standard_Integer _PMLNum,
			const Standard_Integer _powOrder)
{
  Standard_Real GridIndexLoc = fabs(_distance/_gridStep);
  Standard_Real PBase = GridIndexLoc/((Standard_Real)_PMLNum);
  Standard_Real result = pow(PBase, _powOrder);
  return result;
}


Standard_Real 
PMLDataDefine::
ComputePMLParamtPowFunc(const Standard_Real _gridStep, 
			const Standard_Real _distance, 
			const Standard_Integer _PMLNum)
{
  Standard_Real GridIndexLoc = fabs(_distance/_gridStep);
  Standard_Real PBase = GridIndexLoc/((Standard_Real)_PMLNum);
  Standard_Real result = pow(PBase, m_PowOrder);

  return result;
}



Standard_Real 
PMLDataDefine::
ComputePMLSigmaOpt(const Standard_Real _gridStep, 
		   const Standard_Integer _PMLNum)
{
  Standard_Real PI = 2*asin(1);
  Standard_Real FreeSpace_Impedance = 120.0*PI;
  Standard_Real PML_Thickness =_gridStep*((Standard_Real)_PMLNum);

  Standard_Real Reflection_Error = exp(-16); // exp(-16)~~1e-6;
  Standard_Real Sigma_Opt = -log(Reflection_Error)*(m_PowOrder+1)/(FreeSpace_Impedance*PML_Thickness*2.0);

  return Sigma_Opt;
}



Standard_Real 
PMLDataDefine::
ComputePMLSigma(const Standard_Real _gridStep, 
		const Standard_Real _distance,
		const Standard_Integer _PMLNum)
{
  Standard_Real Sigma_Opt = ComputePMLSigmaOpt(_gridStep, _PMLNum);

  Standard_Real Sigma_Max = m_SigmaRatio * Sigma_Opt;

  Standard_Real result = Sigma_Max * ComputePMLParamtPowFunc(_gridStep, _distance, _PMLNum);

  return result;
}



Standard_Real 
PMLDataDefine::
ComputePMLKappa(const Standard_Real _gridStep, 
		const Standard_Real _distance, 
		const Standard_Integer _PMLNum)
{
  Standard_Integer _order = m_PowOrder-1;
  Standard_Real result = 1.0 + (m_KappaMax-1) * ComputePMLParamtPowFunc(_gridStep, _distance, _PMLNum, _order);
  return result;
}


/*
Standard_Real 
PMLDataDefine::
ComputePMLAlpha(const Standard_Real _gridStep, 
		const Standard_Real _distance, 
		const Standard_Integer _PMLNum)
{
  Standard_Real result = m_Alpha;
  return result;
}
//*/


//*
Standard_Real 
PMLDataDefine::
ComputePMLAlpha(const Standard_Real _gridStep, 
		const Standard_Real _distance, 
		const Standard_Integer _PMLNum)
{
  Standard_Real result = m_Alpha * ComputePMLParamtPowFunc(_gridStep, _distance, _PMLNum, 1);
  if(result<1e-10) result = 0.;
  return result;
}
//*/
