
#ifndef _PMLDataDefine_HeaderFile
#define _PMLDataDefine_HeaderFile

#include <Standard_TypeDefine.hxx>
#include <TxHierAttribSet.h>

class PMLDataDefine{

public:
  PMLDataDefine();
  ~PMLDataDefine();
  
  void SetAttrib(const TxHierAttribSet& tha);

  void  SetSigmaRatio(const Standard_Real _sigmaRatio);
  
  void  SetPowOrder(const Standard_Integer _mm);
  void  SetGeomProg(const Standard_Real _base);
  
  void  SetAlpha(const Standard_Real _alpha);
  
  void  SetKappaMax(const Standard_Real _kappaMax);
  
  Standard_Real  GetSigmaRatio() const;
  
  Standard_Real  GetPowOrder() const;
  
  Standard_Real  GetGeomProg() const;
  
  Standard_Real  GetAlpha() const;
  
  Standard_Real  GetKappaMax() const;
  
  
  /*************************************************************************/  

  Standard_Real ComputePMLParamtPowFunc(const Standard_Real _gridStep, 
					const Standard_Real _distance, 
					const Standard_Integer _PMLNum,
					const Standard_Integer _powOrder);


  Standard_Real  ComputePMLParamtPowFunc(const Standard_Real _gridStep, 
					 const Standard_Real _distance, 
					 const Standard_Integer _PMLNum);
  
  
  Standard_Real  ComputePMLSigmaOpt(const Standard_Real _gridStep, 
				    const Standard_Integer _PMLNum);
  
  
  Standard_Real  ComputePMLSigma(const Standard_Real _gridStep, 
				 const Standard_Real _distance,
				 const Standard_Integer _PMLNum);
  
  
  Standard_Real  ComputePMLKappa(const Standard_Real _gridStep,
				 const Standard_Real _distance,
				 const Standard_Integer _PMLNum);
  
  
  Standard_Real  ComputePMLAlpha(const Standard_Real _gridStep,
				 const Standard_Real _distance,
				 const Standard_Integer _PMLNum);
  /*************************************************************************/  


  /*************************************************************************/  
  Standard_Real  ComputePMLParamtPowFunc_2(const Standard_Real _gridStep, 
					   const Standard_Real _distance, 
					   const Standard_Integer _PMLNum);

  Standard_Real  ComputePMLSigmaOpt_2(const Standard_Real _gridStep, 
				      const Standard_Integer _PMLNum);
  
  
  Standard_Real  ComputePMLSigma_2(const Standard_Real _gridStep, 
				   const Standard_Real _distance,
				   const Standard_Integer _PMLNum);
  
  
  Standard_Real  ComputePMLKappa_2(const Standard_Real _gridStep,
				   const Standard_Real _distance,
				   const Standard_Integer _PMLNum);
  
  
  Standard_Real  ComputePMLAlpha_2(const Standard_Real _gridStep,
				   const Standard_Real _distance,
				   const Standard_Integer _PMLNum);
  /*************************************************************************/  

public:
  void SetMethodKey(Standard_Integer _isPow);
  Standard_Integer GetMethodKey() const;


public:
  Standard_Integer m_MethodKey;

  Standard_Integer m_PowOrder;

  Standard_Real m_GeomProg;

  Standard_Real m_SigmaRatio;
  Standard_Real m_Alpha;
  Standard_Real m_KappaMax;
};



#endif
