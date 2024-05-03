//                     Copyright (C) 2014 by
//  
//                      Wang Yue, China
//  
// This software is furnished in accordance with the terms and conditions
// of the contract and with the inclusion of the above copyright notice.
// This software or any other copy thereof may not be provided or otherwise
// be made available to any other person. No title to an ownership of the
// software is hereby transferred.
//  
// At the termination of the contract, the software and all copies of this
// software must be deleted.
//
#ifndef _OCAF_ICosPeriodEdge_HeaderFile
#define _OCAF_ICosPeriodEdge_HeaderFile

#ifndef _Standard_HeaderFile
#include <Standard.hxx>
#endif

#ifndef _Standard_GUID_HeaderFile
#include <Standard_GUID.hxx>
#endif

#ifndef _TDataStd_TreeNode_HeaderFile
#include <TDataStd_TreeNode.hxx>
#endif

#ifndef _TCollection_ExtendedString_HeaderFile
#include <TCollection_ExtendedString.hxx>
#endif

#ifndef _TDataStd_Real_HeaderFile
#include <TDataStd_Real.hxx>
#endif

#ifndef _OCAF_IFunction_HeaderFile
#include <OCAF_IFunction.hxx>
#endif

#ifndef _Standard_Real_HeaderFile
#include <Standard_Real.hxx>
#endif

#include <Tags.hxx>

class OCAF_ICosPeriodEdge : public OCAF_IFunction 
{
public:
  inline void* operator new(size_t,void* anAddress)  {  return anAddress; }
  inline void* operator new(size_t size)  { return Standard::Allocate(size);  }
  inline void  operator delete(void *anAddress)  {  if (anAddress) Standard::Free((Standard_Address&)anAddress); }
  
  // Methods PUBLIC
  // 
  Standard_EXPORT static const Standard_GUID& GetID() ;
  
  Standard_EXPORT static  Handle(TDataStd_TreeNode) MakeCosPeriodEdge_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
										    Standard_Integer theType,
										    TCollection_ExtendedString& theError);
  
  Standard_EXPORT   Standard_Boolean MakeCosPeriodEdge_Execute(  TCollection_ExtendedString& theError);

  Standard_EXPORT   OCAF_ICosPeriodEdge(const Handle(TDataStd_TreeNode)& aTreeNode);


  Standard_EXPORT   void SetOrgX(double theParam) { SetReal(CPE_PARAM_OX, theParam); };
  Standard_EXPORT   void SetOrgY(double theParam) { SetReal(CPE_PARAM_OY, theParam); };
  Standard_EXPORT   void SetOrgZ(double theParam) { SetReal(CPE_PARAM_OZ, theParam); };
  
  Standard_EXPORT   double GetOrgX() { return GetReal(CPE_PARAM_OX); };
  Standard_EXPORT   double GetOrgY() { return GetReal(CPE_PARAM_OY); };
  Standard_EXPORT   double GetOrgZ() { return GetReal(CPE_PARAM_OZ); };

  Standard_EXPORT   void SetR(double theParam) { SetReal(CPE_PARAM_R, theParam); };
  Standard_EXPORT   double GetR() { return GetReal(CPE_PARAM_R); };

  Standard_EXPORT   void SetRippleDepth(double theParam) { SetReal(CPE_PARAM_RD, theParam); };
  Standard_EXPORT   double GetRippleDepth() { return GetReal(CPE_PARAM_RD); };

  Standard_EXPORT   void SetRipplePeriodLength(double theParam) { SetReal(CPE_PARAM_RPL, theParam); };
  Standard_EXPORT   double GetRipplePeriodLength() { return GetReal(CPE_PARAM_RPL); };

  Standard_EXPORT   void SetPeriodNum(double theParam) { SetReal(CPE_PARAM_PERIODNUM, theParam); };
  Standard_EXPORT   double GetPeriodNum() { return GetReal(CPE_PARAM_PERIODNUM); };

  Standard_EXPORT   void SetPeriodSampleNum(double theParam) { SetReal(CPE_PARAM_ONEPERIODSAMPLENUM, theParam); };
  Standard_EXPORT   double GetPeriodSampleNum() { return GetReal(CPE_PARAM_ONEPERIODSAMPLENUM); };

  Standard_EXPORT   void SetAxisDir(int theParam) { SetInteger(CPE_PARAM_AXISDIR, theParam); };
  Standard_EXPORT   int GetAxisDir() { return GetInteger(CPE_PARAM_AXISDIR); };

  Standard_EXPORT   void SetAmpDir(int theParam) { SetInteger(CPE_PARAM_AMPDIR, theParam); };
  Standard_EXPORT   int GetAmpDir() { return GetInteger(CPE_PARAM_AMPDIR); };


  Standard_EXPORT   void SetPhaseShiftRatio(double theParam) { SetReal(CPE_PARAM_PHASE, theParam); };
  Standard_EXPORT   double GetPhaseShiftRatio() { return GetReal(CPE_PARAM_PHASE); };


  Standard_EXPORT   ~OCAF_ICosPeriodEdge() {};


protected:

 // Methods PROTECTED
 // 


 // Fields PROTECTED
 //


private: 

 // Methods PRIVATE
 // 


 // Fields PRIVATE
 //


};


// other inline functions and methods (like "C++: function call" methods)
//


#endif
