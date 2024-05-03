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
#ifndef _OCAF_IHelixEdge_HeaderFile
#define _OCAF_IHelixEdge_HeaderFile

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

class Standard_GUID;


class OCAF_IHelixEdge : public OCAF_IFunction 
{
public:
  inline void* operator new(size_t,void* anAddress)  {  return anAddress; }
  inline void* operator new(size_t size)  { return Standard::Allocate(size);  }
  inline void  operator delete(void *anAddress)  {  if (anAddress) Standard::Free((Standard_Address&)anAddress); }
  
  // Methods PUBLIC
  // 
  Standard_EXPORT static const Standard_GUID& GetID() ;
  
  Standard_EXPORT static  Handle(TDataStd_TreeNode) MakeHelixEdge_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
										Standard_Integer theType,
										TCollection_ExtendedString& theError);
  
  Standard_EXPORT   Standard_Boolean MakeHelixEdge_Execute(  TCollection_ExtendedString& theError);

  Standard_EXPORT   OCAF_IHelixEdge(const Handle(TDataStd_TreeNode)& aTreeNode);


  Standard_EXPORT   void SetOrgX(double theParam) { SetReal(HELIX_PARAM_OX, theParam); };
  Standard_EXPORT   void SetOrgY(double theParam) { SetReal(HELIX_PARAM_OY, theParam); };
  Standard_EXPORT   void SetOrgZ(double theParam) { SetReal(HELIX_PARAM_OZ, theParam); };
  
  Standard_EXPORT   double GetOrgX() { return GetReal(HELIX_PARAM_OX); };
  Standard_EXPORT   double GetOrgY() { return GetReal(HELIX_PARAM_OY); };
  Standard_EXPORT   double GetOrgZ() { return GetReal(HELIX_PARAM_OZ); };

  Standard_EXPORT   void SetR(double theParam) { SetReal(HELIX_PARAM_R, theParam); };
  Standard_EXPORT   double GetR() { return GetReal(HELIX_PARAM_R); };

  Standard_EXPORT   void SetPeriodLength(double theParam) { SetReal(HELIX_PARAM_L, theParam); };
  Standard_EXPORT   double GetPeriodLength() { return GetReal(HELIX_PARAM_L); };

  Standard_EXPORT   void SetPeriodNum(double theParam) { SetReal(HELIX_PARAM_PERIODNUM, theParam); };
  Standard_EXPORT   double GetPeriodNum() { return GetReal(HELIX_PARAM_PERIODNUM); };

  Standard_EXPORT   void SetAxisDir(int theParam) { SetInteger(HELIX_PARAM_AXISDIR, theParam); };
  Standard_EXPORT   int GetAxisDir() { return GetInteger(HELIX_PARAM_AXISDIR); };


  Standard_EXPORT   ~OCAF_IHelixEdge() {};


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
