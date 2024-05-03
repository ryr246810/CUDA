//                     Copyright (C) 2010,2015 by
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
#ifndef _OCAF_ICylinder_HeaderFile
#define _OCAF_ICylinder_HeaderFile

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

class OCAF_ICylinder : public OCAF_IFunction {

public:

  inline void* operator new(size_t,void* anAddress) 
  {
    return anAddress;
  }
  inline void* operator new(size_t size) 
  { 
    return Standard::Allocate(size); 
  }
  inline void  operator delete(void *anAddress) 
  { 
    if (anAddress) Standard::Free((Standard_Address&)anAddress); 
  }
  
  // Methods PUBLIC
  // 
  Standard_EXPORT static const Standard_GUID& GetID() ;

  Standard_EXPORT static  Handle(TDataStd_TreeNode) MakeCylinder_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
									       Standard_Integer theType,
									       TCollection_ExtendedString& theError);
  
  
  Standard_EXPORT   Standard_Boolean MakeCylinder_Execute(  TCollection_ExtendedString& theError);

  Standard_EXPORT OCAF_ICylinder(const Handle(TDataStd_TreeNode)& aTreeNode);
  Standard_EXPORT ~OCAF_ICylinder() {}

  
  Standard_EXPORT   void                      SetPoint(const Handle(TDataStd_TreeNode)& theRef) { SetReference(CYLINDER_ARG_CENTER, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode) GetPointNode()    { return GetReferenceNode(CYLINDER_ARG_CENTER); };
  Standard_EXPORT   TopoDS_Shape              GetPoint()    { return GetReference(CYLINDER_ARG_CENTER); };

  Standard_EXPORT   void                      SetVector(const Handle(TDataStd_TreeNode)& theRef) { SetReference(CYLINDER_ARG_VECTOR, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode) GetVectorNode()    { return GetReferenceNode(CYLINDER_ARG_VECTOR); };
  Standard_EXPORT   TopoDS_Shape              GetVector()    { return GetReference(CYLINDER_ARG_VECTOR); };


  Standard_EXPORT   void            SetR(double theParam) { SetReal(CYLINDER_ARG_RADIUS, theParam); };
  Standard_EXPORT   Standard_Real   GetR() { return GetReal(CYLINDER_ARG_RADIUS); };

  Standard_EXPORT   void            SetH(double theParam) { SetReal(CYLINDER_ARG_HEIGHT, theParam); };
  Standard_EXPORT   Standard_Real   GetH() { return GetReal(CYLINDER_ARG_HEIGHT); };


  

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
