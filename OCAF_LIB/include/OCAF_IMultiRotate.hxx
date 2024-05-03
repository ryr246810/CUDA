//                     Copyright 2010(C),2012 by
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
#ifndef _OCAF_IMultiRotate_HeaderFile
#define _OCAF_IMultiRotate_HeaderFile

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

#ifndef _TDataStd_Integer_HeaderFile
#include <TDataStd_Integer.hxx>
#endif

#ifndef _TNaming_NamedShape_HeaderFile
#include <TNaming_NamedShape.hxx>
#endif

#ifndef _TopoDS_Shape_HeaderFile
#include <TopoDS_Shape.hxx>
#endif

#ifndef _OCAF_IFunction_HeaderFile
#include <OCAF_IFunction.hxx>
#endif

#ifndef _Standard_Real_HeaderFile
#include <Standard_Real.hxx>
#endif

#ifndef _Standard_Boolean_HeaderFile
#include <Standard_Boolean.hxx>
#endif

#include <OCAF_ITransformParent.hxx>

#include <Tags.hxx>

class OCAF_IMultiRotate : public OCAF_IFunction{

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
  Standard_EXPORT static const Standard_GUID& GetID() ;

  Standard_EXPORT static  Handle(TDataStd_TreeNode) MakeMultiRotate_FunctionNode( const Handle(TDataStd_TreeNode)&  theNode,
										  Standard_Integer theType,
										  TCollection_ExtendedString& theError);
  
  Standard_EXPORT   Standard_Boolean MakeMultiRotate_Execute(  TCollection_ExtendedString& theError);
  
  Standard_EXPORT   OCAF_IMultiRotate(const Handle(TDataStd_TreeNode)& aTreeNode);


  Standard_EXPORT   void                        SetContext(const Handle(TDataStd_TreeNode)& theRef)  { SetReference(MULTIROTATE_ARG_CONTEXT, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetContextNode() { return GetReferenceNode(MULTIROTATE_ARG_CONTEXT); };
  Standard_EXPORT   TopoDS_Shape                GetContext() {return GetReference(MULTIROTATE_ARG_CONTEXT); };  

  Standard_EXPORT   void                        SetAxis(const Handle(TDataStd_TreeNode)& theRef)  { SetReference(MULTIROTATE_ARG_AXIS, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetAxisNode()  { return GetReferenceNode(MULTIROTATE_ARG_AXIS); };
  Standard_EXPORT   TopoDS_Shape                GetAxis()  { return GetReference(MULTIROTATE_ARG_AXIS); };

  Standard_EXPORT   void                        SetAngle(double theParam) { SetReal(MULTIROTATE_ARG_ANGLE, theParam); };
  Standard_EXPORT   double                      GetAngle()                { return GetReal(MULTIROTATE_ARG_ANGLE); };

  Standard_EXPORT   void                        SetPeriodNum(int theParam) { SetInteger(MULTIROTATE_ARG_NUM, theParam); };
  Standard_EXPORT   double                      GetPeriodNum()             { return GetInteger(MULTIROTATE_ARG_NUM); };


  Standard_EXPORT   ~OCAF_IMultiRotate() {}

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
