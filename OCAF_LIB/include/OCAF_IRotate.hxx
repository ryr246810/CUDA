//                     Copyright 2010(C),2012 by
//  
//                         Wang Yue, China
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
#ifndef _OCAF_IRotate_HeaderFile
#define _OCAF_IRotate_HeaderFile

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

class OCAF_IRotate : public OCAF_IFunction , public OCAF_ITransformParent {

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

  Standard_EXPORT static  Handle(TDataStd_TreeNode) MakeRotate_FunctionNode( const Handle(TDataStd_TreeNode)&  theNode,
									     Standard_Integer theType,
									     TCollection_ExtendedString& theError);
  
  Standard_EXPORT   Standard_Boolean MakeRotate_Execute(  TCollection_ExtendedString& theError);
  
  Standard_EXPORT   OCAF_IRotate(const Handle(TDataStd_TreeNode)& aTreeNode);
  

  Standard_EXPORT   void                        SetContext(const Handle(TDataStd_TreeNode)& theRef)  { SetReference(ROTATE_ARG_CONTEXT, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetContextNode() { return GetReferenceNode(ROTATE_ARG_CONTEXT); };
  Standard_EXPORT   TopoDS_Shape                GetContext() {return GetReference(ROTATE_ARG_CONTEXT); };  

  Standard_EXPORT   void                        SetOriginal(const Handle(TDataStd_TreeNode)&  theOriginal) { SetFuncReference(ROTATE_ARG_ORIGINAL, theOriginal); }
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetOriginalNode() { return GetFuncReferenceNode(ROTATE_ARG_ORIGINAL); }
  Standard_EXPORT   TopoDS_Shape                GetOriginal() { return GetFuncReference(ROTATE_ARG_ORIGINAL); }

  Standard_EXPORT   void                        SetMoveAttach(Standard_Boolean isAttach);
  Standard_EXPORT   Standard_Boolean            GetMoveAttach();

  Standard_EXPORT   void                        SetAngle(double theParam) { SetReal(ROTATE_ARG_ANGLE, theParam); };
  Standard_EXPORT   double                      GetAngle()                { return GetReal(ROTATE_ARG_ANGLE); };

  Standard_EXPORT   void                        SetCentPoint(const Handle(TDataStd_TreeNode)& theRef) { SetReference(ROTATE_ARG_CENTER, theRef); }
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetCentPointNode() { return GetReferenceNode(ROTATE_ARG_CENTER); }
  Standard_EXPORT   TopoDS_Shape                GetCentPoint() { return GetReference(ROTATE_ARG_CENTER); }

  Standard_EXPORT   void                        SetPoint1(const Handle(TDataStd_TreeNode)& theRef)  { SetReference(ROTATE_ARG_POINT1, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetPoint1Node()  { return GetReferenceNode(ROTATE_ARG_POINT1); };
  Standard_EXPORT   TopoDS_Shape                GetPoint1()  { return GetReference(ROTATE_ARG_POINT1); };

  Standard_EXPORT   void                        SetPoint2(const Handle(TDataStd_TreeNode)& theRef)  { SetReference(ROTATE_ARG_POINT2, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetPoint2Node()  { return GetReferenceNode(ROTATE_ARG_POINT2); };
  Standard_EXPORT   TopoDS_Shape                GetPoint2()  { return GetReference(ROTATE_ARG_POINT2); };


  Standard_EXPORT   void                        SetAxis(const Handle(TDataStd_TreeNode)& theRef)  { SetReference(ROTATE_ARG_AXIS, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetAxisNode()  { return GetReferenceNode(ROTATE_ARG_AXIS); };
  Standard_EXPORT   TopoDS_Shape                GetAxis()  { return GetReference(ROTATE_ARG_AXIS); };

  Standard_EXPORT   void                        SetCopyMode(const Standard_Boolean _isSolid);
  Standard_EXPORT   Standard_Boolean            GetCopyMode();

  Standard_EXPORT   ~OCAF_IRotate() {}

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
