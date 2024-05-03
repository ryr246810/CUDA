//                     Copyright (C) 2010 by
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
#ifndef _OCAF_IFace_HeaderFile
#define _OCAF_IFace_HeaderFile

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

#ifndef _TopoDS_Face_HeaderFile
#include <TopoDS_Face.hxx>
#endif

#ifndef _TopoDS_Edge_HeaderFile
#include <TopoDS_Edge.hxx>
#endif


#include <Tags.hxx>

class Standard_GUID;


class OCAF_IFace : public OCAF_IFunction {

public:
  inline void* operator new(size_t,void* anAddress)  {  return anAddress; }
  inline void* operator new(size_t size)  { return Standard::Allocate(size);  }
  inline void  operator delete(void *anAddress)  {  if (anAddress) Standard::Free((Standard_Address&)anAddress); }
  
  // Methods PUBLIC
  //
  Standard_EXPORT static const Standard_GUID& GetID() ;
  
  Standard_EXPORT static  Handle(TDataStd_TreeNode) MakeFace_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
									   Standard_Integer theType,
									   TCollection_ExtendedString& theError);
  

  //Standard_EXPORT static  Standard_Boolean MakeBox_Excute( const Handle(TDataStd_TreeNode)& aFunctionNode, TCollection_ExtendedString& theError);
  
  Standard_EXPORT   Standard_Boolean MakeFace_Execute(  TCollection_ExtendedString& theError);

  Standard_EXPORT   OCAF_IFace(const Handle(TDataStd_TreeNode)& aTreeNode);


  Standard_EXPORT   void                        SetFace(const Handle(TDataStd_TreeNode)& theRef) { SetReference(FACE_BUILD_FACEARG_TAG, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetFaceNode() { return GetReferenceNode(FACE_BUILD_FACEARG_TAG); };
  Standard_EXPORT   TopoDS_Shape                GetFace() { return GetReference(FACE_BUILD_FACEARG_TAG); };

  Standard_EXPORT   void                        SetWire(const Handle(TDataStd_TreeNode)& theRef) { SetReference(FACE_BUILD_WIREARG_TAG, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetWireNode() { return GetReferenceNode(FACE_BUILD_WIREARG_TAG); };
  Standard_EXPORT   TopoDS_Shape                GetWire() { return GetReference(FACE_BUILD_WIREARG_TAG); };

  Standard_EXPORT   void                        SetIsPlanar(const Standard_Boolean isPlanar);
  Standard_EXPORT   Standard_Boolean            GetIsPlanar();

  Standard_EXPORT   ~OCAF_IFace() {};


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
