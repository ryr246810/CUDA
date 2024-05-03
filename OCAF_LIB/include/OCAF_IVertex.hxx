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
#ifndef _OCAF_IVertex_HeaderFile
#define _OCAF_IVertex_HeaderFile

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


class OCAF_IVertex : public OCAF_IFunction {

public:

  inline void* operator new(size_t,void* anAddress)  {  return anAddress; }
  inline void* operator new(size_t size)  { return Standard::Allocate(size);  }
  inline void  operator delete(void *anAddress)  {  if (anAddress) Standard::Free((Standard_Address&)anAddress); }
  
  // Methods PUBLIC
  // 
  Standard_EXPORT static const Standard_GUID& GetID() ;
  
  Standard_EXPORT static  Handle(TDataStd_TreeNode) MakeVertex_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
									     Standard_Integer theType,
									     TCollection_ExtendedString& theError);
  
  
  //Standard_EXPORT static  Standard_Boolean MakeBox_Excute( const Handle(TDataStd_TreeNode)& aFunctionNode, TCollection_ExtendedString& theError);
  
  Standard_EXPORT   Standard_Boolean MakeVertex_Execute(  TCollection_ExtendedString& theError);

  Standard_EXPORT   OCAF_IVertex(const Handle(TDataStd_TreeNode)& aTreeNode);

  Standard_EXPORT   void SetX(double theX) { SetReal(VERTEX_ARG_X, theX); };
  Standard_EXPORT   void SetY(double theY) { SetReal(VERTEX_ARG_Y, theY); };
  Standard_EXPORT   void SetZ(double theZ) { SetReal(VERTEX_ARG_Z, theZ); };

  Standard_EXPORT   double GetX() { return GetReal(VERTEX_ARG_X); };
  Standard_EXPORT   double GetY() { return GetReal(VERTEX_ARG_Y); };
  Standard_EXPORT   double GetZ() { return GetReal(VERTEX_ARG_Z); };



  Standard_EXPORT   void                        SetRef(const Handle(TDataStd_TreeNode)& theRef) { SetReference(VERTEX_ARG_REF, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetRefNode() { return GetReferenceNode(VERTEX_ARG_REF); };
  Standard_EXPORT   TopoDS_Shape                GetRef() { return GetReference(VERTEX_ARG_REF); };

  Standard_EXPORT   void                        SetCurve(const Handle(TDataStd_TreeNode)& theRef) { SetReference(VERTEX_ARG_CURVE, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetCurveNode() { return GetReferenceNode(VERTEX_ARG_CURVE); };
  Standard_EXPORT   TopoDS_Shape                GetCurve() { return GetReference(VERTEX_ARG_CURVE); };


  Standard_EXPORT   void                        SetSurface(const Handle(TDataStd_TreeNode)& theRef) { SetReference(VERTEX_ARG_SURFACE, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetSurfaceNode() { return GetReferenceNode(VERTEX_ARG_SURFACE); };
  Standard_EXPORT   TopoDS_Shape                GetSurface() { return GetReference(VERTEX_ARG_SURFACE); };


  Standard_EXPORT   void                        SetLine1(const Handle(TDataStd_TreeNode)& theRef) { SetReference(VERTEX_ARG_LINE1, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetLine1Node() { return GetReferenceNode(VERTEX_ARG_LINE1); };
  Standard_EXPORT   TopoDS_Shape                GetLine1() { return GetReference(VERTEX_ARG_LINE1); };

  Standard_EXPORT   void                        SetLine2(const Handle(TDataStd_TreeNode)& theRef) { SetReference(VERTEX_ARG_LINE2, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetLine2Node() { return GetReferenceNode(VERTEX_ARG_LINE2); };
  Standard_EXPORT   TopoDS_Shape                GetLine2() { return GetReference(VERTEX_ARG_LINE2); };


  Standard_EXPORT   void SetParameterT(double theParam) { SetReal(VERTEX_ARG_PARAM_T, theParam); };
  Standard_EXPORT   double GetParameterT() { return GetReal(VERTEX_ARG_PARAM_T); };

  Standard_EXPORT   void SetParameterU(double theParam) { SetReal(VERTEX_ARG_PARAM_U, theParam); };
  Standard_EXPORT   double GetParameterU() { return GetReal(VERTEX_ARG_PARAM_U); };

  Standard_EXPORT   void SetParameterV(double theParam) { SetReal(VERTEX_ARG_PARAM_V, theParam); };
  Standard_EXPORT   double GetParameterV() { return GetReal(VERTEX_ARG_PARAM_V); };


  Standard_EXPORT   ~OCAF_IVertex() {};


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
