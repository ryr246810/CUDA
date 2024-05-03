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
#ifndef _OCAF_IEdge_HeaderFile
#define _OCAF_IEdge_HeaderFile

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


class OCAF_IEdge : public OCAF_IFunction {

public:

  inline void* operator new(size_t,void* anAddress)  {  return anAddress; }
  inline void* operator new(size_t size)  { return Standard::Allocate(size);  }
  inline void  operator delete(void *anAddress)  {  if (anAddress) Standard::Free((Standard_Address&)anAddress); }
  
  // Methods PUBLIC
  // 
  Standard_EXPORT static const Standard_GUID& GetID() ;

  
  Standard_EXPORT static  Handle(TDataStd_TreeNode) MakeEdge_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
									   Standard_Integer theType,
									   TCollection_ExtendedString& theError);
  
  
  //Standard_EXPORT static  Standard_Boolean MakeBox_Excute( const Handle(TDataStd_TreeNode)& aFunctionNode, TCollection_ExtendedString& theError);
  
  Standard_EXPORT   Standard_Boolean MakeEdge_Execute(  TCollection_ExtendedString& theError);

  Standard_EXPORT   OCAF_IEdge(const Handle(TDataStd_TreeNode)& aTreeNode);


  Standard_EXPORT   void                        SetSurface(const Handle(TDataStd_TreeNode)& theRef) { SetReference(EDGE_ARG_SURFACE, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetSurfaceNode() { return GetReferenceNode(EDGE_ARG_SURFACE); };
  Standard_EXPORT   TopoDS_Shape                GetSurface() { return GetReference(EDGE_ARG_SURFACE); };

  Standard_EXPORT   void                        SetLine(const Handle(TDataStd_TreeNode)& theRef)    { SetReference(EDGE_ARG_LINE, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetLineNode()    { return GetReferenceNode(EDGE_ARG_LINE); };
  Standard_EXPORT   TopoDS_Shape                GetLine()    { return GetReference(EDGE_ARG_LINE); };

  Standard_EXPORT   void                        SetPoint1(const Handle(TDataStd_TreeNode)& theRef)  { SetReference(EDGE_ARG_POINT1, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetPoint1Node()  { return GetReferenceNode(EDGE_ARG_POINT1); };
  Standard_EXPORT   TopoDS_Shape                GetPoint1()  { return GetReference(EDGE_ARG_POINT1); };

  Standard_EXPORT   void                        SetPoint2(const Handle(TDataStd_TreeNode)& theRef)  { SetReference(EDGE_ARG_POINT2, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetPoint2Node()  { return GetReferenceNode(EDGE_ARG_POINT2); };
  Standard_EXPORT   TopoDS_Shape                GetPoint2()  { return GetReference(EDGE_ARG_POINT2); };

  Standard_EXPORT   void SetParameterT1(double theParam) { SetReal(EDGE_ARG_PARAM_T1, theParam); };
  Standard_EXPORT   void SetParameterT2(double theParam) { SetReal(EDGE_ARG_PARAM_T2, theParam); };

  Standard_EXPORT   double GetParameterT1() { return GetReal(EDGE_ARG_PARAM_T1); };
  Standard_EXPORT   double GetParameterT2() { return GetReal(EDGE_ARG_PARAM_T2); };


  Standard_EXPORT   void SetParameterU1(double theParam) { SetReal(EDGE_ARG_PARAM_U1, theParam); };
  Standard_EXPORT   void SetParameterV1(double theParam) { SetReal(EDGE_ARG_PARAM_V1, theParam); };
  Standard_EXPORT   void SetParameterU2(double theParam) { SetReal(EDGE_ARG_PARAM_U2, theParam); };
  Standard_EXPORT   void SetParameterV2(double theParam) { SetReal(EDGE_ARG_PARAM_V2, theParam); };

  Standard_EXPORT   double GetParameterU1() { return GetReal(EDGE_ARG_PARAM_U1); };
  Standard_EXPORT   double GetParameterV1() { return GetReal(EDGE_ARG_PARAM_V1); };
  Standard_EXPORT   double GetParameterU2() { return GetReal(EDGE_ARG_PARAM_U2); };
  Standard_EXPORT   double GetParameterV2() { return GetReal(EDGE_ARG_PARAM_V2); };


  Standard_EXPORT   ~OCAF_IEdge() {};


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
