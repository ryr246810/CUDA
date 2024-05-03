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
#ifndef _OCAF_IVector_HeaderFile
#define _OCAF_IVector_HeaderFile

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


class OCAF_IVector : public OCAF_IFunction {

public:

  inline void* operator new(size_t,void* anAddress)  {  return anAddress; }
  inline void* operator new(size_t size)  { return Standard::Allocate(size);  }
  inline void  operator delete(void *anAddress)  {  if (anAddress) Standard::Free((Standard_Address&)anAddress); }
  
  // Methods PUBLIC
  // 
  Standard_EXPORT static const Standard_GUID& GetID() ;
  
  Standard_EXPORT static  Handle(TDataStd_TreeNode) MakeVector_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
									     Standard_Integer theType,
									     TCollection_ExtendedString& theError);
  
  
  //Standard_EXPORT static  Standard_Boolean MakeBox_Excute( const Handle(TDataStd_TreeNode)& aFunctionNode, TCollection_ExtendedString& theError);
  
  Standard_EXPORT   Standard_Boolean MakeVector_Execute(  TCollection_ExtendedString& theError);

  Standard_EXPORT   OCAF_IVector(const Handle(TDataStd_TreeNode)& aTreeNode);


  /****************************************************************************************************************************************************/
  Standard_EXPORT   void                        SetPoint1(const Handle(TDataStd_TreeNode)& theRef)  { SetReference(VECTOR_ARG_POINT1, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetPoint1Node()  { return GetReferenceNode(VECTOR_ARG_POINT1); };
  Standard_EXPORT   TopoDS_Shape                GetPoint1()  { return GetReference(VECTOR_ARG_POINT1); };

  Standard_EXPORT   void                        SetPoint2(const Handle(TDataStd_TreeNode)& theRef)  { SetReference(VECTOR_ARG_POINT2, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetPoint2Node()  { return GetReferenceNode(VECTOR_ARG_POINT2); };
  Standard_EXPORT   TopoDS_Shape                GetPoint2()  { return GetReference(VECTOR_ARG_POINT2); };
  /****************************************************************************************************************************************************/


  /****************************************************************************************************************************************************/
  Standard_EXPORT   void   SetDX(double theX) { SetReal(VECTOR_ARG_DX, theX); }
  Standard_EXPORT   double GetDX() { return GetReal(VECTOR_ARG_DX); }

  Standard_EXPORT   void   SetDY(double theY) { SetReal(VECTOR_ARG_DY, theY); }
  Standard_EXPORT   double GetDY() { return GetReal(VECTOR_ARG_DY); }

  Standard_EXPORT   void   SetDZ(double theZ) { SetReal(VECTOR_ARG_DZ, theZ); }
  Standard_EXPORT   double GetDZ() { return GetReal(VECTOR_ARG_DZ); }
  /****************************************************************************************************************************************************/


  /****************************************************************************************************************************************************/
  Standard_EXPORT   void                        SetCurve(const Handle(TDataStd_TreeNode)& theRef)  { SetReference(VECTOR_ARG_CURVE, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetCurveNode()  { return GetReferenceNode(VECTOR_ARG_CURVE); };
  Standard_EXPORT   TopoDS_Shape                GetCurve()  { return GetReference(VECTOR_ARG_CURVE); };

  Standard_EXPORT   void   SetParameter(double aValue) { SetReal(VECTOR_ARG_T, aValue); }
  Standard_EXPORT   double GetParameter() { return GetReal(VECTOR_ARG_T); }
  /****************************************************************************************************************************************************/


  /****************************************************************************************************************************************************/
  Standard_EXPORT   void                        SetSurface(const Handle(TDataStd_TreeNode)& theRef) { SetReference(VECTOR_ARG_SURFACE, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetSurfaceNode() { return GetReferenceNode(VECTOR_ARG_SURFACE); };
  Standard_EXPORT   TopoDS_Shape                GetSurface() { return GetReference(VECTOR_ARG_SURFACE); };

  Standard_EXPORT   void                        SetPoint(const Handle(TDataStd_TreeNode)& theRef)  { SetReference(VECTOR_ARG_PNT, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetPointNode()  { return GetReferenceNode(VECTOR_ARG_PNT); };
  Standard_EXPORT   TopoDS_Shape                GetPoint()  { return GetReference(VECTOR_ARG_PNT); };

  Standard_EXPORT   void   SetU(double theX) { SetReal(VECTOR_ARG_U, theX); }
  Standard_EXPORT   double GetU() { return GetReal(VECTOR_ARG_U); }

  Standard_EXPORT   void   SetV(double theY) { SetReal(VECTOR_ARG_V, theY); }
  Standard_EXPORT   double GetV() { return GetReal(VECTOR_ARG_V); }
  /****************************************************************************************************************************************************/


  Standard_EXPORT   ~OCAF_IVector() {};


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
