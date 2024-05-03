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

#ifndef _OCAF_IPlane_HeaderFile
#define _OCAF_IPlane_HeaderFile

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


class OCAF_IPlane : public OCAF_IFunction {

public:

  inline void* operator new(size_t,void* anAddress)  {  return anAddress; }
  inline void* operator new(size_t size)  { return Standard::Allocate(size);  }
  inline void  operator delete(void *anAddress)  {  if (anAddress) Standard::Free((Standard_Address&)anAddress); }
  
  // Methods PUBLIC
  // 
  Standard_EXPORT static const Standard_GUID& GetID() ;

  Standard_EXPORT static  Handle(TDataStd_TreeNode) MakePlane_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
									    Standard_Integer theType,
									    TCollection_ExtendedString& theError);
  
  
  Standard_EXPORT   Standard_Boolean MakePlane_Execute(  TCollection_ExtendedString& theError);

  Standard_EXPORT   OCAF_IPlane(const Handle(TDataStd_TreeNode)& aTreeNode);


  Standard_EXPORT   void                        SetVector(const Handle(TDataStd_TreeNode)& theRef)    { SetReference(PLANE_ARG_VECTOR, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetVectorNode()    { return GetReferenceNode(PLANE_ARG_VECTOR); };
  Standard_EXPORT   TopoDS_Shape                GetVector()    { return GetReference(PLANE_ARG_VECTOR); };

  Standard_EXPORT   void                        SetVector1(const Handle(TDataStd_TreeNode)& theRef)    { SetReference(PLANE_ARG_VECTOR1, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetVector1Node()    { return GetReferenceNode(PLANE_ARG_VECTOR1); };
  Standard_EXPORT   TopoDS_Shape                GetVector1()    { return GetReference(PLANE_ARG_VECTOR1); };

  Standard_EXPORT   void                        SetVector2(const Handle(TDataStd_TreeNode)& theRef)    { SetReference(PLANE_ARG_VECTOR2, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetVector2Node()    { return GetReferenceNode(PLANE_ARG_VECTOR2); };
  Standard_EXPORT   TopoDS_Shape                GetVector2()    { return GetReference(PLANE_ARG_VECTOR2); };

  Standard_EXPORT   void                        SetSize(double theParam) { SetReal(PLANE_ARG_SIZE, theParam); };
  Standard_EXPORT   double                      GetSize() { return GetReal(PLANE_ARG_SIZE); };

  Standard_EXPORT   void                        SetParameterU(double theParam) { SetReal(PLANE_ARG_PARAM_U, theParam); };
  Standard_EXPORT   double                      GetParameterU() { return GetReal(PLANE_ARG_PARAM_U); };

  Standard_EXPORT   void                        SetParameterV(double theParam) { SetReal(PLANE_ARG_PARAM_V, theParam); };
  Standard_EXPORT   double                      GetParameterV() { return GetReal(PLANE_ARG_PARAM_V); };

  Standard_EXPORT   void                        SetOrientation(double theParam) { SetReal(PLANE_ARG_ORIENT, theParam); };
  Standard_EXPORT   double                      GetOrientation() { return GetReal(PLANE_ARG_ORIENT); };

  Standard_EXPORT   void                        SetFace(const Handle(TDataStd_TreeNode)& theRef)  { SetReference(PLANE_ARG_FACE, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetFaceNode()  { return GetReferenceNode(PLANE_ARG_FACE); };
  Standard_EXPORT   TopoDS_Shape                GetFace()  { return GetReference(PLANE_ARG_FACE); };

  Standard_EXPORT   void                        SetLCS(const Handle(TDataStd_TreeNode)& theRef)  { SetReference(PLANE_ARG_LCS, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetLCSNode()  { return GetReferenceNode(PLANE_ARG_LCS); };
  Standard_EXPORT   TopoDS_Shape                GetLCS()  { return GetReference(PLANE_ARG_LCS); };

  Standard_EXPORT   void                        SetPoint(const Handle(TDataStd_TreeNode)& theRef)  { SetReference(PLANE_ARG_POINT1, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetPointNode()  { return GetReferenceNode(PLANE_ARG_POINT1); };
  Standard_EXPORT   TopoDS_Shape                GetPoint()  { return GetReference(PLANE_ARG_POINT1); };

  Standard_EXPORT   void                        SetPoint1(const Handle(TDataStd_TreeNode)& theRef)  { SetReference(PLANE_ARG_POINT1, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetPoint1Node()  { return GetReferenceNode(PLANE_ARG_POINT1); };
  Standard_EXPORT   TopoDS_Shape                GetPoint1()  { return GetReference(PLANE_ARG_POINT1); };

  Standard_EXPORT   void                        SetPoint2(const Handle(TDataStd_TreeNode)& theRef)  { SetReference(PLANE_ARG_POINT2, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetPoint2Node()  { return GetReferenceNode(PLANE_ARG_POINT2); };
  Standard_EXPORT   TopoDS_Shape                GetPoint2()  { return GetReference(PLANE_ARG_POINT2); };

  Standard_EXPORT   void                        SetPoint3(const Handle(TDataStd_TreeNode)& theRef)  { SetReference(PLANE_ARG_POINT3, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetPoint3Node()  { return GetReferenceNode(PLANE_ARG_POINT3); };
  Standard_EXPORT   TopoDS_Shape                GetPoint3()  { return GetReference(PLANE_ARG_POINT3); };



  Standard_EXPORT   ~OCAF_IPlane() {};


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
