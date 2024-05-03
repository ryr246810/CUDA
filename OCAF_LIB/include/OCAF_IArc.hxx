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
#ifndef _OCAF_IArc_HeaderFile
#define _OCAF_IArc_HeaderFile

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


class OCAF_IArc : public OCAF_IFunction {

public:

  inline void* operator new(size_t,void* anAddress)  {  return anAddress; }
  inline void* operator new(size_t size)  { return Standard::Allocate(size);  }
  inline void  operator delete(void *anAddress)  {  if (anAddress) Standard::Free((Standard_Address&)anAddress); }
  
  // Methods PUBLIC
  // 
  Standard_EXPORT static const Standard_GUID& GetID() ;

  Standard_EXPORT static  Handle(TDataStd_TreeNode) MakeArc_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
									  Standard_Integer theType,
									  TCollection_ExtendedString& theError);
  
  
  Standard_EXPORT   Standard_Boolean MakeArc_Execute(  TCollection_ExtendedString& theError);

  Standard_EXPORT   OCAF_IArc(const Handle(TDataStd_TreeNode)& aTreeNode);


  Standard_EXPORT   void                        SetPoint1(const Handle(TDataStd_TreeNode)& theRef)  { SetReference(ARC_ARG_POINT1, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetPoint1Node()  { return GetReferenceNode(ARC_ARG_POINT1); };
  Standard_EXPORT   TopoDS_Shape                GetPoint1()  { return GetReference(ARC_ARG_POINT1); };

  Standard_EXPORT   void                        SetPoint2(const Handle(TDataStd_TreeNode)& theRef)  { SetReference(ARC_ARG_POINT2, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetPoint2Node()  { return GetReferenceNode(ARC_ARG_POINT2); };
  Standard_EXPORT   TopoDS_Shape                GetPoint2()  { return GetReference(ARC_ARG_POINT2); };

  Standard_EXPORT   void                        SetPoint3(const Handle(TDataStd_TreeNode)& theRef)  { SetReference(ARC_ARG_POINT3, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetPoint3Node()  { return GetReferenceNode(ARC_ARG_POINT3); };
  Standard_EXPORT   TopoDS_Shape                GetPoint3()  { return GetReference(ARC_ARG_POINT3); };

  Standard_EXPORT   void                        SetSense(Standard_Boolean theSense) { SetInteger(ARC_ARG_SENSE, theSense); }
  Standard_EXPORT   Standard_Boolean            GetSense() { return GetInteger(ARC_ARG_SENSE); }

  Standard_EXPORT   ~OCAF_IArc() {};


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
