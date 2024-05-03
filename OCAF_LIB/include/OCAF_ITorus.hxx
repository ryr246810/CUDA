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
#ifndef _OCAF_ITorus_HeaderFile
#define _OCAF_ITorus_HeaderFile

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


class OCAF_ITorus : public OCAF_IFunction {

public:

  inline void* operator new(size_t,void* anAddress)  {  return anAddress; }
  inline void* operator new(size_t size)  { return Standard::Allocate(size);  }
  inline void  operator delete(void *anAddress)  {  if (anAddress) Standard::Free((Standard_Address&)anAddress); }
  
  // Methods PUBLIC
  // 
  Standard_EXPORT static const Standard_GUID& GetID() ;
  
  Standard_EXPORT static  Handle(TDataStd_TreeNode) MakeTorus_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
									    Standard_Integer theType,
									    TCollection_ExtendedString& theError);
  
  
  Standard_EXPORT   Standard_Boolean MakeTorus_Execute(  TCollection_ExtendedString& theError);

  Standard_EXPORT   OCAF_ITorus(const Handle(TDataStd_TreeNode)& aTreeNode);


  Standard_EXPORT   void                        SetPoint(const Handle(TDataStd_TreeNode)& theRef) { SetReference(TORUS_ARG_CENTER, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetPointNode() { return GetReferenceNode(TORUS_ARG_CENTER); };
  Standard_EXPORT   TopoDS_Shape                GetPoint() { return GetReference(TORUS_ARG_CENTER); };

  Standard_EXPORT   void                        SetVector(const Handle(TDataStd_TreeNode)& theRef)    { SetReference(TORUS_ARG_VECTOR, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetVectorNode()    { return GetReferenceNode(TORUS_ARG_VECTOR); };
  Standard_EXPORT   TopoDS_Shape                GetVector()    { return GetReference(TORUS_ARG_VECTOR); };

  Standard_EXPORT   void                        SetRMajor(double theParam) { SetReal(TORUS_ARG_RMAJOR, theParam); };
  Standard_EXPORT   double                      GetRMajor() { return GetReal(TORUS_ARG_RMAJOR); };

  Standard_EXPORT   void                        SetRMinor(double theParam) { SetReal(TORUS_ARG_RMINOR, theParam); };
  Standard_EXPORT   double                      GetRMinor() { return GetReal(TORUS_ARG_RMINOR); };


  Standard_EXPORT   ~OCAF_ITorus() {};


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
