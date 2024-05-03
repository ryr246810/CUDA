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
#ifndef _OCAF_ISolid_HeaderFile
#define _OCAF_ISolid_HeaderFile

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

#ifndef _TopoDS_Solid_HeaderFile
#include <TopoDS_Solid.hxx>
#endif

#ifndef _TopoDS_Edge_HeaderFile
#include <TopoDS_Edge.hxx>
#endif


#include <Tags.hxx>


class OCAF_ISolid : public OCAF_IFunction {

public:
  inline void* operator new(size_t,void* anAddress)  {  return anAddress; }
  inline void* operator new(size_t size)  { return Standard::Allocate(size);  }
  inline void  operator delete(void *anAddress)  {  if (anAddress) Standard::Free((Standard_Address&)anAddress); }
  
  // Methods PUBLIC
  //
  Standard_EXPORT static const Standard_GUID& GetID() ;
  
  Standard_EXPORT static  Handle(TDataStd_TreeNode) MakeSolid_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
									    Standard_Integer theType,
									    TCollection_ExtendedString& theError);
  
  
  Standard_EXPORT   Standard_Boolean MakeSolid_Execute(  TCollection_ExtendedString& theError);

  Standard_EXPORT   OCAF_ISolid(const Handle(TDataStd_TreeNode)& aTreeNode);

  Standard_EXPORT Standard_Boolean SetBuildSolidElements(const TDF_LabelMap& ArgumentsMap);
  Standard_EXPORT Standard_Boolean SetBuildSolidElements(const TDF_AttributeMap& ArgumentsMap);

  Standard_EXPORT Standard_Boolean SetBuildSolidElement(Handle(TDataStd_TreeNode) theNode);
  Standard_EXPORT TopoDS_Shape     GetBuildSolidElement(Standard_Integer theNumber);
  Standard_EXPORT void             GetBuildSolidElementsMap(TDF_AttributeMap& ArgumentsMap);


  Standard_EXPORT void             ClearBuildSolidElements();
  Standard_EXPORT void             RemoveBuildSolidElement(Handle(TDataStd_TreeNode) theNode);


  Standard_EXPORT   ~OCAF_ISolid() {};


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
