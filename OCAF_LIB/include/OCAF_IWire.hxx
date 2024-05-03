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
#ifndef _OCAF_IWire_HeaderFile
#define _OCAF_IWire_HeaderFile

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

#ifndef _TopoDS_Wire_HeaderFile
#include <TopoDS_Wire.hxx>
#endif

#ifndef _TopoDS_Edge_HeaderFile
#include <TopoDS_Edge.hxx>
#endif


#include <Tags.hxx>


class OCAF_IWire : public OCAF_IFunction {

public:
  inline void* operator new(size_t,void* anAddress)  {  return anAddress; }
  inline void* operator new(size_t size)  { return Standard::Allocate(size);  }
  inline void  operator delete(void *anAddress)  {  if (anAddress) Standard::Free((Standard_Address&)anAddress); }
  
  // Methods PUBLIC
  //
  Standard_EXPORT static const Standard_GUID& GetID() ;
  
  Standard_EXPORT static  Handle(TDataStd_TreeNode) MakeWire_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
									   Standard_Integer theType,
									   TCollection_ExtendedString& theError);
  
  Standard_EXPORT   Standard_Boolean MakeWire_Execute(  TCollection_ExtendedString& theError);

  Standard_EXPORT   OCAF_IWire(const Handle(TDataStd_TreeNode)& aTreeNode);

  Standard_EXPORT Standard_Real    GetTolerance(){return GetReal(WIRE_BUILD_TOLERANCE_TAG);};
  Standard_EXPORT void             SetTolerance(Standard_Real aTolerance){ SetReal(WIRE_BUILD_TOLERANCE_TAG, aTolerance); };

  Standard_EXPORT Standard_Boolean SetBuildWireElements(const TDF_LabelMap& ArgumentsMap);

  Standard_EXPORT Standard_Boolean SetBuildWireElements(const TDF_AttributeMap& ArgumentsMap);
  Standard_EXPORT Standard_Boolean SetBuildWireElements(const TDF_AttributeList& ArgumentsList);
  Standard_EXPORT Standard_Boolean SetBuildWireElements(const TDF_AttributeSequence& ArgumentsSequence);

  Standard_EXPORT Standard_Boolean SetBuildWireElement(Handle(TDataStd_TreeNode) theNode);
  Standard_EXPORT TopoDS_Shape     GetBuildWireElement(Standard_Integer theNumber);

  Standard_EXPORT void             ClearBuildWireElements();
  Standard_EXPORT void             RemoveBuildWireElement(Handle(TDataStd_TreeNode) theNode);

  Standard_EXPORT void             GetBuildWireElementsMap(TDF_AttributeMap& ArgumentsMap);

  Standard_EXPORT   ~OCAF_IWire() {};


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
