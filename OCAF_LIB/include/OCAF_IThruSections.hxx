#ifndef _OCAF_IThruSections_HeaderFile
#define _OCAF_IThruSections_HeaderFile

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


class OCAF_IThruSections : public OCAF_IFunction {

public:
  
  inline void* operator new(size_t,void* anAddress)  { 
    return anAddress; 
  }
  inline void* operator new(size_t size) { 
    return Standard::Allocate(size);
  }
  inline void  operator delete(void *anAddress) {  
    if (anAddress) Standard::Free((Standard_Address&)anAddress);
  }
  
  // Methods PUBLIC
  // 
  Standard_EXPORT static const Standard_GUID& GetID() ;
  
  Standard_EXPORT static  Handle(TDataStd_TreeNode) MakeThruSections_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
										   Standard_Integer theType,
										   TCollection_ExtendedString& theError);
  
  Standard_EXPORT   Standard_Boolean MakeThruSections_Execute(  TCollection_ExtendedString& theError);
  

  Standard_EXPORT OCAF_IThruSections(const Handle(TDataStd_TreeNode)& aTreeNode);



  Standard_EXPORT static Standard_Boolean IsClosedSection(const Handle(TDataStd_TreeNode)& aTreeNode);

  Standard_EXPORT void SetSolid();
  Standard_EXPORT void SetShell();
  Standard_EXPORT Standard_Boolean IsSolid();
  Standard_EXPORT Standard_Boolean IsShell();


  Standard_EXPORT void SetSolidOrShell(const Standard_Boolean _isSolid);
  Standard_EXPORT Standard_Boolean IsSolidOrShell();

  Standard_EXPORT void SetRuled(const Standard_Boolean _isRuled);
  Standard_EXPORT Standard_Boolean IsRuled();

  Standard_EXPORT Standard_Real    GetTolerance(){return GetReal(THRUSECTION_BUILD_TOLERANCE_TAG);};
  Standard_EXPORT void             SetTolerance(Standard_Real aTolerance){ SetReal(THRUSECTION_BUILD_TOLERANCE_TAG, aTolerance); };

  Standard_EXPORT Standard_Boolean SetSection(Handle(TDataStd_TreeNode) theNode);
  Standard_EXPORT TopoDS_Shape     GetSection(Standard_Integer theNumber);

  Standard_EXPORT Standard_Boolean SetBuildThruSectionsElement(Handle(TDataStd_TreeNode) theNode);
  Standard_EXPORT Standard_Boolean SetBuildThruSectionsElements(const TDF_LabelMap& ArgumentsMap);
  Standard_EXPORT Standard_Boolean SetBuildThruSectionsElements(const TDF_AttributeMap& ArgumentsMap);

  Standard_EXPORT TopoDS_Shape     GetBuildThruSectionsElement(Standard_Integer theNumber);
  Standard_EXPORT void             GetBuildThruSectionsElementsMap(TDF_AttributeMap& ArgumentsMap);

  Standard_EXPORT void             ClearBuildThruSectionsElements();
  Standard_EXPORT void             RemoveBuildThruSectionsElement(Handle(TDataStd_TreeNode) theNode);


  Standard_EXPORT ~OCAF_IThruSections() {}
  
  
  
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
