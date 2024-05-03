#ifndef _OCAF_IPipeShell_HeaderFile
#define _OCAF_IPipeShell_HeaderFile

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

#include <BRepBuilderAPI_TransitionMode.hxx>


class OCAF_IPipeShell : public OCAF_IFunction {

public:

  inline void* operator new(size_t,void* anAddress) { 
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
  
  Standard_EXPORT static  Handle(TDataStd_TreeNode) MakePipeShell_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
										Standard_Integer theType,
										TCollection_ExtendedString& theError);
  
  Standard_EXPORT   Standard_Boolean MakePipeShell_Execute(  TCollection_ExtendedString& theError);
  


  Standard_EXPORT static Standard_Boolean IsClosedSection(const Handle(TDataStd_TreeNode)& aTreeNode);
  Standard_EXPORT OCAF_IPipeShell(const Handle(TDataStd_TreeNode)& aTreeNode);


  Standard_EXPORT void                          SetTransitionMode(BRepBuilderAPI_TransitionMode theMode);
  Standard_EXPORT BRepBuilderAPI_TransitionMode GetTransitionMode();


  Standard_EXPORT void             SetSolidOrShell(const Standard_Boolean _isSolid);
  Standard_EXPORT Standard_Boolean IsSolidOrShell();

  //===========spine==============>>
  Standard_EXPORT Standard_Boolean           SetSpine(Handle(TDataStd_TreeNode) theObjNode);
  Standard_EXPORT TopoDS_Wire                GetSpine();
  Standard_EXPORT Handle(TDataStd_TreeNode)  GetSpineNode();
  Standard_EXPORT void                       ClearSpine();
  //===========spine==============<<



  //===========mode==============>>
  Standard_EXPORT void                      SetFrenet(Standard_Boolean theType);
  Standard_EXPORT Standard_Boolean          IsFrenet();

  Standard_EXPORT Standard_Boolean          SetPipeMode(Handle(TDataStd_TreeNode) theObjNode, Standard_Boolean CurvilinearEquivalence, Standard_Boolean KeepContact);
  Standard_EXPORT TopoDS_Wire               GetPipeMode();
  Standard_EXPORT Handle(TDataStd_TreeNode) GetPipeModeNode();
  Standard_EXPORT Standard_Boolean          IsCurvilinearEquivalence();
  Standard_EXPORT Standard_Boolean          IsKeepContact();
  //===========mode==============<<



  //===========profile==============>>
  Standard_EXPORT TDF_Label        GetProfilesLabel(){ TDF_Label theProfilesLab = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(PIPESHELL_PROFILE_TAG); return theProfilesLab;}

  Standard_EXPORT Standard_Boolean AddProfile(Handle(TDataStd_TreeNode) theObjNode, Standard_Boolean theWithContact, Standard_Boolean theWithCorrection);
  Standard_EXPORT TopoDS_Wire      GetProfile(Standard_Integer theIndex);
  Standard_EXPORT void             GetProfilesMap(TDF_AttributeMap& _ProfilesMap);
  Standard_EXPORT void             GetProfilesSequence(TDF_AttributeSequence& ArgumentsSequence);

  Standard_EXPORT Standard_Boolean IsWithContact(Standard_Integer theIndex);
  Standard_EXPORT Standard_Boolean IsWithCorrection(Standard_Integer theIndex);

  Standard_EXPORT void             ClearProfiles();
  //===========profile==============<<



  Standard_EXPORT ~OCAF_IPipeShell() {}

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
