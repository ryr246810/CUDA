
#ifndef _OCAF_IPipe_HeaderFile
#define _OCAF_IPipe_HeaderFile

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


class OCAF_IPipe : public OCAF_IFunction {

public:

  inline void* operator new(size_t,void* anAddress) {return anAddress; }
  inline void* operator new(size_t size) { return Standard::Allocate(size); }
  inline void  operator delete(void *anAddress) {  if (anAddress) Standard::Free((Standard_Address&)anAddress); }
  
  // Methods PUBLIC
  // 
  Standard_EXPORT static const Standard_GUID& GetID() ;
  
  Standard_EXPORT static  Handle(TDataStd_TreeNode) MakePipe_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
									   Standard_Integer theType,
									   TCollection_ExtendedString& theError);
  
  Standard_EXPORT   Standard_Boolean MakePipe_Execute(  TCollection_ExtendedString& theError);
  


  Standard_EXPORT OCAF_IPipe(const Handle(TDataStd_TreeNode)& aTreeNode);



  //===========spine==============>>
  Standard_EXPORT Standard_Boolean           SetSpine(Handle(TDataStd_TreeNode) theObjNode){  return SetReference(PIPE_SPINE_TAG, theObjNode); };
  Standard_EXPORT TopoDS_Wire                GetSpine();
  Standard_EXPORT Handle(TDataStd_TreeNode)  GetSpineNode(){  return GetReferenceNode(PIPE_SPINE_TAG); };
  Standard_EXPORT void                       ClearSpine();
  //===========spine==============<<


  //===========profile==============>>
  Standard_EXPORT Standard_Boolean           SetProfile(Handle(TDataStd_TreeNode) theObjNode){ return SetReference(PIPE_PROFILE_TAG, theObjNode); };
  Standard_EXPORT TopoDS_Shape               GetProfile(){ return GetReference(PIPE_PROFILE_TAG); };
  Standard_EXPORT Handle(TDataStd_TreeNode)  GetProfileNode(){ return GetReferenceNode(PIPE_PROFILE_TAG); };

  Standard_EXPORT void                       ClearProfile();
  //===========profile==============<<



  Standard_EXPORT ~OCAF_IPipe() {}

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
