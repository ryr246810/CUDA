
#ifndef _OCAF_IPrism_HeaderFile
#define _OCAF_IPrism_HeaderFile

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


class OCAF_IPrism : public OCAF_IFunction {

public:
  inline void* operator new(size_t,void* anAddress)  { return anAddress; }
  inline void* operator new(size_t size)  { return Standard::Allocate(size); }
  inline void  operator delete(void *anAddress)  { if (anAddress) Standard::Free((Standard_Address&)anAddress); }
  
  // Methods PUBLIC
  // 
  Standard_EXPORT static const Standard_GUID& GetID() ;
  
  Standard_EXPORT static Handle(TDataStd_TreeNode) MakePrism_FunctionNode(const Handle(TDataStd_TreeNode)& anObjectNode,
									  Standard_Integer           theType,
									  TCollection_ExtendedString& theError);
  
  Standard_EXPORT Standard_Boolean MakePrism_Execute(TCollection_ExtendedString& theError);
  
  Standard_EXPORT OCAF_IPrism(const Handle(TDataStd_TreeNode)& aTreeNode);
  
  //===========spine==============>>
  Standard_EXPORT Standard_Boolean           SetVector(Handle(TDataStd_TreeNode) theObjNode);
  
  Standard_EXPORT TopoDS_Edge                GetVector();
  
  Standard_EXPORT Handle(TDataStd_TreeNode)  GetVectorNode();
  
  Standard_EXPORT void                       ClearVector();
  //===========spine==============<<
  
  //===========profile==============>>
  Standard_EXPORT Standard_Boolean           SetProfile(Handle(TDataStd_TreeNode) theObjNode);
  
  Standard_EXPORT TopoDS_Shape               GetProfile();
  
  Standard_EXPORT Handle(TDataStd_TreeNode)  GetProfileNode();
  
  Standard_EXPORT void                       ClearProfile();
  
  Standard_EXPORT ~OCAF_IPrism() {}
  
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
