
#ifndef _OCAF_ISelection_HeaderFile
#define _OCAF_ISelection_HeaderFile

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

#ifndef _TopoDS_Shape_HeaderFile
#include <TopoDS_Shape.hxx>
#endif

#ifndef _OCAF_IFunction_HeaderFile
#include <OCAF_IFunction.hxx>
#endif

#ifndef _Standard_Boolean_HeaderFile
#include <Standard_Boolean.hxx>
#endif


class OCAF_ISelection : public OCAF_IFunction{

public:

  inline void* operator new(size_t,void* anAddress) 
  {
    return anAddress;
  }
  inline void* operator new(size_t size) 
  { 
    return Standard::Allocate(size); 
  }
  inline void  operator delete(void *anAddress) 
  { 
    if (anAddress) Standard::Free((Standard_Address&)anAddress); 
  }
  
  // Methods PUBLIC
  // 
  Standard_EXPORT static const Standard_GUID& GetID() ;

  Standard_EXPORT static Standard_Boolean MakeSelect_Prereq( const TopoDS_Shape& theContext,
							     const Handle(TDataStd_TreeNode)& AccessTreeNode);

  Standard_EXPORT static Handle(TDataStd_TreeNode) MakeSelect_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode, 
									    TCollection_ExtendedString& theError);

  Standard_EXPORT Standard_Boolean MakeSelect_Execute( const TopoDS_Shape& theShape, 
						       const TopoDS_Shape& theContext, 
						       TCollection_ExtendedString& theError );
  
  Standard_EXPORT static  Handle(TDataStd_TreeNode) MakeSelect(const TCollection_ExtendedString& theName,
							      const TopoDS_Shape& theShape,const TopoDS_Shape& theContext,
							      const Handle(TDataStd_TreeNode)& theAccessTreeNode,
							      TCollection_ExtendedString& theError) ;

  Standard_EXPORT OCAF_ISelection(const Handle(TDataStd_TreeNode)& aTreeNode);

  Standard_EXPORT   Standard_Boolean Select(const TopoDS_Shape& theShape,
					    const TopoDS_Shape& theContext) ;

  Standard_EXPORT   TopoDS_Shape GetContext() const;
  Standard_EXPORT ~OCAF_ISelection() {}
  
  
  
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


#endif
