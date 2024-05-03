
#ifndef _OCAF_ObjectTool_HeaderFile
#define _OCAF_ObjectTool_HeaderFile


#ifndef _TDataStd_TreeNode_HeaderFile
#include <TDataStd_TreeNode.hxx>
#endif

#ifndef _TDF_Label_HeaderFile
#include <TDF_Label.hxx>
#endif

#ifndef _TopoDS_Shape_HeaderFile
#include <TopoDS_Shape.hxx>
#endif

#ifndef _TCollection_ExtendedString_HeaderFile
#include <TCollection_ExtendedString.hxx>
#endif

#ifndef _TPrsStd_AISPresentation_HeaderFile
#include <TPrsStd_AISPresentation.hxx>
#endif

#ifndef _Standard_HeaderFile
#include <Standard.hxx>
#endif

#ifndef _TDF_AttributeMap_HeaderFile
#include <TDF_AttributeMap.hxx>
#endif

#ifndef _MMgt_TShared_HeaderFile
#include <MMgt_TShared.hxx>
#endif

#ifndef _OCAF_ObjectType_HeaderFile
#include <OCAF_ObjectType.hxx>
#endif

#ifndef _Standard_Boolean_HeaderFile
#include <Standard_Boolean.hxx>
#endif

#ifndef _Standard_Real_HeaderFile
#include <Standard_Real.hxx>
#endif


class OCAF_ObjectTool
{
public:
  inline void* operator new(size_t,void* anAddress)  {  return anAddress;  }
  inline void* operator new(size_t size)  { return Standard::Allocate(size); }
  inline void  operator delete(void *anAddress)  { if (anAddress) Standard::Free((Standard_Address&)anAddress); }
  
  Standard_EXPORT static TDF_Label GetEntry(const Handle(TDataStd_TreeNode)& theTreeNode);
  Standard_EXPORT static Standard_Boolean IsOneObject(const Handle(TDataStd_TreeNode)& theTreeNode);
  Standard_EXPORT static Standard_Boolean IsOneFunctionNode(const Handle(TDataStd_TreeNode)& theTreeNode);
  Standard_EXPORT static Handle(TDataStd_TreeNode) Make_ObjectNode(const TCollection_ExtendedString& theName,
								   const Handle(TDataStd_TreeNode)& AccessTreeNode,
								   TCollection_ExtendedString& theError);

  Standard_EXPORT static Handle(TDataStd_TreeNode) AddObject(const TDF_Label& theAccessLabel) ;
  Standard_EXPORT static Handle(TDataStd_TreeNode) GetObjectNode(const Handle(TDataStd_TreeNode)& theTreeNode);
  Standard_EXPORT static Handle(TDataStd_TreeNode) GetNode(const TopoDS_Shape& theShape, 
							   const TDF_Label& theAccessLabel);
  Standard_EXPORT static Standard_Boolean IsAuxiliryObject(const Handle(TDataStd_TreeNode)& theObject);
  Standard_EXPORT static void RemoveObject(const Handle(TDataStd_TreeNode)& theObjectNode);


  Standard_EXPORT static void SetName(const Handle(TDataStd_TreeNode)& theTreeNode,
				      const TCollection_ExtendedString& aName);

  Standard_EXPORT static Standard_Boolean HasName(const Handle(TDataStd_TreeNode)& theTreeNode);

  Standard_EXPORT static TCollection_ExtendedString GetName(const Handle(TDataStd_TreeNode)& theTreeNode);


  Standard_EXPORT OCAF_ObjectTool();
  Standard_EXPORT ~OCAF_ObjectTool() {}


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
