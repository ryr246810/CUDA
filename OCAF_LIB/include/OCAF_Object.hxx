
#ifndef _OCAF_Object_HeaderFile
#define _OCAF_Object_HeaderFile


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


class OCAF_Object
{
public:
  inline void* operator new(size_t,void* anAddress)  {  return anAddress;  }
  inline void* operator new(size_t size)  { return Standard::Allocate(size); }
  inline void  operator delete(void *anAddress)  { if (anAddress) Standard::Free((Standard_Address&)anAddress); }
  

  Standard_EXPORT static const Standard_GUID& GetObjectID();
  

  // Methods PUBLIC
  Standard_EXPORT   void SetName(const TCollection_ExtendedString& aName); // ok
  Standard_EXPORT   Standard_Boolean HasName(); //ok
  Standard_EXPORT   TCollection_ExtendedString GetName() const; //ok


  Standard_EXPORT   TDF_Label GetEntry() const;  // ok

  Standard_EXPORT   void SetObjectMask(Standard_Integer theMask);  // ok
  Standard_EXPORT   Standard_Integer GetObjectMask() const;  // ok

  Standard_EXPORT   Standard_Boolean SetObjResultMaterial(Standard_Integer theMaterial);  // ok
  Standard_EXPORT   Standard_Integer GetObjResultMaterial() const;  // ok
  Standard_EXPORT   Handle(TPrsStd_AISPresentation) GetPresentation( ) const;  // ok

  Standard_EXPORT   Handle(TDataStd_TreeNode) AddFunction(const Standard_GUID& theID);  // ok
  Standard_EXPORT   Handle(TDataStd_TreeNode) GetLastFunction() const;   // ok
  Standard_EXPORT   TopoDS_Shape GetObjectValue() const;   // ok


  Standard_EXPORT   Standard_Boolean CanRemove(); // ok
  Standard_EXPORT   Standard_Boolean Remove(); //ok
  Standard_EXPORT   Standard_Boolean HasReferencedObjects() const; //ok


  Standard_EXPORT OCAF_Object(const Handle(TDataStd_TreeNode)& aTreeNode);
  Standard_EXPORT ~OCAF_Object() {}

protected:
  
  // Methods PROTECTED
  // 


  // Fields PROTECTED
  //
  Handle(TDataStd_TreeNode) myTreeNode;

private: 
  
  // Methods PRIVATE
  // 


  // Fields PRIVATE
  //

};


#endif
