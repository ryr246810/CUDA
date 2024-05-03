
#ifndef _OCAF_ISTEPImport_HeaderFile
#define _OCAF_ISTEPImport_HeaderFile

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

#ifndef _TopTools_IndexedMapOfShape_HeaderFile
#include <TopTools_IndexedMapOfShape.hxx>
#endif

#ifndef _Standard_Transient_HeaderFile
#include <Standard_Transient.hxx>
#endif

#ifndef _Transfer_TransientProcess_HeaderFile
#include <Transfer_TransientProcess.hxx>
#endif



class OCAF_ISTEPImport : public OCAF_IFunction 
{
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
  
  Standard_EXPORT OCAF_ISTEPImport(const Handle(TDataStd_TreeNode)& aTreeNode);
  


  Standard_EXPORT ~OCAF_ISTEPImport() {}
  
  
public:
  Standard_EXPORT static void StoreName( const Handle(Standard_Transient)        &theEnti,
					 const TopTools_IndexedMapOfShape        &theIndices,
					 const Handle(Transfer_TransientProcess) &theTP,
					 const TDF_Label                         &theShapeLabel);

  Standard_EXPORT static void AppendFile(const TCollection_AsciiString& aNameFile, 
					 const TopoDS_Shape& theShape, 
					 const TDF_Label& theAccessLabel);
  
  Standard_EXPORT static void CheckFile(const TCollection_AsciiString& aFileName, 
					const TopoDS_Shape& aRefShape, 
					const TDF_Label& theAccessLabel, 
					Standard_Integer& theShapeIndex);
  
  Standard_EXPORT static void ReadFile_1(const Standard_CString& theName, 
					 const Standard_Boolean anIsIgnoreUnits,
					 const TDF_Label& theAccessLabel);
  
  
  Standard_EXPORT static void ReadFile_2(const Standard_CString& theName, 
					 const Standard_Boolean anIsIgnoreUnits,
					 const TDF_Label& theAccessLabel);

  Standard_EXPORT   Standard_Boolean SaveFile(const Standard_CString& theName) ;
};





// other inline functions and methods (like "C++: function call" methods)
//


#endif
