
#ifndef _OCAF_IBRepImport_HeaderFile
#define _OCAF_IBRepImport_HeaderFile

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


class OCAF_IBRepImport : public OCAF_IFunction {

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
  Standard_EXPORT static  Handle(TDataStd_TreeNode) AddShape(const TCollection_ExtendedString& theName,
							  const TopoDS_Shape& theShape,
							  const Handle(TDataStd_TreeNode)& AccessTreeNode,
							  TCollection_ExtendedString& theError) ;

Standard_EXPORT OCAF_IBRepImport(const Handle(TDataStd_TreeNode)& aTreeNode);

  Standard_EXPORT static  Handle(TDataStd_TreeNode) ReadFile(const Standard_CString& theName, 
							  const TDF_Label& theAccessLabel) ;

Standard_EXPORT   Standard_Boolean SaveFile(const Standard_CString& theName) ;
Standard_EXPORT ~OCAF_IBRepImport() {}




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
