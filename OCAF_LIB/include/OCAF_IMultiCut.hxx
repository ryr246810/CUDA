
#ifndef _OCAF_IMultiCut_HeaderFile
#define _OCAF_IMultiCut_HeaderFile

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

#ifndef _OCAF_IBooleanOperation_HeaderFile
#include <OCAF_IBooleanOperation.hxx>
#endif

class Standard_GUID;

class OCAF_IMultiCut : public OCAF_IBooleanOperation {

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
  
  Standard_EXPORT static  Handle(TDataStd_TreeNode) MakeMultiCut_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
									       TCollection_ExtendedString& theError);
  

  Standard_EXPORT   Standard_Boolean MakeMultiCut_Execute(  TCollection_ExtendedString& theError);



  Standard_EXPORT OCAF_IMultiCut(const Handle(TDataStd_TreeNode)& aTreeNode);
  Standard_EXPORT ~OCAF_IMultiCut() {}



  Standard_EXPORT Standard_Boolean SetCutMultiTool(Handle(TDataStd_TreeNode) theNode);

  Standard_EXPORT Standard_Boolean SetCutMultiTools(const TDF_AttributeSequence& ArgumentsSequence);
  Standard_EXPORT void             GetCutMultiTools(TDF_AttributeSequence& ArgumentsSequence);

  Standard_EXPORT TopoDS_Shape     GetCutMultiTool(Standard_Integer theNumber);

  Standard_EXPORT void             ClearCutMultiTools();



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
