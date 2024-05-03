
#ifndef _OCAF_ISphere_HeaderFile
#define _OCAF_ISphere_HeaderFile

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

#ifndef _OCAF_IFunction_HeaderFile
#include <OCAF_IFunction.hxx>
#endif

#ifndef _Standard_Real_HeaderFile
#include <Standard_Real.hxx>
#endif


class OCAF_ISphere : public OCAF_IFunction {

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

  Standard_EXPORT static  Handle(TDataStd_TreeNode) MakeSphere_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
									     Standard_Integer theType,
									     TCollection_ExtendedString& theError);
  
  
  Standard_EXPORT   Standard_Boolean MakeSphere_Execute(  TCollection_ExtendedString& theError);


  
  Standard_EXPORT   void                      SetPoint(const Handle(TDataStd_TreeNode)& theRef) { SetReference(SPHERE_ARG_CENTER, theRef); };
  Standard_EXPORT   Handle(TDataStd_TreeNode) GetPointNode()    { return GetReferenceNode(SPHERE_ARG_CENTER); };
  Standard_EXPORT   TopoDS_Shape              GetPoint()    { return GetReference(SPHERE_ARG_CENTER); };

  Standard_EXPORT   void                      SetR(double theParam) { SetReal(SPHERE_ARG_RADIUS, theParam); };
  Standard_EXPORT   Standard_Real             GetR() { return GetReal(SPHERE_ARG_RADIUS); };

  Standard_EXPORT OCAF_ISphere(const Handle(TDataStd_TreeNode)& aTreeNode);
  Standard_EXPORT ~OCAF_ISphere() {}


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
