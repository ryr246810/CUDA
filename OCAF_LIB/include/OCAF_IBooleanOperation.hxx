
#ifndef _OCAF_IBooleanOperation_HeaderFile
#define _OCAF_IBooleanOperation_HeaderFile

#ifndef _Standard_HeaderFile
#include <Standard.hxx>
#endif

#ifndef _TDataStd_TreeNode_HeaderFile
#include <TDataStd_TreeNode.hxx>
#endif
#ifndef _TopoDS_Shape_HeaderFile
#include <TopoDS_Shape.hxx>
#endif


#ifndef _OCAF_IFunction_HeaderFile
#include <OCAF_IFunction.hxx>
#endif
#ifndef _TDataStd_TreeNode_HeaderFile
#include <TDataStd_TreeNode.hxx>
#endif
#ifndef _Standard_Boolean_HeaderFile
#include <Standard_Boolean.hxx>
#endif

#include <Tags.hxx>


class OCAF_IBooleanOperation : public OCAF_IFunction {

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
  Standard_EXPORT OCAF_IBooleanOperation(const Handle(TDataStd_TreeNode)& aTreeNode);

  Standard_EXPORT   void                      SetObject1(Handle(TDataStd_TreeNode) theRef) { SetReference(BOOL_ARG_SHAPE1, theRef); }
  Standard_EXPORT   void                      SetObject2(Handle(TDataStd_TreeNode) theRef) { SetReference(BOOL_ARG_SHAPE2, theRef); }

  Standard_EXPORT   TopoDS_Shape              GetObject1() { return GetReference(BOOL_ARG_SHAPE1); }
  Standard_EXPORT   TopoDS_Shape              GetObject2() { return GetReference(BOOL_ARG_SHAPE2); }

  Standard_EXPORT   Handle(TDataStd_TreeNode) GetObject1Node() { return GetReferenceNode(BOOL_ARG_SHAPE1); }
  Standard_EXPORT   Handle(TDataStd_TreeNode) GetObject2Node() { return GetReferenceNode(BOOL_ARG_SHAPE2); }


  Standard_EXPORT   void                        SetMaskOfDetectSelfIntersections(int theParam) { SetInteger(BOOL_ARG_SELFINTERSECTION, theParam); };
  Standard_EXPORT   Standard_Integer            GetMaskOfDetectSelfIntersections()             { return GetInteger(BOOL_ARG_SELFINTERSECTION); };

  Standard_EXPORT   void                        SetMaskOfRemoveExtraEdges(int theParam) { SetInteger(BOOL_ARG_REMOVEEXTRAEDGES, theParam); };
  Standard_EXPORT   Standard_Integer            GetMaskOfRemoveExtraEdges()             { return GetInteger(BOOL_ARG_REMOVEEXTRAEDGES); };

  Standard_EXPORT   void                        SetTolerance(Standard_Real theParam) { SetReal(BOOL_ARG_TOLERANCE, theParam); };
  Standard_EXPORT   Standard_Real               GetTolerance()                       { return GetReal(BOOL_ARG_TOLERANCE); };


  Standard_EXPORT ~OCAF_IBooleanOperation() {}


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
