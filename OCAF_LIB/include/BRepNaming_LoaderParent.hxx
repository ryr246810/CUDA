
#ifndef _BRepNaming_LoaderParent_HeaderFile
#define _BRepNaming_LoaderParent_HeaderFile

#ifndef _TopAbs_ShapeEnum_HeaderFile
#include <TopAbs_ShapeEnum.hxx>
#endif
#ifndef _Standard_Boolean_HeaderFile
#include <Standard_Boolean.hxx>
#endif

#include <BRepBuilderAPI_MakeShape.hxx>
#include <TopoDS_Shape.hxx>
#include <TNaming_Builder.hxx>
#include <TDF_Label.hxx>
#include <TopTools_DataMapOfShapeShape.hxx>
#include <TopTools_MapOfShape.hxx>


#ifndef _Standard_HeaderFile
#include <Standard.hxx>
#endif
#ifndef _Standard_Macro_HeaderFile
#include <Standard_Macro.hxx>
#endif

class BRepNaming_LoaderParent  {

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
  
  Standard_EXPORT static  void LoadGeneratedDangleShapes(const TopoDS_Shape& ShapeIn,const TopAbs_ShapeEnum GeneratedFrom,TNaming_Builder& GenBuider) ;
  Standard_EXPORT static  Standard_Boolean GetDangleShapes(const TopoDS_Shape& ShapeIn,const TopAbs_ShapeEnum GeneratedFrom,TopTools_DataMapOfShapeShape& Dangles) ;
  
  




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
