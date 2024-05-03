#ifndef _BRepNaming_Loader_HeaderFile
#define _BRepNaming_Loader_HeaderFile

#ifndef _BRepNaming_LoaderParent_HeaderFile
#include <BRepNaming_LoaderParent.hxx>
#endif
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

class BRepNaming_Loader  : public BRepNaming_LoaderParent {

public:

    void* operator new(size_t,void* anAddress) 
      {
        return anAddress;
      }
    void* operator new(size_t size) 
      { 
        return Standard::Allocate(size); 
      }
    void  operator delete(void *anAddress) 
      { 
        if (anAddress) Standard::Free((Standard_Address&)anAddress); 
      }
 // Methods PUBLIC
 // 
Standard_EXPORT static  void LoadGeneratedShapes(BRepBuilderAPI_MakeShape& MakeShape,const TopoDS_Shape& ShapeIn,const TopAbs_ShapeEnum GeneratedFrom,TNaming_Builder& Buider) ;
Standard_EXPORT static  void LoadSeparatelyGeneratedShapes(BRepBuilderAPI_MakeShape& MakeShape,const TopoDS_Shape& ShapeIn,const TopAbs_ShapeEnum GeneratedFrom,const TDF_Label& theFatherlabel) ;
Standard_EXPORT static  void LoadModifiedShapes(BRepBuilderAPI_MakeShape& MakeShape,const TopoDS_Shape& ShapeIn,const TopAbs_ShapeEnum ModifiedFrom,TNaming_Builder& Buider,const Standard_Boolean theBool = Standard_False) ;
Standard_EXPORT static  void LoadSeparatelyModifiedShapes(BRepBuilderAPI_MakeShape& MakeShape,const TopoDS_Shape& ShapeIn,const TopAbs_ShapeEnum ModifiedFrom,const TDF_Label& theFatherlabel) ;
Standard_EXPORT static  void LoadDeletedShapes(BRepBuilderAPI_MakeShape& MakeShape,const TopoDS_Shape& ShapeIn,const TopAbs_ShapeEnum KindOfDeletedShape,TNaming_Builder& Buider) ;
Standard_EXPORT static  void LoadSeparatelyDeletedShapes(BRepBuilderAPI_MakeShape& MakeShape,const TopoDS_Shape& ShapeIn,const TopAbs_ShapeEnum KindOfDeletedShape,const TDF_Label& theFatherlabel) ;
Standard_EXPORT static  void LoadAndOrientGeneratedShapes(BRepBuilderAPI_MakeShape& MakeShape,const TopoDS_Shape& ShapeIn,const TopAbs_ShapeEnum GeneratedFrom,TNaming_Builder& Buider,const TopTools_DataMapOfShapeShape& SubShapesOfResult) ;
Standard_EXPORT static  void LoadAndOrientModifiedShapes(BRepBuilderAPI_MakeShape& MakeShape,const TopoDS_Shape& ShapeIn,const TopAbs_ShapeEnum ModifiedFrom,TNaming_Builder& Buider,const TopTools_DataMapOfShapeShape& SubShapesOfResult) ;
Standard_EXPORT static  void ModifyPart(const TopoDS_Shape& PartShape,const TopoDS_Shape& Primitive,const TDF_Label& Label) ;
Standard_EXPORT static  Standard_Boolean HasDangleShapes(const TopoDS_Shape& ShapeIn) ;
Standard_EXPORT static  void LoadGeneratedDangleShapes(const TopoDS_Shape& ShapeIn,const TopAbs_ShapeEnum GeneratedFrom,TNaming_Builder& GenBuider) ;
Standard_EXPORT static  void LoadGeneratedDangleShapes(const TopoDS_Shape& ShapeIn,const TopAbs_ShapeEnum GeneratedFrom,const TopTools_MapOfShape& OnlyThese,TNaming_Builder& GenBuider) ;
Standard_EXPORT static  void LoadModifiedDangleShapes(BRepBuilderAPI_MakeShape& MakeShape,const TopoDS_Shape& ShapeIn,const TopAbs_ShapeEnum GeneratedFrom,TNaming_Builder& GenBuider) ;
Standard_EXPORT static  void LoadDeletedDangleShapes(BRepBuilderAPI_MakeShape& MakeShape,const TopoDS_Shape& ShapeIn,const TopAbs_ShapeEnum ShapeType,TNaming_Builder& DelBuider) ;
Standard_EXPORT static  void LoadDangleShapes(const TopoDS_Shape& theShape,const TDF_Label& theLabelGenerator) ;
Standard_EXPORT static  void LoadDangleShapes(const TopoDS_Shape& theShape,const TopoDS_Shape& ignoredShape,const TDF_Label& theLabelGenerator) ;
Standard_EXPORT static  Standard_Boolean GetDangleShapes(const TopoDS_Shape& ShapeIn,const TopAbs_ShapeEnum GeneratedFrom,TopTools_DataMapOfShapeShape& Dangles) ;
Standard_EXPORT static  Standard_Boolean GetDangleShapes(const TopoDS_Shape& ShapeIn,const TopAbs_ShapeEnum GeneratedFrom,TopTools_MapOfShape& Dangles) ;
Standard_EXPORT static  Standard_Boolean IsDangle(const TopoDS_Shape& theDangle,const TopoDS_Shape& theShape) ;





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





// other Inline functions and methods (like "C++: function call" methods)
//


#endif
