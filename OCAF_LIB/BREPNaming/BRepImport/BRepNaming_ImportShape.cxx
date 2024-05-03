
#include "BRepNaming_ImportShape.ixx"

#include <Standard_NullObject.hxx>
#include <BRepTools.hxx>

#include <TopoDS.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Wire.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Iterator.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>

#include <TopTools_ListOfShape.hxx>
#include <TopTools_ListIteratorOfListOfShape.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopTools_DataMapOfShapeShape.hxx>
#include <TopTools_DataMapOfShapeListOfShape.hxx>
#include <TopTools_DataMapIteratorOfDataMapOfShapeListOfShape.hxx>
#include <ShapeExtend_WireData.hxx>

#include <TDF_Label.hxx>
#include <TDF_LabelMap.hxx>
#include <TDF_TagSource.hxx>
#include <TDF_ChildIterator.hxx>
#include <TNaming_Builder.hxx>
#include <TNaming_NamedShape.hxx>
#include <BRepNaming_LoaderParent.hxx>

//=======================================================================
//function : BRepNaming_ImportShape
//purpose  : Constructor
//=======================================================================

BRepNaming_ImportShape::BRepNaming_ImportShape() {}

//=======================================================================
//function : BRepNaming_ImportShape
//purpose  : Constructor
//=======================================================================

BRepNaming_ImportShape::BRepNaming_ImportShape(const TDF_Label& L):BRepNaming_TopNaming(L) {}

//=======================================================================
//function : Init
//purpose  : Initialization
//=======================================================================

void BRepNaming_ImportShape::Init(const TDF_Label& Label) {
  if(Label.IsNull()) 
    Standard_NullObject::Raise("BRepNaming_ImportShape::Init The Result label is Null ..."); 
  myResultLabel = Label;
}  

//=======================================================================
//function : Load
//purpose  : To load an ImportShape
//           Use this method for a topological naming of an imported shape
//=======================================================================

void BRepNaming_ImportShape::Load(const TopoDS_Shape& theShape) const {
  ResultLabel().ForgetAllAttributes();
  TNaming_Builder b(ResultLabel());
  b.Generated(theShape);
}
