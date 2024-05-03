
#include <CAGDDefine.hxx>

#include "BRepNaming_Box.ixx"

//#include <BRepNaming_Loader.hxx>
#include <TNaming_Builder.hxx>
#include <TDF_Label.hxx>
#include <Standard_NullObject.hxx>
#include <TopoDS_Solid.hxx>

//=======================================================================
//function : BRepNaming_Box
//purpose  : 
//=======================================================================

BRepNaming_Box::BRepNaming_Box() {}

//=======================================================================
//function : BRepNaming_Box
//purpose  : 
//=======================================================================

BRepNaming_Box::BRepNaming_Box(const TDF_Label& Label)
: BRepNaming_TopNaming(Label) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================

void BRepNaming_Box::Init(const TDF_Label& Label) {
  if(Label.IsNull())
    Standard_NullObject::Raise("BRepNaming_Box::Init The Result label is Null ..."); 
  myResultLabel = Label;
}  

//=======================================================================
//function : Load
//purpose  : 
//=======================================================================
void BRepNaming_Box::Load (TopoDS_Shape& aShape, const BRepNaming_TypeOfPrimitive3D Type) const 
{
  // add TNaming_NamedShape attribute to the ResultLabel
  TNaming_Builder Builder (ResultLabel());
  if (Type == BRepNaming_SOLID){
    Builder.Generated(aShape);
  }
}
