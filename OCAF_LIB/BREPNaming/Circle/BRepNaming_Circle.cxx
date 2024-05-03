// File:	BRepNaming_Circle.cxx

#include <CAGDDefine.hxx>

#include "BRepNaming_Circle.ixx"
#include <TNaming_Builder.hxx>
#include <Standard_NullObject.hxx>
#include <TopoDS_Solid.hxx>
#ifdef MDTV_DEB
#include <TDataStd_Name.hxx>
#endif

//=======================================================================
//function : BRepNaming_Circle
//purpose  : 
//=======================================================================

BRepNaming_Circle::BRepNaming_Circle() {}

//=======================================================================
//function : BRepNaming_Circle
//purpose  : 
//=======================================================================

BRepNaming_Circle::BRepNaming_Circle(const TDF_Label& ResultLabel):
  BRepNaming_TopNaming(ResultLabel) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================

void BRepNaming_Circle::Init(const TDF_Label& ResultLabel) {
  if(ResultLabel.IsNull())
    Standard_NullObject::Raise("BRepNaming_Circle::Init The Result label is Null ..."); 
  myResultLabel = ResultLabel;
}    


//=======================================================================
//function : Load (Circle)
//purpose  : 
//=======================================================================

void BRepNaming_Circle::Load (TopoDS_Shape& aShape, const BRepNaming_TypeOfPrimitive3D Type) const
{
  TNaming_Builder Builder (ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel
  if (Type == BRepNaming_EDGE){
    Builder.Generated ( aShape );
  }
}

