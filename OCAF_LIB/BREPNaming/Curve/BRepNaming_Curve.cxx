// File:	BRepNaming_Curve.cxx

#include <CAGDDefine.hxx>

#include "BRepNaming_Curve.ixx"

#include <TNaming_Builder.hxx>
#include <Standard_NullObject.hxx>
#include <TopoDS_Solid.hxx>
#ifdef MDTV_DEB
#include <TDataStd_Name.hxx>
#endif

//=======================================================================
//function : BRepNaming_Curve
//purpose  : 
//=======================================================================

BRepNaming_Curve::BRepNaming_Curve() {}

//=======================================================================
//function : BRepNaming_Curve
//purpose  : 
//=======================================================================

BRepNaming_Curve::BRepNaming_Curve(const TDF_Label& ResultLabel):
  BRepNaming_TopNaming(ResultLabel) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================

void BRepNaming_Curve::Init(const TDF_Label& ResultLabel) {
  if(ResultLabel.IsNull())
    Standard_NullObject::Raise("BRepNaming_Curve::Init The Result label is Null ..."); 
  myResultLabel = ResultLabel;
}    


//=======================================================================
//function : Load (Curve)
//purpose  : 
//=======================================================================

void BRepNaming_Curve::Load (TopoDS_Shape& aShape, const BRepNaming_TypeOfPrimitive3D Type) const
{
  TNaming_Builder Builder (ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel
  if (Type == BRepNaming_EDGE){
    Builder.Generated ( aShape );
  }
}

