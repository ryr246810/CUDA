// File:	BRepNaming_Ellipse.cxx

#include <CAGDDefine.hxx>

#include "BRepNaming_Ellipse.ixx"
#include <TNaming_Builder.hxx>
#include <Standard_NullObject.hxx>
#include <TopoDS_Solid.hxx>
#ifdef MDTV_DEB
#include <TDataStd_Name.hxx>
#endif

//=======================================================================
//function : BRepNaming_Ellipse
//purpose  : 
//=======================================================================

BRepNaming_Ellipse::BRepNaming_Ellipse() {}

//=======================================================================
//function : BRepNaming_Ellipse
//purpose  : 
//=======================================================================

BRepNaming_Ellipse::BRepNaming_Ellipse(const TDF_Label& ResultLabel):
  BRepNaming_TopNaming(ResultLabel) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================

void BRepNaming_Ellipse::Init(const TDF_Label& ResultLabel) {
  if(ResultLabel.IsNull())
    Standard_NullObject::Raise("BRepNaming_Ellipse::Init The Result label is Null ..."); 
  myResultLabel = ResultLabel;
}    


//=======================================================================
//function : Load (Ellipse)
//purpose  : 
//=======================================================================

void BRepNaming_Ellipse::Load (TopoDS_Shape& aShape, const BRepNaming_TypeOfPrimitive3D Type) const
{
  TNaming_Builder Builder (ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel
  if (Type == BRepNaming_EDGE){
    Builder.Generated ( aShape );
  }
}

