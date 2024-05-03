// File:	BRepNaming_Torus.cxx

#include <CAGDDefine.hxx>

#include "BRepNaming_Torus.ixx"
#include <TNaming_Builder.hxx>
#include <Standard_NullObject.hxx>
#include <TopoDS_Solid.hxx>
#ifdef MDTV_DEB
#include <TDataStd_Name.hxx>
#endif

//=======================================================================
//function : BRepNaming_Torus
//purpose  : 
//=======================================================================

BRepNaming_Torus::BRepNaming_Torus() {}

//=======================================================================
//function : BRepNaming_Torus
//purpose  : 
//=======================================================================

BRepNaming_Torus::BRepNaming_Torus(const TDF_Label& ResultLabel):
  BRepNaming_TopNaming(ResultLabel) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================

void BRepNaming_Torus::Init(const TDF_Label& ResultLabel) {
  if(ResultLabel.IsNull())
    Standard_NullObject::Raise("BRepNaming_Torus::Init The Result label is Null ..."); 
  myResultLabel = ResultLabel;
}    


//=======================================================================
//function : Load (Torus)
//purpose  : 
//=======================================================================

void BRepNaming_Torus::Load (TopoDS_Shape& aShape, const BRepNaming_TypeOfPrimitive3D Type) const
{
  TNaming_Builder Builder (ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel
  if (Type == BRepNaming_SOLID){
    Builder.Generated ( aShape );
  }
}

