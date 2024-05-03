// File:	BRepNaming_Parabola.cxx

#include <CAGDDefine.hxx>

#include "BRepNaming_Parabola.ixx"
#include <TNaming_Builder.hxx>
#include <Standard_NullObject.hxx>
#include <TopoDS_Solid.hxx>
#ifdef MDTV_DEB
#include <TDataStd_Name.hxx>
#endif

//=======================================================================
//function : BRepNaming_Parabola
//purpose  : 
//=======================================================================

BRepNaming_Parabola::BRepNaming_Parabola() {}

//=======================================================================
//function : BRepNaming_Parabola
//purpose  : 
//=======================================================================

BRepNaming_Parabola::BRepNaming_Parabola(const TDF_Label& ResultLabel):
  BRepNaming_TopNaming(ResultLabel) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================

void BRepNaming_Parabola::Init(const TDF_Label& ResultLabel) {
  if(ResultLabel.IsNull())
    Standard_NullObject::Raise("BRepNaming_Parabola::Init The Result label is Null ..."); 
  myResultLabel = ResultLabel;
}    


//=======================================================================
//function : Load (Parabola)
//purpose  : 
//=======================================================================

void BRepNaming_Parabola::Load (TopoDS_Shape& aShape, const BRepNaming_TypeOfPrimitive3D Type) const
{
  TNaming_Builder Builder (ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel
  if (Type == BRepNaming_EDGE){
    Builder.Generated ( aShape );
  }
}

