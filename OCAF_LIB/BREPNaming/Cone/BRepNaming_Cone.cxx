// File:	BRepNaming_Cone.cxx

#include <CAGDDefine.hxx>

#include "BRepNaming_Cone.ixx"
#include <TNaming_Builder.hxx>
#include <Standard_NullObject.hxx>
#include <TopoDS_Solid.hxx>
#ifdef MDTV_DEB
#include <TDataStd_Name.hxx>
#endif

//=======================================================================
//function : BRepNaming_Cone
//purpose  : 
//=======================================================================

BRepNaming_Cone::BRepNaming_Cone() {}

//=======================================================================
//function : BRepNaming_Cone
//purpose  : 
//=======================================================================

BRepNaming_Cone::BRepNaming_Cone(const TDF_Label& ResultLabel):
  BRepNaming_TopNaming(ResultLabel) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================

void BRepNaming_Cone::Init(const TDF_Label& ResultLabel) {
  if(ResultLabel.IsNull())
    Standard_NullObject::Raise("BRepNaming_Cone::Init The Result label is Null ..."); 
  myResultLabel = ResultLabel;
}    


//=======================================================================
//function : Load (Cone)
//purpose  : 
//=======================================================================

void BRepNaming_Cone::Load (TopoDS_Shape& aShape, const BRepNaming_TypeOfPrimitive3D Type) const
{
  TNaming_Builder Builder (ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel
  if (Type == BRepNaming_SOLID){
    Builder.Generated ( aShape );
  }
}

