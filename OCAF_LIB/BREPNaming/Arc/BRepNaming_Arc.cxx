// File:	BRepNaming_Arc.cxx

#include <CAGDDefine.hxx>

#include "BRepNaming_Arc.ixx"
#include <TNaming_Builder.hxx>
#include <Standard_NullObject.hxx>
#include <TopoDS_Solid.hxx>
#ifdef MDTV_DEB
#include <TDataStd_Name.hxx>
#endif

//=======================================================================
//function : BRepNaming_Arc
//purpose  : 
//=======================================================================

BRepNaming_Arc::BRepNaming_Arc() {}

//=======================================================================
//function : BRepNaming_Arc
//purpose  : 
//=======================================================================

BRepNaming_Arc::BRepNaming_Arc(const TDF_Label& ResultLabel):
  BRepNaming_TopNaming(ResultLabel) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================

void BRepNaming_Arc::Init(const TDF_Label& ResultLabel) {
  if(ResultLabel.IsNull())
    Standard_NullObject::Raise("BRepNaming_Arc::Init The Result label is Null ..."); 
  myResultLabel = ResultLabel;
}    


//=======================================================================
//function : Load (Arc)
//purpose  : 
//=======================================================================

void BRepNaming_Arc::Load (TopoDS_Shape& aShape, const BRepNaming_TypeOfPrimitive3D Type) const
{
  TNaming_Builder Builder (ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel
  if (Type == BRepNaming_EDGE){
    Builder.Generated ( aShape );
  }
}

