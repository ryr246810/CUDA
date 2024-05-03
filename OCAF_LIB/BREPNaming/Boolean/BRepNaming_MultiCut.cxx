
#include <CAGDDefine.hxx>

#include "BRepNaming_MultiCut.ixx"

//#include <BRepNaming_Loader.hxx>
#include <TNaming_Builder.hxx>
#include <TDF_Label.hxx>
#include <Standard_NullObject.hxx>
#include <TopoDS_Solid.hxx>
#include <BRepLib.hxx>

#include <BRepNaming_Loader.hxx>

#include <GEOMUtils.hxx>

//=======================================================================
//function : BRepNaming_MultiCut
//purpose  : 
//=======================================================================

BRepNaming_MultiCut::
BRepNaming_MultiCut() {}

//=======================================================================
//function : BRepNaming_MultiCut
//purpose  : 
//=======================================================================

BRepNaming_MultiCut::
BRepNaming_MultiCut(const TDF_Label& Label) : BRepNaming_TopNaming(Label) {}

//=======================================================================
//function : Load
//purpose  : 
//=======================================================================


//=======================================================================
//function : Init
//purpose  : 
//=======================================================================
void 
BRepNaming_MultiCut::
Init(const TDF_Label& Label) 
{
  if(Label.IsNull()) Standard_NullObject::Raise("BRepNaming_MultiCut::Init The Result label is Null ..."); 
  myResultLabel = Label;
}  


//=======================================================================
//function : Load
//purpose  : 
//=======================================================================
void 
BRepNaming_MultiCut::
Load (TopoDS_Shape& aShape, const BRepNaming_TypeOfPrimitive3D Type) const 
{
  TNaming_Builder Builder (ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel
  if (Type == BRepNaming_MULTICUT){
    Builder.Generated ( aShape );
  }
}

