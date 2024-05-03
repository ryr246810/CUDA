
#include "BRepNaming_TopNaming.ixx"

#include <TDF_Label.hxx>

#include <Standard_NullObject.hxx>

//=======================================================================
//function : BRepNaming_TopNaming
//purpose  : Constructor
//=======================================================================

BRepNaming_TopNaming::BRepNaming_TopNaming(){ }

//=======================================================================
//function : BRepNaming_TopNaming
//purpose  : Constructor
//=======================================================================

BRepNaming_TopNaming::BRepNaming_TopNaming(const TDF_Label& Label)
{
  if(Label.IsNull())   Standard_NullObject::Raise("BRepNaming_TopNaming:: The Result label is Null ..."); 
  myResultLabel = Label;
}

