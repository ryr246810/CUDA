// File:	BRepNaming_Selection.cxx
// Created:	2010.06.07.11:17
// Author:	Wang Yue
//		<id_wangyue@hotmail.com>
// Copyright:	Wang Yue 2010 2015


#include <CAGDDefine.hxx>

#include "BRepNaming_Selection.ixx"

//#include <BRepNaming_Loader.hxx>
#include <TNaming_Builder.hxx>
#include <TDF_Label.hxx>
#include <Standard_NullObject.hxx>
#include <TopoDS_Solid.hxx>
#include <BRepLib.hxx>

//=======================================================================
//function : BRepNaming_Selection
//purpose  : 
//=======================================================================
BRepNaming_Selection::BRepNaming_Selection() {}

//=======================================================================
//function : BRepNaming_Selection
//purpose  : 
//=======================================================================
BRepNaming_Selection::BRepNaming_Selection(const TDF_Label& Label): BRepNaming_TopNaming(Label) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================

void BRepNaming_Selection::Init(const TDF_Label& Label) 
{
  if(Label.IsNull()) Standard_NullObject::Raise("BRepNaming_Selection::Init The Result label is Null ..."); 
  myResultLabel = Label;
}  


//=======================================================================
//function : Load
//purpose  : 
//=======================================================================

void BRepNaming_Selection::Load (const TopoDS_Shape& aShape, const BRepNaming_TypeOfPrimitive3D Type) const 
{
  TNaming_Builder Builder (ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel
  if (Type == BRepNaming_SELECTION){
    Builder.Generated ( aShape );
  }
}
