// File:	BRepNaming_Solid.cxx
// Created:	2010.04.02.11:17
// Author:	Wang Yue
//		<id_wangyue@hotmail.com>
// Copyright:	Wang Yue 2010 2015


#include <CAGDDefine.hxx>

#include "BRepNaming_Solid.ixx"

//#include <BRepNaming_Loader.hxx>
#include <TNaming_Builder.hxx>
#include <TDF_Label.hxx>
#include <Standard_NullObject.hxx>
#include <TopoDS_Solid.hxx>
#include <BRepLib.hxx>

//=======================================================================
//function : BRepNaming_Solid
//purpose  : 
//=======================================================================
BRepNaming_Solid::BRepNaming_Solid() {}

//=======================================================================
//function : BRepNaming_Solid
//purpose  : 
//=======================================================================
BRepNaming_Solid::BRepNaming_Solid(const TDF_Label& Label): BRepNaming_TopNaming(Label) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================

void BRepNaming_Solid::Init(const TDF_Label& Label) 
{
  if(Label.IsNull()) Standard_NullObject::Raise("BRepNaming_Solid::Init The Result label is Null ..."); 
  myResultLabel = Label;
}  


//=======================================================================
//function : Load
//purpose  : 
//=======================================================================

void BRepNaming_Solid::Load(TopoDS_Solid& aSolid, const BRepNaming_TypeOfPrimitive3D Type) const 
{
  TNaming_Builder Builder (ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel
  if (Type == BRepNaming_SOLID){
    Builder.Generated (aSolid);
  }
}
