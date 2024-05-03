// File:	BRepNaming_Line.cxx
// Created:	2010.04.02.11:17
// Author:	Wang Yue
//		<id_wangyue@hotmail.com>
// Copyright:	Wang Yue 2010 2015


#include <CAGDDefine.hxx>

#include "BRepNaming_Shell.ixx"

//#include <BRepNaming_Loader.hxx>
#include <TNaming_Builder.hxx>
#include <TDF_Label.hxx>
#include <Standard_NullObject.hxx>
#include <TopoDS_Solid.hxx>
#include <BRepLib.hxx>

//=======================================================================
//function : BRepNaming_Shell
//purpose  : 
//=======================================================================
BRepNaming_Shell::BRepNaming_Shell() {}

//=======================================================================
//function : BRepNaming_Shell
//purpose  : 
//=======================================================================
BRepNaming_Shell::BRepNaming_Shell(const TDF_Label& Label): BRepNaming_TopNaming(Label) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================

void BRepNaming_Shell::Init(const TDF_Label& Label) 
{
  if(Label.IsNull()) Standard_NullObject::Raise("BRepNaming_Shell::Init The Result label is Null ..."); 
  myResultLabel = Label;
}  


//=======================================================================
//function : Load
//purpose  : 
//=======================================================================

void BRepNaming_Shell::Load (TopoDS_Shell& aShell, const BRepNaming_TypeOfPrimitive3D Type) const 
{
  TNaming_Builder Builder (ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel
  if (Type == BRepNaming_SHELL){
    Builder.Generated (aShell);
  }
}
