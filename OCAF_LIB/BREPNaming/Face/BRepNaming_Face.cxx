// File:	BRepNaming_Line.cxx
// Created:	2010.04.02.11:17
// Author:	Wang Yue
//		<id_wangyue@hotmail.com>
// Copyright:	Wang Yue 2010 2015


#include <CAGDDefine.hxx>

#include "BRepNaming_Face.ixx"

//#include <BRepNaming_Loader.hxx>
#include <TNaming_Builder.hxx>
#include <TDF_Label.hxx>
#include <Standard_NullObject.hxx>
#include <TopoDS_Solid.hxx>
#include <BRepLib.hxx>

//=======================================================================
//function : BRepNaming_Face
//purpose  : 
//=======================================================================
BRepNaming_Face::BRepNaming_Face() {}

//=======================================================================
//function : BRepNaming_Face
//purpose  : 
//=======================================================================
BRepNaming_Face::BRepNaming_Face(const TDF_Label& Label): BRepNaming_TopNaming(Label) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================

void BRepNaming_Face::Init(const TDF_Label& Label) 
{
  if(Label.IsNull()) Standard_NullObject::Raise("BRepNaming_Face::Init The Result label is Null ..."); 
  myResultLabel = Label;
}  


//=======================================================================
//function : Load
//purpose  : 
//=======================================================================
void BRepNaming_Face::Load (TopoDS_Face& aFace, const BRepNaming_TypeOfPrimitive3D Type) const 
{
  TNaming_Builder Builder (ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel
  if (Type == BRepNaming_FACE){
    Builder.Generated (aFace);
  }
}
