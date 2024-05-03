// File:	BRepNaming_MultiFuse.cxx
// Created:	2010.03.19.11:17
// Author:	Wang Yue
//		<id_wangyue@hotmail.com>
// Copyright:	Wang Yue 2010


#include <CAGDDefine.hxx>

#include "BRepNaming_MultiFuse.ixx"

//#include <BRepNaming_Loader.hxx>
#include <TNaming_Builder.hxx>
#include <TDF_Label.hxx>
#include <Standard_NullObject.hxx>
#include <TopoDS_Solid.hxx>
#include <BRepLib.hxx>

#include <GEOMUtils.hxx>

//=======================================================================
//function : BRepNaming_MultiFuse
//purpose  : 
//=======================================================================
BRepNaming_MultiFuse::BRepNaming_MultiFuse() {}

//=======================================================================
//function : BRepNaming_MultiFuse
//purpose  : 
//=======================================================================
BRepNaming_MultiFuse::BRepNaming_MultiFuse(const TDF_Label& Label): BRepNaming_TopNaming(Label) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================
void BRepNaming_MultiFuse::Init(const TDF_Label& Label) 
{
  if(Label.IsNull()) Standard_NullObject::Raise("BRepNaming_MultiFuse::Init The Result label is Null ..."); 
  myResultLabel = Label;
}  

//=======================================================================
//function : Load
//purpose  : 
//=======================================================================
void BRepNaming_MultiFuse::Load (TopoDS_Shape& aShape, const BRepNaming_TypeOfPrimitive3D Type) const
{
  TNaming_Builder Builder (ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel
  if (Type == BRepNaming_MULTIFUSE){
    Builder.Generated ( aShape );
  }
}
