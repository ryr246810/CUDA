// File:	BRepNaming_MultiRotate.cxx
// Created:	2010.07.22
// Author:	Wang Yue
//		<id_wangyue@hotmail.com>
// Copyright:	Wang Yue 2010


#include <CAGDDefine.hxx>

#include "BRepNaming_MultiRotate.ixx"

//#include <BRepNaming_Loader.hxx>
#include <TNaming_Builder.hxx>
#include <TDF_Label.hxx>
#include <Standard_NullObject.hxx>
#include <TopoDS_Solid.hxx>
#include <BRepLib.hxx>

//=======================================================================
//function : BRepNaming_MultiRotate
//purpose  : 
//=======================================================================
BRepNaming_MultiRotate::BRepNaming_MultiRotate() {}

//=======================================================================
//function : BRepNaming_MultiRotate
//purpose  : 
//=======================================================================
BRepNaming_MultiRotate::BRepNaming_MultiRotate(const TDF_Label& Label): BRepNaming_TopNaming(Label) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================
void BRepNaming_MultiRotate::Init(const TDF_Label& Label) {
  if(Label.IsNull()) Standard_NullObject::Raise("BRepNaming_MultiRotate::Init The Result label is Null ..."); 
  myResultLabel = Label;
}  

//=======================================================================
//function : Load
//purpose  : 
//=======================================================================
void BRepNaming_MultiRotate::Load(TopoDS_Shape& aShape, const BRepNaming_TypeOfPrimitive3D Type) const {

  TNaming_Builder Builder(ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel

  if (Type == BRepNaming_MULTIROTATE){
    Builder.Generated ( aShape );
  }
}
