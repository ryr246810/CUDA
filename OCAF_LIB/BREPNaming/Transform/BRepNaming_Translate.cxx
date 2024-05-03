// File:	BRepNaming_Translate.cxx
// Created:	2010.07.22
// Author:	Wang Yue
//		<id_wangyue@hotmail.com>
// Copyright:	Wang Yue 2010


#include <CAGDDefine.hxx>

#include "BRepNaming_Translate.ixx"

//#include <BRepNaming_Loader.hxx>
#include <TNaming_Builder.hxx>
#include <TDF_Label.hxx>
#include <Standard_NullObject.hxx>
#include <TopoDS_Solid.hxx>
#include <BRepLib.hxx>

//=======================================================================
//function : BRepNaming_Translate
//purpose  : 
//=======================================================================
BRepNaming_Translate::BRepNaming_Translate() {}

//=======================================================================
//function : BRepNaming_Translate
//purpose  : 
//=======================================================================
BRepNaming_Translate::BRepNaming_Translate(const TDF_Label& Label): BRepNaming_TopNaming(Label) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================
void BRepNaming_Translate::Init(const TDF_Label& Label) {
  if(Label.IsNull()) Standard_NullObject::Raise("BRepNaming_Translate::Init The Result label is Null ..."); 
  myResultLabel = Label;
}  

//=======================================================================
//function : Load
//purpose  : 
//=======================================================================
void BRepNaming_Translate::Load (TopoDS_Shape& aShape, const BRepNaming_TypeOfPrimitive3D Type) const {

  TNaming_Builder Builder (ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel
  if (Type == BRepNaming_TRANSLATE){
    Builder.Generated ( aShape );
  }
}
