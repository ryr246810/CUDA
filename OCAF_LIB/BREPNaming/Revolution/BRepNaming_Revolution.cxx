// File:	BRepNaming_Revolution.cxx
// Created:	2010.03.19.11:17
// Author:	Wang Yue
//		<id_wangyue@hotmail.com>
// Copyright:	Wang Yue 2010


#include <CAGDDefine.hxx>

#include "BRepNaming_Revolution.ixx"

//#include <BRepNaming_Loader.hxx>
#include <TNaming_Builder.hxx>
#include <TDF_Label.hxx>
#include <Standard_NullObject.hxx>
#include <TopoDS_Solid.hxx>
#include <BRepLib.hxx>

//=======================================================================
//function : BRepNaming_Revolution
//purpose  : 
//=======================================================================
BRepNaming_Revolution::BRepNaming_Revolution() {}

//=======================================================================
//function : BRepNaming_Revolution
//purpose  : 
//=======================================================================
BRepNaming_Revolution::BRepNaming_Revolution(const TDF_Label& Label): BRepNaming_TopNaming(Label) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================

void BRepNaming_Revolution::Init(const TDF_Label& Label) {
  if(Label.IsNull()) Standard_NullObject::Raise("BRepNaming_Revolution::Init The Result label is Null ..."); 
  myResultLabel = Label;
}  


//=======================================================================
//function : Load
//purpose  : 
//=======================================================================

void BRepNaming_Revolution::Load (TopoDS_Shape& aShape, const BRepNaming_TypeOfPrimitive3D Type, const Standard_Integer aType) const {

  TNaming_Builder Builder (ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel
  if (Type == BRepNaming_REVOLUTION){
    Builder.Generated ( aShape );
  }
}
