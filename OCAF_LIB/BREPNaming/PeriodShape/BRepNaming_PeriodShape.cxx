// File:	BRepNaming_PeriodShape.cxx
// Created:	2010.07.22
// Author:	Wang Yue
//		<id_wangyue@hotmail.com>
// Copyright:	Wang Yue 2010


#include <CAGDDefine.hxx>

#include "BRepNaming_PeriodShape.ixx"

//#include <BRepNaming_Loader.hxx>
#include <TNaming_Builder.hxx>
#include <TDF_Label.hxx>
#include <Standard_NullObject.hxx>
#include <TopoDS_Solid.hxx>
#include <BRepLib.hxx>

//=======================================================================
//function : BRepNaming_PeriodShape
//purpose  : 
//=======================================================================
BRepNaming_PeriodShape::BRepNaming_PeriodShape() {}

//=======================================================================
//function : BRepNaming_PeriodShape
//purpose  : 
//=======================================================================
BRepNaming_PeriodShape::BRepNaming_PeriodShape(const TDF_Label& Label): BRepNaming_TopNaming(Label) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================
void BRepNaming_PeriodShape::Init(const TDF_Label& Label) {
  if(Label.IsNull()) Standard_NullObject::Raise("BRepNaming_PeriodShape::Init The Result label is Null ..."); 
  myResultLabel = Label;
}  

//=======================================================================
//function : Load
//purpose  : 
//=======================================================================
void BRepNaming_PeriodShape::Load(TopoDS_Shape& aShape, const BRepNaming_TypeOfPrimitive3D Type) const {

  TNaming_Builder Builder(ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel

  if (Type == BRepNaming_PERIODSHAPE){
    Builder.Generated ( aShape );
  }
}
