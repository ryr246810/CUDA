// File:	BRepNaming_RecPeriodEdge.cxx
// Created:	2014.08.27.20:53
// Author:	Wang Yue
//		<id_wangyue@hotmail.com>
// Copyright:	Wang Yue 2014


#include <CAGDDefine.hxx>

#include "BRepNaming_RecPeriodEdge.ixx"

//#include <BRepNaming_Loader.hxx>
#include <TNaming_Builder.hxx>
#include <TDF_Label.hxx>
#include <Standard_NullObject.hxx>
#include <TopoDS_Solid.hxx>
#include <BRepLib.hxx>

//=======================================================================
//function : BRepNaming_RecPeriodEdge
//purpose  : 
//=======================================================================
BRepNaming_RecPeriodEdge::BRepNaming_RecPeriodEdge() {}

//=======================================================================
//function : BRepNaming_RecPeriodEdge
//purpose  : 
//=======================================================================
BRepNaming_RecPeriodEdge::BRepNaming_RecPeriodEdge(const TDF_Label& Label): BRepNaming_TopNaming(Label) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================

void BRepNaming_RecPeriodEdge::Init(const TDF_Label& Label) {
  if(Label.IsNull()) Standard_NullObject::Raise("BRepNaming_RecPeriodEdge::Init The Result label is Null ..."); 
  myResultLabel = Label;
}  


//=======================================================================
//function : Load
//purpose  : 
//=======================================================================

void BRepNaming_RecPeriodEdge::Load (TopoDS_Shape& aShape, const BRepNaming_TypeOfPrimitive3D Type) const {

  TNaming_Builder Builder (ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel
  if (Type == BRepNaming_RECPERIODEDGE){
    Builder.Generated (aShape);
  }
}