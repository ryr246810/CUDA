// File:	BRepNaming_Vector.cxx
// Created:	2010.03.19.11:17
// Author:	Wang Yue
//		<id_wangyue@hotmail.com>
// Copyright:	Wang Yue 2010


#include <CAGDDefine.hxx>

#include "BRepNaming_Vector.ixx"

//#include <BRepNaming_Loader.hxx>
#include <TNaming_Builder.hxx>
#include <TDF_Label.hxx>
#include <Standard_NullObject.hxx>
#include <TopoDS_Solid.hxx>
#include <BRepLib.hxx>

//=======================================================================
//function : BRepNaming_Vector
//purpose  : 
//=======================================================================
BRepNaming_Vector::BRepNaming_Vector() {}

//=======================================================================
//function : BRepNaming_Vector
//purpose  : 
//=======================================================================
BRepNaming_Vector::BRepNaming_Vector(const TDF_Label& Label): BRepNaming_TopNaming(Label) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================

void BRepNaming_Vector::Init(const TDF_Label& Label) {
  if(Label.IsNull()) Standard_NullObject::Raise("BRepNaming_Vector::Init The Result label is Null ..."); 
  myResultLabel = Label;
}  


//=======================================================================
//function : Load
//purpose  : 
//=======================================================================

void BRepNaming_Vector::Load (BRepBuilderAPI_MakeEdge& MS, const BRepNaming_TypeOfPrimitive3D Type, const Standard_Integer aType) const {
  TNaming_Builder Builder (ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel
  if (Type == BRepNaming_EDGE){
    Builder.Generated (MS.Edge());
  }
  else {
#ifdef MDTV_DEB
    cout<<"BRepNaming_Vector::Load(): Unexpected type of result"<<endl;
    Builder.Generated (MS.Shape());
#endif
  }
}

void BRepNaming_Vector::Load (TopoDS_Shape& aShape, const BRepNaming_TypeOfPrimitive3D Type) const {

  TNaming_Builder Builder (ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel
  if (Type == BRepNaming_EDGE){
    Builder.Generated (aShape);
  }
}
