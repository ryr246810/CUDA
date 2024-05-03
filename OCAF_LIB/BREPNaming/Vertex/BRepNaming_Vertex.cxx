// File:	BRepNaming_Vertex.cxx
// Created:	2010.03.19.11:17
// Author:	Wang Yue
//		<id_wangyue@hotmail.com>
// Copyright:	Wang Yue 2010

#include <CAGDDefine.hxx>

#include "BRepNaming_Vertex.ixx"

//#include <BRepNaming_Loader.hxx>
#include <TNaming_Builder.hxx>
#include <TDF_Label.hxx>
#include <Standard_NullObject.hxx>
#include <TopoDS_Solid.hxx>

//=======================================================================
//function : BRepNaming_Vertex
//purpose  : 
//=======================================================================
BRepNaming_Vertex::BRepNaming_Vertex() {}

//=======================================================================
//function : BRepNaming_Vertex
//purpose  : 
//=======================================================================
BRepNaming_Vertex::BRepNaming_Vertex(const TDF_Label& Label): BRepNaming_TopNaming(Label) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================

void BRepNaming_Vertex::Init(const TDF_Label& Label) 
{
  if(Label.IsNull())
    Standard_NullObject::Raise("BRepNaming_Vertex::Init The Result label is Null ..."); 
  myResultLabel = Label;
}  


//=======================================================================
//function : Load
//purpose  : 
//=======================================================================

void BRepNaming_Vertex::Load (TopoDS_Shape& aShape,  const BRepNaming_TypeOfPrimitive3D Type) const 
{
  // add TNaming_NamedShape attribute to the ResultLabel
  TNaming_Builder Builder (ResultLabel());

  if (Type == BRepNaming_VERTEX){
    Builder.Generated (aShape);
  }else{
    Builder.Generated (aShape);
    cout<<"BRepNaming_Vertex::Load---------------------------error"<<endl;
  }

  /*
  if (Type == BRepNaming_VERTEX){
    Builder.Generated (MS.Vertex());
  } else {
    Builder.Generated (MS.Shape());
  }
  //*/
}
