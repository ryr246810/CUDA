// File:	BRepNaming_Extrusion.cxx
// Created:	Fri Nov  5 14:33:04 1999
// Author:	Vladislav ROMASHKO
//		<vro@flox.nnov.matra-dtv.fr>
// Copyright:	Matra Datavision 1999
// Modified by vro, Thu Dec 21 10:34:49 2000
// Modified by vro, Thu Dec 21 10:34:59 2000

//#include "stdafx.h"

#include "BRepNaming_Extrusion.hxx"
#include <BRepNaming_Loader.hxx>
#include <TNaming_Builder.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_DataMapOfShapeShape.hxx>
#include <TDF_Label.hxx>
#include <TDF_TagSource.hxx>
#include <Standard_NullObject.hxx>
#include <TopoDS_Iterator.hxx>
#include <TopExp.hxx>
#include <TColStd_ListOfInteger.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <BRep_Tool.hxx>
#include <TopoDS.hxx>

#ifdef MDTV_DEB
#include <TDataStd_Name.hxx>
#endif

//=======================================================================
//function : BRepNaming_Extrusion
//purpose  : 
//=======================================================================

BRepNaming_Extrusion::BRepNaming_Extrusion() {}

//=======================================================================
//function : BRepNaming_Extrusion
//purpose  : 
//=======================================================================

BRepNaming_Extrusion::BRepNaming_Extrusion(const TDF_Label& Label):BRepNaming_TopNaming(Label) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================

void BRepNaming_Extrusion::Init(const TDF_Label& Label) {
  if(Label.IsNull())
    Standard_NullObject::Raise("BRepNaming_Extrusion::Init The Result label is Null ..."); 
  myResultLabel = Label;
}

//=======================================================================
//function : Bottom
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_Extrusion::Bottom() const {
  const TDF_Label& L = ResultLabel().NewChild();
#ifdef MDTV_DEB
  TDataStd_Name::Set(L, "Bottom");
#endif
  return L;
}

//=======================================================================
//function : Top
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_Extrusion::Top() const {
  const TDF_Label& L = ResultLabel().NewChild();
#ifdef MDTV_DEB
  TDataStd_Name::Set(L, "Top");
#endif
  return L;
}

//=======================================================================
//function : Lateral
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_Extrusion::Lateral() const {
  const TDF_Label& L = ResultLabel().NewChild();
#ifdef MDTV_DEB
  TDataStd_Name::Set(L, "Lateral");
#endif
  return L;
}

//=======================================================================
//function : Degenerated
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_Extrusion::Degenerated() const {
  const TDF_Label& L = ResultLabel().NewChild();
#ifdef MDTV_DEB
  TDataStd_Name::Set(L, "Degenerated");
#endif
  return L;
}

//=======================================================================
//function : Dangle
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_Extrusion::Dangle() const {
  const TDF_Label& L = ResultLabel().NewChild();
#ifdef MDTV_DEB
  TDataStd_Name::Set(L, "Dangle");
#endif
  return L;
}

//=======================================================================
//function : Content
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_Extrusion::Content() const {
  const TDF_Label& L = ResultLabel().NewChild();
#ifdef MDTV_DEB
  TDataStd_Name::Set(L, "Content");
#endif
  return L;
}


//=======================================================================
//function : Load (Prism)
//purpose  : 
//=======================================================================


void BRepNaming_Extrusion::Load (const TopoDS_Shape& aShape,
							 const BRepNaming_TypeOfPrimitive3D Type) const
{
	TNaming_Builder Builder (ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel
	if (Type == BRepNaming_EXTRUSION){
		Builder.Generated (aShape);
	}
}

