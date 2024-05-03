
#include <CAGDDefine.hxx>

#include "BRepNaming_ThruSections.ixx"

#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopTools_ListIteratorOfListOfShape.hxx>
#include <TopTools_MapIteratorOfMapOfShape.hxx>


#include <TopTools_MapOfShape.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <BRepCheck_Shell.hxx>
#include <Standard_NullObject.hxx>

#include <TNaming_Builder.hxx>
#include <TNaming_NamedShape.hxx>
#include <BRepNaming_Loader.hxx>

#include <TopoDS_Compound.hxx>
#include <BRep_Builder.hxx>

#include <Standard_NullObject.hxx>
#include <TDF_Label.hxx>

#ifdef DEB
#include <BRepTools.hxx>
#include <TDataStd_Name.hxx>
#endif
//=======================================================================
//function : BRepNaming_ThruSections
//purpose  : 
//=======================================================================

BRepNaming_ThruSections::BRepNaming_ThruSections()
{}

//=======================================================================
//function : BRepNaming_ThruSections
//purpose  : 
//=======================================================================

BRepNaming_ThruSections::BRepNaming_ThruSections(const TDF_Label& ResultLabel)
  :BRepNaming_TopNaming(ResultLabel)
{
  if(ResultLabel.IsNull()) Standard_NullObject::Raise("BRepNaming_ThruSections:: The Result label is Null ..."); 
  myResultLabel = ResultLabel;
}


//=======================================================================
//function : Init
//purpose  : 
//=======================================================================

void BRepNaming_ThruSections::Init(const TDF_Label& ResultLabel)
{
  if(ResultLabel.IsNull())
    Standard_NullObject::Raise("BRepNaming_ThruSections::Init The Result label is Null ..."); 
  myResultLabel = ResultLabel;
}    

#ifdef DEB
#include <BRepTools.hxx>
static void Write(const TopoDS_Shape& shape, const Standard_CString filename) 
{
  ofstream save;
  save.open(filename);
  save << "DBRep_DrawableShape" << endl << endl;
  if(!shape.IsNull()) BRepTools::Write(shape, save);
  save.close();
}
#endif
//=======================================================================
//function : Load (Pipe thru set of sections)
//purpose  : temporary naming
//=======================================================================

void BRepNaming_ThruSections::Load (BRepOffsetAPI_ThruSections&  theMake, 
				    const TopTools_ListOfShape& theListOfSections,
				    const Standard_Boolean theIsRuled ) const

{
  enum ResultType {
    CLOSED,
    OPENED
  };
  ResultLabel().ForgetAllAttributes();
  TNaming_Builder aBuilder(ResultLabel());
  aBuilder.Generated(theMake.Shape());
  
  ResultType aRType;
  Handle(TDF_TagSource) Tagger = TDF_TagSource::Set(ResultLabel());
  if (Tagger.IsNull()) return;
  Tagger->Set(0);
  
  switch(theMake.Shape().ShapeType()) 
    {
    case TopAbs_SOLID:
      aRType = CLOSED;
      break;
    case TopAbs_SHELL:
      Handle(BRepCheck_Shell) aCheck = new BRepCheck_Shell(TopoDS::Shell(theMake.Shape()));
      if(aCheck->Closed() == BRepCheck_NoError)
	aRType = CLOSED;
      else
	aRType = OPENED;
    }
  
  if(aRType == CLOSED) {
    if(!theMake.FirstShape().IsNull()) {
      //Insert bottom face
      TNaming_Builder aBuilder((First()));
      aBuilder.Generated(theMake.FirstShape());
    }
    if(!theMake.LastShape().IsNull()) {
      //Insert top face
      TNaming_Builder aBuilder((Last()));
      aBuilder.Generated(theMake.LastShape());
    }
  }
  
  //Insert lateral faces
  TopTools_ListOfShape aList;
  for (TopExp_Explorer anExp(theMake.Shape(), TopAbs_FACE); anExp.More(); anExp.Next()) {
    const TopoDS_Shape& aFace = anExp.Current();
    if(aRType == CLOSED) {
      if(aFace.IsSame(theMake.FirstShape()) || aFace.IsSame(theMake.LastShape())) 
	continue;
    } 
    TNaming_Builder aLateralBuilder(Lateral());
    aLateralBuilder.Generated(aFace);
    aList.Append(aFace); //lateral faces
  }

  TopTools_MapOfShape aDangles;
  if(aRType == OPENED) { // load free edges
    GetDangleShapes(aList, TopAbs_FACE, aDangles);
    TopTools_MapIteratorOfMapOfShape anItr(aDangles);
    for (; anItr.More(); anItr.Next()) {
      TNaming_Builder aFEBuilder(FreeEdges()); 
      aFEBuilder.Generated(anItr.Key());
    }
  }
  //  TopTools_MapOfShape aSimEdges;
  if(!theIsRuled) {
    if(aList.Extent() == 1 || aList.Extent() == 2) {
      //add sim edges
      if(aRType == CLOSED) GetDangleShapes(aList, TopAbs_FACE, aDangles);
      TopTools_MapOfShape aSimEdges;
      TopTools_ListIteratorOfListOfShape aListItr(aList);
      for (; aListItr.More(); aListItr.Next()) {
	const TopoDS_Shape& aFace = aListItr.Value();
	for (TopExp_Explorer exp(aFace, TopAbs_EDGE); exp.More(); exp.Next()) {
	  const TopoDS_Shape& anEdge = exp.Current();  
	  if(aDangles.Contains(anEdge)) continue;
	  else 
	    aSimEdges.Add(anEdge);
	}
      }
      TopTools_MapIteratorOfMapOfShape anItr(aSimEdges);
      for (; anItr.More(); anItr.Next()) {      
	TNaming_Builder aBuilder(ResultLabel().NewChild());
	aBuilder.Generated(anItr.Key());
#ifdef DEB
	TDataStd_Name::Set(aBuilder.NamedShape()->Label(), "SimEdge");
#endif
      }
    }
  }
}






  
//=======================================================================
//function : First
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_ThruSections::First() const 
{
  const TDF_Label& L = ResultLabel().NewChild();
#ifdef DEB
  TDataStd_Name::Set(L, "First");
#endif
  return L;
}

//=======================================================================
//function : Last
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_ThruSections::Last() const 
{
  const TDF_Label& L = ResultLabel().NewChild();
#ifdef DEB
  TDataStd_Name::Set(L, "Last");
#endif
  return L;
}

//=======================================================================
//function : Lateral
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_ThruSections::Lateral() const 
{
  const TDF_Label& L = ResultLabel().NewChild();
#ifdef DEB
  TDataStd_Name::Set(L, "Lateral");
#endif
  return L;
}

//=======================================================================
//function : FreeEdges
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_ThruSections::FreeEdges() const 
{
  const TDF_Label& L = ResultLabel().NewChild();
#ifdef DEB
  TDataStd_Name::Set(L, "FreeEdges");
#endif
  return L;
}


//=======================================================================
//function : GetDangleShapes
//purpose  : 
//=======================================================================

Standard_Boolean BRepNaming_ThruSections::GetDangleShapes(const TopTools_ListOfShape& theList,
							 const TopAbs_ShapeEnum GeneratedFrom,
							 TopTools_MapOfShape& theDangles) const
{
  theDangles.Clear();
  TopTools_IndexedDataMapOfShapeListOfShape subShapeAndAncestors;
  TopAbs_ShapeEnum GeneratedTo;
  if (GeneratedFrom == TopAbs_FACE) GeneratedTo = TopAbs_EDGE;
  else if (GeneratedFrom == TopAbs_EDGE) GeneratedTo = TopAbs_VERTEX;
  else return Standard_False;
  TopoDS_Compound aCompound;
  BRep_Builder aShapeBuilder;
  aShapeBuilder.MakeCompound(aCompound);
  TopTools_ListIteratorOfListOfShape aListItr(theList);
  for (; aListItr.More(); aListItr.Next()) {
    const TopoDS_Shape& aShape = aListItr.Value();
    aShapeBuilder.Add(aCompound, aShape);
  }
#ifdef DEB   
//    Standard_CString aFileName = "Lateral.brep";
//    Write(aCompound, aFileName);
#endif
  TopExp::MapShapesAndAncestors(aCompound, GeneratedTo, GeneratedFrom, subShapeAndAncestors);
  for (Standard_Integer i = 1; i <= subShapeAndAncestors.Extent(); i++) {
    const TopoDS_Shape& mayBeDangle = subShapeAndAncestors.FindKey(i);
    const TopTools_ListOfShape& ancestors = subShapeAndAncestors.FindFromIndex(i);
    if (ancestors.Extent() == 1) theDangles.Add(mayBeDangle);
  }
  return theDangles.Extent();
}











// @@SDM: begin

// Copyright Open CASCADE......................................Version    3.0-00
// Lastly modified by : szy                                    Date :  4-06-2003

// File history synopsis (creation,modification,correction)
// +---------------------------------------------------------------------------+
// ! Developer !              Comments                   !   Date   ! Version  !
// +-----------!-----------------------------------------!----------!----------+
// !       szy ! Creation                                ! 2-06-2003! 3.0-00-%L%!
// !       szy ! Modified                                ! 4-06-2003! 3.0-00-%L%!
// +---------------------------------------------------------------------------+

// @@SDM: end
