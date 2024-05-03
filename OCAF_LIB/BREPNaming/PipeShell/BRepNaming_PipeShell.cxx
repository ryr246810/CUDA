#include <CAGDDefine.hxx>

#include "BRepNaming_PipeShell.ixx"
#include <TopTools_ListIteratorOfListOfShape.hxx>
#include <TopTools_MapIteratorOfMapOfShape.hxx>
#include <TopTools_DataMapOfShapeListOfShape.hxx>
#include <TopTools_MapOfShape.hxx>
#include <TopExp_Explorer.hxx>
#include <TopExp.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Vertex.hxx>
#include <BRepOffsetAPI_MakePipeShell.hxx>
#include <BRepCheck_Shell.hxx>
#include <BRepCheck_Wire.hxx>
#include <Standard_NullObject.hxx>

#include <TDataStd_Name.hxx>
#include <TNaming_Builder.hxx>
#include <TNaming_NamedShape.hxx>
#include <BRepNaming.hxx>
#include <BRepNaming_Loader.hxx>

//=======================================================================
//function : BRepNaming_PipeShell
//purpose  : 
//=======================================================================

BRepNaming_PipeShell::BRepNaming_PipeShell()
{}

//=======================================================================
//function : BRepNaming_PipeShell
//purpose  : 
//=======================================================================

BRepNaming_PipeShell::BRepNaming_PipeShell(const TDF_Label& theLabel)
{
  if(theLabel.IsNull())
    Standard_NullObject::Raise("BRepNaming_Pipe:: The Result label is Null ..."); 
  myResultLabel = theLabel;
}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================

void BRepNaming_PipeShell::Init(const TDF_Label& ResultLabel)
{
  if(ResultLabel.IsNull())
    Standard_NullObject::Raise("BRepNaming_PipeShell::Init The Result label is Null ..."); 
  myResultLabel = ResultLabel;
}    

//=======================================================================
//function : FindNewShape
//purpose  : 
//=======================================================================
static void FindNewShapes(const TopTools_ListOfShape& theList1,  const TopTools_ListOfShape& theList2,  TopTools_ListOfShape& theNewList)
{
  theNewList.Clear();
  TopTools_ListIteratorOfListOfShape aList1Itr(theList1);
  for (; aList1Itr.More(); aList1Itr.Next()) {
    const TopoDS_Shape& aShape1 = aList1Itr.Value();
    TopTools_ListIteratorOfListOfShape aList2Itr(theList2);
    for (; aList2Itr.More(); aList2Itr.Next()) {
      const TopoDS_Shape& aShape2 = aList2Itr.Value();
      if(aShape1.IsEqual(aShape2))
	theNewList.Append(aShape1);
    }
  }
}
//=======================================================================
//function : FillList
//purpose  : 
//=======================================================================
static void FillList(const TopTools_ListOfShape& theList1,  TopTools_ListOfShape& theList2)
{
  theList2.Clear();
  TopTools_ListIteratorOfListOfShape aList1Itr(theList1);
  for (; aList1Itr.More(); aList1Itr.Next()) {
    const TopoDS_Shape& aShape = aList1Itr.Value();
    theList2.Append(aShape);
  }
}
//=======================================================================
//function : Contains
//purpose  : 
//=======================================================================
static Standard_Boolean Contains(const TopTools_ListOfShape& theList, TopoDS_Shape& theShape) 
{
  TopTools_ListIteratorOfListOfShape aListItr(theList);
  for (; aListItr.More(); aListItr.Next()) 
    if(theShape.IsEqual(aListItr.Value()))
      return Standard_True;
  return Standard_False;
}
//=======================================================================
//function : FillMap
//purpose  : Key - Generated shape, Value - generators
//=======================================================================
static void FillMap(BRepBuilderAPI_MakeShape& MS, const TopoDS_Wire& theSpine, TopTools_DataMapOfShapeListOfShape& theMap) 
{
  TopoDS_Shape aPrevRoot;
  TopTools_ListOfShape aPrevList;
  Standard_Boolean aFirst = Standard_True;
  TopExp_Explorer SpineExplorer (theSpine, TopAbs_EDGE); //Spine
  for (; SpineExplorer.More(); SpineExplorer.Next ()) {
    const TopoDS_Shape& aRoot = SpineExplorer.Current ();
    const TopTools_ListOfShape& aList = MS.Generated (aRoot);
    TopTools_ListOfShape aList2;
    FillList(aList, aList2);
    if(aFirst) {
      aFirst = Standard_False;
      aPrevList = aList;
      aPrevRoot = aRoot;
    } else {
      TopTools_ListOfShape aSharedShapes;
      FindNewShapes(aPrevList, aList,  aSharedShapes);
      TopTools_ListIteratorOfListOfShape anItr(aPrevList);
      for (; anItr.More(); anItr.Next()) {
	if(theMap.IsBound(anItr.Value()))
	  continue;
	TopTools_ListOfShape aList1;
	aList1.Append(aPrevRoot);
	if(Contains(aSharedShapes, anItr.Value())) 
	  aList1.Append(aRoot);
	theMap.Bind(anItr.Value(), aList1);
      }
      aPrevRoot = aRoot;
      aPrevList = aList;
    }
  }
  TopTools_ListIteratorOfListOfShape anItr(aPrevList);
  for (; anItr.More(); anItr.Next()) {
    if(theMap.IsBound(anItr.Value()))
      continue;
    TopTools_ListOfShape aList1;
    aList1.Append(aPrevRoot);
    theMap.Bind(anItr.Value(), aList1);
  }
}
//=======================================================================
//function : LoadLateralShapes
//purpose  : 
//=======================================================================
static void LoadLateralShapes (BRepBuilderAPI_MakeShape&      MS,
			       const TopoDS_Shape&     ShapeIn1,
			       const TopAbs_ShapeEnum  KindOfShape1,
			       const TopoDS_Shape&     ShapeIn2,
			       const TopAbs_ShapeEnum  KindOfShape2,
			       const TDF_Label& theResultLabel)
{
  TopTools_MapOfShape View1, View2;
  TopTools_DataMapOfShapeListOfShape aMap;
  FillMap(MS, TopoDS::Wire(ShapeIn2), aMap);// Generated -> ListOfGenerators
  TopExp_Explorer SectionExplorer (ShapeIn1, KindOfShape1);//Section

  for (; SectionExplorer.More(); SectionExplorer.Next ()) {
    const TopoDS_Shape& Root1 = SectionExplorer.Current ();
    if (!View1.Add(Root1)) continue;
    const TopTools_ListOfShape& Shapes11 = MS.Generated (Root1);//generated from edge of Section
    TopTools_ListOfShape Shapes1;
    FillList(Shapes11, Shapes1);

    View2.Clear();

    TopTools_ListIteratorOfListOfShape aNewItr(Shapes1);
    for (; aNewItr.More(); aNewItr.Next()) {
      const TopoDS_Shape& aNewShape = aNewItr.Value();      
      if(!aNewShape.IsNull()) {	
	const TDF_Label& aLabel = theResultLabel.NewChild();
	    
	TNaming_Builder aBuilder(aLabel);
	if(!Root1.IsSame(aNewShape)) 
	  aBuilder.Generated (Root1, aNewShape);
	if(aMap.IsBound(aNewShape)) {
	  const TopTools_ListOfShape& Generators = aMap.Find(aNewShape);
	  TopTools_ListIteratorOfListOfShape anItr(Generators);
	  for (; anItr.More(); anItr.Next()) {
	    const TopoDS_Shape& aGenerator = anItr.Value(); 
	    if(!aGenerator.IsSame(aNewShape))
	      aBuilder.Generated (aGenerator, aNewShape);
	  }
	} 
      }	      
    }
  }
}

//=======================================================================
//function : Load (Pipe)
//purpose  : 
//=======================================================================

void BRepNaming_PipeShell::Load (BRepOffsetAPI_MakePipeShell&   theMake, 
				 const TopoDS_Wire&             theSpine,
				 const TopTools_ListOfShape&    theListOfSections ) const

{
  enum ResultType {
    CLOSED,
    OPENED
  };

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
  
  BRepNaming::CleanStructure(ResultLabel());

  TopoDS_Shape aGenerator1, aGenerator2;
  TopTools_ListIteratorOfListOfShape aListItr(theListOfSections);
  for (; aListItr.More(); aListItr.Next()) {
    const TopoDS_Shape& aShape = aListItr.Value();
    const TopTools_ListOfShape& Shapes = theMake.Generated (aShape);
    if(Shapes.Extent()) {
      TopTools_ListIteratorOfListOfShape anItr(Shapes);
      for (; anItr.More(); anItr.Next()) {
	const TopoDS_Shape& aShape2 = anItr.Value();
	if(!aShape2.IsNull()) {
	  if(aShape2.IsEqual(theMake.FirstShape()))
	    aGenerator1 = aShape;
	  else
	    aGenerator2 = aShape;
	}
      }
    }
  }

  if(aGenerator1.IsNull())
    aGenerator1 = theListOfSections.First();
  if(aGenerator2.IsNull())
    aGenerator2 = theListOfSections.Last();

  TNaming_Builder aBuilder(ResultLabel());
  aBuilder.Generated(aGenerator1, theMake.Shape());

  TopoDS_Shape aFirst;
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
  } else {

    aFirst = theMake.FirstShape();
    if(!aFirst.IsNull()) { 
      TNaming_Builder aFirstWireBuilder(First());
      aFirstWireBuilder.Modify(aGenerator1, aFirst);
      //Free Edges
      for (TopExp_Explorer anExp(aFirst, TopAbs_EDGE); anExp.More(); anExp.Next()) {
	const TopoDS_Shape& anEdge = anExp.Current();
	TNaming_Builder anEdgesBuilder(FreeEdges());
	anEdgesBuilder.Generated(aFirst, anEdge);
      }
    } else {
    }
    
    // a Last Wire
    TopoDS_Shape aLast = theMake.LastShape();
    if(!aLast.IsNull()) {
      TNaming_Builder aLastWireBuilder(Last());
      aLastWireBuilder.Modify(aGenerator2, aLast);
      //Free Edges
      for (TopExp_Explorer anExp(aLast, TopAbs_EDGE); anExp.More(); anExp.Next()) {
	const TopoDS_Shape& anEdge = anExp.Current();
	TNaming_Builder anEdgesBuilder(FreeEdges());
	anEdgesBuilder.Generated(aLast, anEdge); 
      }
    } else {
    }
  }
//Insert lateral faces (and may be free edges)
#ifdef OOC2530
  TNaming_Builder aLateralBuilder(Lateral());
  BRepNaming_Loader::LoadGeneratedShapes (theMake, theSpine, TopAbs_EDGE, aLateralBuilder);
  BRepNaming_Loader::LoadGeneratedShapes (theMake, aFirst,   TopAbs_EDGE, aLateralBuilder);
  Handle(BRepCheck_Wire) aCheck = new BRepCheck_Wire(TopoDS::Wire(aFirst));
  if(aCheck->Closed() != BRepCheck_NoError) 
    BRepNaming_Loader::LoadGeneratedShapes (theMake, aFirst,  TopAbs_VERTEX, aLateralBuilder);
#endif
#if !defined OOC2530
//workaround till fixing OCC2530
  LoadLateralShapes (theMake, aFirst, TopAbs_EDGE, theSpine, TopAbs_EDGE, ResultLabel());
  Handle(BRepCheck_Wire) aCheck = new BRepCheck_Wire(TopoDS::Wire(aFirst));
  if(aCheck->Closed() != BRepCheck_NoError)
    LoadLateralShapes (theMake,  aFirst, TopAbs_VERTEX, theSpine, TopAbs_EDGE,ResultLabel());
#endif
}






 
//=======================================================================
//function : First
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_PipeShell::First() const 
{
  const TDF_Label& L = ResultLabel().NewChild();
  return L;
}

//=======================================================================
//function : Last
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_PipeShell::Last() const 
{
  const TDF_Label& L = ResultLabel().NewChild();
  return L;
}

//=======================================================================
//function : Lateral
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_PipeShell::Lateral() const 
{
  const TDF_Label& L = ResultLabel().NewChild();
  return L;
}

//=======================================================================
//function : FreeEdges
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_PipeShell::FreeEdges() const 
{
  const TDF_Label& L = ResultLabel().NewChild();
  return L;
}


//=======================================================================
//function : GetDangleShapes
//purpose  : 
//=======================================================================

Standard_Boolean BRepNaming_PipeShell::GetDangleShapes(const TopTools_ListOfShape& theList,
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

  TopExp::MapShapesAndAncestors(aCompound, GeneratedTo, GeneratedFrom, subShapeAndAncestors);
  for (Standard_Integer i = 1; i <= subShapeAndAncestors.Extent(); i++) {
    const TopoDS_Shape& mayBeDangle = subShapeAndAncestors.FindKey(i);
    const TopTools_ListOfShape& ancestors = subShapeAndAncestors.FindFromIndex(i);
    if (ancestors.Extent() == 1) theDangles.Add(mayBeDangle);
  }
  return theDangles.Extent();
}
