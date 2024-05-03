
#include "BRepNaming_BooleanOperationFeat.ixx"
#include <Standard_NullObject.hxx>
#include <BRep_Tool.hxx>
#include <TopoDS_Iterator.hxx>
#include <TopoDS_Vertex.hxx>
#include <TopoDS.hxx>
#include <TopExp.hxx>
#include <TopTools_ListIteratorOfListOfShape.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
//
#include <TDataStd_Name.hxx>
#include <TDF_Label.hxx>
#include <TDF_TagSource.hxx>
#include <TNaming_Builder.hxx>
#include <TNaming_NamedShape.hxx>

#include <BRepNaming_Loader.hxx>

#ifdef DEB
#include <TDataStd_Name.hxx>
#endif


#include <GEOMUtils.hxx>


//=======================================================================
//function : BRepNaming_BooleanOperationFeat
//purpose  : 
//=======================================================================

BRepNaming_BooleanOperationFeat::BRepNaming_BooleanOperationFeat() {}

//=======================================================================
//function : BRepNaming_BooleanOperationFeat
//purpose  : 
//=======================================================================

BRepNaming_BooleanOperationFeat::BRepNaming_BooleanOperationFeat(const TDF_Label& ResultLabel):BRepNaming_TopNaming(ResultLabel) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================

void BRepNaming_BooleanOperationFeat::Init(const TDF_Label& ResultLabel) 
{  
  if(ResultLabel.IsNull()) 
    Standard_NullObject::Raise("BRepNaming_BooleanOperationFeat::Init The Result label is Null ..."); 
  myResultLabel = ResultLabel; 
}

//=======================================================================
//function : ModifiedFaces
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_BooleanOperationFeat::ModifiedFaces() const 
{
#ifdef DEB
  const TDF_Label& ModifiedFacesLabel = ResultLabel().NewChild();
  TDataStd_Name::Set(ModifiedFacesLabel, "ModifiedFaces");
  return ModifiedFacesLabel;
#endif
  return ResultLabel().NewChild();
}

//=======================================================================
//function : ModifiedEdges
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_BooleanOperationFeat::ModifiedEdges() const 
{
#ifdef DEB
  const TDF_Label& ModifiedEdgesLabel = ResultLabel().NewChild();
  TDataStd_Name::Set(ModifiedEdgesLabel, "ModifiedEdges");
  return ModifiedEdgesLabel;
#endif
  return ResultLabel().NewChild();
}

//=======================================================================
//function : DeletedFaces
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_BooleanOperationFeat::DeletedFaces() const 
{
#ifdef DEB
  const TDF_Label& DeletedFacesLabel = ResultLabel().NewChild();
  TDataStd_Name::Set(DeletedFacesLabel, "DeletedFaces");
  return DeletedFacesLabel;
#endif
  return ResultLabel().NewChild();
}

//=======================================================================
//function : DeletedEdges
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_BooleanOperationFeat::DeletedEdges() const 
{
#ifdef DEB
  const TDF_Label& DeletedEdgesLabel = ResultLabel().NewChild();
  TDataStd_Name::Set(DeletedEdgesLabel, "DeletedEdges");
  return DeletedEdgesLabel;
#endif
  return ResultLabel().NewChild();
}

//=======================================================================
//function : DeletedVertices
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_BooleanOperationFeat::DeletedVertices() const 
{
#ifdef DEB
  const TDF_Label& DeletedVerticesLabel = ResultLabel().NewChild();
  TDataStd_Name::Set(DeletedVerticesLabel, "DeletedVertices");
  return DeletedVerticesLabel;
#endif
  return ResultLabel().NewChild();
}

//=======================================================================
//function : NewShapes
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_BooleanOperationFeat::NewShapes() const 
{
#ifdef DEB
  const TDF_Label& NewShapesLabel = ResultLabel().NewChild();
  TDataStd_Name::Set(NewShapesLabel, "NewShapes");
  return NewShapesLabel;
#endif
  return ResultLabel().NewChild();
}

//=======================================================================
//function : Content
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_BooleanOperationFeat::Content() const 
{
#ifdef DEB
  const TDF_Label& ContentLabel = ResultLabel().NewChild();
  TDataStd_Name::Set(ContentLabel, "Content");
  return ContentLabel;
#endif
  return ResultLabel().NewChild();
}

//=======================================================================
//function : DeletedDegeneratedEdges
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_BooleanOperationFeat::DeletedDegeneratedEdges() const 
{
#ifdef DEB
  const TDF_Label& DegeneratedLabel = ResultLabel().NewChild();
  TDataStd_Name::Set(DegeneratedLabel, "DeletedDegeneratedEdges");
  return DegeneratedLabel;
#endif
  return ResultLabel().NewChild();
}

//=======================================================================
//function : ShapeType
//purpose  : 
//=======================================================================

TopAbs_ShapeEnum BRepNaming_BooleanOperationFeat::ShapeType(const TopoDS_Shape& theShape) const 
{
  TopAbs_ShapeEnum TypeSh = theShape.ShapeType();
  if (TypeSh == TopAbs_COMPOUND || TypeSh == TopAbs_COMPSOLID) {
    TopoDS_Iterator itr(theShape);
    if (!itr.More()) return TypeSh; 
    TypeSh = ShapeType(itr.Value());
    if(TypeSh == TopAbs_COMPOUND) return TypeSh;
    itr.Next();
    for(; itr.More(); itr.Next()) 
      if(ShapeType(itr.Value()) != TypeSh) return TopAbs_COMPOUND;      
  }
  return TypeSh;
} 

//=======================================================================
//function : GetShape
//purpose  : 
//=======================================================================

TopoDS_Shape BRepNaming_BooleanOperationFeat::GetShape(const TopoDS_Shape& theShape) const 
{
  if (theShape.ShapeType() == TopAbs_COMPOUND || theShape.ShapeType() == TopAbs_COMPSOLID) {
    TopoDS_Iterator itr(theShape);
    if (itr.More()) return itr.Value();
  }
  return theShape;
}

//=======================================================================
//function : LoadWire
//purpose  : 
//=======================================================================

void BRepNaming_BooleanOperationFeat::LoadWire(BRepAlgoAPI_BooleanOperation& MS) const 
{
  // Naming of modified edges:
  TNaming_Builder ModBuilder(ModifiedEdges());
  BRepNaming_Loader::LoadModifiedShapes (MS, MS.Shape1(), TopAbs_EDGE, ModBuilder);

  // load generated vertexes
  if(MS.HasGenerated()) {  
    TNaming_Builder nBuilder (NewShapes());
    BRepNaming_Loader::LoadGeneratedShapes (MS, MS.Shape1(), TopAbs_EDGE, nBuilder);
    BRepNaming_Loader::LoadGeneratedShapes (MS, MS.Shape2(), TopAbs_FACE, nBuilder);
  }
  // Naming of deleted edges, dangle vertices
  if(MS.HasDeleted()){ 
    TNaming_Builder DelEBuilder(DeletedEdges());
    BRepNaming_Loader::LoadDeletedShapes(MS, MS.Shape1(), TopAbs_EDGE, DelEBuilder);
    TNaming_Builder DelVBuilder(DeletedVertices());
    BRepNaming_Loader::LoadDeletedShapes(MS, MS.Shape1(), TopAbs_VERTEX, DelEBuilder);
  }
 }

//=======================================================================
//function : LoadShell
//purpose  : 
//=======================================================================

void BRepNaming_BooleanOperationFeat::LoadShell(BRepAlgoAPI_BooleanOperation& MS) const 
{
// Naming of modified faces and dangle edges
  TNaming_Builder ModFBuilder(ModifiedFaces());
  BRepNaming_Loader::LoadModifiedShapes(MS, MS.Shape1(), TopAbs_FACE, ModFBuilder);
  TNaming_Builder ModEBuilder(ModifiedEdges());    
  BRepNaming_Loader::LoadModifiedShapes(MS, MS.Shape1(), TopAbs_EDGE, ModEBuilder);
  
  if(MS.HasGenerated()) {  
    TNaming_Builder nBuilder (NewShapes());
//  generated Edges
    BRepNaming_Loader::LoadGeneratedShapes (MS, MS.Shape2(), TopAbs_FACE, nBuilder);
    BRepNaming_Loader::LoadGeneratedShapes (MS, MS.Shape1(), TopAbs_FACE, nBuilder);
  }
  // Naming of deleted faces edges:
  if(MS.HasDeleted()){ 
    TNaming_Builder DelFBuilder(DeletedFaces());
    BRepNaming_Loader::LoadDeletedShapes(MS, MS.Shape1(), TopAbs_FACE, DelFBuilder);

    TNaming_Builder DelEBuilder(DeletedEdges());
    BRepNaming_Loader::LoadDeletedShapes(MS, MS.Shape1(), TopAbs_EDGE, DelEBuilder);
  }
}

//=======================================================================
//function : LoadContent
//purpose  : 
//=======================================================================

void BRepNaming_BooleanOperationFeat::LoadContent(BRepAlgoAPI_BooleanOperation& MS) const 
{
  if (MS.Shape().ShapeType() == TopAbs_COMPSOLID || MS.Shape().ShapeType() == TopAbs_COMPOUND) {
    TopoDS_Iterator itr(MS.Shape());
    Standard_Integer nbShapes = 0;
    while (itr.More()) {
      nbShapes++;
      itr.Next();
    }
    if (nbShapes > 1) {
      for (itr.Initialize(MS.Shape()); itr.More(); itr.Next()) {
	TNaming_Builder bContent(Content());
	bContent.Generated(itr.Value());      
      }
    }
  } 
}  

//=======================================================================
//function : LoadResult
//purpose  : 
//=======================================================================



void BRepNaming_BooleanOperationFeat::LoadResult(BRepAlgoAPI_BooleanOperation& MS) const 
{
  Handle(TDF_TagSource) Tagger = TDF_TagSource::Set(ResultLabel());
  if (Tagger.IsNull()) return;
  Tagger->Set(0);

  TNaming_Builder Builder (ResultLabel());

  TopoDS_Shape aResult = MS.Shape();

  GEOMUtils::FixShapeAfterBooleanOperation(aResult);

  if (MS.Shape1().IsNull()) Builder.Generated(aResult);
  else Builder.Modify(MS.Shape1(), aResult);  
}



void BRepNaming_BooleanOperationFeat::LoadResultWithRemoveExtraEdges(BRepAlgoAPI_BooleanOperation& MS) const 
{
  Handle(TDF_TagSource) Tagger = TDF_TagSource::Set(ResultLabel());
  if (Tagger.IsNull()) return;
  Tagger->Set(0);

  TNaming_Builder Builder (ResultLabel());

  TopoDS_Shape aResult = MS.Shape();


  GEOMUtils::FixShapeAfterBooleanOperation(aResult);
  aResult = GEOMUtils::RemoveExtraEdges(aResult);


  if (MS.Shape1().IsNull()) Builder.Generated(aResult);
  else Builder.Modify(MS.Shape1(), aResult);  
}



//=======================================================================
//function : LoadDegenerated
//purpose  : 
//=======================================================================

void BRepNaming_BooleanOperationFeat::LoadDegenerated(BRepAlgoAPI_BooleanOperation& MS) const 
{
  TopTools_IndexedMapOfShape allEdges;
  TopExp::MapShapes(MS.Shape1(), TopAbs_EDGE, allEdges);
  Standard_Integer i = 1;
  for (; i <= allEdges.Extent(); i++) {
    if (BRep_Tool::Degenerated(TopoDS::Edge(allEdges.FindKey(i)))) {
      if (MS.IsDeleted(allEdges.FindKey(i))) {
	TNaming_Builder DegeneratedBuilder(DeletedDegeneratedEdges()); 
	DegeneratedBuilder.Generated(allEdges.FindKey(i));
#ifdef DEB
	TDataStd_Name::Set(DegeneratedBuilder.NamedShape()->Label(), "DeletedDegenerated");
#endif
      }      
    }
  }
}

//=======================================================================
//function : IsResultChanged
//purpose  : 
//=======================================================================

Standard_Boolean BRepNaming_BooleanOperationFeat::IsResultChanged(BRepAlgoAPI_BooleanOperation& MS) const 
{
  TopoDS_Shape ResSh = MS.Shape();
  if (MS.Shape().ShapeType() == TopAbs_COMPOUND) {
    Standard_Integer nbSubResults = 0;
    TopoDS_Iterator itr(MS.Shape());
    for (; itr.More(); itr.Next()) nbSubResults++;
    if (nbSubResults == 1) {
      itr.Initialize(MS.Shape());
      if (itr.More()) ResSh = itr.Value();
    }
  }
  return MS.Shape1().IsSame(ResSh);
}

// @@SDM: begin

// File history synopsis (creation,modification,correction)
// +---------------------------------------------------------------------------+
// ! Developer !              Comments                   !   Date   ! Version  !
// +-----------!-----------------------------------------!----------!----------+
// !       szy ! Creation                                !27-09-1999!3.0-00-4!
// !       vro ! Class became deffered                   !31-10-2000!3.0-00-4!
// !       vro ! Redesign                                !13-12-2000! 3.0-00-4!
// !       vro ! Result control                          !07-03-2001! 3.0-00-4!
// !       szy ! Modified LoadShell & LoadWire           ! 8-05-2003! 3.0-00-%L%!
// !       szy ! Adopted                                 ! 9-06-2003! 3.0-00-%L%!
// +---------------------------------------------------------------------------+
// Lastly modified by : szy                                    Date :  9-06-2003

// @@SDM: end
