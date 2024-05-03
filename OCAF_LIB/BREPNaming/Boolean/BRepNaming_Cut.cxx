#include "BRepNaming_Cut.ixx"
#include <TopoDS_Iterator.hxx>
#include <TopoDS_Shell.hxx>
#include <BRep_Builder.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopTools_ListIteratorOfListOfShape.hxx>
//
#include <TNaming_NamedShape.hxx>
#include <TNaming_Tool.hxx>
#include <TNaming_Builder.hxx>
#include <TNaming_Builder.hxx>
#include <BRepNaming_Loader.hxx>


//=======================================================================
//function : BRepNaming_Cut
//purpose  : 
//=======================================================================

BRepNaming_Cut::BRepNaming_Cut() {}

//=======================================================================
//function : BRepNaming_Cut
//purpose  : 
//=======================================================================

BRepNaming_Cut::BRepNaming_Cut(const TDF_Label& ResultLabel) :BRepNaming_BooleanOperationFeat(ResultLabel) {}

//=======================================================================
//function : Load
//purpose  : 
//=======================================================================

void 
BRepNaming_Cut::
Load(BRepAlgoAPI_BooleanOperation& MS) const 
{
  TopoDS_Shape ResSh = MS.Shape();
  if (ResSh.IsNull()) {
#ifdef DEB
    cout<<"BRepNaming_Cut::Load(): The result of the boolean operation is null"<<endl;
#endif
    return;
  }
  // Naming of the result:
  LoadResult(MS);
}


void 
BRepNaming_Cut::
Load (TopoDS_Shape& aShape, const BRepNaming_TypeOfPrimitive3D Type) const 
{
  TNaming_Builder Builder (ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel
  if (Type == BRepNaming_CUT){
    Builder.Generated ( aShape );
  }
}



/*

void BRepNaming_Cut::Load(BRepAlgoAPI_BooleanOperation& MS) const {
  TopoDS_Shape ResSh = MS.Shape();
  const TopoDS_Shape& ObjSh = MS.Shape1();
  const TopoDS_Shape& ToolSh = MS.Shape2();
  const TopAbs_ShapeEnum& TypeSh = ShapeType(ObjSh);

  if (ResSh.IsNull()) {
#ifdef DEB
    cout<<"BRepNaming_Cut::Load(): The result of the boolean operation is null"<<endl;
#endif
    return;
  }

  // If the shapes are the same - select the result and exit:
  if (IsResultChanged(MS)) {
#ifdef DEB
    cout<<"BRepNaming_Cut::Load(): The object and the result of CUT operation are the same"<<endl;
#endif
    if (MS.Shape().ShapeType() == TopAbs_COMPOUND) {
      Standard_Integer nbSubResults = 0;
      TopoDS_Iterator itr(MS.Shape());
      for (; itr.More(); itr.Next()) nbSubResults++;
      if (nbSubResults == 1) { //
	itr.Initialize(MS.Shape());
	if (itr.More()) ResSh = itr.Value();
      } //
    }    
    TNaming_Builder aBuilder(ResultLabel());
    aBuilder.Select(ResSh, ObjSh);
    return;
  }

  // Naming of the result:
  LoadResult(MS);
  
  // Naming of modified, deleted and new sub shapes:
  if (TypeSh == TopAbs_WIRE || TypeSh == TopAbs_EDGE) {//LoadWire(MS);
    //Modified
    TNaming_Builder ModEBuilder(ModifiedEdges());    
    BRepNaming_Loader::LoadModifiedShapes(MS, ObjSh, TopAbs_EDGE, ModEBuilder, Standard_True);
    //Generated vertexes
    if(MS.HasGenerated()) {  
      TNaming_Builder nBuilder (NewShapes());
      BRepNaming_Loader::LoadGeneratedShapes (MS, ObjSh,  TopAbs_EDGE, nBuilder);
      BRepNaming_Loader::LoadGeneratedShapes (MS, ToolSh, TopAbs_FACE, nBuilder);
    }
    //Deleted (Faces, Edges, Vertexes)
    if(MS.HasDeleted()){ 
      TNaming_Builder DelFBuilder(DeletedFaces()); // all deleted shapes
      BRepNaming_Loader::LoadDeletedShapes(MS, ObjSh,  TopAbs_EDGE,   DelFBuilder);
      BRepNaming_Loader::LoadDeletedShapes(MS, ObjSh,  TopAbs_VERTEX, DelFBuilder);
      BRepNaming_Loader::LoadDeletedShapes(MS, ToolSh, TopAbs_FACE,   DelFBuilder);
    }

  }
  else if (TypeSh == TopAbs_SHELL || TypeSh == TopAbs_FACE) {//LoadShell(MS);
    //Modified
    TNaming_Builder ModFBuilder(ModifiedFaces());
    BRepNaming_Loader::LoadModifiedShapes(MS, ObjSh, TopAbs_FACE, ModFBuilder, Standard_True);
    TNaming_Builder ModEBuilder(ModifiedEdges());    
    BRepNaming_Loader::LoadModifiedShapes(MS, ObjSh, TopAbs_EDGE, ModEBuilder, Standard_True);
    //Generated edges (edges of free boundaries)
    if(MS.HasGenerated()) {  
      TNaming_Builder nBuilder (NewShapes());
      BRepNaming_Loader::LoadGeneratedShapes (MS, ObjSh,  TopAbs_FACE, nBuilder);
      BRepNaming_Loader::LoadGeneratedShapes (MS, ToolSh, TopAbs_FACE, nBuilder);
    }
    //Deleted
    if(MS.HasDeleted()){ 
      TNaming_Builder DelFBuilder(DeletedFaces());
      BRepNaming_Loader::LoadDeletedShapes(MS, ObjSh,  TopAbs_FACE, DelFBuilder);     
      BRepNaming_Loader::LoadDeletedShapes(MS, ObjSh,  TopAbs_EDGE, DelFBuilder);
      BRepNaming_Loader::LoadDeletedShapes(MS, ToolSh, TopAbs_FACE, DelFBuilder); 
    }
  }
  else { // Solid
    if(MS.HasModified()){
      TNaming_Builder ModBuilder(ModifiedFaces());    
      BRepNaming_Loader::LoadModifiedShapes (MS, ObjSh,  TopAbs_FACE, ModBuilder, Standard_True);
      BRepNaming_Loader::LoadModifiedShapes (MS, ToolSh, TopAbs_FACE, ModBuilder, Standard_True);
    }
    
    if(MS.HasDeleted()){
      TNaming_Builder DelBuilder(DeletedFaces());
      BRepNaming_Loader::LoadDeletedShapes (MS, ObjSh,  TopAbs_FACE, DelBuilder);
      BRepNaming_Loader::LoadDeletedShapes (MS, ToolSh, TopAbs_FACE, DelBuilder);     
    }
  }
  LoadDegenerated(MS);
    
  // Naming of the content:
  if (ShapeType(ObjSh) == TopAbs_SOLID) LoadContent(MS);
}
//*/


// @@SDM: begin

// Lastly modified by : vro                                    Date : 31-10-2000

// File history synopsis (creation,modification,correction)
// +---------------------------------------------------------------------------+
// ! Developer !              Comments                   !   Date   ! Version  !
// +-----------!-----------------------------------------!----------!----------+
// !       vro ! Creation                                !31-10-2000!3.0-00-3!
// !       vro ! Redesign                                !13-12-2000! 3.0-00-3!
// !       vro ! Result control                          !07-03-2001! 3.0-00-3!
// !       vro ! Result may be null                      !19-03-2001! 3.0-00-3!
// !       szy ! Modified Load                           ! 8-05-2003! 3.0-00-%L%!
// !       szy ! Modified Load                           !21-05-2003! 3.0-00-%L%!
// +---------------------------------------------------------------------------+
// Lastly modified by : szy                                    Date : 22-05-2003

// @@SDM: end
