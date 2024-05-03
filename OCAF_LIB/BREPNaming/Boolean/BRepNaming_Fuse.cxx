
#include "BRepNaming_Fuse.ixx"
#include <TNaming_Builder.hxx>
#include <BRepNaming_Loader.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_ListIteratorOfListOfShape.hxx>
#include <TNaming_NamedShape.hxx>
#include <TopoDS_Iterator.hxx>
#include <TNaming_Tool.hxx>

//=======================================================================
//function : BRepNaming_Fuse
//purpose  : 
//=======================================================================

BRepNaming_Fuse::BRepNaming_Fuse() {}

//=======================================================================
//function : BRepNaming_Fuse
//purpose  : 
//=======================================================================

BRepNaming_Fuse::BRepNaming_Fuse(const TDF_Label& ResultLabel)
     :BRepNaming_BooleanOperationFeat(ResultLabel) {}



//=======================================================================
//function : Load
//purpose  : 
//=======================================================================
void BRepNaming_Fuse::Load(BRepAlgoAPI_BooleanOperation& MS) const 
{
  const TopoDS_Shape& ResSh = MS.Shape();
  if (ResSh.IsNull()) {
#ifdef DEB
    cout<<"BRepNaming_Fuse::Load(): The result of the boolean operation is null"<<endl;
#endif
    return;
  }
  // Naming of the result:
  LoadResult(MS);
  //LoadResultWithRemoveExtraEdges(MS);
}



void 
BRepNaming_Fuse::
Load (TopoDS_Shape& aShape, const BRepNaming_TypeOfPrimitive3D Type) const 
{
  TNaming_Builder Builder (ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel
  if (Type == BRepNaming_FUSE){
    Builder.Generated ( aShape );
  }
}


/*
void BRepNaming_Fuse::Load(BRepAlgoAPI_BooleanOperation& MS) const 
{
  const TopoDS_Shape& ResSh = MS.Shape();
  const TopoDS_Shape& ObjSh = MS.Shape1();
  const TopoDS_Shape& ToolSh = MS.Shape2();

  if (ResSh.IsNull()) {
#ifdef DEB
    cout<<"BRepNaming_Fuse::Load(): The result of the boolean operation is null"<<endl;
#endif
    return;
  }

  // Naming of the result:
  LoadResult(MS);

  // Naming of modified faces: 
  TNaming_Builder ModBuilder(ModifiedFaces());    
  BRepNaming_Loader::LoadModifiedShapes (MS, ObjSh,  TopAbs_FACE, ModBuilder, Standard_True);  
  BRepNaming_Loader::LoadModifiedShapes (MS, ToolSh, TopAbs_FACE, ModBuilder, Standard_True);    

  // Naming of deleted faces:
  if(MS.HasDeleted()){
    TNaming_Builder DelBuilder(DeletedFaces());
    BRepNaming_Loader::LoadDeletedShapes  (MS, ObjSh,  TopAbs_FACE, DelBuilder);
    BRepNaming_Loader::LoadDeletedShapes  (MS, ToolSh, TopAbs_FACE, DelBuilder);
  }

  // Naming of the content of the result:
  LoadContent(MS);
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
// !       vro ! Result may be null                      !19-03-2001! 3.0-00-3!
// !       szy ! Modified Load                           ! 8-05-2003! 3.0-00-%L%!
// !       szy ! Modified Load                           !21-05-2003! 3.0-00-%L%!
// +---------------------------------------------------------------------------+
// Lastly modified by : szy                                    Date : 22-05-2003

// @@SDM: end
