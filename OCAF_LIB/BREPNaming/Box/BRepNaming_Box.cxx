
#include <CAGDDefine.hxx>

#include "BRepNaming_Box.ixx"

//#include <BRepNaming_Loader.hxx>
#include <TNaming_Builder.hxx>
#include <TDF_Label.hxx>
#include <Standard_NullObject.hxx>
#include <TopoDS_Solid.hxx>

//=======================================================================
//function : BRepNaming_Box
//purpose  : 
//=======================================================================

BRepNaming_Box::BRepNaming_Box() {}

//=======================================================================
//function : BRepNaming_Box
//purpose  : 
//=======================================================================

BRepNaming_Box::BRepNaming_Box(const TDF_Label& Label)
: BRepNaming_TopNaming(Label) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================

void BRepNaming_Box::Init(const TDF_Label& Label) {
  if(Label.IsNull())
    Standard_NullObject::Raise("BRepNaming_Box::Init The Result label is Null ..."); 
  myResultLabel = Label;
}  

//=======================================================================
//function : Load
//purpose  : 
//=======================================================================

void BRepNaming_Box::Load (BRepPrimAPI_MakeBox& MS, const BRepNaming_TypeOfPrimitive3D Type) const {
  //Load the faces of the box :
  TopoDS_Face BottomFace = MS.BottomFace ();
  TNaming_Builder BottomFaceIns (Bottom ()); 
  BottomFaceIns.Generated (BottomFace);
 
  TopoDS_Face TopFace = MS.TopFace ();
  TNaming_Builder TopFaceIns (Top ()); 
  TopFaceIns.Generated (TopFace); 

  TopoDS_Face FrontFace = MS.FrontFace ();
  TNaming_Builder FrontFaceIns (Front ()); 
  FrontFaceIns.Generated (FrontFace); 

  TopoDS_Face RightFace = MS.RightFace ();
  TNaming_Builder RightFaceIns (Right ()); 
  RightFaceIns.Generated (RightFace); 

  TopoDS_Face BackFace = MS.BackFace ();
  TNaming_Builder BackFaceIns (Back ()); 
  BackFaceIns.Generated (BackFace); 

  TopoDS_Face LeftFace = MS.LeftFace ();
  TNaming_Builder LeftFaceIns (Left ()); 
  LeftFaceIns.Generated (LeftFace); 

  // add TNaming_NamedShape attribute to the ResultLabel
  TNaming_Builder Builder (ResultLabel());
  if (Type == BRepNaming_SOLID) Builder.Generated (MS.Solid());
  else if (Type == BRepNaming_SHELL) Builder.Generated (MS.Shell());
  else {
#ifdef MDTV_DEB
    cout<<"BRepNaming_Box::Load(): Unexpected type of result"<<endl;
    Builder.Generated (MS.Shape());
#endif
  }
}

//=======================================================================
//function : Back
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_Box::Back () const {
  return ResultLabel().FindChild(1,Standard_True); 
}

//=======================================================================
//function : Front
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_Box::Front () const {
  return ResultLabel().FindChild(2,Standard_True); 
}

//=======================================================================
//function : Left
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_Box::Left () const {
  return ResultLabel().FindChild(3,Standard_True); 
}

//=======================================================================
//function : Right
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_Box::Right () const {
  return ResultLabel().FindChild(4,Standard_True); 
}

//=======================================================================
//function : Bottom
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_Box::Bottom () const {
  return ResultLabel().FindChild(5,Standard_True); 
}

//=======================================================================
//function : Top
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_Box::Top () const {
  return ResultLabel().FindChild(6,Standard_True); 
}

