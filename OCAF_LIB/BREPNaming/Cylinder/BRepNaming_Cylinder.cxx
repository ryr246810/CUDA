#include <CAGDDefine.hxx>

#include "BRepNaming_Cylinder.ixx"
#include <TNaming_Builder.hxx>
#include <Standard_NullObject.hxx>
#include <TopoDS_Solid.hxx>
#ifdef MDTV_DEB
#include <TDataStd_Name.hxx>
#endif

//=======================================================================
//function : BRepNaming_Cylinder
//purpose  : 
//=======================================================================

BRepNaming_Cylinder::BRepNaming_Cylinder() {}

//=======================================================================
//function : BRepNaming_Cylinder
//purpose  : 
//=======================================================================

BRepNaming_Cylinder::BRepNaming_Cylinder(const TDF_Label& ResultLabel):
       BRepNaming_TopNaming(ResultLabel) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================

void BRepNaming_Cylinder::Init(const TDF_Label& ResultLabel) {
  if(ResultLabel.IsNull())
    Standard_NullObject::Raise("BRepNaming_Cylinder::Init The Result label is Null ..."); 
  myResultLabel = ResultLabel;
}    

//=======================================================================
//function : Bottom
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_Cylinder::Bottom() const {
#ifdef MDTV_DEB
  TDataStd_Name::Set(ResultLabel().FindChild(1, Standard_True), "Bottom");
#endif
  return ResultLabel().FindChild(1, Standard_True);
}

//=======================================================================
//function : Top
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_Cylinder::Top() const {
#ifdef MDTV_DEB
  TDataStd_Name::Set(ResultLabel().FindChild(2, Standard_True), "Top");
#endif
  return ResultLabel().FindChild(2, Standard_True);
}

//=======================================================================
//function : Lateral
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_Cylinder::Lateral() const {
#ifdef MDTV_DEB
  TDataStd_Name::Set(ResultLabel().FindChild(3, Standard_True), "Lateral");
#endif
  return ResultLabel().FindChild(3, Standard_True);
}

//=======================================================================
//function : StartSide
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_Cylinder::StartSide() const {
#ifdef MDTV_DEB
  TDataStd_Name::Set(ResultLabel().FindChild(4, Standard_True), "StartSide");
#endif
  return ResultLabel().FindChild(4, Standard_True);
}

//=======================================================================
//function : EndSide
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_Cylinder::EndSide() const {
#ifdef MDTV_DEB
  TDataStd_Name::Set(ResultLabel().FindChild(5, Standard_True), "EndSide");
#endif
  return ResultLabel().FindChild(5, Standard_True);
}

//=======================================================================
//function : Load (Cylinder)
//purpose  : 
//=======================================================================

void BRepNaming_Cylinder::Load (BRepPrimAPI_MakeCylinder& mkCylinder,
				const BRepNaming_TypeOfPrimitive3D Type) const
{
  BRepPrim_Cylinder& S = mkCylinder.Cylinder();

  if (S.HasBottom()) {
    TopoDS_Face BottomFace = S.BottomFace();
    TNaming_Builder BottomFaceIns(Bottom()); 
    BottomFaceIns.Generated(BottomFace); 
  }

  if (S.HasTop()) {
    TopoDS_Face TopFace = S.TopFace();
    TNaming_Builder TopFaceIns(Top()); 
    TopFaceIns.Generated(TopFace); 
  }

  TopoDS_Face LateralFace = S.LateralFace();
  TNaming_Builder LateralFaceIns(Lateral()); 
  LateralFaceIns.Generated(LateralFace); 

  if (S.HasSides()) {
    TopoDS_Face StartFace = S.StartFace();
    TNaming_Builder StartFaceIns(StartSide()); 
    StartFaceIns.Generated(StartFace); 
    TopoDS_Face EndFace = S.EndFace();
    TNaming_Builder EndFaceIns(EndSide()); 
    EndFaceIns.Generated(EndFace); 
  }

  TNaming_Builder Builder (ResultLabel());
  if (Type == BRepNaming_SOLID) Builder.Generated (mkCylinder.Solid());
  else if (Type == BRepNaming_SHELL) Builder.Generated (mkCylinder.Shell());
  else {
#ifdef DEB
    cout<<"BRepNaming_Cylinder::Load(): Unexpected type of result"<<endl;
    Builder.Generated (mkCylinder.Shape());
#endif
  }
}

