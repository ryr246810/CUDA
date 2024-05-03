// File:	OCAF_CylinderDriver.cxx
// Created:	2010.07.18
// Author:	wang yue
//		id_wangyue@hotmail.com

#include <CAGDDefine.hxx>

#include "OCAF_CylinderDriver.ixx"
#include <OCAF_ICylinder.hxx>
#include <OCAF_IFunction.hxx>

#include <Tags.hxx>

#include <TDataStd_Real.hxx>
#include <TDataStd_TreeNode.hxx>
#include <TopoDS_Shape.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>

#include <TDocStd_Modified.hxx>
#include <BRepNaming_Cylinder.hxx>
#include <BRepNaming_TypeOfPrimitive3D.hxx>
#include <TDF_ChildIterator.hxx>

#include <TNaming_Builder.hxx>
#include <TNaming_NamedShape.hxx>
#include <TNaming.hxx>
#include <TopLoc_Location.hxx>

#include <gp_Ax2.hxx>
#include <BRepAlgo.hxx>

#include <Standard_ConstructionError.hxx>
#include <Standard_TypeMismatch.hxx>
#include <Standard_NullObject.hxx>
#include <StdFail_NotDone.hxx>

#define OK_CYLINDER 0
#define X_NOT_FOUND 1
#define Y_NOT_FOUND 2
#define Z_NOT_FOUND 3
#define RADIUS_NOT_FOUND 4
#define HEIGHT_NOT_FOUND 5
#define EMPTY_CYLINDER 6
#define CYLINDER_NOT_DONE 7
#define NULL_CYLINDER 8

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================

OCAF_CylinderDriver::OCAF_CylinderDriver():OCAF_Driver() {}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================

Standard_Integer OCAF_CylinderDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const {
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_CYLINDER;


  TopLoc_Location aLocation;
  Handle(TNaming_NamedShape) aPrevNS;
  TDF_Label aPrevLabel = aNode->Label().FindChild(RESULTS_TAG);
  if(!aPrevLabel.IsNull() && aPrevLabel.FindAttribute(TNaming_NamedShape::GetID(), aPrevNS)) {
    if(!aPrevNS->IsEmpty())
      aLocation = aPrevNS->Get().Location();
  }

  
  OCAF_ICylinder anInterface(aNode);
  Standard_Integer aType = anInterface.GetType();
  TDF_Label aResultLabel;

  gp_Pnt aP;
  gp_Vec aV;

  if (aType == CYLINDER_R_H) {
    aP = gp::Origin();
    aV = gp::DZ();
  }
  else if (aType == CYLINDER_PNT_VEC_R_H) {
    TopoDS_Shape aRefPoint  = anInterface.GetPoint();
    TopoDS_Shape aRefVector = anInterface.GetVector();
    if (aRefPoint.IsNull() || aRefVector.IsNull()) {
      Standard_NullObject::Raise("Cylinder creation aborted: point or vector is not defined");
    }
    if (aRefPoint.ShapeType() != TopAbs_VERTEX ||
        aRefVector.ShapeType() != TopAbs_EDGE) {
      Standard_TypeMismatch::Raise("Cylinder creation aborted: point or vector shapes has wrong type");
    }

    aP = BRep_Tool::Pnt(TopoDS::Vertex(aRefPoint));

    TopoDS_Edge anE = TopoDS::Edge(aRefVector);
    TopoDS_Vertex V1, V2;
    TopExp::Vertices(anE, V1, V2, Standard_True);
    if (V1.IsNull() || V2.IsNull()) {
      Standard_NullObject::Raise("Cylinder creation aborted: vector is not defined");
    }
    aV = gp_Vec(BRep_Tool::Pnt(V1), BRep_Tool::Pnt(V2));
  }
  else {
    return 0;
  }

  if (anInterface.GetH() < 0.0) aV.Reverse();
  gp_Ax2 anAxes (aP, aV);

  BRepPrimAPI_MakeCylinder mkCylinder(anAxes, anInterface.GetR(), Abs(anInterface.GetH()));

  /*
  cout<<"anInterface.GetR()\t=\t"<<anInterface.GetR()<<endl;
  cout<<"anInterface.GetH()\t=\t"<<anInterface.GetH()<<endl;
  //*/

  mkCylinder.Build();
  if (!mkCylinder.IsDone()) {
    StdFail_NotDone::Raise("Cylinder can't be computed from the given parameters");
  }

  if (!mkCylinder.IsDone()) return CYLINDER_NOT_DONE;
  if (mkCylinder.Shape().IsNull()) return NULL_CYLINDER;
  if (!BRepAlgo::IsValid(mkCylinder.Shape())) return CYLINDER_NOT_DONE;


  //Name result
  
  aResultLabel = Label().FindChild(RESULTS_TAG);
  
  BRepNaming_Cylinder aNaming(aResultLabel);
  aNaming.Load(mkCylinder, BRepNaming_SOLID);
  
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  if(!aLocation.IsIdentity()) 
    TNaming::Displace(aResultLabel, aLocation, Standard_True);
  
  return OK_CYLINDER;
}
