// File:	OCAF_ConeDriver.cxx
// Created:	2010.07.15
// Author:	Wang Yue
//		id_wangyue@hotmail.com

#include <CAGDDefine.hxx>


#include "OCAF_ConeDriver.ixx"
#include <OCAF_ICone.hxx>
#include <OCAF_IFunction.hxx>
#include <BRepNaming_Cone.hxx>

#include <Tags.hxx>


#include <BRepPrimAPI_MakeCone.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRep_Tool.hxx>


#include <BRepNaming_TypeOfPrimitive3D.hxx>


#include <TDF_ChildIterator.hxx>

#include <TDataStd_Real.hxx>
#include <TDataStd_TreeNode.hxx>
#include <TDocStd_Modified.hxx>
#include <TNaming_Builder.hxx>
#include <TNaming_NamedShape.hxx>
#include <TNaming.hxx>
#include <TopLoc_Location.hxx>


#include <TopoDS.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Vertex.hxx>
#include <TopAbs.hxx>
#include <TopExp.hxx>

#include <Precision.hxx>

#include <gp.hxx>
#include <gp_Pnt.hxx>
#include <gp_Vec.hxx>
#include <gp_Elips.hxx>
#include <Geom2d_Curve.hxx>

#include <Standard_ConstructionError.hxx>
#include <Standard_TypeMismatch.hxx>
#include <Standard_NullObject.hxx>
#include <StdFail_NotDone.hxx>

#include <BRepAlgo.hxx>
#include <BRepLib.hxx>



#define OK_CONE 0
#define EMPTY_CONE 1
#define CONE_NOT_DONE 2
#define NULL_CONE 3

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_ConeDriver::OCAF_ConeDriver():OCAF_Driver() {}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_ConeDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  // 1. check whether this driver's label(initialized from TFunction_Function's label) has a TDataStd_TreeNode attribute
  // 1.1    if have, get the tree node and set as "aNode"
  // 1.2    else return "EMPTY_CONE", that is to say "no box is created"
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_CONE;

  // 4. construct an instance of OCAF_IBox "anInterface"
  OCAF_ICone anInterface(aNode);
  Standard_Integer aType = anInterface.GetType();
  TDF_Label aResultLabel;

  TopoDS_Shape aShape;

  gp_Pnt aP;
  gp_Vec aV;

  Standard_Real aR1 = anInterface.GetR1();
  Standard_Real aR2 = anInterface.GetR2();

  if (aType == CONE_R1_R2_H) {
    aP = gp::Origin();
    aV = gp::DZ();

  }
  else if (aType == CONE_PNT_VEC_R1_R2_H) {
    TopoDS_Shape aRefPoint  = anInterface.GetPoint();
    TopoDS_Shape aRefVector = anInterface.GetVector();
    if (aRefPoint.IsNull() || aRefVector.IsNull()) {
      Standard_NullObject::Raise("Cone creation aborted: point or vector is not defined");
    }
    if (aRefPoint.ShapeType() != TopAbs_VERTEX || aRefVector.ShapeType() != TopAbs_EDGE) {
      Standard_TypeMismatch::Raise("Cone creation aborted: point or vector shapes has wrong type");
    }

    aP = BRep_Tool::Pnt(TopoDS::Vertex(aRefPoint));

    TopoDS_Edge anE = TopoDS::Edge(aRefVector);
    TopoDS_Vertex V1, V2;
    TopExp::Vertices(anE, V1, V2, Standard_True);
    if (V1.IsNull() || V2.IsNull()) {
      Standard_NullObject::Raise("Cylinder creation aborted: vector is not defined");
    }
    aV = gp_Vec(BRep_Tool::Pnt(V1), BRep_Tool::Pnt(V2));

  } else {
  }

  if (anInterface.GetH() < 0.0) aV.Reverse();
  gp_Ax2 anAxes (aP, aV);

  // Cone does not work if same radius
  if (fabs(aR1 - aR2) <= Precision::Confusion()) {
    BRepPrimAPI_MakeCylinder MC (anAxes, (aR1 + aR2)/2.0, Abs(anInterface.GetH()));
    MC.Build();
    if (!MC.IsDone()) {
      StdFail_NotDone::Raise("Cylinder can't be computed from the given parameters");
    }
    aShape = MC.Shape();
  } else {
    BRepPrimAPI_MakeCone MC (anAxes, anInterface.GetR1(), anInterface.GetR2(), Abs(anInterface.GetH()));
    MC.Build();
    if (!MC.IsDone()) {
      StdFail_NotDone::Raise("Cylinder can't be computed from the given parameters");
    }
    aShape = MC.Shape();
  }

  //////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////
  if (aShape.IsNull()) return NULL_CONE;
  if (!BRepAlgo::IsValid(aShape)) return CONE_NOT_DONE;
  // 6_3. create a child label of this driver's label
  aResultLabel = Label().FindChild(RESULTS_TAG);
  // 7_3. append a TNaming_NamedShape attribute to "aResultLabel"
  BRepNaming_Cone aNaming(aResultLabel);
  aNaming.Load(aShape, BRepNaming_SOLID);

  // 8. record the evolution information using TDocStd_Modified
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  /*
  if(!aLocation.IsIdentity()) 
    TNaming::Displace(aResultLabel, aLocation, Standard_True);
  */

  return OK_CONE;
}
