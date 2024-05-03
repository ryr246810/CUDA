// File:	OCAF_TorusDriver.cxx
// Created:	2010.07.15
// Author:	Wang Yue
//		id_wangyue@hotmail.com

#include <CAGDDefine.hxx>


#include "OCAF_TorusDriver.ixx"
#include <OCAF_ITorus.hxx>
#include <OCAF_IFunction.hxx>

#include <Tags.hxx>

#include <BRepPrimAPI_MakeTorus.hxx>

#include <BRepNaming_Torus.hxx>
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



#define OK_TORUS 0
#define EMPTY_TORUS 1
#define TORUS_NOT_DONE 2
#define NULL_TORUS 3

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_TorusDriver::OCAF_TorusDriver():OCAF_Driver() {}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_TorusDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  // 1. check whether this driver's label(initialized from TFunction_Function's label) has a TDataStd_TreeNode attribute
  // 1.1    if have, get the tree node and set as "aNode"
  // 1.2    else return "EMPTY_TORUS", that is to say "no box is created"
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_TORUS;

  // 4. construct an instance of OCAF_IBox "anInterface"
  OCAF_ITorus anInterface(aNode);
  Standard_Integer aType = anInterface.GetType();
  TDF_Label aResultLabel;

  TopoDS_Shape aShape;

  if (aType == TORUS_RR) {
    aShape = BRepPrimAPI_MakeTorus(anInterface.GetRMajor(), anInterface.GetRMinor()).Shape();
  } 
  else if (aType == TORUS_PNT_VEC_RR) {
    TopoDS_Shape aRefPoint  = anInterface.GetPoint();
    TopoDS_Shape aRefVector = anInterface.GetVector();

    if (aRefPoint.ShapeType() != TopAbs_VERTEX) {
      Standard_TypeMismatch::Raise("Torus Center must be a vertex");
    }
    if (aRefVector.ShapeType() != TopAbs_EDGE) {
      Standard_TypeMismatch::Raise("Torus Axis must be an edge");
    }

    gp_Pnt aP = BRep_Tool::Pnt(TopoDS::Vertex(aRefPoint));
    TopoDS_Edge anE = TopoDS::Edge(aRefVector);
    TopoDS_Vertex V1, V2;
    TopExp::Vertices(anE, V1, V2, Standard_True);
    if (V1.IsNull() || V2.IsNull()) {
      Standard_ConstructionError::Raise("Bad edge for the Torus Axis given");
    }

    gp_Vec aV (BRep_Tool::Pnt(V1), BRep_Tool::Pnt(V2));
    if (aV.Magnitude() < Precision::Confusion()) {
      Standard_ConstructionError::Raise("End vertices of edge, defining the Torus Axis, are too close");
    }

    gp_Ax2 anAxes (aP, aV);
    BRepPrimAPI_MakeTorus MT (anAxes, anInterface.GetRMajor(), anInterface.GetRMinor());
    if (!MT.IsDone()) MT.Build();
    if (!MT.IsDone()) StdFail_NotDone::Raise("Torus construction algorithm has failed");
    aShape = MT.Shape();
  } else {
  }

  //////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////
  if (aShape.IsNull()) return NULL_TORUS;
  if (!BRepAlgo::IsValid(aShape)) return TORUS_NOT_DONE;
  // 6_3. create a child label of this driver's label
  aResultLabel = Label().FindChild(RESULTS_TAG);
  // 7_3. append a TNaming_NamedShape attribute to "aResultLabel"
  BRepNaming_Torus aNaming(aResultLabel);
  aNaming.Load(aShape, BRepNaming_SOLID);

  // 8. record the evolution information using TDocStd_Modified
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  /*
  if(!aLocation.IsIdentity()) 
    TNaming::Displace(aResultLabel, aLocation, Standard_True);
  */

  return OK_TORUS;
}
