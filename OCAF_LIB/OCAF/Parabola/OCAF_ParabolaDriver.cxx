// File:	OCAF_ParabolaDriver.cxx
// Created:	2010.07.15
// Author:	Wang Yue
//		id_wangyue@hotmail.com

#include <CAGDDefine.hxx>


#include "OCAF_ParabolaDriver.ixx"
#include <OCAF_IParabola.hxx>
#include <OCAF_IFunction.hxx>

#include <Tags.hxx>

#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepExtrema_DistShapeShape.hxx>

#include <BRepNaming_Parabola.hxx>
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
#include <gp_Parab.hxx>
#include <Geom2d_Curve.hxx>

#include <Standard_ConstructionError.hxx>
#include <Standard_TypeMismatch.hxx>
#include <Standard_NullObject.hxx>

#include <BRepAlgo.hxx>
#include <BRepLib.hxx>



#define OK_PARABOLA 0
#define EMPTY_PARABOLA 1
#define PARABOLA_NOT_DONE 2
#define NULL_PARABOLA 3

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_ParabolaDriver::OCAF_ParabolaDriver():OCAF_Driver() {}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_ParabolaDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  // 1. check whether this driver's label(initialized from TFunction_Function's label) has a TDataStd_TreeNode attribute
  // 1.1    if have, get the tree node and set as "aNode"
  // 1.2    else return "EMPTY_PARABOLA", that is to say "no box is created"
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_PARABOLA;

  // 4. construct an instance of OCAF_IParabola "anInterface"
  OCAF_IParabola anInterface(aNode);
  Standard_Integer aType = anInterface.GetType();
  TDF_Label aResultLabel;


  TopoDS_Shape aShape;
  if (aType == PARABOLA_PNT_VV_F) {
    // Center
    gp_Pnt aP = gp::Origin();
    TopoDS_Shape aRefPoint = anInterface.GetCenter();
    if (aRefPoint.ShapeType() != TopAbs_VERTEX) {
      Standard_ConstructionError::Raise("Parabola creation aborted: invalid center argument, must be a point");
    }
    aP = BRep_Tool::Pnt(TopoDS::Vertex(aRefPoint));


    // Normal
    gp_Vec aV = gp::DZ();
    TopoDS_Shape aRefVector = anInterface.GetVector();

    if (aRefVector.ShapeType() != TopAbs_EDGE) {
      Standard_ConstructionError::Raise("Parabola creation aborted: invalid normal vector argument, must be a vector or an edge");
    }
    TopoDS_Edge anE1 = TopoDS::Edge(aRefVector);
    TopoDS_Vertex V1, V2;
    TopExp::Vertices(anE1, V1, V2, Standard_True);
    if (!V1.IsNull() && !V2.IsNull()) {
      aV = gp_Vec(BRep_Tool::Pnt(V1), BRep_Tool::Pnt(V2));
      if (aV.Magnitude() < gp::Resolution()) {
	Standard_ConstructionError::Raise("Parabola creation aborted: normal vector of zero length is given");
      }
    }
    // Axes
    gp_Ax2 anAxes (aP, aV);

    // Main Axis vector
    TopoDS_Shape aRefVectorMaj = anInterface.GetVectorMajor();
    if (aRefVectorMaj.ShapeType() != TopAbs_EDGE) {
      Standard_ConstructionError::Raise("Parabola creation aborted: invalid major axis vector argument, must be a vector or an edge");
    }

    TopoDS_Edge anE2 = TopoDS::Edge(aRefVectorMaj);
    TopoDS_Vertex VV1, VV2;
    TopExp::Vertices(anE2, VV1, VV2, Standard_True);
    if (!VV1.IsNull() && !VV2.IsNull()) {
      gp_Vec aVM (BRep_Tool::Pnt(VV1), BRep_Tool::Pnt(VV2));
      if (aVM.Magnitude() < gp::Resolution()) {
	Standard_ConstructionError::Raise("Parabola creation aborted: major axis vector of zero length is given");
      }
      if (aV.IsParallel(aVM, Precision::Angular())) {
	Standard_ConstructionError::Raise("Parabola creation aborted: normal and major axis vectors are parallel");
      }
      // Axes defined with main axis vector
      anAxes  = gp_Ax2 (aP, aV, aVM);
    }
    // Parabola
    gp_Parab anParab ( anAxes, anInterface.GetFocal() );
    aShape = BRepBuilderAPI_MakeEdge(anParab, anInterface.GetParamt1(), anInterface.GetParamt2() ).Edge();
  }
  else {
  }
  //////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////
  if (aShape.IsNull()) return NULL_PARABOLA;
  if (!BRepAlgo::IsValid(aShape)) return PARABOLA_NOT_DONE;
  // 6_3. create a child label of this driver's label
  aResultLabel = Label().FindChild(RESULTS_TAG);
  // 7_3. append a TNaming_NamedShape attribute to "aResultLabel"
  BRepNaming_Parabola aNaming(aResultLabel);
  aNaming.Load(aShape, BRepNaming_EDGE);

  // 8. record the evolution information using TDocStd_Modified
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  /*
  if(!aLocation.IsIdentity()) 
    TNaming::Displace(aResultLabel, aLocation, Standard_True);
  */

  return OK_PARABOLA;
}

