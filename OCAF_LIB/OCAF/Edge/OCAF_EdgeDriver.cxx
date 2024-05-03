// File:	OCAF_EdgeDriver.cxx
// Created:	2010.03.19
// Author:	Wang Yue
//		<id_wangyue@hotmail.com>

#include <CAGDDefine.hxx>


#include "OCAF_EdgeDriver.ixx"
#include <OCAF_IEdge.hxx>
#include <OCAF_IFunction.hxx>

#include <Tags.hxx>

#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepExtrema_DistShapeShape.hxx>

#include <BRepNaming_Edge.hxx>
#include <BRepNaming_TypeOfPrimitive3D.hxx>


#include <TDF_ChildIterator.hxx>

#include <TDataStd_Real.hxx>
#include <TDataStd_TreeNode.hxx>
#include <TDocStd_Modified.hxx>
#include <TNaming_Builder.hxx>
#include <TNaming_NamedShape.hxx>
#include <TNaming.hxx>
#include <TopLoc_Location.hxx>
#include <TopoDS_Shape.hxx>
#include <gp_Pnt.hxx>


#include <Standard_TypeMismatch.hxx>
#include <Standard_NullObject.hxx>

#include <BRepAlgo.hxx>

#include <BRepLib.hxx>

#include <Geom2d_Curve.hxx>

#define OK_EDGE 0
#define EMPTY_EDGE 1
#define EDGE_NOT_DONE 2
#define NULL_EDGE 3

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_EdgeDriver::OCAF_EdgeDriver():OCAF_Driver() {}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_EdgeDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  // 1. check whether this driver's label(initialized from TFunction_Function's label) has a TDataStd_TreeNode attribute
  // 1.1    if have, get the tree node and set as "aNode"
  // 1.2    else return "EMPTY_EDGE", that is to say "no box is created"
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_EDGE;


  /*
  TopLoc_Location aLocation;
  Handle(TNaming_NamedShape) aPrevNS;

  // 2. create a child label "aPrevLabel" of the lable of "aNode"
  TDF_Label aPrevLabel = aNode->Label().FindChild(RESULTS_TAG);
  // 3. check whether "aPrevLabel" have a TNaming_NamedShape attribute "aPrevNS"
  if(!aPrevLabel.IsNull() && aPrevLabel.FindAttribute(TNaming_NamedShape::GetID(), aPrevNS)) {
    // 3.1 if aPrevNS is not Empty, use "aPrevNS" to set "aLocation"
    if(!aPrevNS->IsEmpty())aLocation = aPrevNS->Get().Location();
  }
  //*/

  // 4. construct an instance of OCAF_IBox "anInterface"
  OCAF_IEdge anInterface(aNode);
  Standard_Integer aType = anInterface.GetType();
  TDF_Label aResultLabel;

  if (aType == EDGE_TWO_PNT) {
    TopoDS_Shape aShape1 = anInterface.GetPoint1();
    TopoDS_Shape aShape2 = anInterface.GetPoint2();

    if (aShape1.ShapeType() != TopAbs_VERTEX || aShape2.ShapeType() != TopAbs_VERTEX) {
      Standard_ConstructionError::Raise("Wrong arguments: two points must be given");
    }
    if (aShape1.IsSame(aShape2)) {
      Standard_ConstructionError::Raise("The end points must be different");
    }

    gp_Pnt P1 = BRep_Tool::Pnt(TopoDS::Vertex(aShape1));
    gp_Pnt P2 = BRep_Tool::Pnt(TopoDS::Vertex(aShape2));

    if (P1.Distance(P2) < Precision::Confusion()) {
      Standard_ConstructionError::Raise("The end points are too close");
    }

    // 5_1. make a Edge using the BRepNaming_Edge method.
    BRepBuilderAPI_MakeEdge mkEdge( P1, P2 );
    mkEdge.Build();
    if (!mkEdge.IsDone()) return EDGE_NOT_DONE;
    if (mkEdge.Edge().IsNull()) return NULL_EDGE;
    if (!BRepAlgo::IsValid(mkEdge.Edge())) return EDGE_NOT_DONE;
    // 6_1. create a child label of this driver's label
    aResultLabel = Label().FindChild(RESULTS_TAG);
    // 7_1. append a TNaming_NamedShape attribute to "aResultLabel"
    BRepNaming_Edge aNaming(aResultLabel);
    aNaming.Load(mkEdge, BRepNaming_EDGE, aType);
  }
  else if (aType == EDGE_PNT_DIR) {
    TopoDS_Shape aShape1 =  anInterface.GetPoint1();
    TopoDS_Shape aShape2 =  anInterface.GetLine();
    if (aShape1.ShapeType() != TopAbs_VERTEX) {
      Standard_ConstructionError::Raise("Wrong first argument: must be point");
    }
    if (aShape2.ShapeType() != TopAbs_EDGE) {
      Standard_ConstructionError::Raise("Wrong second argument: must be vector");
    }
    if (aShape1.IsSame(aShape2)) {
      Standard_ConstructionError::Raise("The end points must be different");
    }
    gp_Pnt P1 = BRep_Tool::Pnt(TopoDS::Vertex(aShape1));

    TopoDS_Edge anE = TopoDS::Edge(aShape2);
    TopoDS_Vertex V1, V2;
    TopExp::Vertices(anE, V1, V2, Standard_True);
    if (V1.IsNull() || V2.IsNull()) {
      Standard_NullObject::Raise("Edge creation aborted: vector is not defined");
    }
    gp_Pnt PV1 = BRep_Tool::Pnt(V1);
    gp_Pnt PV2 = BRep_Tool::Pnt(V2);
    if (PV1.Distance(PV2) < Precision::Confusion()) {
      Standard_ConstructionError::Raise("Vector with null magnitude");
    }
    gp_Pnt P2 (P1.XYZ() + PV2.XYZ() - PV1.XYZ());

    // 5_2. make a Edge using the BRepNaming_Edge method.
    BRepBuilderAPI_MakeEdge mkEdge( P1, P2 );
    mkEdge.Build();
    if (!mkEdge.IsDone()) return EDGE_NOT_DONE;
    if (mkEdge.Edge().IsNull()) return NULL_EDGE;

    if (!BRepAlgo::IsValid(mkEdge.Edge())) return EDGE_NOT_DONE;
    // 6_2. create a child label of this driver's label
    aResultLabel = Label().FindChild(RESULTS_TAG);
    // 7_2. append a TNaming_NamedShape attribute to "aResultLabel"
    BRepNaming_Edge aNaming(aResultLabel);
    aNaming.Load(mkEdge, BRepNaming_EDGE, aType);
  }
  else if (aType == EDGE_ON_SURFACE) {
    TopoDS_Shape aRefShape =  anInterface.GetSurface();
    if (aRefShape.ShapeType() != TopAbs_FACE) {
      Standard_TypeMismatch::Raise("Point On Surface creation aborted : surface shape is not a face");
    }
    TopoDS_Face F = TopoDS::Face(aRefShape);
    Handle(Geom_Surface) aSurf = BRep_Tool::Surface(F);
    Standard_Real U1,U2,V1,V2;
    Standard_Real u1,u2,v1,v2;

    ShapeAnalysis::GetFaceUVBounds(F,U1,U2,V1,V2);
    u1 = U1 + (U2-U1) * anInterface.GetParameterU1();
    v1 = V1 + (V2-V1) * anInterface.GetParameterV1();
    u2 = U1 + (U2-U1) * anInterface.GetParameterU2();
    v2 = V1 + (V2-V1) * anInterface.GetParameterV2();

    gp_Pnt2d P2d_first(u1, v1), P2d_last(u2, v2);

    //Handle(Geom2d_Curve) aCurve =  Handle(Geom2d_Curve)::DownCast(GCE2d_MakeSegment(P2d_first, P2d_last).Value());
    const Handle(Geom2d_TrimmedCurve)& aCurve = GCE2d_MakeSegment(P2d_first, P2d_last).Value();

    Standard_Real T1,T2,t1,t2;
    T1 = aCurve->FirstParameter();
    T2 = aCurve->LastParameter();
    t1 = T1 + (T2-T1) * anInterface.GetParameterT1();
    t2 = T1 + (T2-T1) * anInterface.GetParameterT2();
 
    // 5_3. make a Edge using the BRepNaming_Edge method.
    BRepBuilderAPI_MakeEdge mkEdge(aCurve, aSurf, t1, t2);
    mkEdge.Build();
    if (!mkEdge.IsDone()) return EDGE_NOT_DONE;
    if (mkEdge.Edge().IsNull()) return NULL_EDGE;

    TopoDS_Edge aEdge = mkEdge.Edge();
    BRepLib::BuildCurves3d(aEdge);

    if (!BRepAlgo::IsValid(aEdge)) return EDGE_NOT_DONE;


    // 6_3. create a child label of this driver's label
    aResultLabel = Label().FindChild(RESULTS_TAG);
    // 7_3. append a TNaming_NamedShape attribute to "aResultLabel"
    BRepNaming_Edge aNaming(aResultLabel);
    aNaming.Load(aEdge, BRepNaming_EDGE);
  }

  // 8. record the evolution information using TDocStd_Modified
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  /*
  if(!aLocation.IsIdentity()) 
    TNaming::Displace(aResultLabel, aLocation, Standard_True);
  */

  return OK_EDGE;
}
