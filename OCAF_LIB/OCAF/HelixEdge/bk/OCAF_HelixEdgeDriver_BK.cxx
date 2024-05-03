// File:	OCAF_HelixEdgeDriver.cxx
// Created:	2014.03.27
// Author:	Wang Yue
//		<id_wangyue@hotmail.com>

#include <CAGDDefine.hxx>
#include "OCAF_HelixEdgeDriver.ixx"
#include <OCAF_IHelixEdge.hxx>
#include <OCAF_IFunction.hxx>

#include <Tags.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepNaming_HelixEdge.hxx>
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
#include <TColgp_HArray1OfPnt.hxx>
#include <GeomAPI_Interpolate.hxx>

#define OK_HELIX 0
#define EMPTY_HELIX 1
#define HELIX_NOT_DONE 2
#define NULL_HELIX 3

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_HelixEdgeDriver::OCAF_HelixEdgeDriver():OCAF_Driver() {}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_HelixEdgeDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  // 1. check whether this driver's label(initialized from TFunction_Function's label) has a TDataStd_TreeNode attribute
  // 1.1    if have, get the tree node and set as "aNode"
  // 1.2    else return "EMPTY_HELIX", that is to say "no box is created"
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_HELIX;


  OCAF_IHelixEdge anInterface(aNode);
  TDF_Label aResultLabel;

  Standard_Integer Num_Period_Points=20;

  Standard_Real r = anInterface.GetR();

  Standard_Real L = anInterface.GetPeriodLength();
  Standard_Integer n = (Standard_Integer)anInterface.GetPeriodNum();
  Standard_Real mainAxisDelta  = L/((Standard_Real)Num_Period_Points);

  Standard_Integer dir0 = anInterface.GetAxisDir();
  Standard_Integer dir1 = (dir0+1)%3;
  Standard_Integer dir2 = (dir0+2)%3;


  Standard_Integer Num_Points_Total=n*Num_Period_Points+1;

  Standard_Real orgPnt[3];
  orgPnt[0] = anInterface.GetOrgX();
  orgPnt[1] = anInterface.GetOrgY();
  orgPnt[2] = anInterface.GetOrgZ();

  Standard_Real tmpPnt[3];
  tmpPnt[0] = anInterface.GetOrgX();
  tmpPnt[1] = anInterface.GetOrgY();
  tmpPnt[2] = anInterface.GetOrgZ();

  Handle(TColgp_HArray1OfPnt) Polygon_Points = new TColgp_HArray1OfPnt(1,Num_Points_Total);

  for(Standard_Integer i=1;i<=Num_Points_Total;i++){
    Standard_Real ang = 2.0*M_PI*(Standard_Real(i-1))/Standard_Real(Num_Period_Points);

    tmpPnt[dir0] = orgPnt[dir0] + mainAxisDelta * Standard_Real(i-1);
    tmpPnt[dir1] = orgPnt[dir1] + r*cos(ang);
    tmpPnt[dir2] = orgPnt[dir2] + r*sin(ang);

    Polygon_Points->SetValue(i, gp_Pnt(tmpPnt[0], tmpPnt[1], tmpPnt[2]));
  }

  GeomAPI_Interpolate P_to_Interpolate(Polygon_Points,false,1.0e-10);
  P_to_Interpolate.Perform();
  Handle(Geom_BSplineCurve) Point_to_C = P_to_Interpolate.Curve();

  BRepBuilderAPI_MakeEdge mkEdge(Point_to_C);

  TopoDS_Shape aShape = mkEdge.Edge();

  mkEdge.Build();
  if (!mkEdge.IsDone()) return HELIX_NOT_DONE;
  if (aShape.IsNull()) return NULL_HELIX;
  if (!BRepAlgo::IsValid(aShape)) return HELIX_NOT_DONE;

  // 6_1. create a child label of this driver's label
  aResultLabel = Label().FindChild(RESULTS_TAG);

  // 7_1. append a TNaming_NamedShape attribute to "aResultLabel"
  BRepNaming_HelixEdge aNaming(aResultLabel);
  aNaming.Load(aShape, BRepNaming_HELIXEDGE);

  // 8. record the evolution information using TDocStd_Modified
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  return OK_HELIX;
}
