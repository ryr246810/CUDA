// File:	OCAF_RecPeriodEdgeDriver.cxx
// Created:	2014.03.27
// Author:	Wang Yue
//		<id_wangyue@hotmail.com>

#include <CAGDDefine.hxx>
#include "OCAF_RecPeriodEdgeDriver.ixx"
#include <OCAF_IRecPeriodEdge.hxx>
#include <OCAF_IFunction.hxx>

#include <Tags.hxx>
#include <BRepBuilderAPI_MakePolygon.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepNaming_RecPeriodEdge.hxx>
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


#define OK_RPEDGE 0
#define EMPTY_RPEDGE 1
#define RPEDGE_NOT_DONE 2
#define NULL_RPEDGE 3

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_RecPeriodEdgeDriver::OCAF_RecPeriodEdgeDriver():OCAF_Driver() {}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_RecPeriodEdgeDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  // 1. check whether this driver's label(initialized from TFunction_Function's label) has a TDataStd_TreeNode attribute
  // 1.1    if have, get the tree node and set as "aNode"
  // 1.2    else return "EMPTY_RPEDGE", that is to say "no box is created"
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_RPEDGE;


  OCAF_IRecPeriodEdge anInterface(aNode);
  TDF_Label aResultLabel;

  Standard_Real r = anInterface.GetR();
  Standard_Real d = anInterface.GetRippleDepth();

  Standard_Real L1 = anInterface.GetFirstSegmentLength();
  Standard_Real L2 = anInterface.GetSecondSegmentLength();

  Standard_Integer n = (Standard_Integer)anInterface.GetPeriodNum();

  Standard_Integer dir0 = anInterface.GetAxisDir();
  Standard_Integer dir1 = anInterface.GetAmpDir();

  Standard_Real orgPnt[3];
  orgPnt[0] = anInterface.GetOrgX();
  orgPnt[1] = anInterface.GetOrgY();
  orgPnt[2] = anInterface.GetOrgZ();

  BRepBuilderAPI_MakePolygon aMakePoly;


  Standard_Real tmpPnt0[3];
  Standard_Real tmpPnt1[3];
  Standard_Real tmpPnt2[3];
  Standard_Real tmpPnt3[3];
  Standard_Real tmpPnt4[3];

  for(Standard_Integer i=0; i<3; i++){
    tmpPnt0[i] = orgPnt[i];
    tmpPnt1[i] = orgPnt[i];
    tmpPnt2[i] = orgPnt[i];
    tmpPnt3[i] = orgPnt[i];
    tmpPnt4[i] = orgPnt[i];
  }

  for(Standard_Integer i=0;i<n;i++){
    tmpPnt0[dir0] = orgPnt[dir0] + ((Standard_Real)i)*(L1+L2);
    tmpPnt0[dir1] = orgPnt[dir1] + r + d;

    tmpPnt1[dir0] = tmpPnt0[dir0] + L1;
    tmpPnt1[dir1] = tmpPnt0[dir1];

    tmpPnt2[dir0] = tmpPnt1[dir0];
    tmpPnt2[dir1] = orgPnt[dir1] + r - d;

    tmpPnt3[dir0] = tmpPnt2[dir0] + L2;
    tmpPnt3[dir1] = tmpPnt2[dir1];

    aMakePoly.Add(gp_Pnt(tmpPnt0[0], tmpPnt0[1], tmpPnt0[2]));
    aMakePoly.Add(gp_Pnt(tmpPnt1[0], tmpPnt1[1], tmpPnt1[2]));
    aMakePoly.Add(gp_Pnt(tmpPnt2[0], tmpPnt2[1], tmpPnt2[2]));
    aMakePoly.Add(gp_Pnt(tmpPnt3[0], tmpPnt3[1], tmpPnt3[2]));

    if(i==(n-1)){
      Standard_Real tmpPnt4[3];
      tmpPnt3[dir0] = tmpPnt3[dir0];
      tmpPnt3[dir1] = tmpPnt3[dir1] + 2.0*d;
      aMakePoly.Add(gp_Pnt(tmpPnt3[0], tmpPnt3[1], tmpPnt3[2]));
    }
  }

  if (!aMakePoly.IsDone())  return RPEDGE_NOT_DONE;
  TopoDS_Shape aShape = aMakePoly.Wire();

  if (aShape.IsNull()) return NULL_RPEDGE;
  if (!BRepAlgo::IsValid(aShape)) return RPEDGE_NOT_DONE;

  // 6_1. create a child label of this driver's label
  aResultLabel = Label().FindChild(RESULTS_TAG);

  // 7_1. append a TNaming_NamedShape attribute to "aResultLabel"
  BRepNaming_RecPeriodEdge aNaming(aResultLabel);
  aNaming.Load(aShape, BRepNaming_RECPERIODEDGE);

  // 8. record the evolution information using TDocStd_Modified
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  return OK_RPEDGE;
}
