// File:	OCAF_CosPeriodEdgeDriver.cxx
// Created:	2014.03.27
// Author:	Wang Yue
//		<id_wangyue@hotmail.com>

#include <CAGDDefine.hxx>
#include "OCAF_CosPeriodEdgeDriver.ixx"
#include <OCAF_ICosPeriodEdge.hxx>
#include <OCAF_IFunction.hxx>

#include <Tags.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepNaming_CosPeriodEdge.hxx>
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

#include <BRepBuilderAPI_MakePolygon.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>

#include <ShapeFix_Wire.hxx>
#include <ShapeFix_Edge.hxx>

#define OK_CPEDGE 0
#define EMPTY_CPEDGE 1
#define CPEDGE_NOT_DONE 2
#define NULL_CPEDGE 3

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_CosPeriodEdgeDriver::OCAF_CosPeriodEdgeDriver():OCAF_Driver() {}



Handle(Geom_BSplineCurve)
OCAF_CosPeriodEdgeDriver::
InterpolateOnePeriod (const Handle(TDataStd_TreeNode)& aNode,
		      const Standard_Integer thePeriodIndx) const
{
  OCAF_ICosPeriodEdge anInterface(aNode);

  Standard_Real phaseShiftAng = anInterface.GetPhaseShiftRatio()*M_PI;

  Standard_Real r = anInterface.GetR();
  Standard_Real d = anInterface.GetRippleDepth();

  Standard_Real L = anInterface.GetRipplePeriodLength();

  Standard_Integer dir0 = anInterface.GetAxisDir();
  Standard_Integer dir1 = anInterface.GetAmpDir();

  const Standard_Integer Num_Period_Points = anInterface.GetPeriodSampleNum();
  const Standard_Integer Num_Points_Total=Num_Period_Points+1;

  Standard_Real orgPnt[3];
  orgPnt[0] = anInterface.GetOrgX();
  orgPnt[1] = anInterface.GetOrgY();
  orgPnt[2] = anInterface.GetOrgZ();

  Standard_Real tmpPnt[3];
  tmpPnt[0] = anInterface.GetOrgX();
  tmpPnt[1] = anInterface.GetOrgY();
  tmpPnt[2] = anInterface.GetOrgZ();

  Standard_Integer dir2 = 3-dir0-dir1;
  Standard_Real theTangent[3];

  Standard_Real deltaRatio = 0.01;
  theTangent[dir0] = 2.0*deltaRatio*M_PI/Standard_Real(Num_Period_Points); 
  theTangent[dir1] = cos(phaseShiftAng + deltaRatio*M_PI/Standard_Real(Num_Period_Points)) - cos(phaseShiftAng - deltaRatio*M_PI/Standard_Real(Num_Period_Points) );
  theTangent[dir2] = 0.0;
  gp_Vec aTangent(theTangent[0], theTangent[1], theTangent[2]);

  if( fabs(theTangent[dir1])<Precision::Confusion()){
    aTangent.Normalize();
  }

  Handle(TColgp_HArray1OfPnt) Polygon_Points = new TColgp_HArray1OfPnt(1,Num_Points_Total);
  for(Standard_Integer i=1;i<=Num_Points_Total;i++)
  {
    Standard_Real tt = Standard_Real(i-1)/Standard_Real(Num_Period_Points) + Standard_Real(thePeriodIndx-1.0);
    Standard_Real ang = 2.0*M_PI*tt + phaseShiftAng;
    Standard_Real value = d*cos(ang)+r;

    tmpPnt[dir0] = orgPnt[dir0] + L*tt;
    tmpPnt[dir1] = orgPnt[dir1] + value;

    Polygon_Points->SetValue(i, gp_Pnt(tmpPnt[0], tmpPnt[1], tmpPnt[2]));
  }

  GeomAPI_Interpolate P_to_Interpolate(Polygon_Points,Standard_False, Precision::Confusion());
  // Here we must apply additional tangency constraints at the ends,
  // to ensure correct smooth connection between sequential periods
  P_to_Interpolate.Load(aTangent,aTangent,Standard_False);
  //P_to_Interpolate.Load(aTangent,aTangent);

  P_to_Interpolate.Perform();

  return P_to_Interpolate.Curve();
}


TopoDS_Shape 
OCAF_CosPeriodEdgeDriver::
Interpolate_1(const Handle(TDataStd_TreeNode)& aNode) const
{
  OCAF_ICosPeriodEdge anInterface(aNode);
  Standard_Integer N_Period = (Standard_Integer)anInterface.GetPeriodNum();

  BRepBuilderAPI_MakeWire mkWire;
  // Build sequential periods, add resulted edges to wire making tool
  // Record also start and end points of the sequence
  for(Standard_Integer i=1;i<=N_Period;i++){
    Handle(Geom_BSplineCurve) Point_to_C = this->InterpolateOnePeriod(aNode,i);
    TopoDS_Edge aEdge = BRepBuilderAPI_MakeEdge(Point_to_C);
    mkWire.Add(aEdge);
  }

  TopoDS_Shape aShape  = mkWire.Wire();

  return aShape;
}



Handle(Geom_BSplineCurve)
OCAF_CosPeriodEdgeDriver::
InterpolateOnePeriod (const Handle(TDataStd_TreeNode)& aNode,
		      const Standard_Integer theSectionNum,
		      const Standard_Integer theSectionIndx,
		      const Standard_Integer thePeriodIndx) const
{
  OCAF_ICosPeriodEdge anInterface(aNode);

  Standard_Real phaseShiftAng = anInterface.GetPhaseShiftRatio()*M_PI;

  Standard_Real r = anInterface.GetR();
  Standard_Real d = anInterface.GetRippleDepth();

  Standard_Real RPL = anInterface.GetRipplePeriodLength();
  Standard_Real frac = 1.0/Standard_Real(theSectionNum);
  Standard_Real L = frac*RPL;


  Standard_Integer dir0 = anInterface.GetAxisDir();
  Standard_Integer dir1 = anInterface.GetAmpDir();

  const Standard_Integer Num_Period_Points = anInterface.GetPeriodSampleNum();
  const Standard_Integer Num_Points_Total = Num_Period_Points+1;

  Standard_Real orgPnt[3];
  orgPnt[0] = anInterface.GetOrgX();
  orgPnt[1] = anInterface.GetOrgY();
  orgPnt[2] = anInterface.GetOrgZ();

  Standard_Real tmpPnt[3];
  tmpPnt[0] = anInterface.GetOrgX();
  tmpPnt[1] = anInterface.GetOrgY();
  tmpPnt[2] = anInterface.GetOrgZ();

  Handle(TColgp_HArray1OfPnt) Polygon_Points = new TColgp_HArray1OfPnt(1,Num_Points_Total);
  for(Standard_Integer i=1;i<=Num_Points_Total;i++)
  {
    Standard_Real tt = Standard_Real(i-1)/Standard_Real(Num_Period_Points) + Standard_Real(theSectionIndx);
    Standard_Real ang = frac*2.0*M_PI*tt + phaseShiftAng;
    Standard_Real value = d*cos(ang)+r;

    tmpPnt[dir0] = orgPnt[dir0] + L*tt + Standard_Real(thePeriodIndx-1.0)*RPL;
    tmpPnt[dir1] = orgPnt[dir1] + value;

    Polygon_Points->SetValue(i, gp_Pnt(tmpPnt[0], tmpPnt[1], tmpPnt[2]));
  }

  GeomAPI_Interpolate P_to_Interpolate(Polygon_Points,Standard_False, Precision::Confusion());
  P_to_Interpolate.Perform();
  return P_to_Interpolate.Curve();
}


TopoDS_Shape 
OCAF_CosPeriodEdgeDriver::
Interpolate_2(const Handle(TDataStd_TreeNode)& aNode) const
{
  OCAF_ICosPeriodEdge anInterface(aNode);
  Standard_Integer N_Period = (Standard_Integer)anInterface.GetPeriodNum();

  BRepBuilderAPI_MakeWire mkWire;
  // Build sequential periods, add resulted edges to wire making tool
  // Record also start and end points of the sequence
  Standard_Integer N_Section = 2;
  for(Standard_Integer i=1;i<=N_Period;i++){
    for(Standard_Integer j=0; j<N_Section; j++){
      Handle(Geom_BSplineCurve) Point_to_C = this->InterpolateOnePeriod(aNode,N_Section,j,i);
      TopoDS_Edge aEdge = BRepBuilderAPI_MakeEdge(Point_to_C);
      mkWire.Add(aEdge);
    }
  }
  TopoDS_Shape aShape  = mkWire.Wire();
  return aShape;
}


TopoDS_Shape 
OCAF_CosPeriodEdgeDriver::
Interpolate_0(const Handle(TDataStd_TreeNode)& aNode) const
{
  OCAF_ICosPeriodEdge anInterface(aNode);

  Standard_Real phaseShiftAng = anInterface.GetPhaseShiftRatio()*M_PI;

  Standard_Real r = anInterface.GetR();
  Standard_Real d = anInterface.GetRippleDepth();

  Standard_Real L = anInterface.GetRipplePeriodLength();

  Standard_Integer dir0 = anInterface.GetAxisDir();
  Standard_Integer dir1 = anInterface.GetAmpDir();

  Standard_Real orgPnt[3];
  orgPnt[0] = anInterface.GetOrgX();
  orgPnt[1] = anInterface.GetOrgY();
  orgPnt[2] = anInterface.GetOrgZ();

  Standard_Real tmpPnt[3];
  tmpPnt[0] = anInterface.GetOrgX();
  tmpPnt[1] = anInterface.GetOrgY();
  tmpPnt[2] = anInterface.GetOrgZ();

  Standard_Integer Num_Period = (Standard_Integer)anInterface.GetPeriodNum();
  Standard_Integer Num_SectionNum = Num_Period + 1;
  Standard_Integer Num_Period_Points = anInterface.GetPeriodSampleNum();
  Standard_Integer Num_Points_Total=  Num_SectionNum * (Num_Period_Points-1) + 1;

  Standard_Real frac = ((Standard_Real)Num_Period)/ ((Standard_Real)Num_SectionNum);
  L = L*frac;

  Handle(TColgp_HArray1OfPnt) Polygon_Points = new TColgp_HArray1OfPnt(1,Num_Points_Total);


  for(Standard_Integer i=1;i<=Num_Points_Total;i++)
  {
    Standard_Real tt = Standard_Real(i-1)/Standard_Real(Num_Period_Points-1);
    Standard_Real ang = frac*2.0*M_PI*tt + phaseShiftAng;
    Standard_Real value = d*cos(ang)+r;
    tmpPnt[dir0] = orgPnt[dir0] + L*tt;
    tmpPnt[dir1] = orgPnt[dir1] + value;
    Polygon_Points->SetValue(i, gp_Pnt(tmpPnt[0], tmpPnt[1], tmpPnt[2]));
  }


  GeomAPI_Interpolate P_to_Interpolate(Polygon_Points,Standard_False, Precision::Confusion());
  P_to_Interpolate.Perform();

  Handle(Geom_BSplineCurve) Point_to_C = P_to_Interpolate.Curve();

  TopoDS_Shape aShape = BRepBuilderAPI_MakeEdge(Point_to_C);

  return aShape;
}



TopoDS_Shape 
OCAF_CosPeriodEdgeDriver::
Polygon_0(const Handle(TDataStd_TreeNode)& aNode) const
{
  OCAF_ICosPeriodEdge anInterface(aNode);

  Standard_Real r = anInterface.GetR();
  Standard_Real d = anInterface.GetRippleDepth();

  Standard_Real L = anInterface.GetRipplePeriodLength();
  Standard_Integer N_Period = (Standard_Integer)anInterface.GetPeriodNum();

  Standard_Integer dir0 = anInterface.GetAxisDir();
  Standard_Integer dir1 = anInterface.GetAmpDir();

  Standard_Real phaseShiftAng = anInterface.GetPhaseShiftRatio()*M_PI;

  Standard_Real orgPnt[3];
  orgPnt[0] = anInterface.GetOrgX();
  orgPnt[1] = anInterface.GetOrgY();
  orgPnt[2] = anInterface.GetOrgZ();

  Standard_Real tmpPnt[3];
  tmpPnt[0] = anInterface.GetOrgX();
  tmpPnt[1] = anInterface.GetOrgY();
  tmpPnt[2] = anInterface.GetOrgZ();

  Standard_Integer Num_Period_Points = anInterface.GetPeriodSampleNum();
  if(Num_Period_Points==0){
    Num_Period_Points = 10;
    anInterface.SetPeriodSampleNum(Num_Period_Points);
  }

  Standard_Integer Num_Points_Total=N_Period*(Num_Period_Points-1) + 1;
  Handle(TColgp_HArray1OfPnt) Polygon_Points = new TColgp_HArray1OfPnt(1,Num_Points_Total);
  BRepBuilderAPI_MakePolygon aMakePoly;
  
  for(Standard_Integer i=1;i<=Num_Points_Total;i++){
    Standard_Real tt = Standard_Real(i-1)/Standard_Real(Num_Period_Points-1);
    
    Standard_Real ang = 2.0*M_PI*tt + phaseShiftAng;
    Standard_Real value = d*cos(ang)+r;
    
    tmpPnt[dir0] = orgPnt[dir0] + L*tt;
    tmpPnt[dir1] = orgPnt[dir1] + value;
    
    aMakePoly.Add(gp_Pnt(tmpPnt[0], tmpPnt[1], tmpPnt[2]));
  }

  TopoDS_Shape aShape = aMakePoly.Wire();
    
  return aShape;
}



TopoDS_Shape 
OCAF_CosPeriodEdgeDriver::
Polygon_1(const Handle(TDataStd_TreeNode)& aNode) const
{
  OCAF_ICosPeriodEdge anInterface(aNode);

  Standard_Real r = anInterface.GetR();
  Standard_Real d = anInterface.GetRippleDepth();

  Standard_Real L = anInterface.GetRipplePeriodLength();

  Standard_Integer dir0 = anInterface.GetAxisDir();
  Standard_Integer dir1 = anInterface.GetAmpDir();

  Standard_Real phaseShiftAng = anInterface.GetPhaseShiftRatio()*M_PI;

  Standard_Real orgPnt[3];
  orgPnt[0] = anInterface.GetOrgX();
  orgPnt[1] = anInterface.GetOrgY();
  orgPnt[2] = anInterface.GetOrgZ();

  Standard_Real tmpPnt[3];
  tmpPnt[0] = anInterface.GetOrgX();
  tmpPnt[1] = anInterface.GetOrgY();
  tmpPnt[2] = anInterface.GetOrgZ();


  Standard_Integer Num_Period = (Standard_Integer)anInterface.GetPeriodNum();
  Standard_Integer Num_SectionNum = Num_Period + 1;
  Standard_Integer Num_Period_Points = anInterface.GetPeriodSampleNum();
  Standard_Integer Num_Points_Total=  Num_SectionNum * (Num_Period_Points-1) + 1;

  Standard_Real frac = ((Standard_Real)Num_Period)/ ((Standard_Real)Num_SectionNum);
  L = L*frac;

  BRepBuilderAPI_MakePolygon aMakePoly;

  for(Standard_Integer i=1;i<=Num_Points_Total;i++)
  {
    Standard_Real tt = Standard_Real(i-1)/Standard_Real(Num_Period_Points-1);
    Standard_Real ang = frac*2.0*M_PI*tt + phaseShiftAng;
    Standard_Real value = d*cos(ang)+r;
    tmpPnt[dir0] = orgPnt[dir0] + L*tt;
    tmpPnt[dir1] = orgPnt[dir1] + value;

    aMakePoly.Add(gp_Pnt(tmpPnt[0], tmpPnt[1], tmpPnt[2]));
  }

  TopoDS_Shape aShape = aMakePoly.Wire();
  return aShape;
}






//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_CosPeriodEdgeDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  // 1. check whether this driver's label(initialized from TFunction_Function's label) has a TDataStd_TreeNode attribute
  // 1.1    if have, get the tree node and set as "aNode"
  // 1.2    else return "EMPTY_CPEDGE", that is to say "no box is created"
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_CPEDGE;

  TDF_Label aResultLabel;

  OCAF_ICosPeriodEdge anInterface(aNode);
  Standard_Integer aType = anInterface.GetType();

  TopoDS_Shape aShape;

  if (aType == COSPERIODEDGE_POLYGON) {
    aShape = Polygon_0(aNode);
  }else{  //}else if(aType == COSPERIODEDGE_SMOOTH){
    aShape = Interpolate_1(aNode);
  }

  if (aShape.IsNull()) return NULL_CPEDGE;
  if (!BRepAlgo::IsValid(aShape)) return CPEDGE_NOT_DONE;

  // 6_1. create a child label of this driver's label
  aResultLabel = Label().FindChild(RESULTS_TAG);

  // 7_1. append a TNaming_NamedShape attribute to "aResultLabel"
  BRepNaming_CosPeriodEdge aNaming(aResultLabel);
  aNaming.Load(aShape, BRepNaming_COSPERIODEDGE);

  // 8. record the evolution information using TDocStd_Modified
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  return OK_CPEDGE;
}
