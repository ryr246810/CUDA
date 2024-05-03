// File:	OCAF_CurveDriver.cxx
// Created:	2010.07.15
// Author:	Wang Yue
//		id_wangyue@hotmail.com

#include <CAGDDefine.hxx>


#include "OCAF_CurveDriver.ixx"
#include <OCAF_ICurve.hxx>

#include <Tags.hxx>

#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepExtrema_DistShapeShape.hxx>

#include <BRepNaming_Curve.hxx>
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
#include <gp_Circ.hxx>
#include <GC_MakeCircle.hxx>
#include <Geom_Circle.hxx>
#include <Geom2d_Curve.hxx>

#include <Standard_ConstructionError.hxx>
#include <Standard_TypeMismatch.hxx>
#include <Standard_NullObject.hxx>

#include <BRepAlgo.hxx>
#include <BRepLib.hxx>

#include <TColgp_HArray1OfPnt.hxx>
#include <GeomAPI_Interpolate.hxx>
#include <GeomAPI_PointsToBSpline.hxx>

#include <BRepBuilderAPI_MakePolygon.hxx>

#define OK_CURVE 0
#define EMPTY_CURVE 1
#define CURVE_NOT_DONE 2
#define NULL_CURVE 3

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_CurveDriver::OCAF_CurveDriver():OCAF_Driver() {}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_CurveDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  // 1. check whether this driver's label(initialized from TFunction_Function's label) has a TDataStd_TreeNode attribute
  // 1.1    if have, get the tree node and set as "aNode"
  // 1.2    else return "EMPTY_CURVE", that is to say "no box is created"
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_CURVE;

  // 4. construct an instance of OCAF_IBox "anInterface"
  OCAF_ICurve anInterface(aNode);
  TDF_Label aResultLabel;

  TopoDS_Shape aShape;
  /////////////////////////////////////////////////////////////  
  Standard_Integer numOfRow = anInterface.GetRowCount();
  Handle(TColStd_HArray1OfReal) Curve_Real = anInterface.GetArray();
  Standard_Integer dIndx_a = Curve_Real->Lower();


  Standard_Integer aType = anInterface.GetType();


  if(aType==CURVE_INTERPOLATE){
    Standard_Integer pIndx_a = 1;
    Standard_Integer pIndx_z = pIndx_a + numOfRow - 1;

    Handle(TColgp_HArray1OfPnt) Curve_Points = new TColgp_HArray1OfPnt(pIndx_a, pIndx_z);

    for(Standard_Integer i=pIndx_a; i<=pIndx_z; i++){
      Standard_Integer j= dIndx_a + (i-pIndx_a)*3;
      Curve_Points->SetValue(i, gp_Pnt(Curve_Real->Value(j),
				       Curve_Real->Value(j+1),
				       Curve_Real->Value(j+2)));
    }

    GeomAPI_Interpolate P_to_Interpolate(Curve_Points,false,1.0e-7);
    //GeomAPI_PointsToBSpline P_to_Interpolate(*Curve_Points);
    P_to_Interpolate.Perform();
    Handle(Geom_BSplineCurve) Point_to_C = P_to_Interpolate.Curve();
    
    BRepBuilderAPI_MakeEdge mkEdge(Point_to_C);
    
    aShape = mkEdge.Edge();
    
    mkEdge.Build();
    if (!mkEdge.IsDone()) return CURVE_NOT_DONE; 
  }else if(aType == CURVE_POLYGON){
    BRepBuilderAPI_MakePolygon aMakePoly;
    for(Standard_Integer i=0; i<numOfRow; i++){
      Standard_Integer j= dIndx_a + i*3;
      aMakePoly.Add(gp_Pnt(Curve_Real->Value(j),
			   Curve_Real->Value(j+1),
			   Curve_Real->Value(j+2)));
    }
    aShape = aMakePoly.Wire();
  }else{
    return EMPTY_CURVE;
  }

  // //////////////////////////////////////////////////////////////////////////////////////////
  if (aShape.IsNull()) return NULL_CURVE;
  if (!BRepAlgo::IsValid(aShape)) return CURVE_NOT_DONE;
  // 6_3. create a child label of this driver's label
  aResultLabel = Label().FindChild(RESULTS_TAG);
  // 7_3. append a TNaming_NamedShape attribute to "aResultLabel"
  BRepNaming_Curve aNaming(aResultLabel);
  aNaming.Load(aShape, BRepNaming_EDGE);

  // 8. record the evolution information using TDocStd_Modified
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);
  
  return OK_CURVE;
}

