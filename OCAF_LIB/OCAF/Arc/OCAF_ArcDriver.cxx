// File:	OCAF_ArcDriver.cxx
// Created:	2010.07.15
// Author:	Wang Yue
//		id_wangyue@hotmail.com

#include <CAGDDefine.hxx>

#include <OCAF_IArc.hxx>
#include <OCAF_IFunction.hxx>

#include <Tags.hxx>

#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepExtrema_DistShapeShape.hxx>

#include <BRepNaming_Arc.hxx>
#include <BRepNaming_TypeOfPrimitive3D.hxx>


#include <TDF_ChildIterator.hxx>

#include <TDataStd_Real.hxx>
#include <TDataStd_TreeNode.hxx>
#include <TDocStd_Modified.hxx>
#include <TNaming_Builder.hxx>
#include <TNaming_NamedShape.hxx>
#include <TNaming.hxx>
#include <TopLoc_Location.hxx>

#include <Precision.hxx>

#include <TopoDS.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Vertex.hxx>
#include <TopAbs.hxx>
#include <TopExp.hxx>

#include <GC_MakeArcOfCircle.hxx>
#include <GC_MakeCircle.hxx>
#include <GC_MakeArcOfEllipse.hxx>
#include <GC_MakeEllipse.hxx>
#include <Standard_ConstructionError.hxx>
#include <Precision.hxx>
#include <gp_Pnt.hxx>
#include <gp_Vec.hxx>
#include <gp_Circ.hxx>
#include <gp_Elips.hxx>
#include <Geom_Circle.hxx>
#include <Geom_Ellipse.hxx>



#include <BRepAlgo.hxx>
#include <BRepLib.hxx>
#include <BRep_Tool.hxx>


#include "OCAF_ArcDriver.ixx"

#define OK_ARC 0
#define EMPTY_ARC 1
#define ARC_NOT_DONE 2
#define NULL_ARC 3

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_ArcDriver::OCAF_ArcDriver():OCAF_Driver() {}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_ArcDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  // 1. check whether this driver's label(initialized from TFunction_Function's label) has a TDataStd_TreeNode attribute
  // 1.1    if have, get the tree node and set as "aNode"
  // 1.2    else return "EMPTY_ARC", that is to say "no box is created"
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_ARC;

  // 4. construct an instance of OCAF_IBox "anInterface"
  OCAF_IArc anInterface(aNode);
  Standard_Integer aType = anInterface.GetType();
  TDF_Label aResultLabel;


  TopoDS_Shape aShape;


  if ((aType == CIRC_ARC_THREE_PNT) || (aType == CIRC_ARC_CENTER) || (aType == ELLIPSE_ARC_CENTER_TWO_PNT)) {
    TopoDS_Shape aRefPoint1 = anInterface.GetPoint1();
    TopoDS_Shape aRefPoint2 = anInterface.GetPoint2();
    TopoDS_Shape aRefPoint3 = anInterface.GetPoint3();

    if (aRefPoint1.ShapeType() == TopAbs_VERTEX &&
        aRefPoint2.ShapeType() == TopAbs_VERTEX &&
        aRefPoint3.ShapeType() == TopAbs_VERTEX){

      gp_Pnt aP1 = BRep_Tool::Pnt(TopoDS::Vertex(aRefPoint1));
      gp_Pnt aP2 = BRep_Tool::Pnt(TopoDS::Vertex(aRefPoint2));
      gp_Pnt aP3 = BRep_Tool::Pnt(TopoDS::Vertex(aRefPoint3));

      if (aP1.Distance(aP2) < gp::Resolution() ||
          aP1.Distance(aP3) < gp::Resolution() ||
          aP2.Distance(aP3) < gp::Resolution())
        Standard_ConstructionError::Raise("Arc creation aborted: coincident points given");

      if (gp_Vec(aP1, aP2).IsParallel(gp_Vec(aP1, aP3), Precision::Angular()))
        Standard_ConstructionError::Raise("Arc creation aborted: points lay on one line");

      if (aType == CIRC_ARC_THREE_PNT)
      {
        GC_MakeArcOfCircle arc(aP1, aP2, aP3);
	Handle(Geom_Curve)  theArcCurve = Handle(Geom_Curve)::DownCast(arc.Value()); 
        aShape = BRepBuilderAPI_MakeEdge(theArcCurve).Edge();
      }
      else if ( aType == CIRC_ARC_CENTER ) { // CIRC_ARC_CENTER
        Standard_Boolean sense = anInterface.GetSense();

        Standard_Real aRad = aP1.Distance(aP2);
        gp_Vec aV1 (aP1, aP2);
        gp_Vec aV2 (aP1, aP3);
        gp_Vec aN = aV1 ^ aV2;

        if (sense)
          aN = -aN;

        GC_MakeCircle circ (aP1, aN, aRad);
        Handle(Geom_Circle) aGeomCirc = circ.Value();

        GC_MakeArcOfCircle arc (aGeomCirc->Circ(), aP2, aP3, Standard_True);

	Handle(Geom_Curve)  theArcCurve = Handle(Geom_Curve)::DownCast(arc.Value()); 

        aShape = BRepBuilderAPI_MakeEdge(theArcCurve).Edge();
      }
      else if ( aType == ELLIPSE_ARC_CENTER_TWO_PNT ) { // ELLIPSE_ARC_CENTER_TWO_PNT
	if ( aP1.Distance(aP2) <= aP1.Distance(aP3) ) {
	  cout << "aP1.Distance(aP2) <= aP1.Distance(aP3)" << endl;
	  gp_Pnt aTmpP = aP2;
	  aP2 = aP3;
	  aP3 = aTmpP;
	}

	GC_MakeEllipse ellipse (aP2, aP3, aP1);
	Handle(Geom_Ellipse) aGeomEllipse = ellipse.Value();

	/*
        gp_Vec aV1 (aP1, aP2);
        gp_Vec aV2 (aP1, aP3);

	double alpha = fabs(aV1.Angle(aV2));
	//*/
	GC_MakeArcOfEllipse arc (aGeomEllipse->Elips(), aP2, aP3, Standard_True);

	Handle(Geom_Curve)  theArcCurve = Handle(Geom_Curve)::DownCast(arc.Value()); 

	aShape = BRepBuilderAPI_MakeEdge(theArcCurve).Edge();
      }
    }
  }
  else {
  }
 

  //////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////
  if (aShape.IsNull()) return NULL_ARC;
  if (!BRepAlgo::IsValid(aShape)) return ARC_NOT_DONE;
  // 6_3. create a child label of this driver's label
  aResultLabel = Label().FindChild(RESULTS_TAG);
  // 7_3. append a TNaming_NamedShape attribute to "aResultLabel"
  BRepNaming_Arc aNaming(aResultLabel);
  aNaming.Load(aShape, BRepNaming_EDGE);

  // 8. record the evolution information using TDocStd_Modified
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  /*
  if(!aLocation.IsIdentity()) 
    TNaming::Displace(aResultLabel, aLocation, Standard_True);
  */

  return OK_ARC;
}

