// File:	OCAF_CircleDriver.cxx
// Created:	2010.07.15
// Author:	Wang Yue
//		id_wangyue@hotmail.com

#include <CAGDDefine.hxx>


#include "OCAF_CircleDriver.ixx"
#include <OCAF_ICircle.hxx>
#include <OCAF_IFunction.hxx>

#include <Tags.hxx>

#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepExtrema_DistShapeShape.hxx>

#include <BRepNaming_Circle.hxx>
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



#define OK_CIRCLE 0
#define EMPTY_CIRCLE 1
#define CIRCLE_NOT_DONE 2
#define NULL_CIRCLE 3

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_CircleDriver::OCAF_CircleDriver():OCAF_Driver() {}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_CircleDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  // 1. check whether this driver's label(initialized from TFunction_Function's label) has a TDataStd_TreeNode attribute
  // 1.1    if have, get the tree node and set as "aNode"
  // 1.2    else return "EMPTY_CIRCLE", that is to say "no box is created"
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_CIRCLE;

  // 4. construct an instance of OCAF_IBox "anInterface"
  OCAF_ICircle anInterface(aNode);
  Standard_Integer aType = anInterface.GetType();
  TDF_Label aResultLabel;


  TopoDS_Shape aShape;

  if (aType == CIRCLE_PNT_VEC_R) {

    // Center
    gp_Pnt aP = gp::Origin();
    TopoDS_Shape aRefPoint = anInterface.GetPoint();

    if (aRefPoint.ShapeType() != TopAbs_VERTEX) {
      Standard_ConstructionError::Raise("Circle creation aborted: invalid center argument, must be a point");
    }
    aP = BRep_Tool::Pnt(TopoDS::Vertex(aRefPoint));


    // Normal
    gp_Vec aV = gp::DZ();

    TopoDS_Shape aRefVector = anInterface.GetVector();
    if (aRefVector.ShapeType() != TopAbs_EDGE) {
      Standard_ConstructionError::Raise("Circle creation aborted: invalid vector argument, must be a vector or an edge");
    }
    TopoDS_Edge anE = TopoDS::Edge(aRefVector);
    TopoDS_Vertex V1, V2;
    TopExp::Vertices(anE, V1, V2, Standard_True);
    if (!V1.IsNull() && !V2.IsNull()) {
      aV = gp_Vec(BRep_Tool::Pnt(V1), BRep_Tool::Pnt(V2));
      if (aV.Magnitude() < gp::Resolution()) {
	Standard_ConstructionError::Raise("Circle creation aborted: vector of zero length is given");
      }
    }
    // Axes
    gp_Ax2 anAxes (aP, aV);
    // Radius
    double anR = anInterface.GetR();
    if (anR < Precision::Confusion())
      Standard_ConstructionError::Raise("Circle creation aborted: radius value less than 1e-07 is not acceptable");
    // Circle
    gp_Circ aCirc (anAxes, anR);
    aShape = BRepBuilderAPI_MakeEdge(aCirc).Edge();
  }
  else if (aType == CIRCLE_CENTER_TWO_PNT) {
    TopoDS_Shape aRefPoint1 = anInterface.GetPoint1();
    TopoDS_Shape aRefPoint2 = anInterface.GetPoint2();
    TopoDS_Shape aRefPoint3 = anInterface.GetPoint3();
    if (aRefPoint1.ShapeType() == TopAbs_VERTEX &&
        aRefPoint2.ShapeType() == TopAbs_VERTEX &&
        aRefPoint3.ShapeType() == TopAbs_VERTEX)
    {
      gp_Pnt aP1 = BRep_Tool::Pnt(TopoDS::Vertex(aRefPoint1));
      gp_Pnt aP2 = BRep_Tool::Pnt(TopoDS::Vertex(aRefPoint2));
      gp_Pnt aP3 = BRep_Tool::Pnt(TopoDS::Vertex(aRefPoint3));

      if (aP1.Distance(aP2) < gp::Resolution() ||
          aP1.Distance(aP3) < gp::Resolution() ||
          aP2.Distance(aP3) < gp::Resolution())
        Standard_ConstructionError::Raise("Circle creation aborted: coincident points given");

      if (gp_Vec(aP1, aP2).IsParallel(gp_Vec(aP1, aP3), Precision::Angular()))
        Standard_ConstructionError::Raise("Circle creation aborted: points lay on one line");

      double x, y, z, x1, y1, z1, x2, y2, z2, dx, dy, dz, dx2, dy2, dz2, dx3, dy3, dz3, aRadius;
      //Calculations for Radius
      x = aP1.X(); y = aP1.Y(); z = aP1.Z();
      x1 = aP2.X(); y1 = aP2.Y(); z1 = aP2.Z();
      dx = x1 - x;
      dy = y1 - y;
      dz = z1 - z;
      aRadius = sqrt(dx*dx + dy*dy + dz*dz);
      //Calculations for Plane Vector
      x2 = aP3.X(); y2 = aP3.Y(); z2 = aP3.Z();
      dx2 = x2 - x; dy2 = y2 - y; dz2 = z2 - z;

      dx3 = ((dy*dz2) - (dy2*dz))/100;
      dy3 = ((dx2*dz) - (dx*dz2))/100;
      dz3 = ((dx*dy2) - (dx2*dy))/100;
      //Make Plane Vector
      gp_Dir aDir ( dx3, dy3, dz3 );
      //Make Circle
      gp_Ax2 anAxes (aP1, aDir);
      gp_Circ aCirc (anAxes, aRadius);
      aShape = BRepBuilderAPI_MakeEdge(aCirc).Edge();  
    }
  }
  else if (aType == CIRCLE_THREE_PNT) {
    TopoDS_Shape aRefPoint1 = anInterface.GetPoint1();
    TopoDS_Shape aRefPoint2 = anInterface.GetPoint2();
    TopoDS_Shape aRefPoint3 = anInterface.GetPoint3();
    if (aRefPoint1.ShapeType() == TopAbs_VERTEX &&
        aRefPoint2.ShapeType() == TopAbs_VERTEX &&
        aRefPoint3.ShapeType() == TopAbs_VERTEX) {
      gp_Pnt aP1 = BRep_Tool::Pnt(TopoDS::Vertex(aRefPoint1));
      gp_Pnt aP2 = BRep_Tool::Pnt(TopoDS::Vertex(aRefPoint2));
      gp_Pnt aP3 = BRep_Tool::Pnt(TopoDS::Vertex(aRefPoint3));
      if (aP1.Distance(aP2) < gp::Resolution() ||
          aP1.Distance(aP3) < gp::Resolution() ||
          aP2.Distance(aP3) < gp::Resolution())
        Standard_ConstructionError::Raise("Circle creation aborted: coincident points given");
      if (gp_Vec(aP1, aP2).IsParallel(gp_Vec(aP1, aP3), Precision::Angular()))
        Standard_ConstructionError::Raise("Circle creation aborted: points lay on one line");
      Handle(Geom_Circle) aCirc = GC_MakeCircle(aP1, aP2, aP3).Value();
      aShape = BRepBuilderAPI_MakeEdge(aCirc).Edge();
    }
  }
  else {
  }


  //////////////////////////////////////////////////////////////////////////////////////////
  if (aShape.IsNull()) return NULL_CIRCLE;
  if (!BRepAlgo::IsValid(aShape)) return CIRCLE_NOT_DONE;
  // 6_3. create a child label of this driver's label
  aResultLabel = Label().FindChild(RESULTS_TAG);
  // 7_3. append a TNaming_NamedShape attribute to "aResultLabel"
  BRepNaming_Circle aNaming(aResultLabel);
  aNaming.Load(aShape, BRepNaming_EDGE);

  // 8. record the evolution information using TDocStd_Modified
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  /*
  if(!aLocation.IsIdentity()) 
    TNaming::Displace(aResultLabel, aLocation, Standard_True);
  */

  return OK_CIRCLE;
}

