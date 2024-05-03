// File:	OCAF_IRevolution.cxx
// Created:	2010.05.31
// Author:	Wang Yue
//		<id_wangyue@hotmail.com>



#include <CAGDDefine.hxx>


#include "OCAF_RevolutionDriver.ixx"
#include <OCAF_IRevolution.hxx>
#include <OCAF_IFunction.hxx>

#include <Tags.hxx>

#include <BRepPrimAPI_MakeRevol.hxx>
#include <BRepExtrema_DistShapeShape.hxx>

#include <BRepNaming_Revolution.hxx>
#include <BRepNaming_TypeOfPrimitive3D.hxx>
#include <BRepBuilderAPI_Transform.hxx>

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
#include <gp_Lin.hxx>


#include <Standard_TypeMismatch.hxx>
#include <Standard_NullObject.hxx>

#include <StdFail_NotDone.hxx>

#include <BRepAlgo.hxx>

#include <BRepLib.hxx>

#include <BRepBuilderAPI_NurbsConvert.hxx>
#include <BRepMesh_IncrementalMesh.hxx>


#define OK_REVOLUTION 0
#define EMPTY_REVOLUTION 1
#define REVOLUTION_NOT_DONE 2
#define NULL_REVOLUTION 3

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_RevolutionDriver::OCAF_RevolutionDriver():OCAF_Driver() {}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_RevolutionDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  // 1. check whether this driver's label(initialized from TFunction_Function's label) has a TDataStd_TreeNode attribute
  // 1.1    if have, get the tree node and set as "aNode"
  // 1.2    else return "EMPTY_REVOLUTION", that is to say "no box is created"
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_REVOLUTION;

  // 4. construct an instance of OCAF_IBox "anInterface"
  OCAF_IRevolution anInterface(aNode);
  Standard_Integer aType = anInterface.GetType();
  TDF_Label aResultLabel;

  TopoDS_Shape aShape;

  if (aType == REVOLUTION_BASE_AXIS_ANGLE || aType == REVOLUTION_BASE_AXIS_ANGLE_2WAYS) {
    TopoDS_Shape aShapeBase = anInterface.GetBase();
    TopoDS_Shape aShapeAxis = anInterface.GetAxis();
    if (aShapeAxis.ShapeType() != TopAbs_EDGE) {
      Standard_TypeMismatch::Raise("Revolution Axis must be an edge");
    }

    TopoDS_Edge anE = TopoDS::Edge(aShapeAxis);
    TopoDS_Vertex V1, V2;
    TopExp::Vertices(anE, V1, V2, Standard_True);
    if (V1.IsNull() || V2.IsNull()) {
      Standard_ConstructionError::Raise("Bad edge for the Revolution Axis given");
    }

    gp_Vec aV (BRep_Tool::Pnt(V1), BRep_Tool::Pnt(V2));
    if (aV.Magnitude() < Precision::Confusion()) {
      Standard_ConstructionError::Raise("End vertices of edge, defining the Revolution Axis, are too close");
    }

    if (aShapeBase.ShapeType() == TopAbs_VERTEX) {
      gp_Lin aLine( BRep_Tool::Pnt(V1), gp_Dir(aV) );
      Standard_Real d = aLine.Distance(BRep_Tool::Pnt(TopoDS::Vertex(aShapeBase)));
      if (d < Precision::Confusion()) {
	Standard_ConstructionError::Raise("Vertex to be rotated is too close to Revolution Axis");
      }
    }

    double anInputAngle = anInterface.GetAngle();
    double anAngle = M_PI*anInputAngle/180.0;

    gp_Ax1 anAxis (BRep_Tool::Pnt(V1), aV);
    if (aType == REVOLUTION_BASE_AXIS_ANGLE_2WAYS) {
      gp_Trsf aTrsf;
      aTrsf.SetRotation(anAxis, ( -anAngle ));
      BRepBuilderAPI_Transform aTransformation(aShapeBase, aTrsf, Standard_False);
      aShapeBase = aTransformation.Shape();
      anAngle = anAngle * 2;
    }

    BRepPrimAPI_MakeRevol MR (aShapeBase, anAxis, anAngle, Standard_False);
    if (!MR.IsDone()) MR.Build();
    if (!MR.IsDone()) {
      StdFail_NotDone::Raise("Revolution algorithm has failed");
      return REVOLUTION_NOT_DONE;
    }
    aShape = MR.Shape();
    if (aShape.IsNull()) return NULL_REVOLUTION;
    if (!BRepAlgo::IsValid(aShape)) return REVOLUTION_NOT_DONE;

    /*
    BRepBuilderAPI_NurbsConvert aNurbsConverter;
    aNurbsConverter.Perform(aShape);
    aShape = aNurbsConverter.Shape();

    if (aShape.IsNull()) return NULL_REVOLUTION;
    if (!BRepAlgo::IsValid(aShape)) return REVOLUTION_NOT_DONE;
    //*/

  } else {
  }


  /*
  BRepMesh_IncrementalMesh Inc(aShape, 1.0);
  Inc.Perform();
  aShape = Inc.Shape();
  //*/


  // 6. create a child label of this driver's label
  aResultLabel = Label().FindChild(RESULTS_TAG);
  // 7. append a TNaming_NamedShape attribute to "aResultLabel"
  BRepNaming_Revolution aNaming(aResultLabel);
  aNaming.Load(aShape, BRepNaming_REVOLUTION, aType);
  
  // 8. record the evolution information using TDocStd_Modified
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  return OK_REVOLUTION;
}
