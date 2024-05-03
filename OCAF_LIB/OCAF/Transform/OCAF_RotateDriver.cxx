#include <CAGDDefine.hxx>

#include "OCAF_RotateDriver.ixx"
#include <OCAF_IRotate.hxx>
#include <OCAF_IFunction.hxx>
#include <BRepNaming_Rotate.hxx>


#include <Tags.hxx>

#include <TDF_Reference.hxx>
#include <TDF_Tool.hxx>
#include <TDF_Label.hxx>
#include <TDF_ChildIterator.hxx>
#include <TDF_AttributeMap.hxx>
#include <TDF_MapIteratorOfAttributeMap.hxx>

#include <TFunction_Function.hxx>

#include <TDataStd_Real.hxx>
#include <TDataStd_Integer.hxx>
#include <TDataStd_TreeNode.hxx>
#include <TDocStd_Modified.hxx>

#include <TNaming.hxx>
#include <TNaming_Tool.hxx>
#include <TNaming_Builder.hxx>
#include <TNaming_Iterator.hxx>
#include <TNaming_NamedShape.hxx>

#include <TopoDS_Shape.hxx>
#include <TopoDS.hxx>
#include <TopAbs.hxx>
#include <TopLoc_Location.hxx>

#include <gp_Vec.hxx>
#include <gp_Trsf.hxx>

#include <TopExp_Explorer.hxx>

#include <TDF_Tool.hxx>
#include <TCollection_AsciiString.hxx>

#include <OCAF_Object.hxx>
#include <OCAF_ObjectTool.hxx>

#define OK_ROTATE 0
#define ROTATE_NOT_DONE 1
#define NULL_ROTATE 2
#define CONTEXT_IS_INVALID 3
#define VERTEX_IS_INVALID 4
#define AXIS_IS_INVALID 5
#define ORIGIN_IS_INVALID 6

#define ROTATE_XYZ 1
#define ROTATE_TWO_POINTS 2
#define ROTATE_VECTOR 3
#define ROTATE_VECTOR_DISTANCE 4

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================

OCAF_RotateDriver::OCAF_RotateDriver():OCAF_Driver() {}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_RotateDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  //find the RotateFunction Node
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return NULL_ROTATE;

  OCAF_IRotate anInterface(aNode);
  Standard_Integer aType = anInterface.GetType();
  Standard_Boolean isCopyMode = anInterface.GetCopyMode();
  gp_Trsf aTrsf;
  gp_Pnt aCP, aP1, aP2;
  TDF_Label aResultLabel;
  TopoDS_Shape anOriginal, aShape;

  // 1. get the original function's result

  if(isCopyMode){
    Handle(TDataStd_TreeNode) aContext = anInterface.GetContextNode();
    OCAF_Object anInterface1(aContext );
    anOriginal = anInterface1.GetObjectValue();
  }
  else{
    Handle(TDataStd_TreeNode) anOriginalFunction = anInterface.GetOriginalNode();
    if (anOriginalFunction.IsNull()) return ORIGIN_IS_INVALID;

    OCAF_IFunction anInterface1(anOriginalFunction);
    anOriginal = anInterface1.GetFunctionResult();
  }

  if (anOriginal.IsNull()) return ORIGIN_IS_INVALID;

  if (aType == ROTATE) {
    TopoDS_Shape anAxis = anInterface.GetAxis();
    if (anAxis.IsNull() || anAxis.ShapeType() != TopAbs_EDGE) return AXIS_IS_INVALID;
    TopoDS_Edge anEdge = TopoDS::Edge(anAxis);

    gp_Pnt aP1 = BRep_Tool::Pnt(TopExp::FirstVertex(anEdge));
    gp_Pnt aP2 = BRep_Tool::Pnt(TopExp::LastVertex(anEdge));
    gp_Dir aDir(gp_Vec(aP1, aP2));
    gp_Ax1 anAx1(aP1, aDir);
    //Standard_Real anAngle = anInterface.GetAngle();
    Standard_Real anAngle = M_PI*( anInterface.GetAngle() )/180.0;
    if (fabs(anAngle) < Precision::Angular()) anAngle += 2*M_PI; // NPAL19665,19769
    aTrsf.SetRotation(anAx1, anAngle);
  }
  else if (aType ==  ROTATE_THREE_POINTS) {
    TopoDS_Shape aCV = anInterface.GetCentPoint();
    TopoDS_Shape aV1 = anInterface.GetPoint1();
    TopoDS_Shape aV2 = anInterface.GetPoint2();

    if(aCV.IsNull() || aCV.ShapeType() != TopAbs_VERTEX) return VERTEX_IS_INVALID;
    if(aV1.IsNull() || aV1.ShapeType() != TopAbs_VERTEX) return VERTEX_IS_INVALID;
    if(aV2.IsNull() || aV2.ShapeType() != TopAbs_VERTEX) return VERTEX_IS_INVALID;

    aCP = BRep_Tool::Pnt(TopoDS::Vertex(aCV));
    aP1 = BRep_Tool::Pnt(TopoDS::Vertex(aV1));
    aP2 = BRep_Tool::Pnt(TopoDS::Vertex(aV2));

    gp_Vec aVec1 (aCP, aP1);
    gp_Vec aVec2 (aCP, aP2);
    gp_Dir aDir (aVec1 ^ aVec2);
    gp_Ax1 anAx1 (aCP, aDir);
    Standard_Real anAngle = aVec1.Angle(aVec2);
    if (fabs(anAngle) < Precision::Angular()) anAngle += 2*M_PI;
    aTrsf.SetRotation(anAx1, anAngle);
  }

  // 3. set the Location
  TopLoc_Location aLocOrig = anOriginal.Location();
  gp_Trsf aTrsfOrig = aLocOrig.Transformation();
  //TopLoc_Location aLocRes (aTrsf * aTrsfOrig); // gp_Trsf::Multiply() has a bug
  aTrsfOrig.PreMultiply(aTrsf);
  TopLoc_Location aLocRes (aTrsfOrig);

  // 4. get the Result Shape
  aShape = anOriginal.Located(aLocRes);

  // 6. create a child label of this driver's label
  aResultLabel = Label().FindChild(RESULTS_TAG);
  // 7. append a TNaming_NamedShape attribute to "aResultLabel"
  BRepNaming_Rotate aNaming(aResultLabel);
  aNaming.Load(aShape, BRepNaming_ROTATE);

  // 8. record the evolution information using TDocStd_Modified
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  return OK_ROTATE;
}












