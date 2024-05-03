#include <CAGDDefine.hxx>

#include "OCAF_MirrorDriver.ixx"
#include <OCAF_IMirror.hxx>
#include <OCAF_IFunction.hxx>
#include <BRepNaming_Mirror.hxx>

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


#include <BRep_Tool.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <Geom_Plane.hxx>

#include <OCAF_Object.hxx>
#include <OCAF_ObjectTool.hxx>


#define OK_MIRROR 0
#define MIRROR_NOT_DONE 1
#define NULL_MIRROR 2
#define CONTEXT_IS_INVALID 3
#define VERTEX_IS_INVALID 4
#define AXIS_IS_INVALID 5
#define ORIGIN_IS_INVALID 6
#define PLANE_IS_INVALID 7

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================

OCAF_MirrorDriver::OCAF_MirrorDriver():OCAF_Driver() {}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_MirrorDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  //find the MirrorFunction Node
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return NULL_MIRROR;

  OCAF_IMirror anInterface(aNode);
  Standard_Integer aType = anInterface.GetType();
  Standard_Boolean isCopyMode = anInterface.GetCopyMode();
  gp_Trsf aTrsf;
  TDF_Label aResultLabel;
  TopoDS_Shape anOriginal, aShape;


  // 1. get the original function's result
  if(isCopyMode){
    Handle(TDataStd_TreeNode) aContext = anInterface.GetContextNode();
    OCAF_Object anInterface1(aContext );
    anOriginal = anInterface1.GetObjectValue();
  }else{
    Handle(TDataStd_TreeNode) anOriginalFunction = anInterface.GetOriginalNode();
    if (anOriginalFunction.IsNull()) return ORIGIN_IS_INVALID;
    OCAF_IFunction anInterface1(anOriginalFunction);
    anOriginal = anInterface1.GetFunctionResult();
  }
  if (anOriginal.IsNull()) return ORIGIN_IS_INVALID;

  // Bug 12158: Check for standalone (not included in faces) degenerated edges
  TopTools_IndexedDataMapOfShapeListOfShape aEFMap;
  TopExp::MapShapesAndAncestors(anOriginal, TopAbs_EDGE, TopAbs_FACE, aEFMap);
  Standard_Integer i, nbE = aEFMap.Extent();
  for (i = 1; i <= nbE; i++) {
    TopoDS_Shape anEdgeSh = aEFMap.FindKey(i);
    if (BRep_Tool::Degenerated(TopoDS::Edge(anEdgeSh))) {
      const TopTools_ListOfShape& aFaces = aEFMap.FindFromIndex(i);
      if (aFaces.IsEmpty())
        Standard_ConstructionError::Raise("Mirror aborted : cannot process standalone degenerated edge");
    }
  }

  // 2. Perform Mirror
  if (aType == MIRROR_PLANE) {
    TopoDS_Shape aPlane = anInterface.GetPlane();
    if (aPlane.IsNull() || aPlane.ShapeType() != TopAbs_FACE) return  PLANE_IS_INVALID;
    TopoDS_Face aFace = TopoDS::Face(aPlane);

    Handle(Geom_Surface) surf = BRep_Tool::Surface(aFace);
    Handle(Geom_Plane) myPlane = Handle(Geom_Plane)::DownCast(surf);
    const gp_Ax3 pos = myPlane->Position();
    const gp_Pnt loc = pos.Location();  /* location of the plane */
    const gp_Dir dir = pos.Direction(); /* Main direction of the plane (Z axis) */
    gp_Ax2 aPln (loc, dir);
    aTrsf.SetMirror(aPln);
  }
  else if (aType == MIRROR_AXIS) {
    TopoDS_Shape anAxis = anInterface.GetAxis();
    if (anAxis.IsNull() || anAxis.ShapeType() != TopAbs_EDGE) return AXIS_IS_INVALID;
    TopoDS_Edge anEdge = TopoDS::Edge(anAxis);

    gp_Pnt aP1 = BRep_Tool::Pnt(TopExp::FirstVertex(anEdge));
    gp_Pnt aP2 = BRep_Tool::Pnt(TopExp::LastVertex (anEdge));
    gp_Vec aV (aP1, aP2);
    gp_Ax1 anAx1 (aP1, aV);
    aTrsf.SetMirror(anAx1);
  } 
  else if (aType == MIRROR_POINT) {
    TopoDS_Shape aPoint = anInterface.GetPoint();
    if (aPoint.IsNull() || aPoint.ShapeType() != TopAbs_VERTEX) return VERTEX_IS_INVALID;
    TopoDS_Vertex aVertex = TopoDS::Vertex(aPoint);

    gp_Pnt aP = BRep_Tool::Pnt(aVertex);
    aTrsf.SetMirror(aP);
  }
  else {
  }


  BRepBuilderAPI_Transform aTransformation (anOriginal, aTrsf, Standard_False);
  aShape = aTransformation.Shape();

  // 6. create a child label of this driver's label
  aResultLabel = Label().FindChild(RESULTS_TAG);
  // 7. append a TNaming_NamedShape attribute to "aResultLabel"
  BRepNaming_Mirror aNaming(aResultLabel);
  aNaming.Load(aShape, BRepNaming_MIRROR);

  // 8. record the evolution information using TDocStd_Modified
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  return OK_MIRROR;
}












