#include <CAGDDefine.hxx>

#include "OCAF_TranslateDriver.ixx"
#include <OCAF_ITranslate.hxx>
#include <OCAF_IFunction.hxx>
#include <BRepNaming_Translate.hxx>

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

#define OK_TRANSLATE 0
#define TRANSLATE_NOT_DONE 1
#define NULL_TRANSLATE 2
#define CONTEXT_IS_INVALID 3
#define VERTEX_IS_INVALID 4
#define VECTOR_IS_INVALID 5
#define ORIGIN_IS_INVALID 6

#define TRANSLATE_XYZ 1
#define TRANSLATE_TWO_POINTS 2
#define TRANSLATE_VECTOR 3
#define TRANSLATE_VECTOR_DISTANCE 4

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================

OCAF_TranslateDriver::OCAF_TranslateDriver():OCAF_Driver() {}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_TranslateDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  //find the TranslateFunction Node
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return NULL_TRANSLATE;

  OCAF_ITranslate anInterface(aNode);
  Standard_Integer aType = anInterface.GetType();
  Standard_Boolean isCopyMode = anInterface.GetCopyMode();
  gp_Trsf aTrsf;
  gp_Pnt aP1, aP2;
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

  // 2. set the aTrsf
  if (aType == TRANSLATE_XYZ) {
    gp_Vec aVec (anInterface.GetDX(), anInterface.GetDY(), anInterface.GetDZ());
    aTrsf.SetTranslation(aVec);
  }
  else if (aType == TRANSLATE_TWO_POINTS) {
    TopoDS_Shape aV1 = anInterface.GetPoint1();
    TopoDS_Shape aV2 = anInterface.GetPoint2();
    if(aV1.IsNull() || aV1.ShapeType() != TopAbs_VERTEX) return VERTEX_IS_INVALID;
    if(aV2.IsNull() || aV2.ShapeType() != TopAbs_VERTEX) return VERTEX_IS_INVALID;
    aP1 = BRep_Tool::Pnt(TopoDS::Vertex(aV1));
    aP2 = BRep_Tool::Pnt(TopoDS::Vertex(aV2));

    aTrsf.SetTranslation(aP1, aP2);
  }
  else if (aType == TRANSLATE_VECTOR) {
    TopoDS_Shape aV  = anInterface.GetVector();
    if(aV.IsNull() || aV.ShapeType() != TopAbs_EDGE) return VECTOR_IS_INVALID;
    TopoDS_Edge anEdge = TopoDS::Edge(aV);
    aP1 = BRep_Tool::Pnt(TopExp::FirstVertex(anEdge));
    aP2 = BRep_Tool::Pnt(TopExp::LastVertex(anEdge));

    aTrsf.SetTranslation(aP1, aP2);
  }
  else if (aType == TRANSLATE_VECTOR_DISTANCE) {
    TopoDS_Shape aV  = anInterface.GetVector();
    if(aV.IsNull() || aV.ShapeType() != TopAbs_EDGE) return VECTOR_IS_INVALID;
    TopoDS_Edge anEdge = TopoDS::Edge(aV);
    aP1 = BRep_Tool::Pnt(TopExp::FirstVertex(anEdge));
    aP2 = BRep_Tool::Pnt(TopExp::LastVertex(anEdge));
    gp_Vec aVec (aP1, aP2);
    aVec.Normalize();

    double aDistance = anInterface.GetDistance();
    aTrsf.SetTranslation(aVec * aDistance);
  }

  // 3. set the Location
  TopLoc_Location aLocOrig = anOriginal.Location();
  gp_Trsf aTrsfOrig = aLocOrig.Transformation();
  TopLoc_Location aLocRes (aTrsf * aTrsfOrig);

  // 4. get the Result Shape
  aShape = anOriginal.Located(aLocRes);

  // 6. create a child label of this driver's label
  aResultLabel = Label().FindChild(RESULTS_TAG);
  // 7. append a TNaming_NamedShape attribute to "aResultLabel"
  BRepNaming_Translate aNaming(aResultLabel);
  aNaming.Load(aShape, BRepNaming_TRANSLATE);

  // 8. record the evolution information using TDocStd_Modified
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);


  return OK_TRANSLATE;
}












