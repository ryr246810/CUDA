#include <CAGDDefine.hxx>

#include "OCAF_MultiRotateDriver.ixx"
#include <OCAF_IMultiRotate.hxx>
#include <OCAF_IFunction.hxx>

#include <BRepNaming_MultiRotate.hxx>

#include <BRepAlgoAPI_Fuse.hxx>

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

#include <TNaming_CopyShape.hxx>
#include <TColStd_IndexedDataMapOfTransientTransient.hxx>

#define OK_MULTIROTATE 0
#define MULTIROTATE_NOT_DONE 1
#define NULL_MULTIROTATE 2
#define AXIS_IS_INVALID 3

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================

OCAF_MultiRotateDriver::OCAF_MultiRotateDriver():OCAF_Driver() {}


//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_MultiRotateDriver::Execute(TFunction_Logbook& theLogbook) const 
{
  //find the TranslateFunction Node
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return NULL_MULTIROTATE;

  OCAF_IMultiRotate anInterface(aNode);
  Standard_Integer periodNum = anInterface.GetPeriodNum();

  // 1. get the context
  TopoDS_Shape aContext = anInterface.GetContext();

  // 2.0 get the axise
  TopoDS_Shape anAxis = anInterface.GetAxis();
  if (anAxis.IsNull() || anAxis.ShapeType() != TopAbs_EDGE) return AXIS_IS_INVALID;
  TopoDS_Edge anEdge = TopoDS::Edge(anAxis);
  gp_Pnt aP1 = BRep_Tool::Pnt(TopExp::FirstVertex(anEdge));
  gp_Pnt aP2 = BRep_Tool::Pnt(TopExp::LastVertex(anEdge));
  gp_Dir aDir(gp_Vec(aP1, aP2));
  gp_Ax1 anAx(aP1, aDir);

  Standard_Real anAngle = M_PI*( anInterface.GetAngle() )/180.0;
  if (fabs(anAngle) < Precision::Angular()) anAngle += 2*M_PI; // NPAL19665,19769

  TopoDS_Shape currShape = aContext;
  TopoDS_Shape aNewShape =  aContext;

  for(Standard_Integer i=1; i<=periodNum; i++){
    TopoDS_Shape anOriginal = aNewShape;
    // 3. set the Location

    gp_Trsf aTrsf;
    aTrsf.SetRotation(anAx, anAngle);

    // 4. set the Location
    TopLoc_Location aLocOrig = anOriginal.Location();
    gp_Trsf aTrsfOrig = aLocOrig.Transformation();
    aTrsfOrig.PreMultiply(aTrsf);
    //aTrsfOrig.Multiply(aTrsf);
    TopLoc_Location aLocRes(aTrsfOrig);
    
    // 5. get the Result Shape
    aNewShape = anOriginal.Located(aLocRes);

    // 6.0 make fuse    



    //*
    TopoDS_Shape aShapeCopy1;
    TopoDS_Shape aShapeCopy2;
    TColStd_IndexedDataMapOfTransientTransient aMapTShapes;
    TNaming_CopyShape::CopyTool(currShape, aMapTShapes, aShapeCopy1);
    TNaming_CopyShape::CopyTool(aNewShape, aMapTShapes, aShapeCopy2);
    
    Standard_Real Fuz = Precision::Confusion();
    
    BRepAlgoAPI_Fuse mkFuse;
    TopTools_ListOfShape L1, L2;
    L1.Append(aShapeCopy1);
    L2.Append(aShapeCopy2);
    mkFuse.SetArguments(L1);
    mkFuse.SetTools(L2);
    mkFuse.SetFuzzyValue(Fuz);
    mkFuse.Build();
    //*/ 


    //BRepAlgoAPI_Fuse mkFuse(currShape, aNewShape);
    currShape = mkFuse.Shape();
  }

  // 7. create a child label of this driver's label
  TDF_Label aResultLabel = Label().FindChild(RESULTS_TAG);
  // 8. append a TNaming_NamedShape attribute to "aResultLabel"
  BRepNaming_MultiRotate aNaming(aResultLabel);
  aNaming.Load(currShape, BRepNaming_MULTIROTATE);

  // 9. record the evolution information using TDocStd_Modified
  // 9.1   to mark "aNode" father's label as modified 

  OCAF_IFunction::AddLabels(aNode, theLogbook);
  TDocStd_Modified::Add(aNode->Father()->Label());
  // 9.2   to mark "aNode" as modified
  theLogbook.SetImpacted(Label());
  TDocStd_Modified::Add(Label()); 
  // 9.3   to mark "aResultLabel" as modified
  theLogbook.SetImpacted(aResultLabel);
  // 9.4   to mark "aResultLabel" children as Impacted!
  TDF_ChildIterator anIterator(aResultLabel);
  for(; anIterator.More(); anIterator.Next()) {
    theLogbook.SetImpacted(anIterator.Value());
  }

  return OK_MULTIROTATE;
}

