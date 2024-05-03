#include "OCAF_CommonDriver.ixx"
#include <OCAF_ICommon.hxx>
#include <OCAF_IFunction.hxx>

#include <Tags.hxx>

#include <BRepAlgoAPI_Common.hxx>
#include <TDF_Reference.hxx>
#include <TDocStd_Modified.hxx>
#include <TNaming_Builder.hxx>
#include <TNaming_NamedShape.hxx>
#include <TopoDS_Shape.hxx>

#include <BRepNaming_Common.hxx>
#include <TDF_ChildIterator.hxx>

#include <TDataStd_TreeNode.hxx>

#include <BRepAlgo.hxx>
#include <Precision.hxx>

#include <TColStd_IndexedDataMapOfTransientTransient.hxx>
#include <TNaming_CopyShape.hxx>

#include <GEOMUtils.hxx>

#define OK_OPERATION 0
#define TREENODE_NOT_FOUND 1
#define LABEL_NOT_FOUND 2
#define NAMED_SHAPE_NOT_FOUND 3
#define NAMED_SHAPE_EMPTY 4
#define OPERATION_NOT_DONE 8
#define NULL_OPERATION 9

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================

OCAF_CommonDriver::OCAF_CommonDriver()
{
}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================

Standard_Integer OCAF_CommonDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const
{
  Handle(TDF_Reference) aReference;
  TopoDS_Shape aMaster, aTool;

  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return TREENODE_NOT_FOUND;

  OCAF_ICommon anInterface(aNode);
  aMaster = anInterface.GetObject1();
  aTool = anInterface.GetObject2();
 

  bool isCheckSelfInte = false;
  if(anInterface.GetMaskOfDetectSelfIntersections()){
    isCheckSelfInte = true;
  }
  if(isCheckSelfInte){
    GEOMUtils::CheckSI(aMaster);
    GEOMUtils::CheckSI(aTool);
  }


  TopoDS_Shape aShapeCopy1;
  TopoDS_Shape aShapeCopy2;
  TColStd_IndexedDataMapOfTransientTransient aMapTShapes;
  TNaming_CopyShape::CopyTool(aMaster, aMapTShapes, aShapeCopy1);
  TNaming_CopyShape::CopyTool(aTool, aMapTShapes, aShapeCopy2);


  //Standard_Real Fuz = Precision::Confusion();
  Standard_Real Fuz = anInterface.GetTolerance();

  BRepAlgoAPI_Common mkCommon;
  TopTools_ListOfShape L1, L2;
  L1.Append(aShapeCopy1);
  L2.Append(aShapeCopy2);
  mkCommon.SetArguments(L1);
  mkCommon.SetTools(L2);
  mkCommon.SetFuzzyValue(Fuz);
  mkCommon.Build();


  if (!mkCommon.IsDone()) return OPERATION_NOT_DONE;

  TopoDS_Shape aResult = mkCommon.Shape();
  GEOMUtils::FixShapeAfterBooleanOperation(aResult);


  if (aResult.IsNull()) return NULL_OPERATION;
  //if (!BRepAlgo::IsValid(aResult)) return OPERATION_NOT_DONE;

  // Name result
  TDF_Label ResultLabel = Label().FindChild(RESULTS_TAG);
  
  BRepNaming_Common aNaming(ResultLabel);
  //aNaming.Load(mkCommon);
  aNaming.Load(aResult, BRepNaming_COMMON);

  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  return OK_OPERATION;
}
