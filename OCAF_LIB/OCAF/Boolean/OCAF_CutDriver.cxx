
#include "OCAF_CutDriver.ixx"
#include <OCAF_ICut.hxx>
#include <OCAF_IFunction.hxx>
  

#include <Tags.hxx>

#include <BRepAlgoAPI_Cut.hxx>
#include <TDF_Reference.hxx>
#include <TDocStd_Modified.hxx>
#include <TNaming_Builder.hxx>
#include <TNaming_NamedShape.hxx>
#include <TopoDS_Shape.hxx>
#include <BRepAlgo.hxx>
#include <Precision.hxx>


#include <BRepNaming_Cut.hxx>
#include <TDF_ChildIterator.hxx>

#include <TDataStd_TreeNode.hxx>

#include <TColStd_IndexedDataMapOfTransientTransient.hxx>
#include <TNaming_CopyShape.hxx>

#include <GEOMUtils.hxx>

#define OK_OPERATION 0
#define REFERENCE_NOT_FOUND 1
#define TREENODE_NOT_FOUND 2
#define NAMED_SHAPE_NOT_FOUND 3
#define NAMED_SHAPE_EMPTY 4
#define OPERATION_NOT_DONE 8
#define NULL_OPERATION 9
#define BAD_OPERATION 10

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================

OCAF_CutDriver::OCAF_CutDriver()
{
}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================

Standard_Integer OCAF_CutDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const
{
  Handle(TDF_Reference) aReference;
  TopoDS_Shape aMaster, aTool;

  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return TREENODE_NOT_FOUND;

  OCAF_ICut anInterface(aNode);
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
  //cout<<"Precision::Confusion()\t=\t"<<Precision::Confusion()<<endl;
  Standard_Real Fuz = anInterface.GetTolerance();
  //cout<<"Fuz\t=\t"<<Fuz<<endl;

  BRepAlgoAPI_Cut mkCut;
  TopTools_ListOfShape L1, L2;
  L1.Append(aShapeCopy1);
  L2.Append(aShapeCopy2);
  mkCut.SetArguments(L1);
  mkCut.SetTools(L2);
  mkCut.SetFuzzyValue(Fuz);
  mkCut.Build();


  if (!mkCut.IsDone()) return OPERATION_NOT_DONE;

  TopoDS_Shape aResult = mkCut.Shape();
  GEOMUtils::FixShapeAfterBooleanOperation(aResult);

  /*
  TColStd_IndexedDataMapOfTransientTransient aMapTResultShapes;
  TopoDS_Shape aResultCopy;
  TNaming_CopyShape::CopyTool(aResult, aMapTResultShapes, aResultCopy);
  //*/

  if (aResult.IsNull()) return NULL_OPERATION;
  if (!BRepAlgo::IsValid(aResult)) return OPERATION_NOT_DONE;

  // Name result
  TDF_Label ResultLabel = Label().FindChild(RESULTS_TAG);
  
  BRepNaming_Cut aNaming(ResultLabel);
  //aNaming.Load(mkCut);
  aNaming.Load(aResult, BRepNaming_CUT);

  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  return OK_OPERATION;
}
