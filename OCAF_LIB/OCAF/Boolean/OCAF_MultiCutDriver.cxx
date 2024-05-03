#include <OCAF_IMultiCut.hxx>
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


#include <BRepNaming_MultiCut.hxx>
#include <TDF_ChildIterator.hxx>

#include <TDataStd_TreeNode.hxx>

#include <TColStd_IndexedDataMapOfTransientTransient.hxx>
#include <TNaming_CopyShape.hxx>

#include <GEOMUtils.hxx>


#include "OCAF_MultiCutDriver.ixx"

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

OCAF_MultiCutDriver::OCAF_MultiCutDriver()
{
}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================

Standard_Integer OCAF_MultiCutDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const
{
  Handle(TDF_Reference) aReference;
  TopoDS_Shape aMaster, aTool;

  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return TREENODE_NOT_FOUND;

  OCAF_IMultiCut anInterface(aNode);

  bool isCheckSelfInte = false;
  if(anInterface.GetMaskOfDetectSelfIntersections()){
    isCheckSelfInte = true;
  }


  aMaster = anInterface.GetObject1();   // get the first shape

  if(isCheckSelfInte){
    GEOMUtils::CheckSI(aMaster);
  }

  TopoDS_Shape aShapeCopy1;
  TColStd_IndexedDataMapOfTransientTransient aMapTShapes;
  TNaming_CopyShape::CopyTool(aMaster, aMapTShapes, aShapeCopy1);

  TopTools_ListOfShape L1, L2;
  L1.Append(aShapeCopy1);


  TDF_AttributeSequence aRefSeq;
  anInterface.GetCutMultiTools(aRefSeq);
  Standard_Integer nbelements = aRefSeq.Length();

  Standard_Integer ind;

  for(ind = 1; ind<= nbelements; ind++){
    TopoDS_Shape currShape = anInterface.GetCutMultiTool(ind);

    if(isCheckSelfInte){
      GEOMUtils::CheckSI(currShape);
    }

    TopoDS_Shape currShapeCopy;
    TNaming_CopyShape::CopyTool(currShape, aMapTShapes, currShapeCopy);
    L2.Append(currShapeCopy);
  }


  //Standard_Real Fuz = Precision::Confusion();
  Standard_Real Fuz = anInterface.GetTolerance();

  BRepAlgoAPI_Cut mkCut;

  mkCut.SetArguments(L1);
  mkCut.SetTools(L2);
  mkCut.SetFuzzyValue(Fuz);
  mkCut.Build();

  if (!mkCut.IsDone()) return OPERATION_NOT_DONE;

  TopoDS_Shape aResult = mkCut.Shape();
  GEOMUtils::FixShapeAfterBooleanOperation(aResult);


  if (aResult.IsNull()) return NULL_OPERATION;

  // Name result
  TDF_Label ResultLabel = Label().FindChild(RESULTS_TAG);
  
  BRepNaming_MultiCut aNaming(ResultLabel);
  aNaming.Load(aResult, BRepNaming_MULTICUT);


  OCAF_IFunction::AddLogBooks(aNode, theLogbook);


  return OK_OPERATION;
}
