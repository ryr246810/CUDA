#include <OCAF_IMultiFuse.hxx>
#include <OCAF_IFunction.hxx>

#include <Tags.hxx>

#include <BRepAlgoAPI_Fuse.hxx>
#include <TDF_Reference.hxx>
#include <TDocStd_Modified.hxx>
#include <TNaming_Builder.hxx>
#include <TNaming_NamedShape.hxx>
#include <TopoDS_Shape.hxx>

#include <BRepNaming_MultiFuse.hxx>
#include <TDF_ChildIterator.hxx>

#include <TDataStd_TreeNode.hxx>

#include <BRepAlgo.hxx>
#include <Precision.hxx>


#include <TColStd_IndexedDataMapOfTransientTransient.hxx>
#include <TNaming_CopyShape.hxx>


#include <BRepNaming_TypeOfPrimitive3D.hxx>

#include <GEOMUtils.hxx>

#include "OCAF_MultiFuseDriver.ixx"

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

OCAF_MultiFuseDriver::OCAF_MultiFuseDriver()
{
}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================

Standard_Integer OCAF_MultiFuseDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const
{
  Handle(TDF_Reference) aReference;
  TopoDS_Shape aMaster, aTool;

  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return TREENODE_NOT_FOUND;

  OCAF_IMultiFuse anInterface(aNode);

  bool isRmExtraEdges = false;
  bool isCheckSelfInte = false;
  if(anInterface.GetMaskOfRemoveExtraEdges()){
    isRmExtraEdges = true;
  }
  if(anInterface.GetMaskOfDetectSelfIntersections()){
    isCheckSelfInte = true;
  }


  TDF_AttributeSequence aRefSeq;
  anInterface.GetMultiFuseElements(aRefSeq);
  Standard_Integer nbelements = aRefSeq.Length();

  Standard_Integer ind;

  TopoDS_Shape aShape = anInterface.GetMultiFuseElement(1);
  if(isCheckSelfInte){
    GEOMUtils::CheckSI(aShape);
  }

  TopoDS_Shape aShapeCopy;
  TColStd_IndexedDataMapOfTransientTransient aMapTShapes;
  TNaming_CopyShape::CopyTool(aShape, aMapTShapes, aShapeCopy);

  TopTools_ListOfShape L1, L2;
  L1.Append(aShapeCopy);

  for(ind = 2; ind<= nbelements; ind++){
    TopoDS_Shape currShape = anInterface.GetMultiFuseElement(ind);

    if(isCheckSelfInte){
      GEOMUtils::CheckSI(currShape);
    }

    TopoDS_Shape currShapeCopy;
    TNaming_CopyShape::CopyTool(currShape, aMapTShapes, currShapeCopy);
    L2.Append(currShapeCopy);
  }


  //Standard_Real Fuz = Precision::Confusion();
  Standard_Real Fuz = anInterface.GetTolerance();

  BRepAlgoAPI_Fuse mkFuse;
  mkFuse.SetArguments(L1);
  mkFuse.SetTools(L2);
  mkFuse.SetFuzzyValue(Fuz);
  mkFuse.Build();

  if (!mkFuse.IsDone()) {
    return OPERATION_NOT_DONE;
  }


  TopoDS_Shape aResult = mkFuse.Shape();
  GEOMUtils::FixShapeAfterBooleanOperation(aResult);
  if (isRmExtraEdges) {
    aResult = GEOMUtils::RemoveExtraEdges(aResult);
  }


  if (aResult.IsNull()) return NULL_OPERATION;


  // Name result
  TDF_Label ResultLabel = Label().FindChild(RESULTS_TAG);
  
  BRepNaming_MultiFuse aNaming(ResultLabel);
  aNaming.Load(aResult, BRepNaming_MULTIFUSE);


  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  return OK_OPERATION;
}
