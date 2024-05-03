
#include <OCAF_IMultiCut.hxx>

#include <OCAF_MultiCutDriver.hxx>

#include <TDF_Label.hxx>

#include <TDocStd_Modified.hxx>

#include <TFunction_Function.hxx>
#include <TFunction_Logbook.hxx>

#include <Standard_ConstructionError.hxx>

#include <OCAF_Object.hxx>
#include <OCAF_ObjectTool.hxx>

#include <Tags.hxx>

//=======================================================================
//function : GetID
//purpose  :
//=======================================================================

const Standard_GUID& OCAF_IMultiCut::GetID()
{
  static Standard_GUID anID("22D22E86-C69A-11d4-8F1A-0060B0EE18E8");
  return anID;
}


//=======================================================================
//function : MakeCut
//purpose  : Adds a box object to the document
//=======================================================================
Handle(TDataStd_TreeNode) OCAF_IMultiCut::MakeMultiCut_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
								     TCollection_ExtendedString& theError)
{
  if (anObjectNode.IsNull() || anObjectNode->Label().IsNull()) {
    theError = NULL_ACCESS_NODE;
    return NULL;
  }

  // 2. Construct an interface "anInterface" used for call OCAF_Object's AddFunction
  OCAF_Object anInterface(anObjectNode);

  // 3. To use "anInterface" to call AddFunction of OCAF_Object
  Handle(TDataStd_TreeNode) aFunctionNode = anInterface.AddFunction(GetID());
  

  if(aFunctionNode.IsNull()) {
    theError = NOTDONE;
    return NULL;
  }

  /*
  OCAF_IMultiCut ICut(aFunctionNode);
  ICut.SetName("Cut function");
  */

  OCAF_IFunction aFunctionInterface(aFunctionNode);
  aFunctionInterface.SetName("MultiCut function");

  theError = DONE;
  return aFunctionNode;
}



//=======================================================================
//function : MakeCut
//purpose  : Adds a box object to the document
//=======================================================================
Standard_Boolean OCAF_IMultiCut::MakeMultiCut_Execute( TCollection_ExtendedString& theError )
{
  if( myTreeNode.IsNull()) {
    theError = NOTDONE;
    return Standard_False;
  }
  // 5. setup a TFunction_Function's driver "aDriver" from OCAF_CutDriver
  OCAF_MultiCutDriver aDriver;
  // 5.1  Initialize the Label of "aDriver"
  aDriver.Init(myTreeNode->Label());
  Handle(TFunction_Logbook) log;
  // 5.2  exacute the function driver
  if(aDriver.Execute(log) > 0) {
    theError = ALGO_FAILED; 
    return Standard_True;
  }

  theError = DONE;
  return Standard_True;
}


//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================

OCAF_IMultiCut::OCAF_IMultiCut(const Handle(TDataStd_TreeNode)& aTreeNode)
  :OCAF_IBooleanOperation(aTreeNode)
{
  if(myTreeNode.IsNull()) Standard_ConstructionError::Raise("Null TreeNode");  
}







//=======================================================================
//function :  
//purpose  :  Modified by Wang Yue 2010.04.01
//=======================================================================
TopoDS_Shape 
OCAF_IMultiCut::
GetCutMultiTool(Standard_Integer theChildTag)
{
  TopoDS_Shape aShape;
  aShape = GetReferenceOfTag(MULTICUT_TAG, theChildTag);

  return aShape;
}



//=======================================================================
//function :  
//purpose  :  
//=======================================================================
void 
OCAF_IMultiCut::
GetCutMultiTools(TDF_AttributeSequence& ArgumentsSequence)
{
  GetArgumentsSequenceOfTag(MULTICUT_TAG, ArgumentsSequence);
}
//=======================================================================
//function :  
//purpose  :  
//=======================================================================
void 
OCAF_IMultiCut::
ClearCutMultiTools()
{
  ClearAllArgumentsOfTag(MULTICUT_TAG); 
}


//=======================================================================
//function : 
//purpose  :
//=======================================================================
Standard_Boolean 
OCAF_IMultiCut::
SetCutMultiTools(const TDF_AttributeSequence& ArgumentsSequence) 
{
  Standard_Boolean result = Standard_True;

  Standard_Integer theLength = ArgumentsSequence.Length();
  Standard_Integer ind;

  for( ind = 1 ; ind<=theLength ; ind++){
    Handle(TDataStd_TreeNode) theNode = Handle(TDataStd_TreeNode)::DownCast(ArgumentsSequence(ind));
    if(theNode.IsNull()) {
      continue;
    }
    result = result && SetCutMultiTool(theNode);
  }
  return result;
}




//=======================================================================
//function : 
//purpose  :
//=======================================================================
Standard_Boolean 
OCAF_IMultiCut::
SetCutMultiTool(Handle(TDataStd_TreeNode) theNode) 
{
  //1. get the ARGUMENTS_TAG's Lable "L"
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(MULTICUT_TAG);
  
  //2. deside the current tag "aCurNum", according the Children Number of "L"
  Standard_Integer aCurNum = L.NbChildren() + 1;

  //3. test whether the perious BuildPolygon element is NULL
  //3.1 find a label tag "i" whose BuildPolygon element is empty
  int i = 1;
  for(i = 1; i < aCurNum; i++){
    if(GetCutMultiTool(i).IsNull()){
      break;
    }
  }
  if(i != aCurNum){
  }
  //4. Select theNode as the BuildPolygon element
  Standard_Boolean result = SetReferenceOfTag(MULTICUT_TAG, i, theNode);

  /*
  if(OCAF_IFunction::GetObjectType(theNode->First()) == OCAF_Selection){
    theNode->Remove();
  }
  //*/

  return result;
}
