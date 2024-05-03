
#include <OCAF_IMultiFuse.hxx>

#include <OCAF_MultiFuseDriver.hxx>

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

const Standard_GUID& OCAF_IMultiFuse::GetID()
{
  static Standard_GUID anID("22D22E63-C69A-11d4-8F1A-0060B0EE18E8");
  return anID;
}


//=======================================================================
//function : MakeFuse
//purpose  : Adds a box object to the document
//=======================================================================
Handle(TDataStd_TreeNode) 
OCAF_IMultiFuse::
MakeMultiFuse_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
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

  OCAF_IFunction aFunctionInterface(aFunctionNode);
  aFunctionInterface.SetName("MultiFuse function");

  theError = DONE;
  return aFunctionNode;
}



//=======================================================================
//function : MakeFuse
//purpose  : Adds a box object to the document
//=======================================================================
Standard_Boolean 
OCAF_IMultiFuse::
MakeMultiFuse_Execute( TCollection_ExtendedString& theError )
{
  if( myTreeNode.IsNull()) {
    theError = NOTDONE;
    return Standard_False;
  }
  // 5. setup a TFunction_Function's driver "aDriver" from OCAF_MultiFuseDriver
  OCAF_MultiFuseDriver aDriver;
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

OCAF_IMultiFuse::
OCAF_IMultiFuse(const Handle(TDataStd_TreeNode)& aTreeNode)
  :OCAF_IBooleanOperation(aTreeNode)
{
  if(myTreeNode.IsNull()) Standard_ConstructionError::Raise("Null TreeNode");  
}




//=======================================================================
//function :  
//purpose  :  Modified by Wang Yue 2010.04.01
//=======================================================================
TopoDS_Shape 
OCAF_IMultiFuse::
GetMultiFuseElement(Standard_Integer theChildTag)
{
  TopoDS_Shape aShape;
  aShape = GetReferenceOfTag(MULTIFUSE_TAG, theChildTag);

  return aShape;
}



//=======================================================================
//function :  
//purpose  :  
//=======================================================================
void 
OCAF_IMultiFuse::
GetMultiFuseElements(TDF_AttributeSequence& ArgumentsSequence)
{
  GetArgumentsSequenceOfTag(MULTIFUSE_TAG, ArgumentsSequence);
}
//=======================================================================
//function :  
//purpose  :  
//=======================================================================
void 
OCAF_IMultiFuse::
ClearMultiFuseElements()
{
  ClearAllArgumentsOfTag(MULTIFUSE_TAG); 
}


//=======================================================================
//function : 
//purpose  :
//=======================================================================
Standard_Boolean 
OCAF_IMultiFuse::
SetMultiFuseElements(const TDF_AttributeSequence& ArgumentsSequence) 
{
  Standard_Boolean result = Standard_True;

  Standard_Integer theLength = ArgumentsSequence.Length();
  Standard_Integer ind;

  for( ind = 1 ; ind<=theLength ; ind++){
    Handle(TDataStd_TreeNode) theNode = Handle(TDataStd_TreeNode)::DownCast(ArgumentsSequence(ind));
    if(theNode.IsNull()) {
      continue;
    }
    result = result && SetMultiFuseElement(theNode);
  }
  return result;
}




//=======================================================================
//function : 
//purpose  :
//=======================================================================
Standard_Boolean 
OCAF_IMultiFuse::
SetMultiFuseElement(Handle(TDataStd_TreeNode) theNode) 
{
  //1. get the ARGUMENTS_TAG's Lable "L"
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(MULTIFUSE_TAG);
  
  //2. deside the current tag "aCurNum", according the Children Number of "L"
  Standard_Integer aCurNum = L.NbChildren() + 1;

  //3. test whether the perious BuildPolygon element is NULL
  //3.1 find a label tag "i" whose BuildPolygon element is empty
  int i = 1;
  for(i = 1; i < aCurNum; i++){
    if(GetMultiFuseElement(i).IsNull()){
      break;
    }
  }
  if(i != aCurNum){
  }
  //4. Select theNode as the BuildPolygon element
  Standard_Boolean result = SetReferenceOfTag(MULTIFUSE_TAG, i, theNode);

  /*
  if(OCAF_IFunction::GetObjectType(theNode->First()) == OCAF_Selection){
    theNode->Remove();
  }
  //*/

  return result;
}
