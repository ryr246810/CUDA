
#include <OCAF_ICut.hxx>

#include <OCAF_CutDriver.hxx>

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

const Standard_GUID& OCAF_ICut::GetID()
{
  static Standard_GUID anID("22D22E61-C69A-11d4-8F1A-0060B0EE18E8");
  return anID;
}


//=======================================================================
//function : MakeCut
//purpose  : Adds a box object to the document
//=======================================================================
Handle(TDataStd_TreeNode) OCAF_ICut::MakeCut_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
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
  OCAF_ICut ICut(aFunctionNode);
  ICut.SetName("Cut function");
  */

  OCAF_IFunction aFunctionInterface(aFunctionNode);
  aFunctionInterface.SetName("Cut function");

  theError = DONE;
  return aFunctionNode;
}



//=======================================================================
//function : MakeCut
//purpose  : Adds a box object to the document
//=======================================================================
Standard_Boolean OCAF_ICut::MakeCut_Execute( TCollection_ExtendedString& theError )
{
  if( myTreeNode.IsNull()) {
    theError = NOTDONE;
    return Standard_False;
  }
  // 5. setup a TFunction_Function's driver "aDriver" from OCAF_CutDriver
  OCAF_CutDriver aDriver;
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

OCAF_ICut::OCAF_ICut(const Handle(TDataStd_TreeNode)& aTreeNode)
  :OCAF_IBooleanOperation(aTreeNode)
{
  if(myTreeNode.IsNull()) Standard_ConstructionError::Raise("Null TreeNode");  
}
