
#include <OCAF_IFuse.hxx>

#include <OCAF_FuseDriver.hxx>

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

const Standard_GUID& OCAF_IFuse::GetID()
{
  static Standard_GUID anID("22D22E62-C69A-11d4-8F1A-0060B0EE18E8");
  return anID;
}


//=======================================================================
//function : MakeFuse
//purpose  : Adds a box object to the document
//=======================================================================
Handle(TDataStd_TreeNode) OCAF_IFuse::MakeFuse_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
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
  OCAF_IFuse IFuse(aFunctionNode);
  IFuse.SetName("Fuse function");
  */

  OCAF_IFunction aFunctionInterface(aFunctionNode);
  aFunctionInterface.SetName("Fuse function");

  theError = DONE;
  return aFunctionNode;
}



//=======================================================================
//function : MakeFuse
//purpose  : Adds a box object to the document
//=======================================================================
Standard_Boolean OCAF_IFuse::MakeFuse_Execute( TCollection_ExtendedString& theError )
{
  if( myTreeNode.IsNull()) {
    theError = NOTDONE;
    return Standard_False;
  }
  // 5. setup a TFunction_Function's driver "aDriver" from OCAF_FuseDriver
  OCAF_FuseDriver aDriver;
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

OCAF_IFuse::OCAF_IFuse(const Handle(TDataStd_TreeNode)& aTreeNode)
  :OCAF_IBooleanOperation(aTreeNode)
{
  if(myTreeNode.IsNull()) Standard_ConstructionError::Raise("Null TreeNode");  
}
