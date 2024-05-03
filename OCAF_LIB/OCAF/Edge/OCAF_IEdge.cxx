// File:	OCAF_IEdge.cxx
// Created:	2010.03.19.03:28
// Author:	Wang Yue
//		<id_wangyue@hotmail.com>


#include <CAGDDefine.hxx>

#include <OCAF_IEdge.hxx>
#include <OCAF_EdgeDriver.hxx>

#include <Tags.hxx>

#include <TDF_Data.hxx>
#include <TDF_Label.hxx>

#include <TDataStd_Real.hxx>

#include <TDocStd_Modified.hxx>

#include <TFunction_Function.hxx>
#include <TFunction_Logbook.hxx>

#include <Standard_ConstructionError.hxx>

#include <OCAF_Object.hxx>
#include <OCAF_ObjectTool.hxx>
//=======================================================================
//function : GetID
//purpose  :
//=======================================================================

const Standard_GUID& OCAF_IEdge::GetID() {
  static Standard_GUID anID("22D22E11-C69A-11d4-8F1A-0060B0EE18E8");
  return anID;
}



//=======================================================================
//function : MakeEdge
//purpose  : Adds a line object to the document
//=======================================================================
Handle(TDataStd_TreeNode) OCAF_IEdge::MakeEdge_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
							     Standard_Integer theType,
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
  OCAF_IEdge IEdge(aFunctionNode);
  IEdge.SetName("Edge function");
  IEdge.SetType(theType);  
  */

  OCAF_IFunction aFunctionInterface(aFunctionNode);
  aFunctionInterface.SetName("Edge function");
  aFunctionInterface.SetType(theType);  

  theError = DONE;
  return aFunctionNode;
}

//=======================================================================
//function : MakeEdge
//purpose  : Adds a line object to the document
//=======================================================================
Standard_Boolean OCAF_IEdge::MakeEdge_Execute( TCollection_ExtendedString& theError )
{
  if( myTreeNode.IsNull()) {
    theError = NOTDONE;
    return Standard_False;
  }
  // 5. setup a TFunction_Function's driver "aDriver" from OCAF_EdgeDriver
  OCAF_EdgeDriver aDriver;
  // 5.1  Initialize the Label of "aDriver"
  aDriver.Init(myTreeNode->Label());
  Handle(TFunction_Logbook) log;
  // 5.2  exacute the function driver
  int a = aDriver.Execute(log);
  //if(aDriver.Execute(log) > 0) {
  if(a > 0) {
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
OCAF_IEdge::OCAF_IEdge(const Handle(TDataStd_TreeNode)& aTreeNode):OCAF_IFunction(aTreeNode)
{
  if(myTreeNode.IsNull()) Standard_ConstructionError::Raise("Null TreeNode");  
}
