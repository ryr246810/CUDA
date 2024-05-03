// File:        OCAF_IShell.cxx
// Created:     2010.04.02.09:32 AM
// Author:      Wang Yue
// email        <id_wangyue@hotmail.com>


#include <CAGDDefine.hxx>

#include <OCAF_ISolid.hxx>
#include <OCAF_SolidDriver.hxx>

#include <Tags.hxx>

#include <TDF_Data.hxx>
#include <TDF_Label.hxx>
#include <TDF_LabelMap.hxx>
#include <TDF_MapIteratorOfLabelMap.hxx>

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

const Standard_GUID& OCAF_ISolid::GetID() {
  static Standard_GUID anID("22D22E15-C69A-11d4-8F1A-0060B0EE18E8");
  return anID;
}


//=======================================================================
//function : MakeSolid
//purpose  : Adds a point object to the document
//=======================================================================
Handle(TDataStd_TreeNode) OCAF_ISolid::MakeSolid_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
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
  OCAF_ISolid ISolid(aFunctionNode);
  ISolid.SetName("Solid function");
  ISolid.SetType(theType);  
  //*/

  OCAF_IFunction aFunctionInterface(aFunctionNode);
  aFunctionInterface.SetName("Solid function");
  aFunctionInterface.SetType(theType);  

  theError = DONE;
  return aFunctionNode;
}

//=======================================================================
//function : MakeSolid
//purpose  : Adds a point object to the document
//=======================================================================
Standard_Boolean OCAF_ISolid::MakeSolid_Execute( TCollection_ExtendedString& theError )
{
  if( myTreeNode.IsNull()) {
    theError = NOTDONE;
    return Standard_False;
  }
  // 5. setup a TFunction_Function's driver "aDriver" from OCAF_SolidDriver
  OCAF_SolidDriver aDriver;
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
OCAF_ISolid::OCAF_ISolid(const Handle(TDataStd_TreeNode)& aTreeNode):OCAF_IFunction(aTreeNode)
{
  if(myTreeNode.IsNull()) Standard_ConstructionError::Raise("Null TreeNode");  
}


//=======================================================================
//function : 
//purpose  :
//=======================================================================
Standard_Boolean OCAF_ISolid::SetBuildSolidElement(Handle(TDataStd_TreeNode) aNode) 
{
  //1. get the ARGUMENTS_TAG's Lable "L"
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(SOLID_BUILD_TAG);
  
  //2. deside the current tag "aCurNum", according the Children Number of "L"
  Standard_Integer aCurNum = L.NbChildren() + 1;

  //3. test whether the perious BuildSolid element is NULL
  //3.1 find a label tag "i" whose BuildSolid element is empty
  int i = 1;
  for(i = 1; i < aCurNum; i++){
    if(GetBuildSolidElement(i).IsNull()){
      break;
    }
  }
  //4. Select aNode as the BuildSolid element
  Standard_Boolean result = SetReferenceOfTag(SOLID_BUILD_TAG, i, aNode);

  /*
  if(OCAF_IFunction::GetObjectType(aNode->First()) == OCAF_Selection){
    aNode->Remove();
  }
  //*/

  return result;
}

//=======================================================================
//function : 
//purpose  :
//=======================================================================
Standard_Boolean OCAF_ISolid::SetBuildSolidElements(const TDF_LabelMap& ArgumentsMap) 
{
  TDF_MapIteratorOfLabelMap aIter;
  aIter.Initialize(ArgumentsMap);

  Standard_Boolean result = Standard_True;

  for(;aIter.More();aIter.Next()){
    TDF_Label aLabel = aIter.Key();
    Handle(TDataStd_TreeNode) theNode;
    aLabel.FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(),theNode);
    if(theNode.IsNull()) continue;
    result = result && SetBuildSolidElement(theNode);
  }
  return result;
}


//=======================================================================
//function : 
//purpose  :
//=======================================================================
Standard_Boolean OCAF_ISolid::SetBuildSolidElements(const TDF_AttributeMap& ArgumentsMap) 
{
  TDF_MapIteratorOfAttributeMap aIter;
  aIter.Initialize(ArgumentsMap);

  Standard_Boolean result = Standard_True;

  for(;aIter.More();aIter.Next()){
    Handle(TDataStd_TreeNode) theNode = Handle(TDataStd_TreeNode)::DownCast(aIter.Key());
    if(theNode.IsNull()) {
      continue;
    }
    result = result && SetBuildSolidElement(theNode);
  }
  return result;
}




//=======================================================================
//function : GetSection
//purpose  :  Modified by Wang Yue 2010.04.01
//=======================================================================

TopoDS_Shape OCAF_ISolid::GetBuildSolidElement(Standard_Integer theNumber)
{
  TopoDS_Solid aResult;
  TopoDS_Shape aShape;
  aShape = GetReferenceOfTag(SOLID_BUILD_TAG, theNumber);
  //aShape = GetSelectionArgument(theNumber);
  return aShape;
}

//=======================================================================
//function :
//purpose  :
//=======================================================================
void OCAF_ISolid::ClearBuildSolidElements()
{
  ClearAllArgumentsOfTag(SOLID_BUILD_TAG); 
}

//=======================================================================
//function :
//purpose  :
//=======================================================================
void OCAF_ISolid::GetBuildSolidElementsMap(TDF_AttributeMap& ArgumentsMap)
{
  GetArgumentsMapOfTag(SOLID_BUILD_TAG, ArgumentsMap);
}
