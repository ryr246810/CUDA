// Created:     2010.04.02.09:32 AM
// Author:      Wang Yue
// email        <id_wangyue@hotmail.com>


#include <CAGDDefine.hxx>

#include <OCAF_IPolygon.hxx>
#include <OCAF_PolygonDriver.hxx>

#include <Tags.hxx>

#include <TDF_Data.hxx>
#include <TDF_Label.hxx>
#include <TDF_LabelMap.hxx>
#include <TDF_MapIteratorOfLabelMap.hxx>
#include <TDF_ListIteratorOfAttributeList.hxx>

#include <TDataStd_Integer.hxx>

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

const Standard_GUID& OCAF_IPolygon::GetID() {
  static Standard_GUID anID("22D22E17-C69A-11d4-8F1A-0060B0EE18E8");
  return anID;
}



//=======================================================================
//function : MakePolygon
//purpose  : Adds a point object to the document
//=======================================================================
Handle(TDataStd_TreeNode) OCAF_IPolygon::MakePolygon_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
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
  OCAF_IPolygon IPolygon(aFunctionNode);
  IPolygon.SetName("Polygon function");
  IPolygon.SetType(theType);  
  //*/

  OCAF_IFunction aFunctionInterface(aFunctionNode);
  aFunctionInterface.SetName("Polygon function");
  aFunctionInterface.SetType(theType);  

  theError = DONE;
  return aFunctionNode;
}

//=======================================================================
//function : MakePolygon
//purpose  : Adds a point object to the document
//=======================================================================
Standard_Boolean OCAF_IPolygon::MakePolygon_Execute( TCollection_ExtendedString& theError )
{
  if( myTreeNode.IsNull()) {
    theError = NOTDONE;
    return Standard_False;
  }
  // 5. setup a TFunction_Function's driver "aDriver" from OCAF_PolygonDriver
  OCAF_PolygonDriver aDriver;
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
OCAF_IPolygon::OCAF_IPolygon(const Handle(TDataStd_TreeNode)& aTreeNode):OCAF_IFunction(aTreeNode)
{
  if(myTreeNode.IsNull()) Standard_ConstructionError::Raise("Null TreeNode");  
}


//=======================================================================
//function : 
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IPolygon::SetBuildPolygonElement(Handle(TDataStd_TreeNode) theNode) 
{
  //1. get the ARGUMENTS_TAG's Lable "L"
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(POLYGON_BUILD_TAG);
  
  //2. deside the current tag "aCurNum", according the Children Number of "L"
  Standard_Integer aCurNum = L.NbChildren() + 1;

  //3. test whether the perious BuildPolygon element is NULL
  //3.1 find a label tag "i" whose BuildPolygon element is empty
  int i = 1;
  for(i = 1; i < aCurNum; i++){
    if(GetBuildPolygonElement(i).IsNull()){
      break;
    }
  }
  if(i != aCurNum){
  }
  //4. Select theNode as the BuildPolygon element
  Standard_Boolean result = SetReferenceOfTag(POLYGON_BUILD_TAG, i, theNode);

  /*
  if(OCAF_IFunction::GetObjectType(theNode->First()) == OCAF_Selection){
    theNode->Remove();
  }
  //*/

  return result;
}

//=======================================================================
//function :  GetBuildPolygonElement
//purpose  :  Modified by Wang Yue 2010.04.01
//=======================================================================
TopoDS_Shape OCAF_IPolygon::GetBuildPolygonElement(Standard_Integer theChildTag)
{
  TopoDS_Shape aShape;
  aShape = GetReferenceOfTag(POLYGON_BUILD_TAG, theChildTag);

  return aShape;
}

//=======================================================================
//function :  
//purpose  :  
//=======================================================================
void OCAF_IPolygon::GetBuildPolygonElements(TDF_AttributeMap& ArgumentsMap)
{
  GetArgumentsMapOfTag(POLYGON_BUILD_TAG, ArgumentsMap);
}

//=======================================================================
//function :  
//purpose  :  
//=======================================================================
void OCAF_IPolygon::GetBuildPolygonElements(TDF_AttributeSequence& ArgumentsSequence)
{
  GetArgumentsSequenceOfTag(POLYGON_BUILD_TAG, ArgumentsSequence);
}
//=======================================================================
//function :  
//purpose  :  
//=======================================================================
void OCAF_IPolygon::ClearBuildPolygonElements()
{
  ClearAllArgumentsOfTag(POLYGON_BUILD_TAG); 
}

//=======================================================================
//function : 
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IPolygon::SetBuildPolygonElements(const TDF_LabelMap& ArgumentsMap) 
{
  TDF_MapIteratorOfLabelMap aIter;
  aIter.Initialize(ArgumentsMap);

  Standard_Boolean result = Standard_True;

  for(;aIter.More();aIter.Next()){
    TDF_Label aLabel = aIter.Key();
    Handle(TDataStd_TreeNode) theNode;
    aLabel.FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(),theNode);
    if(theNode.IsNull()) continue;
    result = result && SetBuildPolygonElement(theNode);
  }
  return result;
}

//=======================================================================
//function : 
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IPolygon::SetBuildPolygonElements(const TDF_AttributeMap& ArgumentsMap) 
{
  TDF_MapIteratorOfAttributeMap aIter;
  aIter.Initialize(ArgumentsMap);

  Standard_Boolean result = Standard_True;

  for(;aIter.More();aIter.Next()){
    Handle(TDataStd_TreeNode) theNode = Handle(TDataStd_TreeNode)::DownCast(aIter.Key());
    if(theNode.IsNull()) {
      continue;
    }
    result = result && SetBuildPolygonElement(theNode);
  }
  return result;
}


//=======================================================================
//function : 
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IPolygon::SetBuildPolygonElements(const TDF_AttributeList& ArgumentsList) 
{
  TDF_ListIteratorOfAttributeList aIter;
  aIter.Initialize(ArgumentsList);

  Standard_Boolean result = Standard_True;

  for(;aIter.More();aIter.Next()){
    Handle(TDataStd_TreeNode) theNode = Handle(TDataStd_TreeNode)::DownCast(aIter.Value());
    if(theNode.IsNull()) {
      continue;
    }
    result = result && SetBuildPolygonElement(theNode);
  }
  return result;
}


//=======================================================================
//function : 
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IPolygon::SetBuildPolygonElements(const TDF_AttributeSequence& ArgumentsSequence) 
{
  Standard_Boolean result = Standard_True;

  Standard_Integer theLength = ArgumentsSequence.Length();
  Standard_Integer ind;

  for( ind = 1 ; ind<=theLength ; ind++){
    Handle(TDataStd_TreeNode) theNode = Handle(TDataStd_TreeNode)::DownCast(ArgumentsSequence(ind));
    if(theNode.IsNull()) {
      continue;
    }
    result = result && SetBuildPolygonElement(theNode);
  }
  return result;
}




//=======================================================================
//function : 
//purpose  :
//=======================================================================
void OCAF_IPolygon::SetClose( const Standard_Boolean _isSolid )
{
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG);
  Handle(TDataStd_Integer) anInt;
  TDF_Label aLabel = L.FindChild(POLYGON_CLOSE_TAG);
  
  Standard_Integer aType = _isSolid ? 1 : 0;

  if(!aLabel.FindAttribute(TDataStd_Integer::GetID(), anInt)) {
      anInt = TDataStd_Integer::Set(aLabel, aType);
  }
  anInt->Set(aType);
}

//=======================================================================
//function : 
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IPolygon::GetIsClose()
{
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG);
  Handle(TDataStd_Integer) anInt;
  TDF_Label aLabel = L.FindChild(POLYGON_CLOSE_TAG);
  
  if(!aLabel.FindAttribute(TDataStd_Integer::GetID(), anInt)) {
    anInt = TDataStd_Integer::Set(aLabel, 0);
  }
  return anInt->Get() ? Standard_True : Standard_False;
}
