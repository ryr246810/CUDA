// File:        OCAF_IPoint.cxx
// Created:     2010.04.02.09:32 AM
// Author:      Wang Yue
// email        <id_wangyue@hotmail.com>


#include <CAGDDefine.hxx>

#include <OCAF_IFace.hxx>
#include <OCAF_FaceDriver.hxx>

#include <Tags.hxx>

#include <TDF_Data.hxx>
#include <TDF_Label.hxx>
#include <TDF_LabelMap.hxx>
#include <TDF_MapIteratorOfLabelMap.hxx>
#include <TDF_ListIteratorOfAttributeList.hxx>

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

const Standard_GUID& OCAF_IFace::GetID() {
  static Standard_GUID anID("22D22E13-C69A-11d4-8F1A-0060B0EE18E8");
  return anID;
}


//=======================================================================
//function : MakeFace
//purpose  : Adds a point object to the document
//=======================================================================
Handle(TDataStd_TreeNode) OCAF_IFace::MakeFace_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
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

  OCAF_IFunction aFunctionInterface(aFunctionNode);
  aFunctionInterface.SetName("Face function");
  aFunctionInterface.SetType(theType);  

  theError = DONE;
  return aFunctionNode;
}

//=======================================================================
//function : MakeFace
//purpose  : Adds a point object to the document
//=======================================================================
Standard_Boolean OCAF_IFace::MakeFace_Execute( TCollection_ExtendedString& theError )
{
  if( myTreeNode.IsNull()) {
    theError = NOTDONE;
    return Standard_False;
  }
  // 5. setup a TFunction_Function's driver "aDriver" from OCAF_FaceDriver
  OCAF_FaceDriver aDriver;
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
OCAF_IFace::OCAF_IFace(const Handle(TDataStd_TreeNode)& aTreeNode):OCAF_IFunction(aTreeNode)
{
  if(myTreeNode.IsNull()) Standard_ConstructionError::Raise("Null TreeNode");  
}


//=======================================================================
//function : SetIsPlanar
//purpose  : 
//=======================================================================
void OCAF_IFace::SetIsPlanar(const Standard_Boolean isPlanar)
{
  if(isPlanar)
    SetInteger(FACE_BUILD_ISPLANAR_TAG, 1);
  else if(!isPlanar)
    SetInteger(FACE_BUILD_ISPLANAR_TAG, 0);
}

//=======================================================================
//function : GetIsPlannar
//purpose  : 
//=======================================================================
Standard_Boolean OCAF_IFace::GetIsPlanar()
{
  if(GetInteger(FACE_BUILD_ISPLANAR_TAG)==1)
    return Standard_True;
  else
    return Standard_False;
}
