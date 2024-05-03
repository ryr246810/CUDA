#include <CAGDDefine.hxx>

#include <OCAF_IPipeShell.hxx>

#include <Tags.hxx>

#include <TopoDS.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepCheck_Shell.hxx>

#include <OCAF_ISelection.hxx>

#include <TDF_ChildIterator.hxx>
#include <TDF_Data.hxx>
#include <TDF_Label.hxx>
#include <TDF_Reference.hxx>
#include <TDF_Tool.hxx>
#include <TDataStd_Real.hxx>
#include <TDataStd_IntegerArray.hxx>

#include <TDocStd_Modified.hxx>

#include <TFunction_Function.hxx>
#include <TFunction_Logbook.hxx>

#include <TNaming_NamedShape.hxx>

#include <Standard_ConstructionError.hxx>

#include <TColStd_Array1OfBoolean.hxx>

#include <OCAF_PipeShellDriver.hxx>

#include <OCAF_Object.hxx>
#include <OCAF_ObjectTool.hxx>
//=======================================================================
//function : GetID
//purpose  :
//=======================================================================
const Standard_GUID& OCAF_IPipeShell::GetID() 
{
  static Standard_GUID anID("22D22E22-C69A-11d4-8F1A-0060B0EE18E8");
  return anID;
}


//=======================================================================
//function :
//purpose  : Adds a point object to the document
//=======================================================================
Handle(TDataStd_TreeNode) OCAF_IPipeShell::MakePipeShell_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
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
  aFunctionInterface.SetName("ThruSection function");
  aFunctionInterface.SetType(theType);  

  theError = DONE;
  return aFunctionNode;
}

//=======================================================================
//function :
//purpose  : Adds a point object to the document
//=======================================================================
Standard_Boolean OCAF_IPipeShell::MakePipeShell_Execute( TCollection_ExtendedString& theError )
{
  if( myTreeNode.IsNull()) {
    theError = NOTDONE;
    return Standard_False;
  }
  // 5. setup a TFunction_Function's driver "aDriver" from OCAF_ThruSectionDriver
  OCAF_PipeShellDriver aDriver;
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
//function : IsClosedSection
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IPipeShell::IsClosedSection(const Handle(TDataStd_TreeNode)& aTreeNode)
{
  OCAF_Object anInt(aTreeNode);

  TopoDS_Shape aShape = anInt.GetObjectValue();

  if(aShape.ShapeType() == TopAbs_SHELL) {
    Handle(BRepCheck_Shell) aCheck = new BRepCheck_Shell(TopoDS::Shell(aShape));
    return (aCheck->Closed() == BRepCheck_NoError);
  } else
    return Standard_True; 
}

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_IPipeShell::OCAF_IPipeShell(const Handle(TDataStd_TreeNode)& aTreeNode)
  :OCAF_IFunction(aTreeNode)
{
  if(myTreeNode.IsNull()) Standard_ConstructionError::Raise("Null TreeNode"); 
}

//=======================================================================
//function : 
//purpose  :
//=======================================================================
void OCAF_IPipeShell::SetSolidOrShell( const Standard_Boolean _isSolid )
{
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG);
  Handle(TDataStd_Integer) anInt;
  TDF_Label aLabel = L.FindChild(PIPESHELL_IS_SOLID_TAG);
  
  Standard_Integer aType = _isSolid ? 1 : 0;

  if(!aLabel.FindAttribute(TDataStd_Integer::GetID(), anInt)) {
      anInt = TDataStd_Integer::Set(aLabel, aType);
  }
  anInt->Set(aType);
}

//=======================================================================
//function : IsSolid
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IPipeShell::IsSolidOrShell()
{
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG);
  Handle(TDataStd_Integer) anInt;
  TDF_Label aLabel = L.FindChild(PIPESHELL_IS_SOLID_TAG);
  
  if(!aLabel.FindAttribute(TDataStd_Integer::GetID(), anInt)) {
    anInt = TDataStd_Integer::Set(aLabel, 1);
  }
  return anInt->Get() ? Standard_True : Standard_False;
}

//=======================================================================
//function : SetTransitionMode
//purpose  :
//=======================================================================
void OCAF_IPipeShell::SetTransitionMode(BRepBuilderAPI_TransitionMode theMode) 
{
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG);
  Handle(TDataStd_Integer) anInt;  
  TDF_Label aLabel = L.FindChild(PIPESHEL_TRANSITION_MODE_TAG);
  
  if(!aLabel.FindAttribute(TDataStd_Integer::GetID(), anInt)) {
    anInt = TDataStd_Integer::Set(aLabel, theMode);
  } else {
    anInt->Set(theMode);
  }
}

//=======================================================================
//function : GetTransitionMode
//purpose  :
//=======================================================================
BRepBuilderAPI_TransitionMode OCAF_IPipeShell::GetTransitionMode() 
{
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG);
  Handle(TDataStd_Integer) anInt;  
  TDF_Label aLabel = L.FindChild(PIPESHEL_TRANSITION_MODE_TAG);
  
  if(!aLabel.FindAttribute(TDataStd_Integer::GetID(), anInt)) {
    anInt = TDataStd_Integer::Set(aLabel, BRepBuilderAPI_Transformed);
  }
  BRepBuilderAPI_TransitionMode aResult;
  aResult = (BRepBuilderAPI_TransitionMode) anInt->Get();
  return aResult;
}

//=======================================================================
//function : AddSpine
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IPipeShell::SetSpine(Handle(TDataStd_TreeNode) theObjNode)
{
  if(!SetReference(PIPESHELL_SPINE_TAG , theObjNode))
    return Standard_False;
  return Standard_True;
}

//=======================================================================
//function : GetSpine
//purpose  :
//=======================================================================
TopoDS_Wire OCAF_IPipeShell::GetSpine() 
{
  TopoDS_Shape aShape;
  TopoDS_Wire aResult;

  aShape = GetReference(PIPESHELL_SPINE_TAG);

  if(!aShape.IsNull()) {
    if(aShape.ShapeType() == TopAbs_WIRE) {
      aResult = TopoDS::Wire(aShape);
    } else if(aShape.ShapeType() == TopAbs_EDGE) {
      aResult = BRepBuilderAPI_MakeWire(TopoDS::Edge(aShape)).Wire();
    }
  }
  return aResult;
}

//=======================================================================
//function : GetSpine
//purpose  :
//=======================================================================
Handle(TDataStd_TreeNode) OCAF_IPipeShell::GetSpineNode() 
{
  return GetReferenceNode(PIPESHELL_SPINE_TAG);
}

//=======================================================================
//function :
//purpose  :
//=======================================================================
void  OCAF_IPipeShell::ClearSpine()
{
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(PIPESHELL_SPINE_TAG);
  L.ForgetAllAttributes(Standard_True);
}

//=======================================================================
//function : SetPipeMode
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IPipeShell::SetPipeMode(Handle(TDataStd_TreeNode) theObjNode, Standard_Boolean CurvilinearEquivalence, Standard_Boolean KeepContact)
{
  if(!SetReference(PIPESHELL_MODE_TAG, theObjNode)) 
    return Standard_False;

  TDF_Label theModeLab = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(PIPESHELL_MODE_TAG);

  Handle(TDataStd_IntegerArray) aData;
  aData = TDataStd_IntegerArray::Set(theModeLab, 0, 1);
  aData->SetValue(0, CurvilinearEquivalence);
  aData->SetValue(1, KeepContact);

  /*
  if(OCAF_IFunction::GetObjectType(theObjNode->First()) == OCAF_Selection) {
    theObjNode->Remove();
  }
  //*/

  return Standard_True;
}

//=======================================================================
//function : GetProfile
//purpose  :
//=======================================================================
TopoDS_Wire OCAF_IPipeShell::GetPipeMode()
{
  TopoDS_Shape aShape;
  TopoDS_Wire aResult;
  
  aShape = GetReference(PIPESHELL_MODE_TAG);

  if(!aShape.IsNull()) {
    if(aShape.ShapeType() == TopAbs_WIRE) 
      aResult = TopoDS::Wire(aShape);
    else{
      if(aShape.ShapeType() == TopAbs_EDGE) {
	aResult = BRepBuilderAPI_MakeWire(TopoDS::Edge(aShape)).Wire();
      }
    }
  }
  return aResult;
}

//=======================================================================
//function : GetSpine
//purpose  :
//=======================================================================
Handle(TDataStd_TreeNode) OCAF_IPipeShell::GetPipeModeNode() 
{
  return GetReferenceNode(PIPESHELL_MODE_TAG);
}

//=======================================================================
//function : IsCurvilinearEquivalence()
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IPipeShell::IsCurvilinearEquivalence()
{
  Standard_Boolean aResult = Standard_False;
  Handle(TDataStd_IntegerArray) aData;
  
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(PIPESHELL_MODE_TAG);
  
  if( L.FindAttribute(TDataStd_IntegerArray::GetID(), aData) ){
    aResult = aData->Value(0);
  }
  return aResult;
}

//=======================================================================
//function : IsKeepContact
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IPipeShell::IsKeepContact()
{
  Standard_Boolean aResult = Standard_False;
  Handle(TDataStd_IntegerArray) aData;
  
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(PIPESHELL_MODE_TAG);
  
  if( L.FindAttribute(TDataStd_IntegerArray::GetID(), aData) ){
    aResult = aData->Value(1);
  }
  return aResult;
}

//=======================================================================
//function : AddProfile
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IPipeShell::AddProfile(Handle(TDataStd_TreeNode) theObjNode, Standard_Boolean theWithContact, Standard_Boolean theWithCorrection) 
{
  // find "aNewArgLab" with tag of  "aNewTag", "aNewArgLab" used to store "theObjNode" 
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(PIPESHELL_PROFILE_TAG);
  Standard_Integer aProfileNum = L.NbChildren() + 1;

  int i = 1;
  for(i = 1; i < aProfileNum; i++){
    if(GetProfile(i).IsNull()){
      break;
    }
  }
  if(!SetReferenceOfTag(PIPESHELL_PROFILE_TAG, i, theObjNode)) 
    return Standard_False;

  TDF_Label theNewArgLab = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(PIPESHELL_PROFILE_TAG).FindChild(i);

  Handle(TDataStd_IntegerArray) aData;
  aData = TDataStd_IntegerArray::Set(theNewArgLab, 0, 1);

  Standard_Integer aType;
  aType = theWithContact?1:0;
  aData->SetValue(0, aType);
  aType = theWithCorrection?1:0;
  aData->SetValue(1, aType);

  /*
  if(OCAF_IFunction::GetObjectType(theObjNode->First()) == OCAF_Selection) {
    theObjNode->Remove();
  }
  //*/  

  return Standard_True;
}

//=======================================================================
//function : GetProfile
//purpose  :
//=======================================================================
TopoDS_Wire OCAF_IPipeShell::GetProfile(Standard_Integer theIndex)
{
  TopoDS_Shape aShape;
  TopoDS_Wire aResult;
  
  aShape = GetReferenceOfTag(PIPESHELL_PROFILE_TAG, theIndex);

  if(!aShape.IsNull()) {
    if(aShape.ShapeType() == TopAbs_WIRE) 
      aResult = TopoDS::Wire(aShape);
    else{
      if(aShape.ShapeType() == TopAbs_EDGE) {
	aResult = BRepBuilderAPI_MakeWire(TopoDS::Edge(aShape)).Wire();
      }
    }
  }
  return aResult;
}

void OCAF_IPipeShell::GetProfilesMap(TDF_AttributeMap& ArgumentsMap)
{
  GetArgumentsMapOfTag(PIPESHELL_PROFILE_TAG, ArgumentsMap);
}

void OCAF_IPipeShell::GetProfilesSequence(TDF_AttributeSequence& ArgumentsSequence)
{
  GetArgumentsSequenceOfTag(PIPESHELL_PROFILE_TAG, ArgumentsSequence);
}

//=======================================================================
//function : IsWithContact
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IPipeShell::IsWithContact(Standard_Integer theIndex)
{
  Standard_Boolean aResult = Standard_False;
  Handle(TDataStd_IntegerArray) aData;
  
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(PIPESHELL_PROFILE_TAG).FindChild(theIndex);
  
  if( L.FindAttribute(TDataStd_IntegerArray::GetID(), aData) ){
    aResult = aData->Value(0);
  }
  return aResult;
}

//=======================================================================
//function : IsWithCorrection
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IPipeShell::IsWithCorrection(Standard_Integer theIndex)
{
  Standard_Boolean aResult = Standard_False;
  Handle(TDataStd_IntegerArray) aData;
  
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(PIPESHELL_PROFILE_TAG).FindChild(theIndex);

  if( L.FindAttribute(TDataStd_IntegerArray::GetID(), aData) ){
    aResult = aData->Value(1);
  }  
  return aResult;
}

//=======================================================================
//function : 
//purpose  :
//=======================================================================
void OCAF_IPipeShell::ClearProfiles()
{
  ClearAllArgumentsOfTag(PIPESHELL_PROFILE_TAG);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//=======================================================================
//function : SetFrenet
//purpose  :
//=======================================================================
void OCAF_IPipeShell::SetFrenet(Standard_Boolean theType)
{
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG);
  Handle(TDataStd_Integer) anInt;
  Standard_Integer aType = theType ? 1 : 0;
  TDF_Label aLabel = L.FindChild(PIPESHELL_IS_FRENET_TAG);
  
  if(!aLabel.FindAttribute(TDataStd_Integer::GetID(), anInt)) {
    anInt = TDataStd_Integer::Set(aLabel, aType);
  } else
    anInt->Set(aType);
}

//=======================================================================
//function : IsFrenet
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IPipeShell::IsFrenet() 
{
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG);
  Handle(TDataStd_Integer) anInt;  
  TDF_Label aLabel = L.FindChild(PIPESHELL_IS_FRENET_TAG);
  
  if(!aLabel.FindAttribute(TDataStd_Integer::GetID(), anInt)) {
    anInt = TDataStd_Integer::Set(aLabel, 0);
  } 
  return anInt->Get() ? Standard_True : Standard_False;
}
