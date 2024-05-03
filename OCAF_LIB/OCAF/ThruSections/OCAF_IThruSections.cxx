
#include <CAGDDefine.hxx>

#include <OCAF_IThruSections.hxx>
#include <OCAF_ThruSectionsDriver.hxx>

#include <Tags.hxx>

#include <TopoDS.hxx>
#include <BRepCheck_Shell.hxx>

#include <OCAF_ISelection.hxx>

#include <TDF_ChildIterator.hxx>
#include <TDF_Data.hxx>
#include <TDF_Label.hxx>
#include <TDF_Reference.hxx>
#include <TDF_Tool.hxx>

#include <TDataStd_Real.hxx>

#include <TDocStd_Modified.hxx>

#include <TFunction_Function.hxx>
#include <TFunction_Logbook.hxx>

#include <TNaming_NamedShape.hxx>

#include <Standard_ConstructionError.hxx>

#include <TDF_MapIteratorOfLabelMap.hxx>

#include <OCAF_Object.hxx>
#include <OCAF_ObjectTool.hxx>

//=======================================================================
//function : GetID
//purpose  :
//=======================================================================
const Standard_GUID& OCAF_IThruSections::GetID()
{
  static Standard_GUID anID("22D22E21-C69A-11d4-8F1A-0060B0EE18E8");
  return anID;
}


//=======================================================================
//function : MakeThruSection
//purpose  : Adds a point object to the document
//=======================================================================
Handle(TDataStd_TreeNode) OCAF_IThruSections::MakeThruSections_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
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
//function : MakeThruSection
//purpose  : Adds a point object to the document
//=======================================================================
Standard_Boolean OCAF_IThruSections::MakeThruSections_Execute( TCollection_ExtendedString& theError )
{
  if( myTreeNode.IsNull()) {
    theError = NOTDONE;
    return Standard_False;
  }
  // 5. setup a TFunction_Function's driver "aDriver" from OCAF_ThruSectionDriver
  OCAF_ThruSectionsDriver aDriver;
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
OCAF_IThruSections::OCAF_IThruSections(const Handle(TDataStd_TreeNode)& aTreeNode)
  :OCAF_IFunction(aTreeNode)
{
  if(myTreeNode.IsNull()) Standard_ConstructionError::Raise("Null TreeNode"); 
}

//=======================================================================
//function : IsClosedSection
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IThruSections::IsClosedSection(const Handle(TDataStd_TreeNode)& aTreeNode) 
{
  OCAF_Object anInt(aTreeNode);
  TopoDS_Shape aShape = anInt.GetObjectValue();

  if(aShape.ShapeType() == TopAbs_SHELL) {
    Handle(BRepCheck_Shell) aCheck = new BRepCheck_Shell(TopoDS::Shell(aShape));
    return (aCheck->Closed() == BRepCheck_NoError);
  } 
  else
    return Standard_True; 
}

//=======================================================================
//function : 
//purpose  :
//=======================================================================
void OCAF_IThruSections::SetSolid()
{
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG);
  Handle(TDataStd_Integer) anInt;
  TDF_Label aLabel = L.FindChild(THRUSECTION_BUILD_ISSOLID_TAG);
  
  if(!aLabel.FindAttribute(TDataStd_Integer::GetID(), anInt)) {
    anInt = TDataStd_Integer::Set(aLabel, 1);
  }
  anInt->Set(1);
}

//=======================================================================
//function : 
//purpose  :
//=======================================================================
void OCAF_IThruSections::SetShell()
{
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG);
  Handle(TDataStd_Integer) anInt;
  TDF_Label aLabel = L.FindChild(THRUSECTION_BUILD_ISSOLID_TAG);
  
  if(!aLabel.FindAttribute(TDataStd_Integer::GetID(), anInt)) {
    anInt = TDataStd_Integer::Set(aLabel, 0);
  }
  anInt->Set(0);
}

//=======================================================================
//function : 
//purpose  :
//=======================================================================
void OCAF_IThruSections::SetSolidOrShell( const Standard_Boolean _isSolid )
{
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG);
  Handle(TDataStd_Integer) anInt;
  TDF_Label aLabel = L.FindChild(THRUSECTION_BUILD_ISSOLID_TAG);
  
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
Standard_Boolean OCAF_IThruSections::IsSolidOrShell()
{
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG);
  Handle(TDataStd_Integer) anInt;
  TDF_Label aLabel = L.FindChild(THRUSECTION_BUILD_ISSOLID_TAG);
  
  if(!aLabel.FindAttribute(TDataStd_Integer::GetID(), anInt)) {
    anInt = TDataStd_Integer::Set(aLabel, 1);
  }

  return anInt->Get() ? Standard_True : Standard_False;
}

//=======================================================================
//function : 
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IThruSections::IsSolid()
{
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG);
  Handle(TDataStd_Integer) anInt;
  TDF_Label aLabel = L.FindChild(THRUSECTION_BUILD_ISSOLID_TAG);
  
  if(!aLabel.FindAttribute(TDataStd_Integer::GetID(), anInt)) {
    anInt = TDataStd_Integer::Set(aLabel, 1);
  }

  return anInt->Get() ? Standard_True : Standard_False;
}


//=======================================================================
//function : 
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IThruSections::IsShell()
{
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG);
  Handle(TDataStd_Integer) anInt;
  TDF_Label aLabel = L.FindChild(THRUSECTION_BUILD_ISSOLID_TAG);
  
  if(!aLabel.FindAttribute(TDataStd_Integer::GetID(), anInt)) {
    anInt = TDataStd_Integer::Set(aLabel, 1);
  }

  return anInt->Get() ? Standard_False : Standard_True;
}

//=======================================================================
//function : 
//purpose  :
//=======================================================================
Standard_EXPORT void OCAF_IThruSections::SetRuled(Standard_Boolean theType) 
{
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG);
  Handle(TDataStd_Integer) anInt;
  Standard_Integer aType = theType ? 1 : 0;
  TDF_Label aLabel = L.FindChild(THRUSECTION_BUILD_ISRULED_TAG);
  
  if(!aLabel.FindAttribute(TDataStd_Integer::GetID(), anInt)) {
    anInt = TDataStd_Integer::Set(aLabel, aType);
  } 
  else
    anInt->Set(aType);
}

//=======================================================================
//function : 
//purpose  :
//=======================================================================
Standard_EXPORT Standard_Boolean OCAF_IThruSections::IsRuled()
{
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG);
  Handle(TDataStd_Integer) anInt;  
  TDF_Label aLabel = L.FindChild(THRUSECTION_BUILD_ISRULED_TAG);
  
  if(!aLabel.FindAttribute(TDataStd_Integer::GetID(), anInt)) {
    anInt = TDataStd_Integer::Set(aLabel, 0);
  } 
  return anInt->Get() ? Standard_True : Standard_False;
}

//=======================================================================
//function : 
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IThruSections::SetSection(Handle(TDataStd_TreeNode) theNode) 
{
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(THRUSECTION_BUILD_SECTION_TAG);
  
  Standard_Integer aCurNum = L.NbChildren() + 1;

  int i;
  for(i = 1; i < aCurNum; i++){
    if(GetSection(i).IsNull()){
      break;
    }
  }

  Standard_Boolean result = SetReferenceOfTag(THRUSECTION_BUILD_SECTION_TAG, i, theNode);  

  /*
  // modified 2014.11.27
  if(OCAF_IFunction::GetObjectType(theNode->First()) == OCAF_Selection)
    theNode->Remove();
  //*/
  return result;
}

//=======================================================================
//function : 
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IThruSections::SetBuildThruSectionsElement(Handle(TDataStd_TreeNode) theNode) 
{
  //1. get the ARGUMENTS_TAG's Lable "L"
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(THRUSECTION_BUILD_SECTION_TAG);
  
  //2. deside the current tag "aCurNum", according the Children Number of "L"
  Standard_Integer aCurNum = L.NbChildren() + 1;

  //3. test whether the perious BuildThruSection element is NULL
  //3.1 find a label tag "i" whose BuildThruSection element is empty
  int i = 1;
  for(i = 1; i < aCurNum; i++){
    if(GetBuildThruSectionsElement(i).IsNull()){
      break;
    }
  }
  //4. Select theNode as the BuildThruSection element
  Standard_Boolean result = SetReferenceOfTag(THRUSECTION_BUILD_SECTION_TAG, i, theNode);

  /*  
  // modified 2014.11.27
  if(OCAF_IFunction::GetObjectType(theNode->First()) == OCAF_Selection){
    theNode->Remove();
  }
  //*/
  return result;
}

//=======================================================================
//function : 
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IThruSections::SetBuildThruSectionsElements(const TDF_LabelMap& ArgumentsMap) 
{
  TDF_MapIteratorOfLabelMap aIter;
  aIter.Initialize(ArgumentsMap);

  Standard_Boolean result = Standard_True;

  for(;aIter.More();aIter.Next()){
    TDF_Label aLabel = aIter.Key();
    Handle(TDataStd_TreeNode) theNode;
    aLabel.FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(),theNode);
    if(theNode.IsNull()) continue;
    result = result && SetBuildThruSectionsElement(theNode);
  }
  return result;
}

//=======================================================================
//function : 
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IThruSections::SetBuildThruSectionsElements(const TDF_AttributeMap& ArgumentsMap) 
{
  TDF_MapIteratorOfAttributeMap aIter;
  aIter.Initialize(ArgumentsMap);

  Standard_Boolean result = Standard_True;

  for(;aIter.More();aIter.Next()){
    Handle(TDataStd_TreeNode) theNode = Handle(TDataStd_TreeNode)::DownCast(aIter.Key());
    if(theNode.IsNull()) {
      continue;
    }
    result = result && SetBuildThruSectionsElement(theNode);
  }
  return result;
}

//=======================================================================
//function : GetSection
//purpose  :  Modified by Wang Yue 2010.04.01
//=======================================================================
TopoDS_Shape OCAF_IThruSections::GetSection(Standard_Integer theNumber)
{
  TopoDS_Shape aShape;
  aShape = GetReferenceOfTag(THRUSECTION_BUILD_SECTION_TAG, theNumber);
  return aShape;
}

//=======================================================================
//function : GetSection
//purpose  :  Modified by Wang Yue 2010.05.10
//=======================================================================
TopoDS_Shape OCAF_IThruSections::GetBuildThruSectionsElement(Standard_Integer theNumber)
{
  TopoDS_Shape aShape;
  aShape = GetReferenceOfTag(THRUSECTION_BUILD_SECTION_TAG, theNumber);
  return aShape;
}

void OCAF_IThruSections::GetBuildThruSectionsElementsMap(TDF_AttributeMap& ArgumentsMap)
{
  GetArgumentsMapOfTag(THRUSECTION_BUILD_SECTION_TAG, ArgumentsMap);
}

void OCAF_IThruSections::ClearBuildThruSectionsElements()
{
  ClearAllArgumentsOfTag(THRUSECTION_BUILD_SECTION_TAG); 
}
