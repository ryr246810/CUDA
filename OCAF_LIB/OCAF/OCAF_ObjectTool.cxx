#include <OCAF_ObjectTool.hxx>

#include <Tags.hxx>
#include <CAGDDefine.hxx>


#include <OCAF_ISelection.hxx>
#include <OCAF_ITransformParent.hxx>
#include <OCAF_ITranslate.hxx>

#include <OCAF_IDisplayer.hxx>

#include <TDF_ChildIterator.hxx>
#include <TDF_Label.hxx>
#include <TDF_MapIteratorOfLabelMap.hxx>
#include <TDF_Reference.hxx>
#include <TDF_Tool.hxx>

#include <TDataStd_ChildNodeIterator.hxx>
#include <TDataStd_Integer.hxx>
#include <TDataStd_Name.hxx>
#include <TDataStd_Real.hxx>
#include <TDataStd_UAttribute.hxx>

#include <TDocStd_Modified.hxx>

#include <TFunction_Function.hxx>

#include <TNaming_NamedShape.hxx>
#include <TNaming_Tool.hxx>
#include <TNaming_RefShape.hxx>
#include <TNaming_UsedShapes.hxx>
#include <TNaming_PtrNode.hxx>

#include <TPrsStd_AISPresentation.hxx>

#include <TDF_AttributeMap.hxx>
#include <TDF_MapIteratorOfAttributeMap.hxx>

#include <TCollection_AsciiString.hxx>

#include <Standard_ConstructionError.hxx>

#include <OCAF_Object.hxx>

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_ObjectTool::OCAF_ObjectTool()
{

}



//=======================================================================
//function : GetEntry
//purpose  :
//=======================================================================
TDF_Label 
OCAF_ObjectTool::
GetEntry(const Handle(TDataStd_TreeNode)& theTreeNode)
{
  if (theTreeNode.IsNull()) return TDF_Label();
  return theTreeNode->Label();
}



//=======================================================================
//function : IsOneObject
//purpose  :
//=======================================================================
Standard_Boolean
OCAF_ObjectTool::
IsOneObject(const Handle(TDataStd_TreeNode)& theTreeNode)
{
  Standard_Boolean result = Standard_False;
  if(theTreeNode.IsNull()){
    result = Standard_False;
  }else if(!theTreeNode->IsAttribute(OCAF_Object::GetObjectID())){
    result = Standard_False;
  }else{
    result = Standard_True;
  }
  return result;
}



//=======================================================================
//function : IsOneFunctionNode
//purpose  :
//=======================================================================
Standard_Boolean
OCAF_ObjectTool::
IsOneFunctionNode(const Handle(TDataStd_TreeNode)& theTreeNode)
{
  Standard_Boolean result = Standard_False;
  if(theTreeNode.IsNull()){
    result = Standard_False;
  }else if(!theTreeNode->IsAttribute(TFunction_Function::GetID())){
    result = Standard_False;
  }else{
    result = Standard_True;
  }
  return result;
}




//=======================================================================
//function : MakeSelect
//purpose  : Adds a point object to the document
//=======================================================================
Handle(TDataStd_TreeNode) 
OCAF_ObjectTool::
Make_ObjectNode(const TCollection_ExtendedString& theName,
		const Handle(TDataStd_TreeNode)& AccessTreeNode,
		TCollection_ExtendedString& theError)
{
  if (AccessTreeNode.IsNull() || AccessTreeNode->Label().IsNull()) {
    theError = NULL_ACCESS_NODE;
    return NULL;
  }
 
  // 1. Add a new object to root label 
  //        (root label can be get from the label of "AccessTreeNode" by using "Root()"---see OCAF_Object's AddObject() )
  Handle(TDataStd_TreeNode) anObjectNode = AddObject(AccessTreeNode->Label());

  // 2. Construct an interface "anInterface"
  OCAF_Object anInterface(anObjectNode);
  anInterface.SetName(theName);

  theError = DONE;

  return anObjectNode;
}



//=======================================================================
//function : AddObject
//purpose  :
//=======================================================================
Handle(TDataStd_TreeNode) 
OCAF_ObjectTool::
AddObject(const TDF_Label& theAccessLabel) 
{
  Handle(TDataStd_TreeNode) aNode;
  if (theAccessLabel.IsNull()) return aNode;

  TDF_Label aRoot = theAccessLabel.Root();

  // if aRoot(root label) have no a Attribute of TDataStd_TreeNode, return a empty TDataStd_TreeNode
  // else find the attribute aNode by using FindAttribute;
  if(!aRoot.FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return aNode; 

  ///////////////////////////////////////////////////////////////////////////////
  // Beginning to build a new object's TDataStd_TreeNode
  ///////////////////////////////////////////////////////////////////////////////
  Handle(TDataStd_TreeNode) anObjectNode;

  //1. build a child labie of aRoot(root label)
  TDF_Label anObjectLabel = aRoot.NewChild();

  //2. use TDataStd_UAttribute to mark anObjectLabel (a user defined attribute ID)  !
  TDataStd_UAttribute::Set(anObjectLabel, OCAF_Object::GetObjectID());

  //3. to add a TDataStd_TreeNode attribute to anObjectLabel
  anObjectNode = TDataStd_TreeNode::Set(anObjectLabel);

  //4. append anObjectNode as a child node of aNode
  aNode->Append(anObjectNode);

  return anObjectNode;
}



//=======================================================================
//function : GetObjectNode
//purpose  :
//=======================================================================
Handle(TDataStd_TreeNode) 
OCAF_ObjectTool::
GetObjectNode(const Handle(TDataStd_TreeNode)& theTreeNode)
{
  Handle(TDataStd_TreeNode) aNode = theTreeNode;
  
  while(!aNode.IsNull()) {  
    if(aNode->IsAttribute(OCAF_Object::GetObjectID())) return aNode;
    aNode = aNode->Father();
  }
  
  return aNode;
}




//=======================================================================
//function : GetNode
//purpose  :
//=======================================================================
Handle(TDataStd_TreeNode) 
OCAF_ObjectTool::
GetNode(const TopoDS_Shape& theShape, const TDF_Label& theAccessLabel)
{
  Handle(TDataStd_TreeNode) aNode;
  if (theAccessLabel.IsNull()) return aNode;

  Handle(TNaming_NamedShape) aNS = TNaming_Tool::NamedShape(theShape, theAccessLabel);
  if(aNS.IsNull()) return aNode;

  TDF_Label aLabel = aNS->Label().Father();  //Function label
  aLabel.FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode); //Node of the function

  if(!aNode.IsNull()) aNode = aNode->Father(); //Node of the object

  return aNode;
}



//=======================================================================
//function : IsAuxiliryObject
//purpose  :
//=======================================================================
Standard_Boolean 
OCAF_ObjectTool::
IsAuxiliryObject(const Handle(TDataStd_TreeNode)& theObject)
{
  if(theObject->First().IsNull()) return Standard_False;
  
  Handle(TFunction_Function) aFunction;
  if(!theObject->First()->FindAttribute(TFunction_Function::GetID(), aFunction)) return Standard_False;	
  
  if(aFunction->GetDriverGUID() == OCAF_ISelection::GetID()) return Standard_True;  //modified 2010.06.07 wangyue
  
  return Standard_False;
}



//=======================================================================
//function : RemoveObject
//purpose  : theObject
//=======================================================================
void 
OCAF_ObjectTool::
RemoveObject(const Handle(TDataStd_TreeNode)& theObjectNode)
{
  if(theObjectNode.IsNull()) return;
  
  OCAF_Object anInterface(theObjectNode);
  
  Handle(TDataStd_TreeNode) aCurrentNode, aRefNode, aNode = anInterface.GetLastFunction();
  if(aNode.IsNull()) return;

  Handle(TDF_Reference) aRef;
  TDF_ChildIterator anIterator;
  TDF_Label aLabel;
  
  while(!aNode.IsNull()) {
    aCurrentNode = aNode->Previous();
    // aNode is a TFunction_Function TreeNode
    anIterator.Initialize(aNode->Label().FindChild(ARGUMENTS_TAG), Standard_True);

    for(; anIterator.More(); anIterator.Next()) {
      if(!anIterator.Value().FindAttribute(TDF_Reference::GetID(), aRef)) continue;
      if(aRef->Get().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aRefNode)){
	if(aRefNode->IsChild(aNode)) {
	  aLabel = aRef->Get().Father();  // get the object's Label
	  if(!aLabel.FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) continue;
	  if(IsAuxiliryObject(aNode)) {
	    OCAF_Object IShape(aNode);
	    IShape.Remove(); 
	  }else{
	    RemoveObject(aNode);
	  }
	}
      }
    }
    aNode = aCurrentNode;
  }

  theObjectNode->Remove();
  theObjectNode->ForgetAllAttributes(Standard_True);
}







//=======================================================================
//function : SetName
//purpose  :
//=======================================================================
void 
OCAF_ObjectTool::
SetName(const Handle(TDataStd_TreeNode)& theTreeNode,
	const TCollection_ExtendedString& aName)
{
  if (theTreeNode.IsNull()) return;

  TDataStd_Name::Set(theTreeNode->Label(), aName);
  TDocStd_Modified::Add(theTreeNode->Label());
}


//=======================================================================
//function : HasName
//purpose  :
//=======================================================================

Standard_Boolean 
OCAF_ObjectTool::
HasName(const Handle(TDataStd_TreeNode)& theTreeNode)
{
  if (theTreeNode.IsNull()) return Standard_False;

  Handle(TDataStd_Name) aName;
  return theTreeNode->Label().FindAttribute(TDataStd_Name::GetID(),aName);
}


//=======================================================================
//function : GetName
//purpose  :
//=======================================================================

TCollection_ExtendedString 
OCAF_ObjectTool::
GetName(const Handle(TDataStd_TreeNode)& theTreeNode)
{
  TCollection_ExtendedString anExtendedString;
  if (theTreeNode.IsNull()) return anExtendedString;

  Handle(TDataStd_Name) aName;
  if (!theTreeNode->Label().FindAttribute(TDataStd_Name::GetID(),aName)) return anExtendedString;

  anExtendedString = aName->Get();
  return anExtendedString;
}
