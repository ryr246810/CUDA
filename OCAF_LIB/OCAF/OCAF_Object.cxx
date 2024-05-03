#include <OCAF_Object.hxx>

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

#include <OCAF_ObjectTool.hxx>


//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================

OCAF_Object::OCAF_Object(const Handle(TDataStd_TreeNode)& aTreeNode):myTreeNode(aTreeNode)
{
  if(myTreeNode.IsNull()) Standard_ConstructionError::Raise("Null TreeNode");
}



//=======================================================================
//function : GetObjectID
//purpose  :
//=======================================================================
const Standard_GUID& 
OCAF_Object::
GetObjectID() {
  static Standard_GUID anID("22D22E01-C69A-11d4-8F1A-0060B0EE18E8");
  return anID;
}



//=======================================================================
//function : GetEntry
//purpose  :
//=======================================================================
TDF_Label 
OCAF_Object::
GetEntry() const
{
  if (myTreeNode.IsNull()) return TDF_Label();
  return myTreeNode->Label();
}

  



//=======================================================================
//function : SetName
//purpose  :
//=======================================================================
void OCAF_Object::SetName(const TCollection_ExtendedString& aName)
{
  if (myTreeNode.IsNull()) return;

  TDataStd_Name::Set(myTreeNode->Label(), aName);
  TDocStd_Modified::Add(myTreeNode->Label());
}


//=======================================================================
//function : HasName
//purpose  :
//=======================================================================

Standard_Boolean OCAF_Object::HasName()
{
  if (myTreeNode.IsNull()) return Standard_False;

  Handle(TDataStd_Name) aName;
  return myTreeNode->Label().FindAttribute(TDataStd_Name::GetID(),aName);
}


//=======================================================================
//function : GetName
//purpose  :
//=======================================================================

TCollection_ExtendedString OCAF_Object::GetName() const
{
  TCollection_ExtendedString anExtendedString;
  if (myTreeNode.IsNull()) return anExtendedString;

  Handle(TDataStd_Name) aName;
  if (!myTreeNode->Label().FindAttribute(TDataStd_Name::GetID(),aName)) return anExtendedString;

  anExtendedString = aName->Get();
  return anExtendedString;
}



//=============================================================================
//function : SetObjectMask
//purpose  :
//=============================================================================
void 
OCAF_Object::
SetObjectMask(Standard_Integer theMask)
{
  if (myTreeNode.IsNull()) return;
  if(!myTreeNode->IsAttribute(GetObjectID())) return;

  TDF_Label _label = myTreeNode->Label();
  Handle(TDataStd_Integer) anInt;

  if(!_label.FindAttribute(TDataStd_Integer::GetID(), anInt)) {
    TDataStd_Integer::Set(_label, theMask);
  }else{
    anInt->Set(theMask);
  }
}



//=============================================================================
//function : GetObjectMask
//purpose  :
//=============================================================================
Standard_Integer 
OCAF_Object::
GetObjectMask() const
{
  if (myTreeNode.IsNull()) return ERROR_MASK;
  if(!myTreeNode->IsAttribute(GetObjectID())) return ERROR_MASK;

  Handle(TDataStd_Integer) aMask;
  TDF_Label _label = myTreeNode->Label();
  if(!_label.FindAttribute(TDataStd_Integer::GetID(), aMask)) return ZERO_MASK;
  return aMask->Get();
}



//=============================================================================
//function : SetResultMaterial
//purpose  :
//=============================================================================
Standard_Boolean 
OCAF_Object::
SetObjResultMaterial(Standard_Integer theMaterial)
{
  Standard_Boolean result = Standard_True;
  if (myTreeNode.IsNull()) result = Standard_False; 
  if(!myTreeNode->IsAttribute(GetObjectID())) return ERROR_MATERIAL;

  Handle(TDF_Reference) aRef;
  if(myTreeNode->FindAttribute(TDF_Reference::GetID(), aRef)){  // refer to function node
    if( !aRef->Get().IsNull() ){
      TDF_Label aLabel = aRef->Get().FindChild(RESULTS_TAG);
      TDataStd_Integer::Set(aLabel,theMaterial);
      //TDocStd_Modified::Add(L);
    }else{
      result = Standard_False; 
    }
  }

  return result; 
}


//=============================================================================
//function : GetResultMaterial
//purpose  :
//=============================================================================
Standard_Integer 
OCAF_Object::
GetObjResultMaterial() const
{
  Standard_Integer theMaterial = ERROR_MATERIAL;

  if (myTreeNode.IsNull()) return ERROR_MATERIAL;
  if(!myTreeNode->IsAttribute(GetObjectID())) return ERROR_MATERIAL;

  Handle(TDF_Reference) aRef;
  if(myTreeNode->FindAttribute(TDF_Reference::GetID(), aRef)){
    if(aRef->Get().IsNull()) return ERROR_MATERIAL;
    TDF_Label aLabel = aRef->Get().FindChild(RESULTS_TAG);
    Handle(TDataStd_Integer) aInt;
    if (!aLabel.FindAttribute(TDataStd_Integer::GetID(), aInt)) return ERROR_MATERIAL;
    theMaterial = aInt->Get();
  }

  return theMaterial;
}



//=======================================================================
//function : GetPresentation
//purpose  :
//=======================================================================
Handle(TPrsStd_AISPresentation) 
OCAF_Object::
GetPresentation( ) const
{
  Handle(TPrsStd_AISPresentation) aPresenation;

  if(OCAF_ObjectTool::IsOneObject(myTreeNode)){
    TDF_Label aLabel = myTreeNode->Label();
    aLabel.FindAttribute(TPrsStd_AISPresentation::GetID(), aPresenation);
  }

  return aPresenation;
}




//=======================================================================
//function : AddFunction
//purpose  :
//=======================================================================
Handle(TDataStd_TreeNode) 
OCAF_Object::
AddFunction(const Standard_GUID& theID)
{
  //1. get the ObjectNode
  Handle(TDataStd_TreeNode) aNode = OCAF_ObjectTool::GetObjectNode(myTreeNode);

  if(aNode.IsNull()) return NULL;	

  //2. build a new child label "aLabel" of the Label of "aNode"
  TDF_Label aLabel = aNode->Label().NewChild(); 

  //TDF_Label aLabel = aNode->Label().FindChild(FUNCTION_TAG);  //modified by wang yue 2010.03.17

  //3. add TDataStd_TreeNode attribute "aFunctionNode" to "aLabel"
  Handle(TDataStd_TreeNode) aFunctionNode = TDataStd_TreeNode::Set(aLabel);
  
  //4. set "aFunctionNode" as a child node of "aNode"
  aNode->Append(aFunctionNode);

  //5. set "TFunction_Function" attribute to "aLabel"
  Handle(TFunction_Function) aFunction = TFunction_Function::Set(aLabel, theID);


  //6. for achieve a function's result, add a TDF_Reference attribute of aNode's label, this reference attribute refer to aLabel 
  Handle(TDF_Reference) aRef;
  TDF_Label aContext;

  if(aNode->FindAttribute(TDF_Reference::GetID(), aRef)){  // already have a function back up the pre defined function node
    aContext = aRef->Get();
  }

  /*
  //6.1 set aNode's Label ( "aNode->Label()" ) refer to aFunctionNode's label "aLabel"
  if(theID != OCAF_ITransformParent::GetID()) {
    TDF_Reference::Set(aNode->Label(), aLabel);
  }
  //*/

  TDF_Reference::Set(aNode->Label(), aLabel);

  if(!aContext.IsNull()){  //  set reference the new function node to the predefined function node
    TDF_Reference::Set(aFunctionNode->Label(), aContext); //Set a context of the function
  }

  return aFunctionNode;
}



//=======================================================================
//function : GetLastFunction
//purpose  : myTreeNode is an object node
//=======================================================================
Handle(TDataStd_TreeNode) 
OCAF_Object::
GetLastFunction() const
{
  Handle(TDataStd_TreeNode) aNode;
  Handle(TFunction_Function) aFunction;
  if(!myTreeNode->First().IsNull()) {  //Find the last function of the object
    TDataStd_ChildNodeIterator anIterator(myTreeNode);
    for(; anIterator.More(); anIterator.Next()) {
      aNode = anIterator.Value();
    }
  }
  return aNode;
}




//=======================================================================
//function : GetObjectValue
//purpose  : myTreeNode is an object node
//=======================================================================
TopoDS_Shape 
OCAF_Object::
GetObjectValue() const
{
  TopoDS_Shape aShape;
  Handle(TNaming_NamedShape) aNamedShape;
  Handle(TDF_Reference) aRef;
  
  if(myTreeNode->FindAttribute(TDF_Reference::GetID(), aRef)){
    if(aRef->Get().IsNull()) return aShape;
    TDF_Label aLabel = aRef->Get().FindChild(RESULTS_TAG);
    if(aLabel.FindAttribute(TNaming_NamedShape::GetID(), aNamedShape)){ 
      //aShape = TNaming_Tool::GetShape(aNamedShape);
      aShape = aNamedShape->Get();  // orginal ----- marked 2015.12.09
    }
  }
  return aShape;
}

