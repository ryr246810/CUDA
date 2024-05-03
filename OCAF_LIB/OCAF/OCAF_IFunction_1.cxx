#include <OCAF_IFunction.hxx>

#include <Tags.hxx>
#include <CAGDDefine.hxx>


#include <OCAF_ISelection.hxx>
#include <OCAF_ITransformParent.hxx>


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

#include <TPrsStd_AISPresentation.hxx>

#include <TDF_AttributeMap.hxx>
#include <TDF_MapIteratorOfAttributeMap.hxx>

#include <TCollection_AsciiString.hxx>

#include <Standard_ConstructionError.hxx>

#include <OCAF_Object.hxx>




//=============================================================================
//function : SetReference
//purpose  :
//=============================================================================
Standard_Boolean 
OCAF_IFunction::
SetReference(const int thePosition,
	     const Handle(TDataStd_TreeNode)& theObject)
{
  if(thePosition <= 0) return Standard_False;
  if(theObject.IsNull()) return Standard_False;

  //1. get the function node of "theObject"
  Handle(TDataStd_TreeNode) aFunctionNode = (OCAF_Object(theObject)).GetLastFunction();
  if(aFunctionNode.IsNull()) return Standard_False;

  Handle(TFunction_Function) aFunction;
  if(!aFunctionNode->Label().FindAttribute(TFunction_Function::GetID(), aFunction))
    return Standard_False;

  TDF_Label aLabel = GetEntry().FindChild(ARGUMENTS_TAG).FindChild((thePosition));

  if(aLabel.IsAttribute(TDF_Reference::GetID()) ) {
    aLabel.ForgetAllAttributes(Standard_True); 
  }

  TDF_Reference::Set(aLabel, aFunctionNode->Label()); 

  return Standard_True;
}


//=============================================================================
//function : GetReference
//purpose  : 
//=============================================================================
TopoDS_Shape 
OCAF_IFunction::
GetReference( int thePosition )
{
  TopoDS_Shape aShape;
  Handle(TNaming_NamedShape) aNamedShape;
  Handle(TDataStd_TreeNode) aNode;
  Handle(TDF_Reference) aReference;
  if(thePosition <= 0) return aShape;

  TDF_Label aLabel = GetEntry().FindChild(ARGUMENTS_TAG).FindChild((thePosition));


  if(aLabel.FindAttribute(TDF_Reference::GetID(), aReference)) {
    aLabel = aReference->Get().FindChild(RESULTS_TAG);
    if(aLabel.FindAttribute(TNaming_NamedShape::GetID(), aNamedShape)){
      return aNamedShape->Get();
    }
  }
  return aShape;
}



//=============================================================================
//function : GetReferenceNode
//purpose  : 
//=============================================================================
Handle(TDataStd_TreeNode) 
OCAF_IFunction::
GetReferenceNode(int thePosition)
{
  Handle(TDataStd_TreeNode) aNode;
  if(thePosition <= 0) return aNode;

  TDF_Label aLabel = GetEntry().FindChild(ARGUMENTS_TAG).FindChild((thePosition));


  Handle(TDF_Reference) aReference;
  if(aLabel.FindAttribute(TDF_Reference::GetID(), aReference)) {
    if(aReference->Get().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) {
      if(aReference->Get().IsAttribute(TFunction_Function::GetID()))
	return aNode->Father();
    }
  }
  return aNode;
}



//=============================================================================
//function : SetFunReference
//purpose  :
//=============================================================================
Standard_Boolean 
OCAF_IFunction::
SetFuncReference(const int thePosition, 
		 const Handle(TDataStd_TreeNode)& theFunction)
{
  if(thePosition <= 0) return Standard_False;
  if(theFunction.IsNull()) return Standard_False;

  Handle(TFunction_Function) aFunction;
  if(!theFunction->Label().FindAttribute(TFunction_Function::GetID(), aFunction))
    return Standard_False;

  TDF_Label aLabel = GetEntry().FindChild(ARGUMENTS_TAG).FindChild((thePosition));

  if(aLabel.IsAttribute(TDF_Reference::GetID()) ) {
    aLabel.ForgetAllAttributes(Standard_True); 
  }

  TDF_Reference::Set(aLabel, theFunction->Label()); 
  return Standard_True;
}


//=============================================================================
//function : GetFuncReference
//purpose  : 
//=============================================================================
TopoDS_Shape 
OCAF_IFunction::
GetFuncReference( int thePosition )
{
  return GetReference(thePosition);
}




//=============================================================================
//function : GetFuncReferenceNode
//purpose  : 
//=============================================================================
Handle(TDataStd_TreeNode) 
OCAF_IFunction::
GetFuncReferenceNode(int thePosition)
{
  Handle(TDataStd_TreeNode) aNode;
  if(thePosition <= 0) return aNode;
  Handle(TDF_Reference) aReference;

  TDF_Label aLabel = GetEntry().FindChild(ARGUMENTS_TAG).FindChild((thePosition));


  if(aLabel.FindAttribute(TDF_Reference::GetID(), aReference)) {
    if(aReference->Get().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) {
      if(aReference->Get().IsAttribute(TFunction_Function::GetID()))
	return aNode;
    }
  }
  return aNode;
}
