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
SetReferenceOfTag(const int theTag, 
		  const int theChildTag, 
		  const Handle(TDataStd_TreeNode)& theObject)
{
  if(theTag<=0 || theChildTag <= 0) return Standard_False;
  if(theObject.IsNull()) return Standard_False;

  //1. get the function node of "theObject"
  Handle(TDataStd_TreeNode) aFunctionNode = (OCAF_Object(theObject)).GetLastFunction();
  if(aFunctionNode.IsNull()) return Standard_False;

  Handle(TFunction_Function) aFunction;
  if(!aFunctionNode->Label().FindAttribute(TFunction_Function::GetID(), aFunction))
    return Standard_False;

  TDF_Label aLabel = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(theTag).FindChild(theChildTag);

  if(aLabel.IsAttribute(TDF_Reference::GetID()) ) {
    aLabel.ForgetAllAttributes(Standard_True); 
  }

  TDF_Reference::Set(aLabel, aFunctionNode->Label()); 

  return Standard_True;
}


//=============================================================================
//function : GetReferenceNodeOfTag
//purpose  : 
//=============================================================================
Handle(TDataStd_TreeNode) 
OCAF_IFunction::
GetReferenceNodeOfTag(int aTag,
		      int theChildTag)
{
  Handle(TDataStd_TreeNode) aNode;
  if(theChildTag <= 0) return aNode;

  TDF_Label aLabel = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(aTag).FindChild(theChildTag);


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
//function : GetReferenceOfTag
//purpose  : 
//=============================================================================
TopoDS_Shape OCAF_IFunction::GetReferenceOfTag(int theTag, int theChildTag)
{
  TopoDS_Shape aShape;
  Handle(TNaming_NamedShape) aNamedShape;
  Handle(TDataStd_TreeNode) aNode;
  Handle(TDF_Reference) aReference;
  if(theTag<=0 || theChildTag <= 0) return aShape;

  TDF_Label aLabel = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(theTag).FindChild(theChildTag);

  if(aLabel.FindAttribute(TDF_Reference::GetID(), aReference)) {
    aLabel = aReference->Get().FindChild(RESULTS_TAG);
    if(aLabel.FindAttribute(TNaming_NamedShape::GetID(), aNamedShape)){
      return aNamedShape->Get();
    }
  }
  return aShape;
}


//=============================================================================
//function : GetArgumentsMapOfTag
//purpose  :
//=============================================================================
void 
OCAF_IFunction::
GetArgumentsMapOfTag(const int theTag,
		     TDF_AttributeMap& anAttributeMap)
{
  anAttributeMap.Clear();
  // parent label for the list of references
  TDF_Label ArgumentsLabel = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(theTag);

  TDF_ChildIterator anIter;
  anIter.Initialize(ArgumentsLabel,Standard_False);

  for (; anIter.More(); anIter.Next()) { 
    TDF_Label aLabel_i = anIter.Value();
    Standard_Integer aTag = aLabel_i.Tag();
    Handle(TDataStd_TreeNode) aNode = GetReferenceNodeOfTag(theTag,aTag);
    if(aNode.IsNull()) continue;
    anAttributeMap.Add( aNode );
  }
}


//=============================================================================
//function : GetArgumentsSequenceOfTag
//purpose  :
//=============================================================================
void 
OCAF_IFunction::
GetArgumentsSequenceOfTag(const int theTag,
			  TDF_AttributeSequence& anAttributeSequence)
{
  anAttributeSequence.Clear();

  // parent label for the list of references
  TDF_Label ArgumentsLabel = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(theTag);

  TDF_ChildIterator anIter;
  anIter.Initialize(ArgumentsLabel,Standard_False);

  for (; anIter.More(); anIter.Next()) { 
    TDF_Label aLabel_i = anIter.Value();
    Standard_Integer aTag = aLabel_i.Tag();
    Handle(TDataStd_TreeNode) aNode = GetReferenceNodeOfTag(theTag,aTag);
    if(aNode.IsNull()) continue;
    anAttributeSequence.Append( aNode );
  }
}


//=============================================================================
//function : ClearAllArgumentsOfTag
//purpose  :
//=============================================================================
void 
OCAF_IFunction::
ClearAllArgumentsOfTag(int aTag)
{
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(aTag);
  L.ForgetAllAttributes(Standard_True);
}
