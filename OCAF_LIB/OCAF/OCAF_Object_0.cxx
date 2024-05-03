#include <OCAF_Object.hxx>

#include <Tags.hxx>
#include <CAGDDefine.hxx>


#include <OCAF_ISelection.hxx>
#include <OCAF_ITransformParent.hxx>
#include <OCAF_ITranslate.hxx>
#include <OCAF_SolverEx.hxx>

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


//=======================================================================
//function : CanRemove
//purpose  :
//=======================================================================
Standard_Boolean 
OCAF_Object::
CanRemove()
{
  if(HasReferencedObjects()) return Standard_False;

  return Standard_True;
}


//=======================================================================
//function : Remove
//purpose  :
//=======================================================================
Standard_Boolean 
OCAF_Object::
Remove()
{
  if(HasReferencedObjects()) return Standard_False;

  myTreeNode->Remove();
  myTreeNode->ForgetAllAttributes(Standard_True);
  myTreeNode.Nullify();

  return Standard_True;
}


//=======================================================================
//function : HasReferencedObjects
//purpose  :
//=======================================================================

Standard_Boolean 
OCAF_Object::
HasReferencedObjects() const
{
  TDF_AttributeMap aMap;
  OCAF_SolverEx aSolver;

  Handle(TDataStd_TreeNode) theLastFunNode = GetLastFunction();

  Handle(TFunction_Function) aFunction, aMyFunction;

  if(!theLastFunNode->FindAttribute(TFunction_Function::GetID(), aMyFunction)) return Standard_False;	

  TDataStd_ChildNodeIterator anIterator(myTreeNode->Root(), Standard_True);
  
  for(; anIterator.More(); anIterator.Next()) {

    if(anIterator.Value() == myTreeNode) continue;
    
    if(anIterator.Value()->FindAttribute(TFunction_Function::GetID(), aFunction)) 
      aSolver.ComputeFunction(aFunction, aMap);
    
    if(aMap.Contains(aMyFunction)) return Standard_True;
  }
  return Standard_False;
}



