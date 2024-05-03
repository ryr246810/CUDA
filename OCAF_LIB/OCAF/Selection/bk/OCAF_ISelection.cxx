
#include <CAGDDefine.hxx>

#include <OCAF_ISelection.hxx>
#include <BRepNaming_Selection.hxx>

#include <TDF_Label.hxx>
#include <TDF_Reference.hxx>
#include <TDocStd_Modified.hxx>
#include <TNaming_Selector.hxx>
#include <TNaming_NamedShape.hxx>
#include <TNaming_Tool.hxx>
#include <TFunction_Function.hxx>
#include <TFunction_Logbook.hxx>

#include <Standard_ConstructionError.hxx>


#include <Tags.hxx>

//=======================================================================
//function : GetCutID
//purpose  :
//=======================================================================
const Standard_GUID& OCAF_ISelection::GetID()
{
  static Standard_GUID anID("22D22E57-C69A-11d4-8F1A-0060B0EE18E8");
  return anID;
}



//=======================================================================
//function : MakeSelect
//purpose  : Adds a point object to the document
//=======================================================================
Standard_Boolean OCAF_ISelection::MakeSelect_Prereq( const TopoDS_Shape& theContext,
						     const Handle(TDataStd_TreeNode)& AccessTreeNode)
{
  if (AccessTreeNode.IsNull() || AccessTreeNode->Label().IsNull()) {
    return Standard_False;
  }

  Handle(TNaming_NamedShape) aNS = TNaming_Tool::NamedShape(theContext, AccessTreeNode->Label());
  if(aNS.IsNull() || !aNS->Label().Father().IsAttribute(TFunction_Function::GetID())) return Standard_False;
  else return Standard_True;
}


//=======================================================================
//function : MakeSelect
//purpose  : Adds a point object to the document
//=======================================================================
Handle(TDataStd_TreeNode) OCAF_ISelection::MakeSelect_ObjectNode(const TCollection_ExtendedString& theName,
								 const Handle(TDataStd_TreeNode)& AccessTreeNode,
								 TCollection_ExtendedString& theError)
{
  if (AccessTreeNode.IsNull() || AccessTreeNode->Label().IsNull()) {
    theError = NULL_ACCESS_NODE;
    return NULL;
  }
 
  // 1. Add a new object to root label 
  //        (root label can be get from the label of "AccessTreeNode" by using "Root()"---see OCAF_IObject's AddObject() )
  Handle(TDataStd_TreeNode) anObjectNode = AddObject(AccessTreeNode->Label());

  // 2. Construct an interface "anInterface" used for call OCAF_IObject's AddFunction
  OCAF_IShape anInterface(anObjectNode);
  anInterface.SetName(theName);

  theError = DONE;

  return anObjectNode;
}


//=======================================================================
//function : MakeSelect
//purpose  : Adds a point object to the document
//=======================================================================
Handle(TDataStd_TreeNode) OCAF_ISelection::MakeSelect_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
								    TCollection_ExtendedString& theError)
{
  if (anObjectNode.IsNull() || anObjectNode->Label().IsNull()) {
    theError = NULL_ACCESS_NODE;
    return NULL;
  }

  // 2. Construct an interface "anInterface" used for call OCAF_IObject's AddFunction
  OCAF_IShape anInterface(anObjectNode);

  // 3. To use "anInterface" to call AddFunction of OCAF_IObject
  Handle(TDataStd_TreeNode) aFunctionNode = anInterface.AddFunction(GetID());
  

  if(aFunctionNode.IsNull()) {
    theError = NOTDONE;
    return NULL;
  }

  OCAF_IShape aFunctionInterface(aFunctionNode);
  aFunctionInterface.SetName("Naming function");

  theError = DONE;
  return aFunctionNode;
}


//=======================================================================
//function : MakeSelect
//purpose  : Adds a point object to the document
//=======================================================================
Standard_Boolean OCAF_ISelection::MakeSelect_Execute( const TopoDS_Shape& theShape,
						      const TopoDS_Shape& theContext,
						      TCollection_ExtendedString& theError )
{
  if( myTreeNode.IsNull()) {
    theError = NOTDONE;
    return Standard_False;
  }

  if(theShape.IsNull() || theContext.IsNull()) {
    theError = NULL_ARGUMENT;
    return Standard_False;
  }

  OCAF_ISelection ISelection(myTreeNode);
  if(!ISelection.Select(theShape, theContext)) {
    theError = UNABLE_DUE_TO_NAMING;
    return Standard_False;
  }

  theError = DONE;
  return Standard_True;
}


//=======================================================================
//function : MakeSelect
//purpose  : Adds a selection object to the document
//=======================================================================
Handle(TDataStd_TreeNode) OCAF_ISelection::MakeSelect(const TCollection_ExtendedString& theName,
						      const TopoDS_Shape& theShape,
						      const TopoDS_Shape& theContext,
						      const Handle(TDataStd_TreeNode)& AccessTreeNode,
						      TCollection_ExtendedString& theError)
{
  if(theShape.IsNull() || theContext.IsNull()) {
    theError = NULL_ARGUMENT;
    return NULL;
  }


  Handle(TDataStd_TreeNode) anObjectNode = AddObject(AccessTreeNode->Label());

  OCAF_IObject anInterface(anObjectNode);
  anInterface.SetName(theName);

  Handle(TDataStd_TreeNode) aFunctionNode = anInterface.AddFunction(GetID());

  if(anObjectNode.IsNull() || aFunctionNode.IsNull()) {
    theError = NOTDONE;
    return NULL;
  }

  OCAF_ISelection ISelection(aFunctionNode);
  ISelection.SetName("Naming function");

  if(!ISelection.Select(theShape, theContext)) {
    theError = UNABLE_DUE_TO_NAMING;
    return NULL;
  }

  theError = DONE;
  return anObjectNode;
}


//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_ISelection::OCAF_ISelection(const Handle(TDataStd_TreeNode)& aTreeNode)
  :OCAF_IShape(aTreeNode)
{
  if(myTreeNode.IsNull()) Standard_ConstructionError::Raise("Null TreeNode");  
}


//=======================================================================
//function : Select
//purpose  : Performs the topological naming of theShape belonging to theContext
//=======================================================================
/*
Standard_Boolean OCAF_ISelection::Select(const TopoDS_Shape& theShape, const TopoDS_Shape& theContext)
{
  TDF_Label aResultLabel = GetEntry().FindChild(RESULTS_TAG);
  TDF_Label aContexLabel = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(SELECTION_CONTEXT_TAG);

  // 1. Find the TNaming_NamedShape according TopoDS_Shape "theShape"
  Handle(TNaming_NamedShape) aNS = TNaming_Tool::NamedShape(theContext, myTreeNode->Label());
  // 2. check whether "aNS" is Null and check whether "aNS" is created by a Function
  if(aNS.IsNull() || !aNS->Label().Father().IsAttribute(TFunction_Function::GetID())) return Standard_False;

  // 3.1 construct a TNaming_Selector "aSelector" at aResultlabel
  TNaming_Selector aSelector(aResultLabel);
  // 3.2 Using "aSelector" to select "theShape" from "theContext"
  Standard_Boolean aStatus = aSelector.Select(theShape, theContext);

  BRepTools::Write(theShape,"SelectedShape_2.brep"); 

  Handle(TNaming_NamedShape) aResultNS = aSelector.NamedShape(); //error

  TopoDS_Shape tmpShape = aResultNS->Get();
  BRepTools::Write(tmpShape,"SelectedShape_tmp.brep"); 

  // 4.0 set a reference from "aContextLabel" to the father label(the TFunction_Function) of "aNS"
  TDF_Reference::Set( aContexLabel, aNS->Label().Father() );  //Set a argument of function (theContext)
  return aStatus;
}
//*/

/*
Standard_Boolean OCAF_ISelection::Select(const TopoDS_Shape& theShape, const TopoDS_Shape& theContext)
{
  TDF_Label aResultLabel = GetEntry().FindChild(RESULTS_TAG);
  TDF_Label aContexLabel = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(SELECTION_CONTEXT_TAG);

  // 1. Find the TNaming_NamedShape according TopoDS_Shape "theShape"
  Handle(TNaming_NamedShape) aNS = TNaming_Tool::NamedShape(theContext, myTreeNode->Label());
  // 2. check whether "aNS" is Null and check whether "aNS" is created by a Function
  if(aNS.IsNull() || !aNS->Label().Father().IsAttribute(TFunction_Function::GetID())) return Standard_False;

  // 3.1 construct a TNaming_Selector "aSelector" at aResultlabel
  TNaming_Selector aSelector(aResultLabel);
  // 3.2 Using "aSelector" to select "theShape" from "theContext"
  Standard_Boolean aStatus = aSelector.Select(theShape, theContext);

  TNaming_Builder Builder(aResultLabel);
  Builder.Generated(theShape);
  TopoDS_Shape tmpShape = (Builder.NamedShape())->Get();
  BRepTools::Write(tmpShape,"SelectedShape_tmp.brep"); 

  // 4.0 set a reference from "aContextLabel" to the father label(the TFunction_Function) of "aNS"
  TDF_Reference::Set( aContexLabel, aNS->Label().Father() );  //Set a argument of function (theContext)
  return aStatus;
}
//*/


Standard_Boolean OCAF_ISelection::Select(const TopoDS_Shape& theShape, const TopoDS_Shape& theContext)
{
  TDF_Label aResultLabel = GetEntry().FindChild(RESULTS_TAG);
  TDF_Label aContexLabel = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(SELECTION_CONTEXT_TAG);

  // 1. Find the TNaming_NamedShape according TopoDS_Shape "theShape"
  Handle(TNaming_NamedShape) aNS = TNaming_Tool::NamedShape(theContext, myTreeNode->Label());
  // 2. check whether "aNS" is Null and check whether "aNS" is created by a Function
  if(aNS.IsNull() || !aNS->Label().Father().IsAttribute(TFunction_Function::GetID())) return Standard_False;

  // 3.1 construct a TNaming_Selector "aSelector" at aResultlabel
  TNaming_Selector aSelector(aResultLabel);
  // 3.2 Using "aSelector" to select "theShape" from "theContext"
  Standard_Boolean aStatus = aSelector.Select(theShape, theContext);

  // 4.0 set a reference from "aContextLabel" to the father label(the TFunction_Function) of "aNS"
  TDF_Reference::Set( aContexLabel, aNS->Label().Father() );  //Set a argument of function (theContext)
  return aStatus;
}


//=======================================================================
//function : GetContext
//purpose  : Returns the context shape 
//=======================================================================
TopoDS_Shape OCAF_ISelection::GetContext() const
{
  TopoDS_Shape aShape;
  Handle(TNaming_NamedShape) aNamedShape;
  Handle(TDF_Reference) aReference;

  TDF_Label aContexLabel = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(SELECTION_CONTEXT_TAG);

  if(aContexLabel.FindAttribute(TDF_Reference::GetID(), aReference)) {
    const TDF_Label& aLabel = aReference->Get().FindChild(RESULTS_TAG, Standard_False);
    if(aLabel.FindAttribute(TNaming_NamedShape::GetID(), aNamedShape))
      return aNamedShape->Get();
  }

  return aShape;
}
