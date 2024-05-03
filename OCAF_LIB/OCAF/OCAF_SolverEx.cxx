#include <CAGDDefine.hxx>

#include <OCAF_SolverEx.hxx>

#include <OCAF_Driver.hxx>
#include <OCAF_Object.hxx>
#include <OCAF_IFunction.hxx>
#include <OCAF_ISelection.hxx>


#include <TDataStd_ChildNodeIterator.hxx>
#include <TDF_LabelMap.hxx>
#include <TDF_AttributeMap.hxx>
#include <TDF_MapIteratorOfLabelMap.hxx>
#include <TDF_ListIteratorOfAttributeList.hxx>

#include <TFunction_DriverTable.hxx>
#include <TDocStd_Modified.hxx>
#include <TFunction_Function.hxx>
#include <TFunction_Logbook.hxx>
#include <TCollection_AsciiString.hxx>



//=======================================================================
//function : SolverEx
//purpose  : Constructor
//=======================================================================
OCAF_SolverEx::OCAF_SolverEx()
{}

//=======================================================================
//function : ComputeExecutionList
//purpose  : to compute all execution list of all model (by interate the root label)
//=======================================================================
void OCAF_SolverEx::ComputeExecutionList(const TDF_Label& theAccessLabel)
{
  //1. get the root label "aRootLabel" and the aRootLabel's TreeNode "aRoot"
  TDF_Label aRootLabel = theAccessLabel.Root();
  Handle(TDataStd_TreeNode) aRoot;
  if (!aRootLabel.FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aRoot)) {
    return;
  }  
  
  myList.Clear();
  
  TDF_AttributeMap aMap;
  Handle(TDataStd_TreeNode) aNode;
  Handle(TFunction_Function) aFunction;

  //2. define an iterator "anIterator"; prepare to iterate the root label's TreeNode "aRoot" 
  TDataStd_ChildNodeIterator anIterator(aRoot, Standard_True);
  
  for(; anIterator.More(); anIterator.Next()) {
    aNode = anIterator.Value();

    //3.1  if "aNode" ia an "OCAF_Object", then check the children of "aNode"
    if(aNode->IsAttribute(OCAF_Object::GetObjectID())) {
      aNode = aNode->First();

      if(aNode.IsNull()) continue;

      //3.2 check whether "aNode" has the attribute of "TFunction_Function"
      while(!aNode.IsNull()) {
	if(aNode->Label().FindAttribute(TFunction_Function::GetID(), aFunction))
	  //3.3 if "aNode" has the attribute of "TFunction_Function", then call "ComputeFunction"
	  //3.3 by calling "ComputeFunction" to decide "aMap"
	  ComputeFunction(aFunction, aMap);
	aNode = aNode->Next();
      }
    }
  }
}

//=======================================================================
//function : Solve
//purpose  : Rebuilds the whole model stored in a document given by theAccessLabel
//=======================================================================
Standard_Boolean OCAF_SolverEx::Solve(const TDF_Label& theAccessLabel)
{
  TDF_Label aRootLabel = theAccessLabel.Root();
  Handle(TDataStd_TreeNode) aRoot;
  if (!aRootLabel.FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aRoot)) {
    return Standard_False;
  }  

  //Clear the modifications
  TDocStd_Modified::Clear(aRootLabel);
  myList.Clear();

  TDF_AttributeMap aMap;
  TDataStd_ChildNodeIterator anIterator(aRoot, Standard_True);
  Handle(TFunction_Function) aFunction;
  Handle(TDataStd_TreeNode) aNode;
  Handle(TFunction_Logbook) aLog;

  //Build myList that contains an order of the execution of functions 
  ComputeExecutionList(theAccessLabel);

  return PerformSolving(aLog, NULL, Standard_False); 
}

//=======================================================================
//function : SolveFrom
//purpose  : Rebuilds the model starting from given by theNode object
//=======================================================================
Standard_Boolean OCAF_SolverEx::SolveFrom(const Handle(TDataStd_TreeNode)& theNode)
{
  if(theNode.IsNull()) return Standard_False;
  TDocStd_Modified::Clear(theNode->Label());  
  myList.Clear();

  TDF_AttributeMap aMap;
  Handle(TFunction_Driver) aDriver;
  Handle(TFunction_Logbook) aLog;
  Handle(TDataStd_TreeNode) aNode;

  //TDataStd_ChildNodeIterator anIterator(theNode->Root(), Standard_True);

  Handle(TFunction_Function) aFunction;

  //1. Build an execution list(a map with arguments of TFunction_Function) for whole model. Not optimal but simple solution
  ComputeExecutionList(theNode->Label());

  //2.1 Find the driver from the DriverTable by aFunction's ID
  if(!theNode->FindAttribute(TFunction_Function::GetID(), aFunction)) return Standard_False;
  TFunction_DriverTable::Get()->FindDriver(aFunction->GetDriverGUID(), aDriver); 
  Handle(OCAF_Driver) anOCAFDriver = Handle(OCAF_Driver)::DownCast(aDriver);

  //2.2 ReExecute the function of "theNode"
  Standard_Integer aStatus;
  if(!anOCAFDriver.IsNull()) {
    anOCAFDriver->Init(aFunction->Label());
    if((aStatus = anOCAFDriver->Execute(aLog)) > 0) {
      return Standard_False;
    }
  }

  //3. Recompute other functions which depends on this function (aFunction)
  //3.1 The order of the execution is given by myList.
  //3.2 Only functions with modified arguments will be recomputed (check MustExecute) 
  return PerformSolving(aLog, aFunction, Standard_True);
}

//=======================================================================
//function : IsCyclicLink
//purpose  : Returns True if link between theFrom and theTo is cyclic
//=======================================================================
Standard_Boolean OCAF_SolverEx::IsCyclicLink(const TDF_Label& theFrom, 
					     const TDF_Label& theTo)
{

  Handle(TFunction_Function) aFromFunction, aToFunction;  
  TDF_AttributeMap aMap;

  //Find functions
  theFrom.FindAttribute(TFunction_Function::GetID(), aFromFunction);
  theTo.FindAttribute(TFunction_Function::GetID(), aToFunction);


  //Find dependencies of the aToFunction
  if(aFromFunction.IsNull() || aToFunction.IsNull()) return Standard_False;

  ComputeFunction(aToFunction, aMap);

  if(aMap.Contains(aFromFunction)) return Standard_True;
  return Standard_False;
}



//=======================================================================
//function : ComputeFunction
//purpose  : Finds dependecies of the function
//=======================================================================
Standard_Boolean OCAF_SolverEx::ComputeFunction(const Handle(TFunction_Function)& theFunction, 
						TDF_AttributeMap& theSolved)
{
  //1. check whether theSolved has already contained "theFunction"
  if(theSolved.Contains(theFunction)) return Standard_True;

  //2.1 Find function driver from DriverTable
  Handle(TFunction_Driver) aDriver;
  TFunction_DriverTable::Get()->FindDriver(theFunction->GetDriverGUID(), aDriver); 
  Handle(OCAF_Driver) anOCAFDriver = Handle(OCAF_Driver)::DownCast(aDriver);

  //2.2 if do not find the accoresponding Function Driver, then append a NULL TFunction_Function
  if(anOCAFDriver.IsNull()) {
    theSolved.Add(theFunction);
    myList.Append(theFunction);
    return Standard_True;
  }

  TDF_LabelMap aMap;

  //3.1 init the Dirver "anOCAFDriver" to theFunction' label 
  anOCAFDriver->Init(theFunction->Label());

  //3.2 set the arguments (all direct and Referenced arguments) of the function to "aMap"
  anOCAFDriver->Arguments(aMap);

  //4. iterator all the arguments of "theFunction"
  TDF_MapIteratorOfLabelMap anIterator(aMap);

  for(; anIterator.More(); anIterator.Next()) {
    Handle(TFunction_Function) aFunction;
    if(anIterator.Key().FindAttribute(TFunction_Function::GetID(), aFunction)) {
      //if(theSolved.Contains(aFunction)) continue;  // removed 2016.04.22
      //4.1 Compute all function that produce arguments
      ComputeFunction(aFunction, theSolved); 
    }
  }

  theSolved.Add(theFunction);
  myList.Append(theFunction);

  return Standard_True;
}


//=======================================================================
//function : PerformSolving
//purpose  : Calls driver method Execute for functions
//=======================================================================
Standard_Boolean OCAF_SolverEx::PerformSolving(Handle(TFunction_Logbook)& theLogbook,
					       const Handle(TFunction_Function)& theSkipFunction,
					       const Standard_Boolean theWithCheck) const 
{
  Handle(TFunction_Function) aFunction;
  Handle(TFunction_Driver) aDriver;
  Standard_Integer aStatus;

  TDF_ListIteratorOfAttributeList anIterator(myList);

  //1. iterator all the functions (TFunction_Function) stored in myList

  for(; anIterator.More(); anIterator.Next()) {
    aFunction = Handle(TFunction_Function)::DownCast(anIterator.Value());
    if(aFunction.IsNull()) return Standard_False; 

    //1.1  skip the funcion that already computed in SolveFrom method
    if(aFunction == theSkipFunction) continue; 

    TFunction_DriverTable::Get()->FindDriver(aFunction->GetDriverGUID(), aDriver); 
    Handle(OCAF_Driver) anOCAFDriver = Handle(OCAF_Driver)::DownCast(aDriver);

    //1.2 execute the functions(stored in myList) exept "theSkipFuncion"
    if(!anOCAFDriver.IsNull()) {
      anOCAFDriver->Init(aFunction->Label());

      if(theWithCheck && !anOCAFDriver->MustExecute(theLogbook)) continue; //No need to execute
      if((aStatus = anOCAFDriver->Execute(theLogbook)) > 0) {
	return Standard_False;
      }
    }
    else{
      cout<<"this execution of function is NULL, Perhaps you need to add something in \" OCAF_Application.cxx \" "<<endl;
    }
  }
  return Standard_True;
}


//=======================================================================
//function : GetExecutionList
//purpose  : Retruns an ordered list of functions 
//=======================================================================
const TDF_AttributeList& OCAF_SolverEx::GetExecutionList() const
{
  return myList;
}

//=======================================================================
//function : GetAttachments
//purpose  : Fills the list with attachemnts of the function
//=======================================================================
void OCAF_SolverEx::GetAttachments(const Handle(TFunction_Function)& theFunction, 
				   TDF_AttributeMap& theMap) const
{

  Handle(TDataStd_TreeNode) aNode;
  Handle(TFunction_Function) aFunction;
  Handle(TDF_Reference) aRef;
  if(theFunction.IsNull()) return;

  // 1. To find the Function Node
  if(!theFunction->FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return;

  // 2. Iterate all the children (with all children's children) of the Root Node
  TDataStd_ChildNodeIterator anIterator(aNode->Root(), Standard_True);
  for(; anIterator.More(); anIterator.Next()) {
    // 2.1 if this iterator is a function
    if(anIterator.Value()->FindAttribute(TFunction_Function::GetID(), aFunction)) {
      if(aFunction == theFunction) continue;
      // 2.2 if the function is a Selection Function
      if(aFunction->GetDriverGUID() == OCAF_ISelection::GetID()) {
	TDF_ChildIterator anItr(aFunction->Label().FindChild(ARGUMENTS_TAG));
	for(; anItr.More(); anItr.Next()) {

	  if(anItr.Value().FindAttribute(TDF_Reference::GetID(), aRef)) {
	    // 2.2.1 if an object is selected shape from the theFunction's result then add the object to theMap
	    if(aRef->Get() == theFunction->Label()) {
	      if(!aFunction->FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) continue;
	      theMap.Add(aNode); 
	      break;
	    }
	  }

	}
      }
    }
  }
  return;
}



//=======================================================================
//function : Dump
//purpose  : Service function
//=======================================================================
void OCAF_SolverEx::Dump() const
{
  TCollection_AsciiString anEntry;
  TDF_ListIteratorOfAttributeList anIterator(myList);
  
  cout << "Execution list contains the following functions: " << endl;
  for(; anIterator.More(); anIterator.Next()) {
    TDF_Tool::Entry(anIterator.Value()->Label(), anEntry);
    cout << "   " << anEntry << endl;
  }
}



