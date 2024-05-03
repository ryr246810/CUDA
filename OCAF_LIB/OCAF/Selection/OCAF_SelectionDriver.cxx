#include <CAGDDefine.hxx>

#include "OCAF_SelectionDriver.ixx"
#include <OCAF_ISelection.hxx>
#include <BRepNaming_Selection.hxx>
#include <OCAF_IFunction.hxx>

#include <Tags.hxx>

#include <TDF_ChildIterator.hxx>

#include <TDataStd_Real.hxx>
#include <TDataStd_TreeNode.hxx>
#include <TDocStd_Modified.hxx>
#include <TNaming_Selector.hxx>
#include <TDF_LabelMap.hxx>
#include <TDF_Tool.hxx>
#include <TDF_MapIteratorOfLabelMap.hxx>
#include <TCollection_AsciiString.hxx>
#include <TNaming_NamedShape.hxx>
#include <TopoDS_Shape.hxx>
#include <gp_Pnt.hxx>

#include <TNaming_Tool.hxx>

#include <TopExp_Explorer.hxx>


#define OK_SELECTION 0
#define EMPTY_SELECTION 1
#define SELECTION_NOT_DONE 2
#define NULL_SELECTION 3

//#include <BRepTools.hxx>

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_SelectionDriver::OCAF_SelectionDriver():OCAF_Driver() {}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_SelectionDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  //1. check the Function Node of this driver
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_SELECTION;

  //2.1 get the RESULTS_TAG's label "aResultLabel"
  TDF_Label aResultLabel = Label().FindChild(RESULTS_TAG);

  //2.2 constructor a TNaming_Selector on "aResultLabel"
  TNaming_Selector aSelector(aResultLabel);
  TDF_LabelMap aMap;
  aMap = theLogbook->GetImpacted();

  if(!aSelector.Solve(aMap)) return SELECTION_NOT_DONE;

  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  return OK_SELECTION;
}




/*
Standard_Integer OCAF_SelectionDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{

  //1. check the Function Node of this driver
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_SELECTION;

  //2.1 get the RESULTS_TAG's label "aResultLabel"
  TDF_Label aResultLabel = Label().FindChild(RESULTS_TAG);

  //2.2 constructor a TNaming_Selector on "aResultLabel"
  TNaming_Selector aSelector(aResultLabel);
  TDF_LabelMap aMap;
  aMap = theLogbook->GetImpacted();

  if(!aSelector.Solve(aMap)) return SELECTION_NOT_DONE;


#ifdef DEB
  TCollection_AsciiString anEntry;
  TDF_Tool::Entry(aNode->Label(), anEntry);
//  cout << "The impacted map for " << anEntry << " contains: " << endl; 
  TDF_MapIteratorOfLabelMap Itr(aMap);
  for(; Itr.More(); Itr.Next()) {
	TDF_Tool::Entry(Itr.Key(), anEntry);
//	cout << "   => " << anEntry << endl; 
  }
#endif

  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  return OK_SELECTION;
}
*/
