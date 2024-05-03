
#include "OCAF_BRepImportDriver.ixx"
#include <OCAF_IFunction.hxx>
#include <Tags.hxx>

#include <TDocStd_Modified.hxx>

#include <TDF_ChildIterator.hxx>

#include <TDataStd_TreeNode.hxx>

#define OK_OPERATION 0
#define TREENODE_NOT_FOUND 1;

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================

OCAF_BRepImportDriver::OCAF_BRepImportDriver()
{
}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================

Standard_Integer OCAF_BRepImportDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const
{
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return TREENODE_NOT_FOUND;

  // Name result
  //TDF_Label aResultLabel = Label().FindChild(RESULTS_TAG);

  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  return OK_OPERATION;
}
