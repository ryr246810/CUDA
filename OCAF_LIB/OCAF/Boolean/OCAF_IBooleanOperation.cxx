
#include <OCAF_IBooleanOperation.hxx>

#include <Tags.hxx>

#include <TDF_ChildIterator.hxx>
#include <TDF_Data.hxx>
#include <TDF_Label.hxx>
#include <TDF_Reference.hxx>

#include <TDocStd_Modified.hxx>

#include <TFunction_Function.hxx>

#include <TNaming_Tool.hxx>
#include <TNaming_NamedShape.hxx>

#include <Standard_ConstructionError.hxx>

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================

OCAF_IBooleanOperation::OCAF_IBooleanOperation(const Handle(TDataStd_TreeNode)& aTreeNode)
:OCAF_IFunction(aTreeNode)
{
  if(myTreeNode.IsNull()) Standard_ConstructionError::Raise("Null TreeNode");  
}

